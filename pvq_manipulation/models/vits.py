"""
This is a wrapper for the TTS VITS model.
TTS.tts.models.vits
https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/vits.py
"""
import os
import numpy as np
import paderbox as pb
import padertorch as pt
import torch

from coqpit import Coqpit
from padertorch.ops._stft import STFT
from pathlib import Path
from pvq_manipulation.helper.utils import VitsAudioConfig_NT, VitsConfig_NT, load_audio

from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.models.vits import Vits, VitsArgs, VitsDataset, spec_to_mel, wav_to_spec
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import embedding_to_torch, numpy_to_torch
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.helpers import generate_path, rand_segments, segment, sequence_mask
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from trainer.trainer import to_cuda
from typing import Dict, List, Union

if not torch.cuda.is_available():
    device = 'cpu'
    from concurrent.futures import ThreadPoolExecutor
else:
    device = 'cuda'


STORAGE_ROOT = Path(os.getenv('STORAGE_ROOT')).expanduser()


class Vits_NT(Vits):
    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
        sample_rate: int = None,
    ):
        super().__init__(
            config,
            ap,
            tokenizer,
            speaker_manager,
            language_manager
        )
        self.sample_rate = sample_rate
        self.embedded_speaker_dim = self.args.d_vector_dim
        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )
        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim if self.config.gan_speaker_conditioning else 0,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )
        self.speaker_manager = self.speaker_manager
        self.speaker_encoder = self.speaker_manager

        self.speaker_manager.eval()

        self.epoch = 0
        self.num_epochs = config['epochs']
        self.lr_lambda = 0
        self.config_solver = config['CONFIG_SOLVER']
        self.config = config

        self.stft = STFT(
            size=self.config.audio.win_length,
            shift=self.config.audio.hop_length,
            window_length=self.config.audio.win_length,
            fading=self.config.audio.fading,
            window=self.config.audio.window,
            pad=self.config.audio.pad
        )

    def get_spectogram_nt(self, wav):
        """
        Extracts spectrogram from audio
        Args:
            wav (torch.Tensor): (Batch_size, Num_samples)
        Returns:
            spectrogram (torch.Tensor): (Batch_size, Frequency_bins, Time) spectrogram
        """
        wav = wav.squeeze(1)
        stft_signal = self.stft(wav)
        stft_signal = torch.einsum('btf-> bft', stft_signal)
        spectrogram = stft_signal.real ** 2 + stft_signal.imag ** 2
        spectrogram = torch.sqrt(spectrogram + 1e-6)
        return spectrogram

    @staticmethod
    def normalize_d_vectors(d_vector, file_path):
        global_mean = pb.io.load(file_path / "mean.json")
        global_mean = torch.tensor(global_mean, dtype=torch.float32)
        d_vector = (d_vector - global_mean)
        d_vector = d_vector / torch.linalg.norm(d_vector, keepdim=True, dim=-1)
        return d_vector

    def get_aux_input_from_test_sentences(self, sentence_info):
        """
        Get aux input for the inference step from test sentences
        Args:
            sentence_info (dict): Expected keys:
                - "d_vector_storage_root" (str)
                - "d_vector" (torch.Tensor)
                - "d_vector_man" (torch.Tensor) (optional)
        Returns:
            aux_input (dict): aux input for the inference step
        """
        if 'd_vector' not in sentence_info.keys():
            d_vector_file = sentence_info['d_vector_storage_root']
            d_vector = torch.load(d_vector_file)
            return {"d_vector": d_vector, **sentence_info}
        else:
            return sentence_info

    @staticmethod
    def init_from_config(
        config: "VitsConfig",
        samples=None,
        verbose=True
    ):
        """
        Initiate model from config
        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        Returns:
            model (Vits): Initialized model.
        """

        
        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()
        assert (upsample_rate == config.audio.hop_length), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        language_manager = LanguageManager.init_from_config(config)
        speaker_manager_config = pb.io.load(Path(config['d_vector_model_file'])/'config.json')
        
        speaker_manager = pt.Configurable.from_config(speaker_manager_config)
        speaker_manager.load_state_dict(
            torch.load(
                Path(config['d_vector_model_file'])/"model.pt", 
                weights_only=True, 
                map_location=device
                )
            )
        speaker_manager.num_speakers = config['num_speakers']
        for param in speaker_manager.parameters():
            param.requires_grad = False

        return Vits_NT(
            new_config,
            ap,
            tokenizer,
            speaker_manager=speaker_manager,
            language_manager=language_manager,
            sample_rate=config['sample_rate'],
        )

    @torch.no_grad()
    def inference(self, x, aux_input=None):
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Args:
            x (torch.Tensor): (batch_size, T_seq) or (T_seq) Input character sequence IDs
            aux_input (dict): Expected keys:
                - d_vector (torch.Tensor): (batch_size, Feature_dim) speaker_embedding
                - x_lengths: (torch.Tensor): (batch_size) length of each text token

        Returns:
            - model_outputs (torch.Tensor): (batch_size, T_wav) Synthesized waveform
        """
        speaker_embedding = aux_input['d_vector']
        if self.config.normalize_vectors:
            speaker_embedding = self.normalize_d_vectors(
                speaker_embedding,
                Path(aux_input['d_vector_storage_root']).parent.parent
            )

        if 'd_vector_man' in aux_input.keys() and aux_input['d_vector_man'] is not None:
            speaker_embedding_man = aux_input['d_vector_man']
        else:
            speaker_embedding_man = speaker_embedding

        aux_input['tokens'] = x
        x_lengths = self._set_x_lengths(x, aux_input)
        x, m_p, logs_p, x_mask = self.text_encoder(
            x,
            x_lengths,
            lang_emb=None
        )
        logw = self.duration_predictor(
            x,
            x_mask,
            g=speaker_embedding[:, :, None],
            lang_emb=None,
        )

        w = torch.exp(logw) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.einsum('blm, bnl -> bnm', attn, m_p)
        logs_p = torch.einsum('blm, bnl -> bnm', attn, logs_p)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale

        z = self.flow(z_p, y_mask, g=speaker_embedding_man[:, :, None], reverse=True)
        z, _, _, y_mask = self.upsampling_z(
            z,
            y_lengths=y_lengths,
            y_mask=y_mask
        )

        if not torch.cuda.is_available():
            num_chunks = 2
            chunk_size = z.shape[-1] // num_chunks
            z_chunks = torch.split(z, chunk_size, dim=-1)

            def decode_chunk(z_chunk):
                return self.waveform_decoder(
                    z_chunk,
                    g=speaker_embedding_man[:, :, None] if self.config.gan_speaker_conditioning else None
                )

            with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                futures = [executor.submit(decode_chunk, chunk) for chunk in z_chunks]
                results = [f.result() for f in futures]

            o = torch.cat(results, dim=-1)

        else:
            o = self.waveform_decoder(
                (z * y_mask)[:, :, : self.max_inference_len],
                g=speaker_embedding_man[:, :, None] if self.config.gan_speaker_conditioning else None
            )
        return o

    def forward(self, x, x_lengths, y, y_lengths, aux_input, inference=False):
        """
        Forward pass of the model.

        Args:
            x (torch.tensor): (Batch, T_seq) Input character sequence IDs
            x_lengths (torch.tensor): (Batch) Input character sequence lengths.
            y (torch.tensor): (Batch_size, Frequency_bins, Time) Input spectrograms.
            y_lengths (torch.tensor): (Batch) Input spectrogram lengths.
            aux_input (dict, optional): Expected keys:
                - d_vector (torch.Tensor): (batch_size, Feature_dim) speaker_embedding
                - waveform: (torch.Tensor): (Batch_size, Num_samples) Target waveform
        Returns:
            Dict: model outputs keyed by the output name.
        """
        outputs = {}
        speaker_embedding = aux_input['d_vector'].detach()[:, :, None]
        x, m_p, logs_p, x_mask = self.text_encoder(
            x,
            x_lengths,
            lang_emb=None
        )
        z, m_q, logs_q, y_mask = self.posterior_encoder(
            y,
            y_lengths,
            g=speaker_embedding,
        )
        z_p = self.flow(z, y_mask, g=speaker_embedding)
        outputs, attn = self.forward_mas(
            outputs,
            z_p,
            m_p,
            logs_p,
            x,
            x_mask,
            y_mask,
            g=speaker_embedding,
            lang_emb=None,
        )
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        z_slice, slice_ids = rand_segments(
            z,
            y_lengths,
            self.spec_segment_size,
            let_short_samples=True,
            pad_short=True
        )
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(
            z_slice,
            slice_ids=slice_ids,
        )

        wav_seg = segment(
            aux_input['waveform'],
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )
        o = self.waveform_decoder(
            z_slice,
            g=speaker_embedding if self.config.gan_speaker_conditioning else None
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            wavs_batch = torch.cat((wav_seg, o), dim=0)
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)
            with torch.no_grad():
                pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
                "z_slice": z_slice,
                "speaker_embedding": speaker_embedding,
            }
        )
        return outputs

    @staticmethod
    def load_model(model_path, checkpoint):
        """
        Load model from checkpoint

        Args:
            model_path (str): model path
            checkpoint (str): checkpoint name

        Returns:
            model (pvq_manipulation.models.vits.Vits_NT): model
        """
        config = pb.io.load_json(model_path / "config.json")
        model_args = VitsArgs(**config['model_args'])
        audio_config = VitsAudioConfig_NT(**config['audio'])
        characters_config = CharactersConfig(**config['characters'])
        del config['audio']
        del config['characters']
        del config['model_args']

        config = VitsConfig_NT(
            model_args=model_args,
            audio=audio_config,
            characters=characters_config,
            **config,
        )
        model = Vits_NT.init_from_config(config)
        model_weights = torch.load(
            model_path / checkpoint,
            map_location=torch.device(device)
        )
        model.load_state_dict(model_weights, strict=False)
        model.to(device)
        model.eval()
        return model

    def synthesize_from_example(self, s_info):
        """
        Synthesize voice from example

        Args:
            s_info (dict): Expected keys:
                - "speaker_id" (str),
                - "example_id" (str),
                - "audio_path" (str),
                - "d_vector_storage_root" (str),
                - "text" (str) specifying the text to synthesize
        Returns:
            wav (torch.Tensor): synthesized waveform
        """
        aux_inputs = self.get_aux_input_from_test_sentences(s_info)
        use_cuda = "cuda" in str(next(self.parameters()).device)

        device = next(self.parameters()).device
        if use_cuda:
            device = "cuda"

        text_inputs = np.asarray(
            self.tokenizer.text_to_ids(aux_inputs["text"], language=None),
            dtype=np.int32,
        )
        if isinstance(aux_inputs["d_vector"], np.ndarray):
            aux_inputs["d_vector"] = embedding_to_torch(aux_inputs["d_vector"], device=device)
        else:
            aux_inputs["d_vector"] = aux_inputs["d_vector"].to(device)

        if "d_vector_man" in aux_inputs.keys(): 
            if isinstance(aux_inputs["d_vector_man"], np.ndarray):
                aux_inputs["d_vector_man"] = embedding_to_torch(aux_inputs["d_vector_man"], device=device)
            else:
                aux_inputs["d_vector_man"] = aux_inputs["d_vector_man"].to(device)

        text_inputs = numpy_to_torch(text_inputs, torch.long, device=device)
        text_inputs = text_inputs.unsqueeze(0)

        wav = self.inference(
            text_inputs,
            aux_input={
                "x_lengths": torch.tensor(
                    text_inputs.shape[1:2]
                ).to(text_inputs.device),
                **aux_inputs
            }
        )[0].data.cpu().numpy().squeeze()
        return wav

    def format_batch_on_device(self, batch):
        """Format batch on device"""
        ac = self.config.audio

        batch['waveform'] = to_cuda(batch['waveform'])
        wav = batch["waveform"]

        batch['spec'] = self.get_spectogram_nt(wav)

        if self.args.encoder_sample_rate:
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def train_step(
        self,
        batch: dict,
        criterion: torch.nn.Module,
        optimizer_idx: int,
    ):
        """
        Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        if optimizer_idx == 0:
            # generator pass
            outputs = self.forward(
                batch["tokens"],
                batch["token_lens"],
                batch["spec"],
                batch["spec_lens"],
                aux_input={
                    **batch,
                },
            )
            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(),
                outputs["waveform_seg"]
            )
            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            # compute melspec segment
            with autocast(enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size
                mel_slice = segment(
                    batch["mel"].float(),
                    self.model_outputs_cache["slice_ids"],
                    spec_segment_size,
                    pad_short=True
                )

                spec = self.get_spectogram_nt(
                    self.model_outputs_cache["model_outputs"].float(),
                )
                mel_slice_hat = spec_to_mel(
                    spec=spec,
                    n_fft=self.config.audio.fft_size,
                    num_mels=self.config.audio.num_mels,
                    sample_rate=self.config.audio.sample_rate,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"],
                self.model_outputs_cache["waveform_seg"],
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=batch["spec_lens"],
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                )
            return self.model_outputs_cache, loss_dict
        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    @torch.no_grad()
    def test_run(self, assets):
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            wav = self.synthesize_from_example(s_info)
            test_audios["{}-audio".format(idx)] = wav
        return {"figures": test_figures, "audios": test_audios}

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        dataset = VitsDataset_NT(
            model_args=self.args,
            speaker_manager=self.speaker_manager,
            config=self.config,
            use_phone_labels=config.use_phone_labels,
            sample_rate=self.sample_rate,
            samples=samples,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            verbose=verbose,
            tokenizer=self.tokenizer,
            start_by_longest=config.start_by_longest,
        )

        # sort input sequences from short to long
        dataset.preprocess_samples()

        # get samplers
        sampler = self.get_sampler(config, dataset, num_gpus)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
        )
        return loader


class VitsDataset_NT(VitsDataset):
    def __init__(
        self,
        model_args,
        speaker_manager,
        sample_rate,
        config,
        use_phone_labels,
        *args,
        **kwargs
    ):
        super().__init__(model_args, *args, **kwargs)
        self.speaker_manager = speaker_manager
        self.sample_rate = sample_rate
        self.config = config
        self.use_phone_labels = use_phone_labels

    def __getitem__(self, idx):
        example = self.samples[idx]
        token_ids = self.get_token_ids(idx, example["text"])

        wav, _ = load_audio(example["audio_file"], target_sr=self.sample_rate)

        speaker_id = example['speaker_name']
        example_id = example['example_id']
        d_vector = None
        for dataset_dict_sub in self.config.dataset_dict['datasets'].values():
            d_vector_file = dataset_dict_sub['d_vector_storage_root']
            if (Path(d_vector_file) / f'{speaker_id}/{example_id}.pth').is_file():
                d_vector = torch.load(Path(d_vector_file) / f'{speaker_id}/{example_id}.pth')
                break
        if d_vector is None:
            raise ValueError(f'Could not find d_vector for example {example_id}')

        if d_vector.dim() == 1:
            d_vector = d_vector[None, :]
        return {
            "raw_text": example['text'],
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "d_vector": d_vector,
            "speaker_name": example["speaker_name"]
        }

    def collate_fn(self, batch):
        """
        Collate a list of samples from a Dataset into a batch for VITS.

        Args:
            batch (dict): Expeted keys:
                - wav (list): list of tensors
                - token_ids (list):
                - token_len (list):
                - speaker_name (list):
                - language_name (list):
                - audiofile_path (list):
                - raw_text (list):
                - wav_d_vector (list):
        Returns:
            - tokens (torch.Tensor): (B, T)
            - token_lens (torch.Tensor): (B)
            - token_rel_lens (torch.Tensor): (B)
            - wav (torch.Tensor): (B, 1, T)
            - wav_lens (torch.Tensor): (B)
            - wav_rel_lens (torch.Tensor): (B)
            - speaker_names (torch.Tensor): (B)
            - language_names (torch.Tensor): (B)
            - audiofile_paths (torch.Tensor): (B)
            - raw_texts (torch.Tensor): (B)
            - audio_unique_names (torch.Tensor): (B)
        """
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(
                [
                    x.size(1) for x in batch["wav"]]
            ),
            dim=0,
            descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)
            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,
            "waveform_lens": wav_lens,
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "raw_text": batch["raw_text"],
            "d_vector": torch.concatenate(batch["d_vector"]) if 'd_vector' in batch.keys() else None,
        }
