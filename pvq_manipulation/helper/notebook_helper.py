import numpy as np 
import librosa
import torch
import paderbox as pb
import padertorch as pt

from collections import defaultdict
from onnxruntime import InferenceSession
from pathlib import Path
from pvq_manipulation.helper.vad import EnergyVAD

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def get_speaker_labels( 
    speaker_labels, 
    pvq_labels,
    manipulation=None,
    manipulation_intensity=0,
    config=None,
    stacked_flow=False,
):
    """
    Build normalized speaker-conditioning labels for flow-based voice manipulation.

    The function reads raw speaker label values, applies label-specific normalization,
    and optionally adds an offset to one selected manipulation attribute.

    Args:
        speaker_labels (dict): Mapping from label name to raw attribute value.
        pvq_labels (list[str]): Ordered list of PVQ labels used as conditioning.
        manipulation (str | None): Label to modify (e.g. "Breathiness").
            If None, no manipulation is applied.
        manipulation_intensity (float): Additive offset for the selected
            manipulation label.
        config (dict | None): Model configuration. Required when
            stacked_flow=True because high-level condition groups are read from
            config['model']['condition_list'].
        stacked_flow (bool): If True, return grouped conditioning tensors per
            flow level. If False, return one flat conditioning tensor.

    Returns:
        torch.Tensor | dict[int, torch.Tensor]:
        - If stacked_flow=False: tensor with shape (1, num_labels).
        - If stacked_flow=True: dictionary mapping flow-level index to tensors
          of shape (1, num_conditions_in_level).
    """
    speaker_conditioning = []
    for label_key in pvq_labels:
        if label_key == 'pitch_mean':
            attribute = speaker_labels[label_key] / 400
        elif label_key in ['Gender', 'Creak_mean']:
            attribute = speaker_labels[label_key] 
        else:
            attribute = speaker_labels[label_key] / 100
        if manipulation is not None and label_key == manipulation:
            attribute += manipulation_intensity
        speaker_conditioning.append(attribute)
    
    if stacked_flow:
        speaker_conditioning_dict = defaultdict(list)
        for idx_high_level, conditions in enumerate(config['model']['condition_list']):
            for condition in conditions:
                for idx, label_key in enumerate(pvq_labels):
                    if label_key == condition:
                        speaker_conditioning_dict[idx_high_level].append(
                            speaker_conditioning[idx]
                        )
        for idx, speaker_conditioning in speaker_conditioning_dict.items():
            speaker_conditioning = torch.tensor([speaker_conditioning], device=device, dtype=torch.float)
            speaker_conditioning_dict[idx] = speaker_conditioning
        return speaker_conditioning_dict
    else:
        return torch.tensor(speaker_conditioning)[None, :]
 
 
def get_manipulation(
    transcription,
    pvq_labels,
    speaker_labels, 
    flow, 
    tts_model,    
    d_vector,
    manipulation,
    manipulation_intensity=1,
    stacked_flow=False,
    config=None,
):
    """
    Synthesizes manipulated speech based on the given manipulation type and intensity.
    Args:
        transcription (str): The text transcription to be synthesized.
        labels (torch.Tensor): (1, num_labels) The original speaker attribute labels.
        flow (pt.modules.NormalizingFlow): The normalizing flow model for manipulation.
        tts_model: The text-to-speech model used for synthesis.
        d_vector (torch.Tensor): (1, feature_dim) The speaker embedding vector.
        manipulation (str): The type of manipulation to apply (e.g., 'Breathiness').
        manipulation_intensity (float): The intensity of the manipulation.
        pvq_labels (list): List of possible speaker attribute labels.
    Returns:
        torch.Tensor: The synthesized audio waveform after manipulation.
    """

    speaker_conditioning = get_speaker_labels(
        speaker_labels=speaker_labels, 
        pvq_labels=pvq_labels,
        manipulation=None,
        manipulation_intensity=0,
        config=config,
        stacked_flow=stacked_flow,
    )

    speaker_conditioning_manipulated = get_speaker_labels( 
        speaker_labels=speaker_labels, 
        pvq_labels=pvq_labels,
        manipulation=manipulation,
        manipulation_intensity=manipulation_intensity,
        config=config,
        stacked_flow=stacked_flow,
    )

    with torch.no_grad():
        sampled_class_manipulated = flow.apply_resampling(
            d_vector.to(device).float(), 
            speaker_conditioning,
            speaker_conditioning_manipulated
        )
    return tts_model.synthesize_from_example({
        'text': transcription,
        'd_vector': d_vector.cpu().numpy(),
        'd_vector_man': sampled_class_manipulated.cpu().numpy(),
    }), speaker_conditioning, speaker_conditioning_manipulated


def extract_speaker_embedding(tts_model, example):
    """
    Extracts the speaker embedding (d-vector) from the given audio example using the TTS model.
    Args:
        tts_model: The text-to-speech model with a speaker manager.
        example (dict): The audio example containing loaded audio data.
                        The dictionary must also include the following keys
                            - 'speaker_id': The unique identifier for the speaker.
                            - 'example_id': The unique identifier for the example.
    Returns:
        torch.Tensor: The extracted speaker embedding (d-vector).
    """
    if 'loaded_audio_data' in example.keys():
        audio_data = example['loaded_audio_data'][16_000]
    else:
        audio_data, sr = pb.io.load_audio(
            example['audio_file'],
            return_sample_rate=True
        )

        if sr != 16_000:
            vad = EnergyVAD(sample_rate=16_000)
            if audio_data.ndim == 1:
                audio_data = audio_data[None, :]
            audio_data = vad({'audio_data': audio_data})['audio_data']

    with torch.no_grad():
        example = tts_model.speaker_manager.prepare_example(
            {'audio_data': {'observation': audio_data}, **example})
        example = pt.data.utils.collate_fn([example])
        example['features'] = torch.tensor(np.array(example['features']), device=device)
        d_vector = tts_model.speaker_manager.forward(example)[0]
    return d_vector


def load_speaker_labels(example, hubert_model, pvq_labels, reg_stor_dir=Path('../saved_models/')):
    """
    Loads speaker labels for the given audio example using a HuBERT model and
    pretrained PVQ Regression models.
    Args:
        example (dict): The audio example containing 'loaded_audio_data'.
        hubert_model: The HuBERT model for feature extraction.
        pvq_labels (list): List of speaker attribute labels.
        reg_stor_dir (Path): Directory containing ONNX models for PVQ labels.
    Returns:
        torch.Tensor: Normalized speaker labels as a tensor.
    """
    audio_data = torch.tensor(example['loaded_audio_data'][16_000], dtype=torch.float).to(device)[None, :]
    num_samples = torch.tensor([audio_data.shape[-1]]).to(device)

    with torch.no_grad():
        features, _ = hubert_model(audio_data, 16_000, sequence_lengths=num_samples)
        features = np.mean(features.squeeze(0).cpu().numpy(), axis=-1)

    pvqd_predictions = {}
    for pvq in pvq_labels:
        session = InferenceSession(
            (reg_stor_dir / f"{pvq}.onnx").read_bytes(), providers=["CPUExecutionProvider"]
        )
        pvqd_predictions[pvq] = session.run(
            None, {"X": features[None]}
        )[0].squeeze(1)[0]

    labels = [pvqd_predictions[pvq] / 100 for pvq in pvq_labels]
    return torch.tensor(labels, device=device).float()


def load_audio_files(example, sample_rates=[16_000, 24_000]):
    """
    Loads audio files and applies Voice Activity Detection (VAD) to filter out non-speech segments.
    Args:
        example (dict or str): If dict, it should contain 'audio_file', 'speaker_id', and 'example_id'.
                               If str, it is treated as the audio file path.
        sample_rates (list): List of sample rates to which the audio will be resampled.
    Returns:
        dict: The updated example dictionary with loaded and processed audio data for each sample rate.
    """
    if isinstance(example, dict):
        audio_file = example['audio_file']
        audio_path = f"../saved_models/{audio_file}.wav"
    else:
        audio_path = example
        example = {'speaker_id': None, 'example_id': None}

    def process_audio(audio, sample_rate):
        vad = EnergyVAD(sample_rate=sample_rate)
        if audio.ndim == 1:
            audio = audio[None, :]
        return vad({'audio_data': audio})['audio_data']

    observation_loaded, sr = pb.io.load_audio(audio_path, return_sample_rate=True)
    example['loaded_audio_data'] = {
        rate: process_audio(
            librosa.resample(observation_loaded, orig_sr=sr, target_sr=rate),
            sample_rate=rate
        )
        for rate in sample_rates
    }
    return example
