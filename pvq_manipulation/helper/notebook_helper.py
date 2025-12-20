import numpy as np 
import librosa
import torch
import paderbox as pb
import padertorch as pt

from onnxruntime import InferenceSession
from pathlib import Path
from pvq_manipulation.helper.vad import EnergyVAD
from pvq_manipulation.helper.creapy_wrapper import process_file

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
 
 
def get_manipulation(
    transcription,
    labels,
    flow, 
    tts_model,    
    d_vector,
    manipulation,
    manipulation_intensity=1,
    pvq_labels=None,
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
    for manipulation_idx, name in enumerate(pvq_labels):
        if name == manipulation:
            break
    else:
        raise NotImplementedError(f"{manipulation} not found in pvq_labels.")
        
    labels_manipulated = labels.clone()
    labels_manipulated[:, manipulation_idx] += manipulation_intensity

    with torch.no_grad():
        output_forward = flow.forward((d_vector.to(device).float(), labels))[0]
        sampled_class_manipulated = flow.sample((output_forward, labels_manipulated))[0]

    return tts_model.synthesize_from_example({
        'text': transcription,
        'd_vector': d_vector.cpu().numpy(),
        'd_vector_man': sampled_class_manipulated.cpu().numpy(),
    })


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


def get_creak_label(example):
    """
    Computes the mean creakiness label for the given audio example.
    Args:
        example (dict): The audio example containing 'loaded_audio_data'.
    Returns:
        float: The mean creakiness label (scaled to 0-100).
    """
    audio_data = example['loaded_audio_data'][16_000]
    _, y_pred, included_indices = process_file(audio_data)
    return np.mean(y_pred[included_indices]) * 100


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
        if pvq == 'Creak':
            pvqd_predictions[pvq] = get_creak_label(example)
        else:
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
