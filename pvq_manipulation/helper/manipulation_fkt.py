import numpy as np 
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
    example, 
    labels,
    flow, 
    tts_model,    
    d_vector,
    manipulation,
    manipulation_intensity=1,
    pvq_labels=None,
):
    for manipulation_idx, name in enumerate(pvq_labels):
        if name == manipulation:
            break
    else:
        raise NotImplementedError(f"{manipulation} not found in pvq_labels.")
        
    labels_manipulated = labels.clone()
    labels_manipulated[:,manipulation_idx] += manipulation_intensity
    
    d_vector = d_vector.to(device)

    output_forward = flow.forward((d_vector.float(), labels))[0]
    sampled_class_manipulated = flow.sample((output_forward, labels_manipulated))[0]

    wav = tts_model.synthesize_from_example({
        'text': example['transcription'],
        'd_vector': d_vector.cpu().detach().numpy(),
        'd_vector_man': sampled_class_manipulated.cpu().detach().numpy(),
    })    
    return wav

def extract_speaker_embedding(tts_model, example):
    audio_data = example['loaded_audio_data']['16_000']    
    with torch.no_grad():
        example = tts_model.speaker_manager.prepare_example({'audio_data': {'observation': audio_data}, **example})
        example = pt.data.utils.collate_fn([example])
        example['features'] = torch.tensor(np.array(example['features']), device=device)
        d_vector = tts_model.speaker_manager.forward(example)[0]
    return d_vector

def get_creak_label(example):
    audio_data = example['loaded_audio_data']['16_000']
    test, y_pred, included_indices = process_file(audio_data)
    mean_creak = np.mean(y_pred[included_indices])
    return mean_creak * 100


def load_speaker_labels(
    example, 
    config_norm_flow, 
    hubert_model, 
    pvq_labels,
    reg_stor_dir=Path('../saved_models/')
):
    audio_data = torch.tensor(example['loaded_audio_data']['16_000'], dtype=torch.float)[None,:]
    num_samples = torch.tensor([audio_data.shape[-1]])

    if torch.cuda.is_available():
        audio_data = audio_data.cuda()
        num_samples = num_samples.cuda()
    providers = ["CPUExecutionProvider"]
    
    with torch.no_grad():
        features, seq_len = hubert_model(
            audio_data, 
            16_000, 
            sequence_lengths=num_samples,
        )
        
        features = np.mean(features.squeeze(0).detach().cpu().numpy(), axis=-1)

        pvqd_predictions = {}
        for pvq in pvq_labels:
            if pvq == 'Creak':
                pvqd_predictions[pvq] = get_creak_label(example)
            else:
                with open(reg_stor_dir / f"{pvq}.onnx", "rb") as fid:
                    onnx = fid.read()
                sess = InferenceSession(onnx, providers=providers)
                pred = sess.run(None, {"X": features[None]})[0].squeeze(1)
                pvqd_predictions[pvq] = pred.tolist()[0]    
    
    labels = []
    for key in pvq_labels:
        labels.append(pvqd_predictions[key]/100)
    return torch.tensor(labels, device=device).float()

def load_audio_files(audio_path):
    if type(audio_path) == dict:
        example = audio_path
        audio_file = example['audio_file']
        audio_path = f"../saved_models/{audio_file}.wav"
    else:
        example = {
            'speaker_id': None,
            'example_id': None
        }
    observation_loaded, sr = pb.io.load_audio(
        audio_path, 
        return_sample_rate=True
    )
    
    example['loaded_audio_data'] = {}
    observation = pb.transform.module_resample.resample_sox(
        observation_loaded, in_rate=sr, out_rate=16_000
        )
    
    vad = EnergyVAD(sample_rate=16_000)
    if observation.ndim == 1:
        observation = observation[None, :]
        
    observation = vad({'audio_data': observation})['audio_data']
    example['loaded_audio_data']['16_000'] = observation
    
    observation = pb.transform.module_resample.resample_sox(observation, in_rate=sr, out_rate=24_000)
    vad = EnergyVAD(sample_rate=24_000)
    if observation.ndim == 1:
        observation = observation[None, :]
    observation = vad({'audio_data': observation})['audio_data']
    example['loaded_audio_data']['24_000'] = observation
    return example
