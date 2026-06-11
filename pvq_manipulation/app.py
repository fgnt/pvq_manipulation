import gradio as gr
import paderbox as pb
import torch
import pandas as pd

from pathlib import Path
from pvq_manipulation.models.vits import Vits_NT
from pvq_manipulation.models.normalizing_flows import CCNF, StackedFlow
from pvq_manipulation.helper.notebook_helper import *

dataset_dict = {
    "1034_121119_000028_000001": {
        'audio_file': "1034_121119_000028_000001",
        'speaker_id': None,
        'example_id': "1034_121119_000028_000001"
    },
    "1088_134315_000094_000000": {
        'audio_file': "1088_134315_000094_000000",
        'speaker_id': None,
        'example_id': "1088_134315_000094_000000"
    },
    "1311_134170_000032_000001": {
        'audio_file': "1311_134170_000032_000001",
        'speaker_id': None,
        'example_id': "1311_134170_000032_000001"
    }
}

storage_dir = Path("../saved_models/")

config_norm_flow = pb.io.load_yaml(
    storage_dir / "config_norm_flow.yaml"
)
normalizing_flow = CCNF.load_model(
    storage_dir, 
    checkpoint="model_norm_flow.pt"
)

config_norm_flow_stacked = pb.io.load_yaml(
    storage_dir / "config_norm_flow_stacked.yaml"
)
normalizing_flow_stacked = StackedFlow.load_model(
    storage_dir, 
    checkpoint="model_norm_flow_stacked.pt"
)

tts_model = Vits_NT.load_model(storage_dir, checkpoint="tts_model.pt")
dataset_labels = pb.io.load_json(storage_dir / "dataset_labels.json")


def delete_cache():
    global cached_example_id, wav_unmanipulated
    del cached_example_id
    del wav_unmanipulated


cached_example_id = None
wav_unmanipulated = None

def update_manipulation(
    example_id,
    transcription,
    manipulation_feature,
    manipulation_intensity,
    stacked_flow,
):
    global cached_example_id, wav_unmanipulated

    if stacked_flow:
        pvq_labels = config_norm_flow_stacked['pvq_labels']
    else:
        pvq_labels = config_norm_flow['pvq_labels']


    example = dataset_dict[example_id]
    example = load_audio_files(example)

    audio_file = example['audio_file']

    if not Path(f"../saved_models/{audio_file}.pth").is_file():
        speaker_embedding = extract_speaker_embedding(tts_model, example)
        torch.save(speaker_embedding, f"../saved_models/{audio_file}.pth")
    else:
        speaker_embedding = torch.load(f"../saved_models/{audio_file}.pth")

    wav_manipulated, speaker_conditioning, speaker_conditioning_manipulated = get_manipulation(
        transcription=transcription, 
        d_vector=speaker_embedding, 
        pvq_labels=pvq_labels,
        speaker_labels=dataset_labels[example['example_id']], 
        flow=normalizing_flow_stacked if stacked_flow else normalizing_flow,
        tts_model=tts_model,
        manipulation=manipulation_feature, 
        manipulation_intensity=manipulation_intensity, 
        stacked_flow=stacked_flow,
        config= config_norm_flow_stacked if stacked_flow else None,
    )


    if cached_example_id != example_id:
        wav_unmanipulated = tts_model.synthesize_from_example({
            'text': transcription,
            'd_vector': speaker_embedding.detach().cpu().numpy(),
            'd_vector_man': speaker_embedding.detach().to(device),
        })

    table = {}
    if stacked_flow:
        for idx_high, conditions in enumerate(config_norm_flow_stacked['model']['condition_list']):
            for idx, condition in enumerate(conditions):
                table[condition] = {
                    "original": speaker_conditioning[idx_high][0, idx].tolist(),
                    "manipulated": speaker_conditioning_manipulated[idx_high][0, idx].tolist()
                }
    else:
        for idx, condition in enumerate(pvq_labels):
            table[condition] = {
                "original": speaker_conditioning[0, idx].tolist(),
                "manipulated": speaker_conditioning_manipulated[0, idx].tolist()
            }
    
    table_df = pd.DataFrame(table).T
    table_df = table_df.round(2) 
    table_df.index.name = "Voice Quality Feature"

    table_df = table_df.reset_index()
    cols = ["Voice Quality Feature"] + [c for c in ["original", "manipulated"] if c in table_df.columns]
    table_df = table_df[cols]

    html_table = table_df.to_html(classes="compact-table", index=False, float_format="%.4f", border=0, escape=False)
    html_table = f'<div style="overflow:auto; max-width:820px;">{html_table}</div>'

    return (24_000, wav_unmanipulated), (24_000, wav_manipulated), html_table


with gr.Blocks(
    css="""
    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.90), rgba(255,255,255,0.72)),
            linear-gradient(135deg, #eef6ff 0%, #f7f1ff 45%, #fff7ef 100%);
        color: #1f2937;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .gradio-container .block {
        border-radius: 20px;
    }
    .hero {
        background: linear-gradient(135deg, rgba(13, 71, 161, 0.96), rgba(88, 28, 135, 0.95));
        color: white !important;
        padding: 22px 26px;
        border-radius: 22px;
        box-shadow: 0 18px 40px rgba(59, 130, 246, 0.20);
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.18);
    }
    .hero * {
        color: white !important;
    }
    .eyebrow {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.82);
        margin-bottom: 8px;
    }
    .intro_box {
        max-width: 100%;
        font-size: 20px;   
        line-height: 1.4; 
    }
    .header {
        font-size: 38px;
        font-weight: bold;  
        line-height: 1.2;
        margin: 0;
    }
    .subtitle {
        margin-top: 8px;
        font-size: 15px;
        line-height: 1.5;
        color: rgba(255,255,255,0.90);
        max-width: 900px;
    }
    .block-label {
        font-size: 14px;
        font-weight: 700;
        color: #334155;
    }
    .gradio-container label,
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container .wrap,
    .gradio-container .prose {
        color: #111827 !important;
    }
    .gradio-container .input,
    .gradio-container .input-container,
    .gradio-container textarea,
    .gradio-container input,
    .gradio-container select {
        background: rgba(255,255,255,0.92) !important;
    }
    /* Compact table styling for results */
    .compact-table {
        border-collapse: collapse;
        font-size: 12px;
        max-width: 100%;
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    .compact-table th, .compact-table td {
        padding: 8px 10px;
        border: 1px solid #e5e7eb;
        text-align: left;
        white-space: nowrap;
    }
    .compact-table thead th {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        color: white;
        font-weight: 700;
    }
    .compact-table tbody tr:nth-child(even) td {
        background: #f8fafc;
    }
    .compact-table tbody tr:hover td {
        background: #eef2ff;
    }
    button.primary {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        border: none !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28);
        border-radius: 14px !important;
    }
    button.primary:hover {
        filter: brightness(1.05);
    }
    """
) as demo:
    gr.Markdown(
        """
        <div class='hero'>
            <div class='eyebrow'>Voice synthesis demo</div>
            <div class='header'>🎙️ Manipulation of Voice Qualities</div>
            <div class='subtitle'>
                Choose a speaker, adjust a voice-quality dimension, and compare the original against the manipulated synthesis.
                The interface is styled for a softer, more modern look.
            </div>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            speaker_dropdown = gr.Dropdown(
                label="Speaker",
                choices=[example_id for example_id in dataset_dict.keys()],
                value="1088_134315_000094_000000",
                type="value"
            )
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text Input",
                value="I am helping phonetic experts to explain the complexities of the human voice.",
                placeholder='Type something'
            )

    with gr.Row():
        with gr.Column(scale=1):
            pvq_dropdown = gr.Dropdown(
                label="Voice Quality",
                choices=[
                    'Creak_mean',
                    'Breathiness',
                    'Roughness',
                    'Resonance',
                    'Weight',
                ],
                value='Roughness',
                type="value"
            )
        with gr.Column(scale=1):
            stacked_flow = gr.Checkbox(
                label="Stacked Flow",
                value=True,   
            )
        with gr.Column(scale=2):
            deviation_slider = gr.Slider(
                label="Manipulation Intensity",
                minimum=0.0,
                maximum=4.0,
                value=1.0,
                step=0.1
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Synthesize", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            unmanip_audio = gr.Audio(label="Unmanipulated Synthesis", type="numpy")
        with gr.Column(scale=1):
            manip_audio = gr.Audio(label="Manipulated Synthesis", type="numpy")

    with gr.Row():
        with gr.Column(scale=2):
            results_table = gr.HTML(
                label="Voice Quality Comparison"
            )

    submit_btn.click(
        fn=update_manipulation,
        inputs=[speaker_dropdown, text_input, pvq_dropdown, deviation_slider, stacked_flow],
        outputs=[unmanip_audio, manip_audio, results_table]
    )

if __name__ == "__main__":
    demo.launch(share=True)



