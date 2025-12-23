# PVQ Manipulation 
---
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-PDF-blue.svg)](https://ieeexplore.ieee.org/document/10888012)
[![ISCA DOI](https://img.shields.io/badge/ISCA/DOI-10.21437/Interspeech.2025--1443-blue.svg)](https://doi.org/10.21437/Interspeech.2025-1443)

Voice quality manipulation aims at modifying perceptual characteristics of speech such as breathiness, creakiness, or related voice attributes, while preserving linguistic content and speaker identity.
This repository provides a research-oriented pipeline for controllable voice quality manipulation based on text-to-speech (TTS).

The system is built on top of a TTS backbone and introduces a dedicated manipulation block, which enables targeted modification of voice qualities.
The manipulation is realized via a conditioned manipulation module based on Continuous Normalizing Flows (CNFs), allowing for smooth and interpretable control over continuous voice quality dimensions.

The TTS backend is based on [YourTTS](https://github.com/coqui-ai/TTS), which is currently supported for Python versions < 3.12.
For newer Python versions (≥ 3.12), the system automatically switches to [Coqui-TTS](https://github.com/idiap/coqui-ai-TTS), which provides ongoing support for modern Python releases.

> **_NOTE:_**
This repository is under active development. The core voice quality manipulation pipeline is functional and can already be used for experimental studies and demonstrations.
Ongoing work focuses on improving documentation, code structure, and usability.
A Jupyter notebook showcasing an example voice quality manipulation is provided and serves as the main point of reference at this stage.
If you encounter version mismatches, backend-specific limitations, or unexpected behavior, please feel free to open an issue so it can be tracked and addressed by the community.

<img src="https://groups.uni-paderborn.de/nt/interspeech_2025_creak_demo/interspeech.png" alt="TTS-System" width="400">



## Installation

```sh
git clone https://github.com/fgnt/pvq_manipulation.git
cd pvq_manipulation
# if python version >= 3.12
pip install -e ".[py12]"
# else
pip install -e ".[legacy]"

gh release download v1.0.0 --repo fgnt/pvq_manipulation --dir ./saved_models
```
If python version >= 3.12
```sh
cd ..
git clone https://gitlab.tugraz.at/speech/creapy.git
cd creapy
nano setup.cfg  # or any text editor of your choice; change the line PyYAML==6.0.0 to PyYAML==6.0.1
pip install -e . 
```

## Manipulation of Voice 
To get started, follow the Example_Notebook.ipynb.
It demonstrates how to load the model, prepare an audio file, and apply perceptual voice quality manipulations step by step. 
## Example Training
To train your own model, follow the toy example in the train_example folder.

## Citation
The manipulation method of manipulation Perceptual Voice Qualities was introduced in the paper ["Speech synthesis along perceptual voice quality dimensions"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10888012)
```sh
@inproceedings{rautenberg2025speech,
  title={Speech synthesis along perceptual voice quality dimensions},
  author={Rautenberg, Frederik and Kuhlmann, Michael and Seebauer, Fritz and Wiechmann, Jana and Wagner, Petra and Haeb-Umbach, Reinhold},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
The approach for controlling creakiness intensity was presented in the paper ["Synthesizing Speech with Selected Perceptual Voice Qualities--A Case Study with Creaky Voice"](https://www.isca-archive.org/interspeech_2025/rautenberg25_interspeech.pdf)

```sh
@inproceedings{rautenberg25_interspeech,
  title     = {{Synthesizing Speech with Selected Perceptual Voice Qualities – A Case Study with Creaky Voice}},
  author    = {Frederik Rautenberg and Fritz Seebauer and Jana Wiechmann and Michael Kuhlmann and Petra Wagner and Reinhold Haeb-Umbach},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1633--1637},
  doi       = {10.21437/Interspeech.2025-1443},
  issn      = {2958-1796},
}
```
