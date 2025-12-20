# PVQ Manipulation 
---

This repository contains code for manipulating perceptual voice quality (PVQ) features.

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
