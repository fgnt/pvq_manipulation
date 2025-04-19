# PVQ Manipulation

This repository contains code for manipulating perceptual voice quality (PVQ) features, intended for experiments and synthesis using models such as YourTTS.

---

## Installation

### Clone the repository and install

```sh
git clone https://github.com/FrederikRautenberg/pvq_manipulation.git
cd pvq_manipulation
pip install -e .
```

### Install [YourTTS](https://github.com/coqui-ai/TTS) from
```
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .[all,dev,notebooks]  # Select the relevant extras
```
### Make sure that [Paderbox](https://github.com/fgnt/paderbox) and [Padertorch](https://github.com/fgnt/padertorch?tab=readme-ov-file) are installed from 
```
git clone https://github.com/fgnt/paderbox.git
cd paderbox
pip install --editable .[all]
git clone https://github.com/fgnt/padertorch.git
cd padertorch && pip install -e .[all]
```
