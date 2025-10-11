# SilenceLLM
# Multimodal LLM Training (Audio + Video + Text)

> Training script for a **multimodal large language model** that jointly learns from **audio**, **video**, and **text** data.  
> Built with [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), and [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

---

## 🧩 Overview

This repository provides a training pipeline for **SilenceQwen3**, a multimodal extension of the [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) model.  
It integrates:
- **Audio encoder**: `openai/whisper-medium`
- **Vision encoder**: `google/siglip2-base-patch16-224`
- **Language backbone**: Qwen3-1.7B
- **Q-Former** for modality alignment (`bert-large-uncased`)
- **LoRA fine-tuning** for efficient parameter adaptation

---

## Installation

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate peft
pip install pillow tqdm
```

---

## Project

```bash

project_root/
├── train.py
├── eval.py
├── silence_trainer.py
├── data/
│   ├──collector.py
│   ├──dataset.py
│   └──mm_utils.py
├── model/
│   ├── submodels
│   ├── silence_llama.py
│   ├── silence_model.py
│   ├── silence_perceiver.py
│   └── silence_qwen.py  
├── script/
│   ├──train.sh
│   └──eval.sh

