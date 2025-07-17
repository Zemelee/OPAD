# OPAD
### Code for ICLR 2025 Paper: On-the-fly Preference Alignment via Principle-Guided Decoding
This repository contains the implementation for the paper "On-the-fly Preference Alignment via Principle-Guided Decoding". The project focuses on generating responses based on task-specific principles to achieve alignment without fine-tuning.

**Principle-Guided Decoding**:

Achieves alignment by dynamically guiding responses based on task-specific principles.

**Supported Models**:

Works with LLaMA-2 and Vicuna models out of the box.

**Supported Datasets**:

[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), Summarization, DSP, and [PSOUPs](https://huggingface.co/datasets/RiverDong/psoups) datasets.

## Getting Started
### Prerequisites
```
pip install -r requirements.txt
```
### Example Usage:
```
bash infer_hh.sh
```
官方论文源码的复现，infer_hh.py 可直接运行。