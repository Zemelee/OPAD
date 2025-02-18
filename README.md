# OPAD
### Code for ICLR 2025 Paper: On-the-fly Preference Alignment via Principle-Guided Decoding
This repository contains the implementation for the paper "On-the-fly Preference Alignment via Principle-Guided Decoding". The project focuses on generating responses based on task-specific principles to achieve alignment without fine-tuning.

**Principle-Guided Decoding**:

Achieves alignment by dynamically guiding responses based on task-specific principles.

**Supported Models**:

Works with LLaMA-2 and Vicuna models out of the box.

**Supported Datasets**:

HH-RLHF, Summarization, DSP, and PSOUPs datasets.

## Getting Started
### Prerequisites
```
pip install -r requirements.txt
```
### Example Usage:
```
bash infer_hh.sh
```
### Change Models:
Modify the conv_type in the shell scripts and add corresponding conversation adapters in conversation.py.
### Experiment with Datasets: 
The codebase is extensible to other datasets by adapting the input format and principles.


### Acknowledgments
This code is built upon the [Linear Alignment](https://github.com/Wizardcoast/Linear_Alignment?tab=readme-ov-file) repository. For more details on the foundational work, please refer to the original repo.
