# FLPedBrain-PyTorch

Official PyTorch implementation of Federated Learning for Pediatric Brain Tumor Segmentation and Classification.

If you find this project useful, please give it a star!

This is the official PyTorch version of [FLPedBrain](https://github.com/edhlee/FLPedBrain). The processed training data is available on [Hugging Face](https://huggingface.co/datasets/edhlee/FLPedBrain-processed), derived from the raw data at [Stanford Digital Repository](https://doi.org/10.25740/bf070wx6289).

## Motivation

This is a PyTorch port of the original TensorFlow FLPedBrain implementation. The migration was motivated to support modern model architectures going forward and future challenges with the federated learning efforts. 

## Overview

This repository implements a comparison between:
- **CDS (Centralized Data Sharing)**: Traditional training with pooled data from all sites
- **FL (Federated Learning)**: Training using FedAvg (FedProx coming soon) across 16 sites.

The model performs simultaneous tumor segmentation and classification for pediatric brain tumor types.


## Model Architecture

- **Encoder**: I3D (Inflated 3D ConvNet) pretrained on ImageNet + Kinetics
- **Decoder**: U-Net style decoder for segmentation
- **Classification Head**: Global average pooling + FC layers
- **Input**: 3D MRI volumes (64 x 256 x 256)

### TensorFlow to PyTorch Migration Notes

We aimed to match the original TensorFlow I3D model architecture as closely as possible. Key differences and considerations:

**Architecture Differences:**
- Pretrained I3D weights are sourced from `piergiaj/pytorch-i3d` (native PyTorch weights)
- Batch normalization momentum/epsilon parameters mapped to PyTorch conventions

**Training Differences:**
- Changed optimizer from Adam to **AdamW** (decoupled weight decay).
- Uses **BF16 mixed precision** (vs FP16 in TF) for modern GPU compatibility (Ampere+)
- Learning rate scheduling uses PyTorch's `StepLR` to match the original decay schedule

**FL Synchronization:**
- Model weight synchronization follows the same FedAvg approach as the TensorFlow version
- Multiple GPU processes run the same `train_fl_single.py` script, each handling a subset of sites
- Weights are synchronized via filesystem (`.pth` files) - GPU 0 aggregates and other GPUs wait for completion signals
- Aggregation uses sample-weighted averaging across sites
- Warmstart strategy (training on largest sites first) preserved from original implementation

## Demo Scripts

The repository includes comprehensive demo notebooks that demonstrate multiple input pipelines:

| Demo | Input Format | Description |
|------|--------------|-------------|
| `dicom_evaluation_pipeline.ipynb` | **DICOM** | Load and process raw DICOM series for inference |
| `dicom_evaluation_pipeline.ipynb` | **PNG slices** | Load pre-extracted PNG slices (matches original data DOI format) |
| `dicom_evaluation_pipeline.ipynb` | **Processed NPY** | Use preprocessed numpy arrays for batch evaluation |

These demos show end-to-end workflows including:
- Loading data from different sources
- Running inference with trained CDS and FL models
- Generating segmentation overlay videos (MP4)
- Comparing model predictions side-by-side

- Example demo of DIPG case in 2026, out of distribution from demos/DIPG_Case_B_cds_vs_fl.mp4. <img width="986" height="366" alt="image" src="https://github.com/user-attachments/assets/17e23840-8183-4cf7-8f49-37aacd06541f" />


### Example Segmentation Videos (CDS vs FL)

| Tumor Type | Video Link |
|------------|------------|
| DIPG | [View Video](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/resolve/main/demos/DIPG_Case_A_cds_vs_fl.mp4) |
| Ependymoma | [View Video](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/resolve/main/demos/EPEN_01_cds_vs_fl.mp4) |
| Medulloblastoma | [View Video](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/resolve/main/demos/MEDU_05_cds_vs_fl.mp4) |
| Pilocytic Astrocytoma | [View Video](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/resolve/main/demos/PILO_07_cds_vs_fl.mp4) |

More demo videos available in the [demos folder on Hugging Face](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/tree/main/demos).

## Data Format
The raw data is on https://purl.stanford.edu/bf070wx6289. To assist researchers and users, we have compiled the train data on Hugging Face: https://huggingface.co/datasets/edhlee/FLPedBrain-processed 


Each `.npy` file contains a dictionary with:
- `xs_uint8`: MRI volumes, shape `(N, 256, 256, 64)`, dtype `uint8`
- `ys_uint8`: Segmentation masks, shape `(N, 256, 256, 64)`, dtype `uint8`
- `label_classes`: Classification tumor labels, shape `(N,)`, dtype `int`

## Pretrained Checkpoint

Our Pytorch-version that was trained with FL. This checkpoint is available on Hugging Face:

| Checkpoint | Description |
|------------|-------------|
| [FLPedBrain_ckpt.pth](https://huggingface.co/datasets/edhlee/FLPedBrain-processed/resolve/main/checkpoints/FLPedBrain_ckpt.pth) | Pretrained model for inference |

To load the checkpoint:

```python
import torch
from model import BrainSegmentationModel

model = BrainSegmentationModel(pretrained=False)
checkpoint = torch.load('checkpoints/FLPedBrain_ckpt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Installation

```bash
conda create -n FLPedBrain-PyTorch python=3.10 -y
conda activate FLPedBrain-PyTorch

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
pip install -r requirements.txt
```


