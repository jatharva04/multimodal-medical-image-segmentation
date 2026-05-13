# Multimodal Medical Image Segmentation

This repository contains preprocessing, registration, segmentation, explainability, and statistical analysis pipelines for multimodal medical image segmentation using CT–MR data and nnU-Net.

## Features
- CT–MR rigid registration
- Multi-organ label map generation
- nnU-Net preprocessing and segmentation workflow
- 2D and 3D explainable AI (XAI) visualization
- Statistical comparison between 2D and 3D models
- Automated quality control utilities

## Requirements
- Python 3.x
- PyTorch
- nnUNetv2
- SimpleITK
- NumPy
- Pandas
- Matplotlib
- SciPy
- Pingouin

## Repository Structure
- `merge_multiorgan_labels.py`  
  Generates multi-class segmentation masks for nnU-Net training.

- `rigid_registration_pipeline.py`  
  Performs CT–MR rigid registration and preprocessing.

- `statistical_comparison_2d_vs_3d.py`  
  Performs Wilcoxon statistical analysis with FDR correction.

- `xai_pipeline_2d.py`  
  Generates 2D explainability visualizations.

- `xai_pipeline_3d.py`  
  Generates 3D explainability visualizations.

## Notes
- Dataset paths are intentionally generalized for portability.
- Users should configure local dataset and output directories before execution.
