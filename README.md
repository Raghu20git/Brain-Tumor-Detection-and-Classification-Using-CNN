# Brain Tumor Detection & Classification Using CNN

## Overview

This project implements an AI-based system for brain tumor classification from MRI images using deep learning. The pipeline performs model inference, uncertainty estimation, explainability (Grad-CAM & Grad-CAM++), and automatic clinical-style PDF report generation.

The system is designed for research and educational purposes only and is not intended for clinical diagnosis.


## Key Features

- EfficientNet-B0 based MRI classifier
- Multi-class tumor detection (4 classes)
- Monte Carlo Dropout uncertainty estimation
- Grad-CAM and Grad-CAM++ visual explanations
- Automatic AI case report (PDF)
- GPU acceleration support (CUDA)
- Modular and reproducible pipeline


## Datasets
- Brain Tumor Classification (MRI): https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- Brain Tumor MRI Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


## Methodology

- Data cleaning performed to remove corrupted, duplicate, and blank MRI images
- Images resized to 224×224, normalized, and converted to tensors during preprocessing
- Applied data augmentation (random flip, rotation, color jitter) to improve generalization
- Used pretrained EfficientNet-B0 as the backbone for transfer learning
- Replaced the original classification head (the final classification layer) with a new 4-class (4 tumor classes) fully connected layer
- Initially froze all backbone layers and trained only the new classification layer
- Subsequently unfroze the backbone layers and fine-tuned the entire network
- Optimized the model using AdamW with low learning rate and weight decay
- Evaluated performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
- Generated visual explanations using Grad-CAM and Grad-CAM++ for interpretability
- Estimated prediction uncertainty using Monte Carlo Dropout during inference
- Saved best and last model checkpoints for reproducibility


## Results

### Best Validation Accuracy

![Best Validation Accuracy](<Results/Best Validation Accuracy.png>)

### Final Test Accuracy

![Final Test Accuracy](<Results/Final Test Accuracy.png>)

### Confusion Matrix

![Confusion Matrix](<Results/Confusion Matrix.png>)

### Classification Report

![Classification Report](<Results/Classification Report.png>)

### Macro ROC-AUC

![Macro ROC-AUC](<Results/Macro ROC-AUC.png>)

### Grad-CAM Output

![GradCAM Output](<Results/Grad-CAM Output.png>)

### Grad-CAM and Grad-CAM++ Overlay

![Grad-CAM and GradCAM++ Overlay](<Results/Grad-CAM and Grad-CAM++ Overlay.png>)

### PDF generated for a random test image

[Generated PDF](<Results/Generated PDF.pdf>)


## Project Structure

Brain Tumor Detection & Classification Using CNN/
│
├── .vscode
│   └── settings.json
├── AI_Reports
├── Checkpoints
├── Dataset/
│   ├── Testing
│   └── Training
├── Results
├── .gitignore
├── main.ipynb
├── README.md
└── requirements.txt

## Environment Setup

### 1. Create virtual environment (Python 3.10.11 recommended)

Run "py -3.10 -m venv venv" in the Powershell (Open the Powershell in the project folder)

### 2. Activate

Run ".\venv\Scripts\Activate" in the Powershell (Open the Powershell in the project folder)

### 3. Do NOT rely on requirements.txt to install torch, torchvision & torchaudio if you want to use NVIDIA GPU (CUDA supported).
Instead do:
Step 1 — install CUDA PyTorch manually by running "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" in the Powershell (Open the Powershell in the project folder)
Step 2 — install the rest by running "pip install -r requirements.txt" in the Powershell (Open the Powershell in the project folder)

### 4. Default (Works Everywhere):
Step 1 — Run "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" in the Powershell (Open the Powershell in the project folder)
Step 2 — Run "pip install -r requirements.txt" in the Powershell (Open the Powershell in the project folder)

### 5. Register Jupyter kernel:
Run "python -m ipykernel install --user --name brain_tumor_py310 --display-name "Python 3.10 (BrainTumor)" in the Powershell (Open the Powershell in the project folder)

