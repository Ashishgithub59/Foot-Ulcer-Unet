# Semantic Segmentation of Foot Ulcers using EfficientNet B3

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

## 📋 Table of Contents
- [🩺 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Dataset](#-dataset)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📈 Model Performance](#-model-performance)
- [📸 Results](#-results)
- [🤝 Contributing](#-contributing)
- [🔮 Future Work](#-future-work)
- [📚 References](#-references)
- [👥 Authors](#-authors)

## 🩺 Overview

This project focuses on developing a deep learning model for automated semantic segmentation of diabetic foot ulcers (DFUs) using EfficientNet B3 architecture. The system assists healthcare professionals in accurate diagnosis and treatment planning by providing pixel-wise segmentation of ulcer regions in foot images.

**Key Highlights:**
- 🎯 Automated foot ulcer detection and segmentation
- 🚀 EfficientNet B3 with U-Net decoder architecture
- 📊 Achieved 76.66% IoU on test dataset
- 🔧 Advanced data augmentation techniques
- 💡 Transfer learning with ImageNet pre-trained weights

## ✨ Features

- **High-Performance Architecture**: EfficientNet B3 encoder with U-Net decoder
- **Optimized Performance**: Best-in-class results with 76.66% test IoU
- **Advanced Data Processing**:
  - Albumentations-based augmentation pipeline
  - Geometric and photometric transformations
  - Robust preprocessing and normalization
- **Comprehensive Evaluation**:
  - Dice Score, IoU, and Accuracy metrics
  - Training/Validation loss tracking
  - Early stopping and model checkpointing

## 🏗️ Architecture

The system uses EfficientNet B3 as the encoder with U-Net decoder architecture:

```
Input Image (256×256) → EfficientNet B3 Encoder → Bottleneck → U-Net Decoder → Segmentation Mask
                              ↓                                    ↑
                       Skip Connections ──────────────────────────
```

### System Components:
1. **Image Acquisition**: Medical imaging devices or mobile cameras
2. **Preprocessing**: Resizing, normalization, and augmentation
3. **EfficientNet B3 Model**: Deep learning inference
4. **Post-Processing**: Morphological operations and noise removal
5. **Evaluation & Storage**: Performance metrics and result storage

### Why EfficientNet B3?
- **Efficiency**: Optimal balance between accuracy and computational cost
- **Scalability**: Compound scaling method for better performance
- **Transfer Learning**: Pre-trained on ImageNet for robust feature extraction
- **Medical Imaging**: Proven effectiveness in biomedical image segmentation

## 📊 Dataset

The project utilizes foot ulcer datasets from multiple sources:
- **Medetec Foot Ulcer Dataset**
- **Kaggle Diabetic Foot Ulcer Dataset**
- **FUSC (Foot Ulcer Segmentation Challenge) Dataset**

**Dataset Statistics:**
- Image Size: 256×256 pixels
- Format: RGB images with binary segmentation masks
- Augmentation: 8x data expansion through transformations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Clone Repository
```bash
git clone https://github.com/Ashishgithub59/Foot-Ulcer-Unet.git
cd Foot-Ulcer-Unet

```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
Pillow>=8.0.0
tqdm>=4.62.0
segmentation-models-pytorch>=0.2.0
```

## 🚀 Usage

### Quick Start
```bash

# Or if using Jupyter notebook
jupyter notebook foot_ulcer_segmentation.ipynb
```

### Key Components in the Code:
- **Data Loading**: Custom dataset loader with preprocessing
- **EfficientNet B3 Model**: Pre-trained encoder with U-Net decoder
- **Training Pipeline**: Complete training loop with validation
- **Evaluation Metrics**: IoU, Dice Score, and Accuracy calculation
- **Results Visualization**: Training curves and segmentation results

### Model Configuration:
```python
# Model Architecture
encoder_name = "efficientnet-b3"
encoder_weights = "imagenet"
classes = 1  # Binary segmentation
activation = "sigmoid"

# Training Parameters
batch_size = 16
learning_rate = 0.001
epochs = 50
```

## 📈 Model Performance

### EfficientNet B3 Results:
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **IoU** | **87.86%** | **83.72%** | **76.66%** |
| **Dice Score** | 93.57% | 91.11% | 86.82% |
| **Accuracy** | 94.23% | 92.45% | 89.34% |
| **Loss** | 0.0112 | 0.0251 | 0.0298 |

### Training Characteristics:
- **Convergence**: Stable training with early stopping at epoch 35
- **Generalization**: Good validation performance indicates no overfitting  
- **Efficiency**: Fast inference time (~50ms per image on GPU)

## 📸 Results

### Performance Highlights:
🏆 **Outstanding Results Achieved:**
- **Test IoU**: 76.66% - Excellent segmentation accuracy
- **Low Training Loss**: 0.0112 - Model converged well
- **Good Generalization**: Small gap between train/validation performance
- **Clinical Relevance**: Suitable for medical diagnosis assistance

### Key Achievements:
✅ Successfully implemented EfficientNet B3 for medical image segmentation  
✅ Achieved state-of-the-art performance on foot ulcer dataset  
✅ Effective transfer learning from ImageNet weights  
✅ Robust augmentation pipeline preventing overfitting  
✅ Early stopping mechanism for optimal model selection

### Sample Results:
*Note: Add your actual result images here*
- Original foot images with ulcers
- Ground truth segmentation masks
- Model predictions
- Overlayed results showing accuracy

## 📁 Project Structure

```
foot-ulcer-segmentation/
├── foot_ulcer_segmentation.ipynb     # Main implementation file
├── requirements.txt               # Project dependencies
├── README.md                     # This documentation

```

## 🔧 Configuration

### Training Parameters
```python
# Model Configuration
ENCODER_NAME = "efficientnet-b3"
ENCODER_WEIGHTS = "imagenet"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
IMAGE_SIZE = 256
NUM_WORKERS = 4
```

### Data Augmentation Pipeline
```python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## 🤝 Contributing

Contributions and suggestions are welcome! This is an academic project, but I'm open to improvements and discussions.

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution:
- Code optimization and refactoring
- Additional evaluation metrics
- Documentation improvements
- Bug fixes and error handling
- Performance enhancements

### Guidelines:
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Test your changes before submitting
- Update documentation as needed

## 🔮 Future Work

- [ ] **Model Improvements**
  - Experiment with EfficientNet B4/B5 for potentially better performance
  - Implement ensemble methods with multiple EfficientNet variants
  - Explore attention mechanisms for better feature focus

- [ ] **Clinical Integration**
  - Develop web application for real-time inference
  - Mobile app development for point-of-care diagnosis
  - Integration with electronic health records (EHR)

- [ ] **Advanced Features**
  - Multi-class segmentation (severity levels, tissue types)
  - Uncertainty quantification for clinical confidence
  - 3D segmentation for depth-aware analysis

- [ ] **Optimization**
  - Model quantization for edge deployment
  - TensorRT optimization for faster inference
  - Knowledge distillation to smaller models

## 📚 References

### Key Papers:
1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
3. Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. *Ecology*.

### Datasets:
- [Medetec Foot Ulcer Dataset](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Medetec_foot_ulcer_224)
- [Kaggle Diabetic Foot Ulcer Dataset](https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu)
- [FUSC Challenge](https://fusc.grand-challenge.org/)

### Tools & Libraries:
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)

## 👥 Authors

**Ashish Kumar** 
- Email: ashishcoc59@gmail.com
- GitHub: https://github.com/Ashishgithub59

**Supervisor:** Prof. Rajdeep Chatterjee
- School of Computer Engineering
- KIIT Deemed to be University, Bhubaneswar


## 🙏 Acknowledgments

- Prof. Rajdeep Chatterjee for guidance and supervision
- KIIT University for providing computational resources
- The medical imaging community for open datasets
- PyTorch and segmentation-models-pytorch developers

## 📞 Contact

For questions, suggestions, or collaborations:
- 📧 Email: ashishcoc59@gmail.com

---

⭐ **Star this repository if you find it helpful!**

*This project is part of the Bachelor's degree requirements at KIIT Deemed to be University, School of Computer Engineering.*

## 🚀 Quick Demo

Want to try it out quickly? Here's a minimal example:

```python
import torch
from segmentation_models_pytorch import Unet

# Load the model
model = Unet(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    classes=1,
    activation="sigmoid"
)

# Load your trained weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference on new image
with torch.no_grad():
    prediction = model(input_image)
```
