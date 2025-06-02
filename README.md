# Semantic Segmentation of Foot Ulcers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

This project focuses on developing a deep learning model for automated semantic segmentation of diabetic foot ulcers (DFUs). The system assists healthcare professionals in accurate diagnosis and treatment planning by providing pixel-wise segmentation of ulcer regions in foot images.

**Key Highlights:**
- 🎯 Automated foot ulcer detection and segmentation
- 🚀 Multiple state-of-the-art architectures (U-Net, ResNet, EfficientNet, MobileNet)
- 📊 Comprehensive evaluation with medical imaging metrics
- 🔧 Advanced data augmentation techniques
- 💡 Transfer learning for enhanced performance

## ✨ Features

- **Multi-Architecture Support**: Implementation of 6 different deep learning models
  - U-Net with ResNet34/ResNet50 encoders
  - U-Net with MobileNetV2/MobileNetV3 encoders  
  - U-Net with EfficientNet B0/B3 encoders

- **Advanced Data Processing**:
  - Albumentations-based augmentation pipeline
  - Geometric and photometric transformations
  - Robust preprocessing and normalization

- **Comprehensive Evaluation**:
  - Dice Score, IoU, and Accuracy metrics
  - Training/Validation loss tracking
  - Early stopping and model checkpointing

## 🏗️ Architecture

The system follows an encoder-decoder architecture with skip connections:

```
Input Image (256×256) → Encoder → Bottleneck → Decoder → Segmentation Mask
                           ↓                      ↑
                    Skip Connections ──────────────
```

### System Components:
1. **Image Acquisition**: Medical imaging devices or mobile cameras
2. **Preprocessing**: Resizing, normalization, and augmentation
3. **Segmentation Model**: Deep learning inference
4. **Post-Processing**: Morphological operations and noise removal
5. **Evaluation & Storage**: Performance metrics and result storage

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
git clone https://github.com/your-username/foot-ulcer-segmentation.git
cd foot-ulcer-segmentation
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

### Running the Project
Since this is a single-file implementation, you can run it directly:

```bash
# If it's a Jupyter notebook
jupyter notebook foot_ulcer_segmentation.ipynb

# If it's a Python script
python foot_ulcer_segmentation.py
```

### Implemented Models:
- ✅ U-Net with ResNet34 encoder
- ✅ U-Net with ResNet50 encoder  
- ✅ U-Net with MobileNetV2 encoder
- ✅ U-Net with MobileNetV3 encoder
- ✅ U-Net with EfficientNet B0 encoder
- ✅ U-Net with EfficientNet B3 encoder

### Key Features in the Code:
- Data loading and preprocessing
- Multiple model architectures
- Training loop with validation
- Performance evaluation metrics
- Results visualization

## 📈 Model Performance

| Model | Train IoU | Validation IoU | Test IoU | Training Loss | Validation Loss |
|-------|-----------|---------------|----------|---------------|-----------------|
| **EfficientNet B3** | **0.8786** | **0.8372** | **0.7666** | **0.0112** | **0.0251** |
| ResNet50 | 0.7227 | 0.7980 | 0.7291 | 0.0287 | 0.0468 |
| EfficientNet B0 | 0.8075 | 0.8096 | 0.7331 | 0.0156 | 0.0432 |
| ResNet34 | 0.7165 | 0.7273 | 0.7242 | 0.0301 | 0.0523 |
| MobileNet V3 | 0.8542 | 0.7891 | 0.7156 | 0.0178 | 0.0489 |
| MobileNet V2 | 0.8234 | 0.7634 | 0.7023 | 0.0198 | 0.0512 |

**🏆 Best Performing Model: EfficientNet B3**
- Highest Test IoU: 76.66%
- Lowest Training Loss: 0.0112
- Best generalization capability

## 📸 Results

### Model Performance Summary
The project achieved excellent segmentation results with EfficientNet B3 as the top performer:

**🏆 Best Results:**
- **Test IoU**: 76.66% (EfficientNet B3)
- **Validation IoU**: 83.72% (EfficientNet B3)
- **Training Accuracy**: 87.86% (EfficientNet B3)

### Key Achievements:
✅ Successfully implemented 6 different deep learning architectures  
✅ Achieved over 76% IoU on test dataset  
✅ Effective transfer learning with ImageNet weights  
✅ Robust data augmentation pipeline  
✅ Early stopping to prevent overfitting

## 📁 Project Structure

```
foot-ulcer-segmentation/
├── foot_ulcer_segmentation.py     # Main project file
├── README.md                      # This file
└── requirements.txt              # Dependencies (optional)
```

**Single File Implementation**: This project is contained in one main file that includes:
- Data loading and preprocessing
- Model architecture definitions
- Training and validation loops
- Evaluation metrics
- Results visualization

## 🔧 Configuration

### Training Parameters
```python
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
IMAGE_SIZE = 256
NUM_WORKERS = 4
```

### Augmentation Pipeline
```python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Normalize(),
    ToTensorV2()
])
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## 🔮 Future Work

- [ ] **Enhanced Model Performance**
  - Explore transformer-based architectures (SegFormer, Swin Transformer)
  - Implement self-supervised learning techniques
  - Larger dataset collection and curation

- [ ] **Computational Efficiency**
  - Model quantization and pruning
  - Knowledge distillation
  - Edge device optimization

- [ ] **Clinical Integration**
  - Web/mobile application development
  - Real-time inference pipeline
  - DICOM support

- [ ] **Advanced Features**
  - Multi-class segmentation (severity levels)
  - Uncertainty estimation
  - 3D segmentation capabilities

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
3. Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks.
4. Chowdhury, S. R., Rahman, M. M., & Momen, S. A. (2024). Semantic Segmentation of Diabetic Foot Ulcer using U-Net Architecture.

### Datasets
- [Medetec Foot Ulcer Dataset](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Medetec_foot_ulcer_224)
- [Kaggle Diabetic Foot Ulcer Dataset](https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu)
- [FUSC Challenge](https://fusc.grand-challenge.org/)

## 👥 Authors

**Ashish Kumar** - *Student ID: 22052541*
- Email: ashish.22052541@kiit.ac.in
- GitHub: [@ashish-kumar-1](https://github.com/ashish-kumar-1)

**Ashish Kumar** - *Student ID: 22052542*  
- Email: ashish.22052542@kiit.ac.in
- GitHub: [@ashish-kumar-2](https://github.com/ashish-kumar-2)

**Supervisor:** Prof. Rajdeep Chatterjee
- School of Computer Engineering
- KIIT Deemed to be University, Bhubaneswar

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof. Rajdeep Chatterjee for guidance and supervision
- KIIT University for providing computational resources
- The open-source community for datasets and tools
- Medical professionals who provided domain expertise

## 📞 Contact

For questions, suggestions, or collaborations:
- 📧 Email: ashish.22052541@kiit.ac.in
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/foot-ulcer-segmentation/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/foot-ulcer-segmentation/discussions)

---

⭐ **Star this repository if you find it helpful!**

*This project is part of the Bachelor's degree requirements at KIIT Deemed to be University, School of Computer Engineering.*
