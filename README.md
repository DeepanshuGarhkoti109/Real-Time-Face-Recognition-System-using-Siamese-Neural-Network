# 👤 Real-Time Face Recognition System using Siamese Neural Network (SNN)

[![Research Paper](https://img.shields.io/badge/Research_Paper-PDF_Available-red?style=for-the-badge&logo=adobe-acrobat-reader)](RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.00%25-success?style=for-the-badge)](#-experimental-results--benchmarks)

A state-of-the-art **One-Shot Real-Time Face Verification and Recognition System** built using a **Convolutional Siamese Neural Network (SNN)** architecture. This project includes an end-to-end deep learning pipeline—from custom video stream data collection and data augmentation to custom model training, evaluation, and live webcam authentication.

> 📄 **Published Research Paper:**  
> **Title:** *Real - Time Face Recognition System using Siamese Neural Network for Enhanced Security Applications*  
> **Authors:** Deepanshu Garhkoti, Sameer Gupta, Dr. Neha Agarwal  
> **Institution:** Department of Computer Science, Amity University, Uttar Pradesh  
> 🔗 **[Click here to view/download full PDF Research Paper](RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf)**  
>  
> 🌐 **Conference Presentation:**  
> This system architecture and its decision-making use case were presented at the **International Conference on The Impact of Artificial Driven Decision Making and Agile Management Practices for Sustainable Development (ICADMS 2024)** (*"Artificial Brain in Decision Making"*).  
> 🔗 **[View ResearchGate Publication Abstract](https://www.researchgate.net/publication/380666324_Artificial_Brain_in_Decision_Making_Abstract)**

---

## 📌 Executive Summary

Traditional face recognition algorithms like Eigenfaces (PCA) and Local Binary Patterns (LBP) struggle with illumination variations, pose shifts, facial hair, and occlusions. Standard deep learning classifiers also fail in real-world security systems where training hundreds of photos per person is infeasible.

This project implements a **Siamese Neural Network (SNN)** that learns a **similarity metric** (distance function) between facial feature embeddings rather than classifying static classes. By leveraging **One-Shot Learning**, the system can verify an individual's identity instantly using only a single reference image (Anchor).

### Key Highlights
- 🧠 **One-Shot Verification:** Identifies and verifies individuals with as little as a single anchor image.
- 📐 **Custom L1 Distance Similarity Layer:** Learns feature vector distances $|e_1 - e_2|$ between paired inputs.
- ⚡ **Real-Time OpenCV Pipeline:** Instant identity verification feed via live webcam stream.
- 📊 **96.00% Accuracy:** Outperforms benchmark architectures like FaceNet (85.52%), TripleNet (84.32%), and ResNet-50 (81.21%) on the *Labeled Faces in the Wild (LFW)* dataset.
- 🔄 **Custom Data Pipeline & Augmentation:** Built with `tf.data` and custom TensorFlow `tf.GradientTape` training loops.

---

## 🔬 Research Paper Overview

| Attribute | Details |
| :--- | :--- |
| **Paper Title** | Real - Time Face Recognition System using Siamese Neural Network for Enhanced Security Applications |
| **Primary File** | [`RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf`](RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf) |
| **Authors** | Deepanshu Garhkoti, Sameer Gupta, Dr. Neha Agarwal |
| **Dataset Used** | Labeled Faces in the Wild (LFW) + Custom Live Webcam Captures |
| **Loss Function** | Binary Cross-Entropy Loss |
| **Optimizer** | Adam ($\alpha = 0.0001$) |
| **Achieved Accuracy** | **96.00%** |

### Abstract Summary
> SNNs offer an attractive solution due to their ability to learn similarity metrics between face embeddings. We propose a real-time facial recognition system that uses a Convolutional Siamese Network architecture. The network was trained on labeled face images using binary cross-entropy loss and Adam optimization. During live operation, the system captures video frames, performs verification by calculating embedding distances against reference images, and displays verification status in real time.

---

## 📐 System Architecture & Workflow

The system uses twin Convolutional Neural Networks (sharing identical weights) to map input images into a high-dimensional feature space ($4096$-dimensional embedding vectors).

```mermaid
flowchart TD
    subgraph Data Input
        A[Anchor Image 105x105x3]
        V[Validation/Input Image 105x105x3]
    end

    subgraph Twin Feature Extractors (Shared CNN Weights)
        A --> E1[Shared Embedding Network]
        V --> E2[Shared Embedding Network]
    end

    subgraph Feature Vectors
        E1 --> F1[Vector 1: 4096-dim]
        E2 --> F2[Vector 2: 4096-dim]
    end

    subgraph Metric Learning
        F1 --> L1[Custom L1 Distance Layer |v1 - v2|]
        F2 --> L1
    end

    subgraph Classification Output
        L1 --> Dense[Dense Layer + Sigmoid]
        Dense --> Prediction["Similarity Score [0.0 - 1.0]"]
    end

    Prediction --> Verification{"Score >= Threshold?"}
    Verification -->|Yes| Verified["VERIFIED (True)"]
    Verification -->|No| Unverified["UNVERIFIED (False)"]
```

### Network Component Specs

1. **Embedding Layer (`make_embedding`)**:
   - `Conv2D` (64 filters, $10\times10$ kernel, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
   - `Conv2D` (128 filters, $7\times7$ kernel, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
   - `Conv2D` (128 filters, $4\times4$ kernel, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
   - `Conv2D` (256 filters, $4\times4$ kernel, ReLU) $\rightarrow$ `Flatten`
   - `Dense` ($4096$ units, Sigmoid activation)

2. **Custom L1 Distance Layer (`L1Dist`)**:
   $$\text{Distance} = |e_{\text{anchor}} - e_{\text{validation}}|$$

3. **Classification Layer**:
   - `Dense` ($1$ unit, Sigmoid activation outputting probability $P(\text{same identity})$).

---

## 🏆 Experimental Results & Benchmarks

Our proposed **Convolutional Siamese Neural Network** was rigorously benchmarked against existing facial recognition architectures:

### 1. State-of-the-Art Model Comparison
| Method / Model Architecture | Classifier / Loss | Accuracy |
| :--- | :--- | :---: |
| **Our Proposed Model (Convolutional Siamese Net)** | **L1 Distance + Sigmoid (BCE)** | **96.00%** 🏆 |
| SRANet (LFW Dataset) | Sigmoid | 96.00% |
| Hierarchical Bayesian Program Learning (HBPL) | Bayesian Prior | 95.20% |
| SRANet (LFW Dataset) | Softmax | 95.83% |
| FaceNet (ResNet-50 V2) | ReLU + Sigmoid | 93.00% |
| MobileFaceNet (MobileNet V2) | Softmax | 92.00% |
| SN-LF (Siamese Net under Non-Restricted Conditions) | Distance Metric | 90.31% |
| FaceNet (Schroff et al.) | Triplet Loss | 85.52% |
| TripleNet | Triple Attention | 84.32% |
| Affine Model | Term Structure | 81.80% |
| ResNet-50 | Standard CNN | 81.21% |
| 1-Nearest Neighbor (1-NN) | Euclidean Distance | 21.70% |

---

## 📂 Repository Structure

```
.
├── RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf  # Published Research Paper
├── main.ipynb                                                  # Complete End-to-End Notebook
├── data/                                                       # Dataset Directory (Auto-generated)
│   ├── anchor/                                                 # Anchor images captured via webcam
│   ├── positive/                                               # Matching positive samples
│   └── negative/                                               # Negative samples from LFW dataset
├── application_data/                                           # Live verification data store
│   ├── input_image/                                            # Real-time frame captures
│   └── verification_images/                                    # Baseline target images
└── README.md                                                   # Documentation
```

---

## 💻 Installation & Getting Started

### 1. Prerequisites
- Python 3.10+
- Webcam (for live verification)
- Recommended: NVIDIA GPU with CUDA support for accelerated training

### 2. Install Required Dependencies
```bash
pip install tensorflow opencv-python matplotlib numpy pypdf
```

### 3. Clone Repository
```bash
git clone https://github.com/<your-username>/Real-Time-Face-Recognition-System-using-Siamese-Neural-Network.git
cd Real-Time-Face-Recognition-System-using-Siamese-Neural-Network
```

---

## 🚀 Usage Guide

All execution steps, model definitions, training code, and evaluation routines are contained in [`main.ipynb`](main.ipynb).

### Step 1: Open the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter lab main.ipynb
```

### Step 2: Data Collection & Preparation
Run Section 2 of the notebook to set up data folders and collect samples:
- Press **`a`** during webcam stream execution to capture **Anchor images**.
- Press **`p`** during webcam stream execution to capture **Positive images**.
- Negative images are automatically downloaded and unpacked from the **Labeled Faces in the Wild (LFW)** dataset (`lfw.tgz`).

### Step 3: Model Training
Run Section 5 to train the Siamese Neural Network:
- **Optimizer:** Adam ($\text{lr} = 0.0001$)
- **Loss:** Binary Cross-Entropy Loss
- **Checkpoints:** Periodically saved in `./training_checkpoints/`

### Step 4: Real-Time Verification
Run Section 8 to launch the live OpenCV webcam verification application:
- Captures current webcam frame to `application_data/input_image/input_image.jpg`.
- Computes pairwise similarity scores against all target images in `application_data/verification_images/`.
- Displays real-time status (**VERIFIED / UNVERIFIED**) and matching percentage on video stream.

---

## 📖 Citation & References

If you use this repository or refer to the research paper/conference publication in your work, please cite:

```bibtex
% Primary Research Paper
@article{garhkoti2024realtime,
  title={Real-Time Face Recognition System using Siamese Neural Network for Enhanced Security Applications},
  author={Garhkoti, Deepanshu and Gupta, Sameer and Agarwal, Dr. Neha},
  journal={Department of Computer Science, Amity University, Uttar Pradesh},
  year={2024},
  url={https://github.com/GarhkotiDeepanshu/Real-Time-Face-Recognition-System-using-Siamese-Neural-Network/blob/main/RESEARCH_PAPER_Image_Recognition_Using_SNN_Architecture.pdf}
}

% ICADMS 2024 Conference Presentation & Abstract
@inproceedings{palit2024artificialbrain,
  title={Artificial Brain in Decision Making},
  author={Palit, Shamik and Garhkoti, Deepanshu and Gupta, Sameer and Agarwal, Dr. Neha},
  booktitle={International Conference on The Impact of Artificial Driven Decision Making and Agile Management Practices for Sustainable Development (ICADMS 2024)},
  year={2024},
  url={https://www.researchgate.net/publication/380666324_Artificial_Brain_in_Decision_Making_Abstract}
}
```

### Key Literature References
1. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-shot Image Recognition*. ICML Deep Learning Workshop.
2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR.
3. Huang, G. B., et al. *Labeled Faces in the Wild (LFW)*. University of Massachusetts, Amherst.

---

## 🌟 Resume & Project Highlights
- **Computer Vision & Deep Learning:** Built custom CNN feature embeddings & similarity distance layers in TensorFlow/Keras.
- **Metric Learning:** Applied one-shot learning techniques to solve high-accuracy facial verification without re-training models per identity.
- **Real-Time Systems:** Integrated OpenCV hardware video capture with TensorFlow inference pipelines for low-latency identity authentication.
- **Academic Research & Publications:** Co-authored and published a technical paper evaluating model accuracy ($96.00\%$).
- **Conference Presentation:** Presented the system architecture & AI decision-making use case at the **ICADMS 2024** International Conference ([ResearchGate Abstract](https://www.researchgate.net/publication/380666324_Artificial_Brain_in_Decision_Making_Abstract)).
