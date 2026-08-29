# 🧠 Facial Recognition on Extended Yale-B Dataset

A comparative study of **deep learning and traditional machine learning** approaches for facial recognition under varying lighting conditions. Built using the [Extended Yale-B dataset](https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full) (28 subjects, ~16,380 images).

---

## 📊 Results Summary

| Model | Test Accuracy |
|---|---|
| FaceNet (Fine-tuned)         | **98.37%**    |
| ResNet50 (Transfer Learning) | **98.13%** |
| CNN from Scratch | 96.13% |
| SVM (RBF Kernel) + PCA | 91.06% |
| Random Forest (Tuned) | 73.34% |
| KNN (k=1) + PCA | 83.15% |

---

## 📁 Project Structure

```
📦 Team05
├── 📂 Team05_Code.py/
│   ├── resnet50_transfer_learning.ipynb   # ResNet50 fine-tuned on Yale-B
│   ├── cnn_from_scratch.ipynb             # Custom 5-block CNN with PyTorch
│   ├── facenet_yaleb_pipeline.ipynb       # FaceNet embeddings + shadow recovery
│   ├── random_forest_model.ipynb          # Random Forest with hyperparameter tuning
│   └── Traditional_ML_Classifiers.ipynb  # KNN & SVM with PCA (Eigenfaces)
├── Team05_Report.pdf
└── Team05_Presentation.pdf
```

---

## 🔍 Approaches
### 1. ResNet50 — Transfer Learning *(98.13%)*
- Pretrained on ImageNet, fine-tuned for 28-class face recognition
- Modified `conv1` layer to accept grayscale input
- Label smoothing + cosine annealing LR scheduler
- Data augmentation: flips, rotations, affine transforms, color jitter

### 2. Custom CNN from Scratch *(96.13%)*
- 5 convolutional blocks (64 → 512 channels) with batch normalization and dropout
- Two fully connected layers (256 → 128) for feature mapping
- Class-weighted cross-entropy loss to handle imbalance
- Early stopping with best-model checkpointing

### 3. FaceNet Pipeline *(Best: 98.37%)*
- Uses `keras-facenet` pretrained embeddings
- Preprocessing with **gamma correction** and **CLAHE** to recover facial features from heavy shadows
- Fine-tuned top 20 layers of the FaceNet base model
- 80/10/10 train/val/test split

### 4. Random Forest *(73.34%)*
- Raw pixel features (128×128 images flattened to 16,384 dims)
- Compared Gini vs. Entropy criteria
- Hyperparameter tuning via `RandomizedSearchCV` (30 iterations)
- Also tested with PCA reduction (accuracy dropped significantly)

### 5. Traditional ML — KNN & SVM + PCA *(Eigenfaces)*
- PCA with 450 components (~95% variance retained)
- KNN: Best accuracy at k=1 (83.15%)
- SVM: RBF kernel outperformed linear and polynomial (91.06%)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn)

- **Deep Learning:** PyTorch, TensorFlow/Keras, keras-facenet
- **ML:** scikit-learn (SVM, KNN, Random Forest, PCA)
- **Image Processing:** OpenCV, torchvision
- **Platform:** Kaggle / Google Colab (GPU)

---

## 📦 Dataset

**Extended Yale-B (Cropped)** — 28 subjects × ~64 illumination conditions each  
Source: [Kaggle — jensdhondt/extendedyaleb-cropped-full](https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full)

The dataset features extreme lighting variation (frontal to near-profile lighting), making it a strong benchmark for illumination-robust face recognition.

---

## ▶️ Running the Notebooks

1. Clone this repository
2. Download the dataset via KaggleHub (handled automatically in each notebook):
   ```python
   import kagglehub
   path = kagglehub.dataset_download("jensdhondt/extendedyaleb-cropped-full")
   ```
3. Open any notebook in Jupyter or Kaggle/Colab and run all cells

> **Note:** Deep learning notebooks require a GPU for reasonable training times.

---

## 📌 Key Takeaways

- Transfer learning (ResNet50) significantly outperforms training from scratch on a moderately sized dataset
- Shadow recovery preprocessing (CLAHE + gamma correction) is critical for the Yale-B dataset
- Traditional ML methods (SVM + PCA/Eigenfaces) remain competitive and interpretable
- Random Forest struggles with raw high-dimensional pixel features — PCA actually hurt performance here

---

## 👥 Team

Team 05 — Mini Project
