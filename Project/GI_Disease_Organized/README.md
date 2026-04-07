# GI Disease Detection using Deep Learning

A deep learning pipeline for classifying gastrointestinal (GI) diseases from endoscopic images using **EfficientNet-B3** trained on the **Kvasir Dataset**. The model achieves **~94.5% accuracy** across 8 disease/normal classes.

---

## 📁 Project Structure

```
GI_Disease_Project/
├── model/
│   ├── model_efficientnet_b3.pth   # Trained EfficientNet-B3 weights
│   └── class_names.json            # Class label mapping (8 classes)
│
├── results/
│   ├── accuracy_curve.png          # Training & validation accuracy over epochs
│   ├── loss_curve.png              # Training & validation loss over epochs
│   ├── confusion_matrix.png        # Per-class confusion matrix heatmap
│   └── metrics_report.txt          # Precision, recall, F1-score report
│
├── data/
│   ├── labels.npy                  # Ground truth labels (numpy array)
│   └── predictions.npy             # Model predictions (numpy array)
│
├── src/
│   └── inference.py                # Run inference on a single image
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🩺 Classes

The model classifies endoscopic images into 8 categories:

| # | Class |
|---|-------|
| 0 | dyed-lifted-polyps |
| 1 | dyed-resection-margins |
| 2 | esophagitis |
| 3 | normal-cecum |
| 4 | normal-pylorus |
| 5 | normal-z-line |
| 6 | polyps |
| 7 | ulcerative-colitis |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **94.5%** |
| Macro Avg Precision | 0.95 |
| Macro Avg Recall | 0.94 |
| Macro Avg F1-Score | 0.94 |

Full per-class breakdown is in [`results/metrics_report.txt`](results/metrics_report.txt).

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/GI_Disease_Project.git
cd GI_Disease_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Inference

```bash
python src/inference.py <path_to_image>
```

**Example:**

```bash
python src/inference.py test_image.jpg
```

**Output:**
```
Predicted class: polyps
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 |
| Framework | PyTorch + timm |
| Input Size | 300 × 300 |
| Dataset | Kvasir Dataset |
| Classes | 8 |
| Test Accuracy | ~94.5% |

---

## 📦 Dependencies

```
torch
torchvision
timm
numpy
scikit-learn
matplotlib
seaborn
Pillow
```

---

## 📌 Dataset

This project uses the [Kvasir Dataset](https://datasets.simula.no/kvasir/), a multi-class endoscopic image dataset for GI tract classification, containing 8,000 images across 8 categories.

---

## 📄 License

This project is for academic and research purposes only.
