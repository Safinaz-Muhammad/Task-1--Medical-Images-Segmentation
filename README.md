<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>

# 🫀 MedSeg3D — Medical Image Segmentation & 3D Organ Viewer

> **Deep learning-powered organ segmentation (Heart · Liver · Lungs) with interactive 3D mesh visualization — built on U-Net and SegNet architectures.**

[Features](#-features) · [Architecture](#-model-architectures) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results) · [Metrics](#-evaluation-metrics) · [Contributing](#-contributing)

---

</div>

## 🧬 Overview

**MedSeg3D** is an end-to-end pipeline for **medical CT image segmentation and 3D reconstruction**. It covers everything from raw mask conversion to deep learning model training, evaluation, and interactive 3D visualization — making it suitable for medical research, education, and clinical prototyping.

```
Raw CT Slices ──► Preprocessing ──► U-Net / SegNet ──► Predicted Masks
                                                              │
                                                     NPY Arrays ──► STL Meshes ──► 3D Viewer GUI
```

---

## ✨ Features

| Category | Details |
|:---|:---|
| 🧠 **Deep Learning** | U-Net and SegNet on 2D CT slices for Heart, Liver, Lungs |
| 📐 **Metrics** | Dice Coefficient, IoU, Accuracy, Precision, Recall |
| 🗂️ **Data Pipeline** | PNG masks → NumPy arrays → STL 3D meshes |
| 🖥️ **3D Viewer** | Interactive GUI with color, opacity, and visibility controls |
| ⌨️ **Keyboard Shortcuts** | `1` Heart · `2` Liver · `3` Lungs |
| 📊 **Evaluation** | Predicted vs. ground-truth comparison with visual overlays |

---

## 🗂️ Repository Structure

```
MedSeg3D/
│
├── 📁 data/
│   ├── GT_heart/                  # Ground truth masks — heart
│   ├── GT_lungs/                  # Ground truth masks — lungs
│   ├── masks_heart/               # Predicted masks — heart
│   ├── masks_liver/               # Predicted masks — liver
│   └── masks_lungs/               # Predicted masks — lungs
│
├── 📁 cleaned_output/             # Generated 3D models (STL)
│   ├── heart_model_clean.stl
│   ├── Liver_model_clean.stl
│   └── Lungs_model_clean.stl
│
├── 📁 notebooks/
│   ├── Heart_Seg.ipynb            # Heart segmentation pipeline
│   ├── Liver_Seg.ipynb            # Liver segmentation pipeline
│   └── Lungs_Seg.ipynb            # Lungs segmentation pipeline
│
├── 🐍 convert_gt_pngs_to_npy.py   # GT PNG masks → NPY arrays
├── 🐍 convert_pngs_to_npy.py      # Predicted PNG masks → NPY arrays
├── 🐍 heart_3d.py                 # Heart NPY → STL mesh
├── 🐍 liver_3d.py                 # Liver NPY → STL mesh
├── 🐍 lung_3d.py                  # Lung NPY → STL mesh
├── 🐍 organ_3d_viewer.py          # Main 3D organ viewer GUI
├── 🐍 organ_viewer_gui.py         # Alternate GUI / support module
├── 🐍 Eval1.py                    # Evaluation script (metrics)
└── 🐍 Eval2.py                    # Additional evaluation script
```

---

## 🧠 Model Architectures

### U-Net

```
Input (128×128×1)
     │
 Encoder ──────────────────────────────────► Skip Connections
  [Conv→BN→ReLU] ×2 + MaxPool (×4 levels)            │
     │                                                 │
  Bottleneck                                           │
  [Conv→BN→ReLU] ×2                                   │
     │                                                 │
 Decoder ◄────────────────────────────── Concat ◄─────┘
  [UpSample + Conv→BN→ReLU] ×4
     │
 Output (128×128×1) ── Sigmoid
```

### SegNet

```
Encoder: VGG-style convolutions + MaxPooling (with pooling indices saved)
Decoder: MaxUnpooling (using saved indices) + Convolutions
Output:  Pixel-wise softmax classification
```

> Both models accept **128×128 grayscale** CT slices and output binary segmentation masks.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MedSeg3D.git
cd MedSeg3D
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```txt
numpy>=1.21
scikit-image>=0.19
Pillow>=9.0
imageio>=2.22
pyvista>=0.37
torch>=1.12
torchvision>=0.13
scikit-learn>=1.0
matplotlib>=3.5
```

---

## 🚀 Usage

### Step 1 — Convert Segmentation Masks to NumPy

```bash
# Ground truth masks
python convert_gt_pngs_to_npy.py

# Predicted masks
python convert_pngs_to_npy.py
```

Outputs `.npy` arrays saved alongside the mask folders.

---

### Step 2 — Generate 3D Meshes (STL)

```bash
python heart_3d.py
python liver_3d.py
python lung_3d.py
```

Cleaned STL files are saved to `cleaned_output/`.

**How it works:** The scripts apply marching cubes on the volumetric mask arrays, smooth the resulting mesh, remove small disconnected components, and export as STL.

---

### Step 3 — Launch the 3D Organ Viewer

```bash
python organ_3d_viewer.py
```

| Control | Action |
|:---|:---|
| `1` | Switch to Heart |
| `2` | Switch to Liver |
| `3` | Switch to Lungs |
| Slider | Adjust opacity |
| Color Picker | Change organ color |
| Toggle | Show / hide organ |

---

### Step 4 — Train a Segmentation Model (Notebooks)

Open the relevant notebook:

```bash
jupyter notebook notebooks/Heart_Seg.ipynb
```

**Dataset loading:**

```python
from data_loader import load_dataset
from sklearn.model_selection import train_test_split

# Load organ dataset
X, y = load_dataset("lungs")   # Options: "heart", "liver", "lungs"

# Split into train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### Step 5 — Evaluate Predictions

```bash
python Eval1.py
python Eval2.py
```

Computes Dice, IoU, Accuracy, Precision, and Recall by comparing predicted masks against ground truth.

---

## 📊 Results

### 🫀 Heart Segmentation

**U-Net**
![Heart U-Net](https://github.com/user-attachments/assets/65e971da-202c-4037-ad64-bb5160296095)

**SegNet**
![Heart SegNet](https://github.com/user-attachments/assets/f97eb396-a8ca-4661-8e6a-c3daf9356f04)

---

### 🫁 Lung Segmentation

**U-Net**
![Lungs U-Net](https://github.com/user-attachments/assets/dd9c5ee9-a213-499d-baff-cd0a6bc7e6b3)

---

### 🧫 Liver Segmentation

**U-Net**
![Liver U-Net](https://github.com/user-attachments/assets/e79203d8-f9c5-4e05-b48d-4464d0b53145)

---

## 📈 Evaluation Metrics

### Metric Definitions

| Metric | Formula | Interpretation |
|:---|:---|:---|
| **Dice Coefficient** | `2·|P∩G| / (|P|+|G|)` | Overlap quality; 1.0 = perfect match |
| **IoU (Jaccard Index)** | `|P∩G| / |P∪G|` | Stricter overlap measure |
| **Accuracy** | `(TP+TN) / Total` | Overall pixel classification rate |
| **Precision** | `TP / (TP+FP)` | How many predicted positives are correct |
| **Recall** | `TP / (TP+FN)` | How many true positives were found |

---

### Training Curves

**U-Net — Liver**
![U-Net Liver Eval](https://github.com/user-attachments/assets/41fc3be5-a3d3-47d3-844a-afb8055cae90)

**SegNet — Heart**
![SegNet Heart Eval](https://github.com/user-attachments/assets/e9628d37-8c60-48a8-9600-013d027a8492)

---

## 🛠️ Extending the Project

### Adding a New Organ

1. Add mask folders: `masks_<organ>/` and `GT_<organ>/`
2. Create `<organ>_3d.py` following the pattern of `heart_3d.py`
3. Register the organ in `organ_3d_viewer.py` under the `ORGANS` config dict
4. Add a keyboard shortcut (e.g., `4`)
5. Create a training notebook `<Organ>_Seg.ipynb`

### Swapping the Backbone

Both U-Net and SegNet encoders can be replaced with pretrained backbones (ResNet, EfficientNet) via `torchvision.models`. Update `model_factory.py` and pass `pretrained=True`.

---

## 🤝 Contributing

Contributions are welcome — bug fixes, new organs, architecture improvements, or documentation updates.

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/add-kidney-segmentation

# 3. Commit your changes
git commit -m "feat: add kidney segmentation pipeline"

# 4. Push and open a Pull Request
git push origin feature/add-kidney-segmentation
```

Please follow [PEP 8](https://peps.python.org/pep-0008/) and include docstrings for new functions.

---

## 📄 License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 📬 Contact

Found a bug or have a suggestion? [Open an issue](../../issues) — contributions and feedback are always welcome.

---

<div align="center">

Made with ❤️ for medical imaging research

</div>
