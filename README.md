# Medical Images Segmentation & 3D Organ Viewer

This repository implements segmentation and 3D visualization for medical images of lungs, liver, and heart. It includes scripts and notebooks for mask conversion, 3D mesh generation, and a GUI application for interactive exploration of organ models.

## Features

- Convert PNG segmentation masks (predicted / ground truth) into NumPy arrays.  
- Generate clean 3D meshes (STL files) from mask arrays.  
- Visualize the 3D organ models (heart, liver, lungs) via a GUI with controls for color, opacity, visibility, and switching organs.  
- Support for keyboard shortcuts:  
  - **1** — Heart  
  - **2** — Liver  
  - **3** — Lungs  

## Repository Structure
├── GT_heart/ # Ground truth masks for heart
├── GT_lungs/ # Ground truth masks for lungs
├── masks_heart/ # Predicted masks for heart
├── masks_liver/ # Predicted masks for liver
├── masks_lungs/ # Predicted masks for lungs
├── cleaned_output/ # Output folder for cleaned STL models
│ ├── heart_model_clean.stl
│ ├── Liver_model_clean.stl
│ ├── Lungs_model_clean.stl
├── convert_gt_pngs_to_npy.py # Script to convert GT PNG masks → NPY
├── convert_pngs_to_npy.py # Script to convert predicted PNG masks → NPY
├── heart_3d.py # Convert heart NPY to STL
├── liver_3d.py # Convert liver NPY to STL
├── lung_3d.py # Convert lung NPY to STL
├── organ_3d_viewer.py # GUI viewer for 3D organ models
├── organ_viewer_gui.py # (Alternate GUI / support)
├── Eval1.py # Evaluation/metrics scripts (if any)
├── Eval2.py # Additional evaluation scripts
└── *.ipynb # Jupyter notebooks (Heart_Seg, Liver_Seg, Lungs_Seg etc.)

## Requirements

- Python 3.8+  
- numpy  
- scikit-image  
- Pillow  
- imageio  
- pyvista  

Usage
Convert segmentation masks
      Run:
      python convert_gt_pngs_to_npy.py
      python convert_pngs_to_npy.py
      

to process ground truth and predicted masks into .npy arrays.
      Generate 3D meshes (STL)
      Run:
      
      python heart_3d.py
      python liver_3d.py
      python lung_3d.py


This will output cleaned STL mesh files into the cleaned_output/ folder.

Launch 3D viewer GUI
      python organ_3d_viewer.py


Then you can interactively:
      Switch organs (1: Heart, 2: Liver, 3: Lungs)
      
      Adjust color / opacity
      
      Toggle visibility

      

Evaluation
      (Optional) Use Eval1.py, Eval2.py, or Jupyter notebooks (*.ipynb) to compute segmentation metrics (e.g., Dice, IoU) comparing predicted masks vs ground truth.
      
      Examples / Screenshots
      
      (You may add images or GIFs here of the 3D viewer in action, segmentation overlays, etc.)

## Segmentation with U-Net and SegNet

This project includes deep learning pipelines for segmenting **lungs, liver, and heart** from medical CT images using **U-Net** and **SegNet** architectures.

### Features

- Train U-Net and SegNet on 2D slices of organs.
- Preprocess images and masks to a consistent size (128×128 by default).
- Metrics supported: Accuracy, Dice Coefficient, IoU, Precision, Recall.
- Visualize predictions overlaying original images with predicted and ground truth masks.
- Split datasets into training and validation sets for robust evaluation.

### Dataset Loading

```python
from data_loader import load_dataset

# Example: load lungs dataset
X, y = load_dataset("lungs")

# Train/Validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

License
      This project is released under the MIT License.

Contributing
      Contributions (bug fixes, enhancements, new organs, etc.) are welcome. Feel free to open issues or pull requests.

Contact
      If you have questions or suggestions, you can open an issue in this repository.
