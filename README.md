# 3D Organ Segmentation Viewer

This project provides a Python-based GUI for visualizing 3D organ models (heart, liver, lungs) generated from medical segmentation masks. It includes scripts for converting mask images to 3D meshes and a PyVista-based viewer with interactive controls for organ switching, color, opacity, and visibility.

## Features
- Visualize 3D models of heart, liver, and lungs
- Switch between organs using keyboard shortcuts (1: Heart, 2: Liver, 3: Lungs)
- Adjust color and opacity of each organ
- Toggle organ visibility
- Automated conversion of PNG masks to NPY arrays and STL meshes

## Folder Structure
```
heart_vis/
├── cleaned_output/
│   ├── heart_model_clean.stl
│   ├── Liver_model_clean.stl
│   ├── Lungs_model_clean.stl
├── GT_heart/
├── GT_lungs/
├── masks_heart/
├── masks_liver/
├── masks_lungs/
├── convert_gt_pngs_to_npy.py
├── convert_pngs_to_npy.py
├── heart_3d.py
├── liver_3d.py
├── lung_3d.py
├── organ_3d_viewer.py
```

## Requirements
- Python 3.8+
- pyvista
- numpy
- scikit-image
- Pillow
- imageio

Install dependencies:
```bash
pip install pyvista numpy scikit-image Pillow imageio
```

## Usage
1. **Convert PNG masks to NPY arrays:**
   - Run `convert_gt_pngs_to_npy.py` and `convert_pngs_to_npy.py` to process ground truth and predicted masks.
2. **Generate STL meshes:**
   - Run `heart_3d.py`, `liver_3d.py`, and `lung_3d.py` to create 3D models in `cleaned_output/`.
3. **Launch the viewer:**
   - Run `organ_3d_viewer.py` to open the GUI and interact with the 3D models.

## Controls
- **Switch organs:** Press 1 (Heart), 2 (Liver), 3 (Lungs)
- **Change color:** Use the color slider
- **Change opacity:** Use the opacity slider
- **Toggle visibility:** Use the checkbox

## How to Upload to GitHub
1. Initialize a git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: 3D organ segmentation viewer"
   ```
2. Create a new repository on GitHub (https://github.com/new)
3. Link your local repo and push:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

## License
MIT License

## Contact
For questions or contributions, open an issue or pull request on GitHub.
