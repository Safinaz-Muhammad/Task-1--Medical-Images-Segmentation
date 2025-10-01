# Lungs_clean_and_debug.py
import os
import glob
import numpy as np
import imageio.v3 as iio
from scipy import ndimage as ndi
from skimage import morphology, measure
import pyvista as pv
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ========== USER CONFIG ==========
MASK_DIR = r"C:\Users\Safinaz's Laptop\heart_vis\masks_lungs"   # <--- set if different
OUT_DIR = os.path.join(os.path.dirname(MASK_DIR), "cleaned_output")
os.makedirs(OUT_DIR, exist_ok=True)
CLEANED_PNG_DIR = os.path.join(OUT_DIR, "cleaned_masks")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")
os.makedirs(CLEANED_PNG_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Cleaning hyperparams (tweak these if needed)
MIN_OBJ_SIZE_3D = 1500      # remove tiny 3D islands (increase if still noisy)
MIN_OBJ_SIZE_2D = 100       # remove tiny 2D blobs per slice
CLOSING_RADIUS = 2          # morphological ball radius
GAUSSIAN_SIGMA = 1.0        # 3D gaussian smoothing prior to marching cubes
MEDIAN_FILTER_SIZE = (3, 3, 3)
Z_PROPAGATION_ITER = 2      # how many dilations to propagate masks to empty slices
# fraction of triangles to remove for fast preview (0.0..0.99)
DECIMATE_REDUCTION = 0.6

# Colors available for interactive color slider
COLORS = ["red", "green", "blue", "yellow",
          "magenta", "cyan", "white", "orange"]

# ==================================

# ---------- load files ----------
files = sorted(
    glob.glob(os.path.join(MASK_DIR, "*.png")) +
    glob.glob(os.path.join(MASK_DIR, "*.PNG")) +
    glob.glob(os.path.join(MASK_DIR, "*.jpg")) +
    glob.glob(os.path.join(MASK_DIR, "*.jpeg")) +
    glob.glob(os.path.join(MASK_DIR, "*.tif")) +
    glob.glob(os.path.join(MASK_DIR, "*.tiff"))
)
if not files:
    raise SystemExit(f"No mask files found in {MASK_DIR!r}")

print(f"Found {len(files)} mask files. Example: {os.path.basename(files[0])}")

# read stack (grayscale)
slices = []
for f in files:
    img = iio.imread(f)
    if img.ndim == 3:
        img = img[..., 0]
    slices.append(img)
volume = np.stack(slices, axis=0).astype(np.uint8)
Z, H, W = volume.shape
print("Raw volume shape:", volume.shape, "dtype:", volume.dtype)

# ---------- diagnostics ----------
areas = (volume > 0).sum(axis=(1, 2))
print("Slice area stats (voxels): min, mean, max =",
      areas.min(), areas.mean(), areas.max())
# find slices with large relative area change vs prev slice
deltas = np.abs(np.diff(areas).astype(float))
if len(deltas):
    median_delta = np.median(deltas)
else:
    median_delta = 0
# these are indices where change occurs between i and i+1
problem_indices = np.where(deltas > max(3*median_delta, 0.2*areas.mean()))[0]
print("Problem slice indices (where area jumps):",
      problem_indices.tolist()[:20])

# ---------- per-slice cleaning ----------
print("Step 1: per-slice fill holes + remove tiny objects")
binary = (volume > 0)
filled = np.zeros_like(binary)
for i in range(Z):
    sl = binary[i]
    sl = ndi.binary_fill_holes(sl)
    # remove small 2D blobs
    sl = morphology.remove_small_objects(
        sl.astype(bool), min_size=MIN_OBJ_SIZE_2D)
    filled[i] = sl

# ---------- 3D median filter ----------
print("Step 2: 3D median filter")
med = ndi.median_filter(filled.astype(np.uint8), size=MEDIAN_FILTER_SIZE)

# ---------- remove small 3D objects ----------
print("Step 3: remove small 3D objects")
clean3d = morphology.remove_small_objects(
    med.astype(bool), min_size=MIN_OBJ_SIZE_3D)
clean3d = clean3d.astype(np.uint8)

# ---------- z-propagation for empty slices ----------
print("Step 4: z-propagation to fill empty slices (if any)")
for i in range(1, Z):
    if clean3d[i].sum() < 10 and clean3d[i-1].sum() > 50:
        # dilate previous slice and use it as starting mask
        filled_slice = ndi.binary_dilation(
            clean3d[i-1], iterations=Z_PROPAGATION_ITER)
        clean3d[i] = filled_slice

# ---------- keep largest 3D component ----------
print("Step 5: keep largest connected 3D component")
labeled, ncomp = ndi.label(clean3d)
if ncomp == 0:
    raise SystemExit(
        "No components after cleaning — try lowering MIN_OBJ_SIZE_3D or MIN_OBJ_SIZE_2D")
counts = np.bincount(labeled.ravel())
# ignore background (index 0)
largest_label = counts[1:].argmax() + 1
clean3d = (labeled == largest_label).astype(np.uint8)
print("Kept largest component voxels:", counts[largest_label])

# ---------- 3D morphological closing to smooth boundaries ----------
print("Step 6: 3D morphological closing")
clean3d = morphology.binary_closing(
    clean3d, morphology.ball(CLOSING_RADIUS)).astype(np.uint8)

# ---------- optional gaussian smooth and re-threshold ----------
if GAUSSIAN_SIGMA > 0:
    print("Applying gaussian smoothing then threshold")
    gauss = ndi.gaussian_filter(clean3d.astype(float), sigma=GAUSSIAN_SIGMA)
    clean3d = (gauss > 0.5).astype(np.uint8)

print("Final cleaned volume shape:", clean3d.shape,
      "counts:", np.bincount(clean3d.ravel()))

# ---------- save cleaned stack (PNG) ----------
print("Saving cleaned PNG stack to:", CLEANED_PNG_DIR)
for i in range(Z):
    out_path = os.path.join(CLEANED_PNG_DIR, f"slice_{i:04d}.png")
    image = (clean3d[i].astype(np.uint8) * 255)
    iio.imwrite(out_path, image)

# ---------- create overlays for a few problem slices ----------
print("Creating overlays for inspection (saved to overlays/)")
inspect_idxs = sorted(set(list(problem_indices) +
                      list(np.argsort(areas)[:3]) + list(np.argsort(-areas)[:3])))
inspect_idxs = [i for i in inspect_idxs if 0 <= i < Z]
for i in inspect_idxs[:12]:
    orig = volume[i]
    cleaned_sl = clean3d[i].astype(np.uint8) * 255
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(orig, cmap='gray')
    ax[0].set_title(f"orig slice {i}")
    ax[1].imshow(orig, cmap='gray')
    ax[1].imshow(cleaned_sl, cmap='Reds', alpha=0.5)
    ax[1].set_title(f"orig + cleaned overlay {i}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OVERLAY_DIR, f"overlay_{i:04d}.png"), dpi=150)
    plt.close(fig)

# --------- marching cubes to mesh ----------
print("Extracting mesh with marching_cubes ...")
verts, faces, normals, vals = measure.marching_cubes(
    clean3d, level=0.5, spacing=(1.0, 1.0, 1.0))
faces = faces.reshape(-1, 3)
faces_packed = np.hstack([np.full((faces.shape[0], 1), 3), faces])
mesh = pv.PolyData(verts, faces_packed.ravel().astype(np.int64))
mesh.clean(inplace=True)
mesh.compute_normals(inplace=True)
print("Full mesh: verts:", mesh.n_points, "cells:", mesh.n_cells)

# ---------- extra mesh cleanup & smoothing ----------
print("Applying extra mesh cleanup...")
# keep only largest connected surface
mesh = mesh.connectivity(largest=True)
# fill holes (adjust value depending on dataset)
mesh = mesh.fill_holes(100.0)
mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)  # smooth surface
# simplify (0.5 → 50% reduction)
mesh = mesh.decimate_pro(0.5)
mesh = mesh.compute_normals(auto_orient_normals=True)
print("Mesh after cleanup: verts:", mesh.n_points, "cells:", mesh.n_cells)

print("Smoothing surface (volume + mesh)...")

# ---------- make a decimated fast mesh for interactive display ----------
fast_mesh = mesh.copy()
try:
    # decimate_pro available in many pyvista versions
    fast_mesh = fast_mesh.decimate_pro(DECIMATE_REDUCTION)
    print("Fast mesh decimated: verts:",
          fast_mesh.n_points, "cells:", fast_mesh.n_cells)
except Exception:
    try:
        fast_mesh = fast_mesh.decimate(DECIMATE_REDUCTION)
        print("Fast mesh decimated (alternate): verts:",
              fast_mesh.n_points, "cells:", fast_mesh.n_cells)
    except Exception:
        print("Decimation not available; using full mesh for display (may be slow)")

# ---------- interactive PyVista viewer ----------
plotter = pv.Plotter(window_size=(1100, 800))
holder = {"actor": None}
state = {"color": "red", "opacity": 0.7, "use_fast": True}


def add_actor(color, opacity, use_fast=True):
    # remove actor
    if holder["actor"] is not None:
        try:
            plotter.remove_actor(holder["actor"], reset_camera=False)
        except Exception:
            pass
    m = fast_mesh if (use_fast and 'fast_mesh' in globals()) else mesh
    holder["actor"] = plotter.add_mesh(
        m, color=color, opacity=opacity, smooth_shading=True)
    plotter.reset_camera()
    plotter.render()


add_actor(state["color"], state["opacity"], state["use_fast"])

# opacity slider


def on_opacity(val):
    state["opacity"] = float(val)
    add_actor(state["color"], state["opacity"], state["use_fast"])


plotter.add_slider_widget(on_opacity, rng=[
                          0.0, 1.0], value=state["opacity"], title="Opacity", pointa=(0.02, 0.02), pointb=(0.30, 0.02))

# color slider (index)


def on_color_idx(val):
    idx = int(round(val))
    idx = max(0, min(idx, len(COLORS)-1))
    state["color"] = COLORS[idx]
    add_actor(state["color"], state["opacity"], state["use_fast"])


plotter.add_slider_widget(on_color_idx, rng=[0, len(
    COLORS)-1], value=0, title="Color index", pointa=(0.7, 0.02), pointb=(0.95, 0.02))

# toggle fast/full mesh


def toggle_fast(full):
    # checkbox gives True when checked; invert to match label
    state["use_fast"] = not full
    add_actor(state["color"], state["opacity"], state["use_fast"])


plotter.add_checkbox_button_widget(
    lambda v: toggle_fast(v), value=True, position=(10, 60))
plotter.add_text("Check to use FAST preview mesh (uncheck for full mesh)",
                 position="lower_left", font_size=8)

# key bindings


def save_mesh():
    out_mesh = os.path.join(OUT_DIR, "Lungs_model_clean.stl")
    mesh.save(out_mesh)
    print("Saved mesh:", out_mesh)


def save_cleaned_pngs():
    print("Cleaned PNGs already saved to:", CLEANED_PNG_DIR)


plotter.add_key_event("s", save_mesh)
plotter.add_key_event("e", save_cleaned_pngs)
plotter.add_text("Keys: 's' save mesh, 'e' show cleaned PNG path",
                 position="upper_right", font_size=9)

# show some debug info
plotter.add_text(
    f"Cleaned voxels: {int(clean3d.sum())}  |  Slices: {Z}", position="upper_left", font_size=10)

print("Overlays saved:", OVERLAY_DIR)
print("Cleaned PNGs saved:", CLEANED_PNG_DIR)
print("Press 's' in the viewer to save full STL, 'e' to view cleaned PNG path.")

plotter.show()
