import os
import numpy as np
from PIL import Image

# Folders containing mask PNGs
folders = {
    'liver': 'masks_liver',
    'lungs': 'masks_lungs',
    'heart': 'masks_heart'
}


for organ, folder in folders.items():
    mask_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    mask_files.sort()  # Ensure consistent order
    masks = []
    for fname in mask_files:
        img = Image.open(os.path.join(folder, fname)).convert('L')
        arr = np.array(img)
        # Convert to binary mask (0 and 1)
        arr_bin = (arr > 0).astype(np.uint8)
        masks.append(arr_bin)
    masks_stack = np.stack(masks, axis=0)
    np.save(f'{organ}_pred.npy', masks_stack)
    print(f"Saved {organ}_pred.npy with shape {masks_stack.shape}")

print("All predicted masks saved as .npy files.")
