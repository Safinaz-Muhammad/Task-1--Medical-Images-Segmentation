import tkinter as tk
from tkinter import colorchooser, ttk
import numpy as np
from PIL import Image, ImageTk
import os

# Load masks if available
organ_files = {
    'Heart': 'heart_gt.npy',
    'Liver': 'liver_gt.npy',
    'Lungs': 'lungs_gt.npy'
}


class OrganViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Organ Mask Viewer')
        self.geometry('800x600')
        self.organs = list(organ_files.keys())
        self.current_organ = self.organs[0]
        self.masks = {}
        self.colors = {org: '#FF0000' for org in self.organs}
        self.opacities = {org: 0.5 for org in self.organs}
        self.visibilities = {org: True for org in self.organs}
        self.load_masks()
        self.create_widgets()
        self.display_mask()

    def load_masks(self):
        for organ, path in organ_files.items():
            if os.path.exists(path):
                arr = np.load(path)
                # Use the first slice for display
                self.masks[organ] = arr[0] if arr.ndim == 3 else arr
            else:
                self.masks[organ] = None

    def create_widgets(self):
        # Organ switch buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, pady=10)
        for organ in self.organs:
            btn = tk.Button(btn_frame, text=organ,
                            command=lambda o=organ: self.switch_organ(o))
            btn.pack(side=tk.LEFT, padx=5)
        # Color, opacity, visibility controls
        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(side=tk.TOP, pady=10)
        self.color_btn = tk.Button(
            ctrl_frame, text='Change Color', command=self.change_color)
        self.color_btn.pack(side=tk.LEFT, padx=5)
        self.opacity_scale = tk.Scale(ctrl_frame, from_=0, to=1, resolution=0.01,
                                      orient=tk.HORIZONTAL, label='Opacity', command=self.change_opacity)
        self.opacity_scale.set(self.opacities[self.current_organ])
        self.opacity_scale.pack(side=tk.LEFT, padx=5)
        self.vis_var = tk.BooleanVar(
            value=self.visibilities[self.current_organ])
        self.vis_check = tk.Checkbutton(
            ctrl_frame, text='Visible', variable=self.vis_var, command=self.toggle_visibility)
        self.vis_check.pack(side=tk.LEFT, padx=5)
        # Canvas for mask display
        self.canvas = tk.Canvas(self, width=512, height=512, bg='black')
        self.canvas.pack(pady=10)

    def switch_organ(self, organ):
        self.current_organ = organ
        self.opacity_scale.set(self.opacities[organ])
        self.vis_var.set(self.visibilities[organ])
        self.display_mask()

    def change_color(self):
        color = colorchooser.askcolor(title='Choose Mask Color')[1]
        if color:
            self.colors[self.current_organ] = color
            self.display_mask()

    def change_opacity(self, val):
        self.opacities[self.current_organ] = float(val)
        self.display_mask()

    def toggle_visibility(self):
        self.visibilities[self.current_organ] = self.vis_var.get()
        self.display_mask()

    def display_mask(self):
        self.canvas.delete('all')
        mask = self.masks.get(self.current_organ)
        if mask is None or not self.visibilities[self.current_organ]:
            return
        # Create RGBA image
        color = self.colors[self.current_organ]
        opacity = int(self.opacities[self.current_organ] * 255)
        rgb = self.winfo_rgb(color)
        rgb = tuple([int(v/256) for v in rgb])
        rgba_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_img[..., :3] = rgb
        rgba_img[..., 3] = mask * opacity
        img = Image.fromarray(rgba_img, 'RGBA').resize((512, 512))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)


if __name__ == '__main__':
    app = OrganViewer()
    app.mainloop()
