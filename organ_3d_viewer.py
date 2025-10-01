
import pyvista as pv
import os
import pyvista as pv
import os
# STL mesh file paths (update if needed)
organ_mesh_files = {
    'Heart': 'cleaned_output/heart_model_clean.stl',
    'Liver': 'cleaned_output/liver_model_clean.stl',
    'Lungs': 'cleaned_output/Lungs_model_clean.stl'
}
organ_colors = {'Heart': 'red', 'Liver': 'green', 'Lungs': 'blue'}
organ_meshes = {}

# Load meshes if available
for organ, mesh_path in organ_mesh_files.items():
    if os.path.exists(mesh_path):
        organ_meshes[organ] = pv.read(mesh_path)


plotter = pv.Plotter(window_size=(1200, 900))
actors = {}
state = {org: {'visible': True,
               'color': organ_colors[org], 'opacity': 0.7} for org in organ_meshes}
if not organ_meshes:
    print("No organ mesh files found. Please ensure STL files exist in cleaned_output.")
    import sys
    sys.exit(1)
current_organ = list(organ_meshes.keys())[0]

# Add all organ meshes
for organ, mesh in organ_meshes.items():
    actors[organ] = plotter.add_mesh(mesh, color=state[organ]['color'],
                                     opacity=state[organ]['opacity'], name=organ, smooth_shading=True)


# Organ switch via keyboard shortcuts
def switch_organ(organ):
    global current_organ
    current_organ = organ
    for org in actors:
        actors[org].SetVisibility(org == organ and state[org]['visible'])
    plotter.render()


# Add key events for organ switching
organ_keys = {'1': 'Heart', '2': 'Liver', '3': 'Lungs'}


def make_switch_func(organ):
    return lambda: switch_organ(organ)


for key, organ in organ_keys.items():
    if organ in actors:
        plotter.add_key_event(key, make_switch_func(organ))

instructions = "Press 1 for Heart, 2 for Liver, 3 for Lungs. Use sliders for color/opacity, checkbox for visibility."
plotter.add_text(instructions, position="upper_left", font_size=10)
# Color slider
color_map = ['red', 'green', 'blue', 'yellow',
             'magenta', 'cyan', 'white', 'orange']


def on_color_idx(val):
    idx = int(round(val))
    color = color_map[idx]
    state[current_organ]['color'] = color
    actors[current_organ].GetProperty().SetColor(pv.parse_color(color))
    plotter.render()


plotter.add_slider_widget(on_color_idx, rng=[0, len(color_map)-1], value=color_map.index(
    state[current_organ]['color']), title="Color", pointa=(0.7, 0.02), pointb=(0.95, 0.02))

# Opacity slider


def on_opacity(val):
    state[current_organ]['opacity'] = float(val)
    actors[current_organ].GetProperty().SetOpacity(float(val))
    plotter.render()


plotter.add_slider_widget(on_opacity, rng=[0.0, 1.0], value=state[current_organ]
                          ['opacity'], title="Opacity", pointa=(0.02, 0.02), pointb=(0.30, 0.02))

# Visibility toggle


def toggle_visibility(val):
    state[current_organ]['visible'] = bool(val)
    actors[current_organ].SetVisibility(bool(val))
    plotter.render()


plotter.add_checkbox_button_widget(
    toggle_visibility, value=True, position=(10, 40))

plotter.add_text("Organ 3D Viewer: Use buttons to switch organs, sliders for color/opacity, checkbox for visibility",
                 position="upper_left", font_size=10)
plotter.show()
