import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

# Directory containing the DICOM folders
base_dir = "/Users/hp024/Documents/maelyne/MRI/Plots/T1_TE=20_TR=600/MRI_t0096"

# Recursively collect all DICOM files
dicom_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".dcm"):
            dicom_files.append(pydicom.dcmread(os.path.join(root, f)))

# Sort DICOM files by slice position
dicom_files.sort(key=lambda x: float(x.SliceLocation))

# Extract pixel data and convert to Hounsfield Units (HU)
slope = float(getattr(dicom_files[0], "RescaleSlope", 1.0))
intercept = float(getattr(dicom_files[0], "RescaleIntercept", 0.0))

volume = np.stack([f.pixel_array for f in dicom_files]).astype(np.float32)
volume = volume * slope + intercept

# Set the desired colormap range (adjust depending on your data)
vmin = 45
vmax = 90

# Generate GIF from axial slices
frames = []

for i, slice_ in enumerate(volume):
    fig, ax = plt.subplots()
    im = ax.imshow(slice_, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"Slice {i}")
    ax.axis("off")

    # Save the figure to memory
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    frames.append(imageio.v2.imread(buf))
    plt.close(fig)

# Save as animated GIF
imageio.mimsave("dicom_animation.gif", frames, fps=5)
output_path = os.path.join(base_dir, "dicom_animation.gif")
imageio.mimsave(output_path, frames, fps=5)
print(f"GIF saved at: {output_path}")




