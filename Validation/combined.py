#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Libraries

# %%
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
from skimage.draw import disk
import pandas as pd

from lut_utils import extract_patient_curves

# %% [markdown]
# # Load patient data

# %%
data   = loadmat("./Patient_1/Combined/CON_0101SP_12072016_tumor_all.mat")
data_3 = loadmat("./Patient_3/Combined/CON_03_V01_tumor_all.mat")

# %% [markdown]
# # Find the DWI slice used in the article
#
# Interactive widget to browse slices and identify the one matching the article figure.
# Patient 1 → slice 36 | Patient 3 → slice 31

# %%
def browse_slices(data_dict, patient_label):
    t_mask = np.transpose(data_dict["tumor_mask"], (0, 2, 1))
    n_slices = t_mask.shape[2]

    def _show(slice_idx):
        plt.figure(figsize=(5, 5))
        plt.imshow(t_mask[:, :, slice_idx], cmap='gray')
        plt.title(f"{patient_label} — slice {slice_idx}")
        plt.axis('off')
        plt.show()

    interact(_show, slice_idx=IntSlider(min=0, max=n_slices - 1, step=1, value=31))

browse_slices(data,   "Patient 1")
browse_slices(data_3, "Patient 3")

# %% [markdown]
# # Find ROI coordinates on DWI
#
# Overlay the article DWI figure on the MRI slice to identify the ROI center.
# Results: Patient 1 → center=(49,49), roi_size=3 | Patient 3 → center=(73,28), roi_size=3

# %%
def find_roi_coordinates(data_dict, patient_label, slice_idx, img_path):
    """Interactive overlay of article figure on MRI slice to find ROI coordinates."""
    mri_slice = np.transpose(data_dict["fullbrain_all"], (0, 2, 1, 3))[:, :, slice_idx, 0]
    t_mask    = np.transpose(data_dict["tumor_mask"],    (0, 2, 1))[:, :, slice_idx].astype(bool)

    try:
        img_article = plt.imread(img_path)
    except FileNotFoundError:
        print(f"[skip] Article image not found: {img_path}")
        return

    H, W = mri_slice.shape

    def _show(y_center, x_center, offset_y, offset_x, zoom, roi_shape,
              roi_width, roi_height, roi_radius, alpha_art=0.5):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(mri_slice, cmap='gray')
        ax.contour(t_mask, colors='cyan', linewidths=1.5)
        ax.imshow(img_article, alpha=alpha_art,
                  extent=(offset_x, W * zoom + offset_x,
                          H * zoom + offset_y, offset_y))

        roi_mask = np.zeros((H, W))
        if roi_shape == "rectangle":
            ys = y_center - roi_height // 2
            xs = x_center - roi_width  // 2
            roi_mask[max(0, ys):min(H, ys + roi_height),
                     max(0, xs):min(W, xs + roi_width)] = 1
        else:
            rr, cc = disk((y_center, x_center), roi_radius, shape=roi_mask.shape)
            roi_mask[rr, cc] = 1

        ax.imshow(roi_mask, cmap='Accent', alpha=0.6 if np.any(roi_mask) else 0)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f"{patient_label} — ROI overlay")
        plt.show()

    interact(
        _show,
        y_center   = IntSlider(min=0, max=H, step=1, value=H // 2),
        x_center   = IntSlider(min=0, max=W, step=1, value=W // 2),
        offset_y   = IntSlider(min=-20, max=20, step=1, value=0),
        offset_x   = IntSlider(min=-20, max=20, step=1, value=0),
        zoom       = FloatSlider(min=0.8, max=1.2, step=0.01, value=1.0),
        roi_shape  = Dropdown(options=["rectangle", "disque"], value="rectangle"),
        roi_width  = IntSlider(min=2, max=50, step=1, value=7),
        roi_height = IntSlider(min=2, max=50, step=1, value=7),
        roi_radius = IntSlider(min=2, max=30, step=1, value=5),
        alpha_art  = (0.0, 1.0, 0.05),
    )

find_roi_coordinates(data,   "Patient 1", slice_idx=36,
                     img_path="./Patient_1/Patient_1_DWI.png")
find_roi_coordinates(data_3, "Patient 3", slice_idx=31,
                     img_path="./Patient_3/Patient_3_DWI.png")

# %% [markdown]
# # Extract DWI curves

# %%
df_p1 = extract_patient_curves(data,   slice_idx=36, center=(49, 49),
                                roi_size=3, roi_type='square')
df_p3 = extract_patient_curves(data_3, slice_idx=31, center=(73, 28),
                                roi_size=3, roi_type='square')

print("df_p1:", df_p1.shape, "| TDs:", df_p1['TD'].unique())
print("df_p3:", df_p3.shape, "| TDs:", df_p3['TD'].unique())

# %% [markdown]
# # Signal decay — both patients, both TDs (log scale)
#
# Raw DWI curves with noise envelope for both patients side by side.

# %%
def plot_signal_decay(df_list, titles, max_b=7500):
    """Log-scale DWI signal decay with noise envelope, one subplot per patient."""
    fig, axes = plt.subplots(1, len(df_list), figsize=(14, 5), sharey=True)
    if len(df_list) == 1:
        axes = [axes]

    for ax, df, title in zip(axes, df_list, titles):
        df_f = df[df['b_value'] <= max_b]
        for td in sorted(df_f['TD'].unique()):
            sub = df_f[df_f['TD'] == td].sort_values('b_value')
            color = 'tab:blue' if td == '19ms' else 'tab:orange'
            ax.fill_between(sub['b_value'],
                            sub['signal'] - sub['noise'],
                            sub['signal'] + sub['noise'],
                            color=color, alpha=0.2)
            ax.plot(sub['b_value'], sub['signal'],
                    'o-', color=color, ms=5, lw=1.5, label=f"TD = {td}")

        ax.set_yscale('log')
        ax.set_xlabel('b-value (s/mm²)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    axes[0].set_ylabel('Normalised signal S/S₀')
    fig.tight_layout()
    plt.show()


plot_signal_decay(
    [df_p1, df_p3],
    ["Patient 1 — signal decay (TD=19ms & 49ms)",
     "Patient 3 — signal decay (TD=19ms & 49ms)"],
)

# %% [markdown]
# # DWI slice visualisation — contextual figure for article
#
# Show the DWI slice with tumor mask contour and the selected ROI overlay.

# %%
def plot_dwi_slice_with_roi(data_dict, patient_label, slice_idx,
                             center, roi_size, b_vol_idx=0):
    """
    Display the DWI slice with tumor mask contour and the selected ROI square.

    Parameters
    ----------
    b_vol_idx : index in the b-value volume to display (0 = first non-zero b-value)
    """
    mri   = np.transpose(data_dict["fullbrain_all"], (0, 2, 1, 3))
    mask  = np.transpose(data_dict["tumor_mask"],    (0, 2, 1))
    bvals = data_dict["bv_all"].flatten()
    nonzero_idx = np.where(bvals > 0)[0]

    vol_idx = nonzero_idx[b_vol_idx] if b_vol_idx < len(nonzero_idx) else nonzero_idx[0]
    b_shown = bvals[vol_idx]

    sl_img  = mri[:, :, slice_idx, vol_idx]
    sl_mask = mask[:, :, slice_idx].astype(bool)

    H, W = sl_img.shape
    cy, cx = center
    roi_img = np.zeros((H, W))
    roi_img[max(0, cy - roi_size):min(H, cy + roi_size + 1),
            max(0, cx - roi_size):min(W, cx + roi_size + 1)] = 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sl_img, cmap='gray')
    ax.contour(sl_mask, colors='cyan', linewidths=1.5)
    ax.contour(roi_img, colors='red',  linewidths=2)
    ax.set_title(f"{patient_label} — slice {slice_idx} | b={b_shown:.0f} s/mm²\n"
                 f"ROI center=({cx},{cy}), size={roi_size}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_dwi_slice_with_roi(data,   "Patient 1", slice_idx=36, center=(49, 49), roi_size=3)
plot_dwi_slice_with_roi(data_3, "Patient 3", slice_idx=31, center=(73, 28), roi_size=3)

# %%
