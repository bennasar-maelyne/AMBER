"""
Sensitivity maps: ∂S/∂θ on the (b, TD) grid  — Approach B.

For each phenotype (operating point = centre of its parameter bounds), computes
numerical partial derivatives via central finite differences through the
Delaunay interpolator.

Physical interpretation
-----------------------
∂S/∂θ(b, TD)  answers: "at this (b, TD), how much does the signal change
if θ shifts by one unit?"  High values → this (b, TD) is informative about θ.

Two quantities are plotted:
  raw        |∂S/∂θ|              — absolute sensitivity (signal units / param unit)
  effective  |∂S/∂θ| / σ_noise   — sensitivity relative to noise (dimensionless SNR)
             with σ_noise(TD) = (1/SNR0) × exp((TE(TD) − TE_ref) / T2)

Parameters evaluated: f, Dex [µm²/ms], rmean [µm]
(rsd excluded — it is a secondary parameter not optimised independently)

Generates
---------
  Figures/sensitivity/fig_sens_<phenotype>.png   — 3×2 grid per phenotype
  Figures/sensitivity/fig_sens_all_raw.png       — all phenotypes × all params (raw)
  Figures/sensitivity/fig_sens_all_eff.png       — all phenotypes × all params (effective)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Reuse LUT utilities from opt_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt_engine import load_lut, build_tri, interpolate_batch
from tumor_catalog import CATALOG

# ── Config ─────────────────────────────────────────────────────────────────────
T2_MS  = 100.0   # ms — tissue T2 (single value, Approach 1)
SNR0   = 40.0    # SNR at b=0, shortest TD

# Finite-difference step sizes (absolute, in parameter units)
# Large enough to cross simplex boundaries, small enough to stay local
STEPS = {"f": 0.04, "Dex": 0.12, "rmean": 0.8, "rsd": 0.4}

PARAM_LABELS = {
    "f":     r"$f$",
    "Dex":   r"$D_{ex}$ (µm²/ms)",
    "rmean": r"$r_{mean}$ (µm)",
}
PARAMS_SENS = ["f", "Dex", "rmean"]   # parameters to differentiate
PARAM_IDX   = {"f": 0, "Dex": 1, "rmean": 2, "rsd": 3}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LUT_PATH = os.path.join(THIS_DIR, "..", "Validation", "lookup_table_optim.mat")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9})


# ── Load LUT ───────────────────────────────────────────────────────────────────
print("Loading LUT...")
params_lut, sig_norm, bval, TD_arr, delta = load_lut(LUT_PATH)
tri = build_tri(params_lut)

b0_idx    = int(np.where(bval == 0)[0][0])
bval_use  = np.delete(bval, b0_idx)
b_use_idx = [i for i in range(len(bval)) if i != b0_idx]
N_BVAL    = len(bval_use)
N_TD      = len(TD_arr)

te_ref = TD_arr[0] + delta   # reference TE (shortest TD)
print(f"  b-values: {bval_use}")
print(f"  TDs: {TD_arr} ms  |  delta: {delta} ms  |  TE_ref: {te_ref} ms")


# ── Noise model σ(TD) ──────────────────────────────────────────────────────────
def sigma_noise(TD_val):
    te = TD_val + delta
    return (1.0 / SNR0) * np.exp((te - te_ref) / T2_MS)

sigma_arr = np.array([sigma_noise(t) for t in TD_arr])   # (N_TD,)


# ── Signal at a single point ───────────────────────────────────────────────────
def signal_at(params_1d):
    """
    Interpolate signal for a single (f, Dex, rmean, rsd) point.

    Returns
    -------
    sig : (N_TD, N_BVAL_USE) or None if outside hull
    """
    pts = params_1d.reshape(1, 4)
    out = interpolate_batch(pts, sig_norm, tri)    # (1, N_TD, N_BVAL_full)
    out_use = out[0][:, b_use_idx]                 # (N_TD, N_BVAL_USE)
    if np.isnan(out_use).any():
        return None
    return out_use


# ── Compute ∂S/∂θ at a centre point ───────────────────────────────────────────
def compute_gradients(center_4d):
    """
    Central finite differences for f, Dex, rmean at the given centre point.

    Returns
    -------
    grads : dict  param -> (N_TD, N_BVAL_USE)   — ∂S/∂θ
    valid : bool  — False if any perturbation falls outside the hull
    """
    grads = {}
    for param in PARAMS_SENS:
        idx  = PARAM_IDX[param]
        step = STEPS[param]

        p_plus  = center_4d.copy(); p_plus[idx]  += step
        p_minus = center_4d.copy(); p_minus[idx] -= step

        s_plus  = signal_at(p_plus)
        s_minus = signal_at(p_minus)

        if s_plus is None or s_minus is None:
            return None, False

        grads[param] = (s_plus - s_minus) / (2.0 * step)

    return grads, True


# ── Helper: cell edges for pcolormesh ─────────────────────────────────────────
def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    e = np.empty(len(c) + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0]  = c[0]  - (c[1]  - c[0])  / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e

b_edges  = cell_edges(bval_use)
td_edges = cell_edges(TD_arr)


# ── Per-phenotype sensitivity maps ─────────────────────────────────────────────
print("\nComputing sensitivity maps per phenotype...")

pheno_grads = {}   # name -> {"raw": {param: (N_TD, N_BVAL)}, "eff": {...}}

for pheno in CATALOG:
    name   = pheno["name"]
    label  = pheno["label"]
    bounds = pheno["bounds"]

    # Operating point: centre of bounds
    center = np.array([
        (bounds["f"][0]     + bounds["f"][1])     / 2,
        (bounds["Dex"][0]   + bounds["Dex"][1])   / 2,
        (bounds["rmean"][0] + bounds["rmean"][1]) / 2,
        (bounds["rsd"][0]   + bounds["rsd"][1])   / 2,
    ])

    grads, ok = compute_gradients(center)
    if not ok:
        print(f"  {name}: operating point outside hull — skipped")
        continue

    # Noise-normalised: |∂S/∂θ| / σ_noise(TD)   (broadcast over b axis)
    eff_grads = {}
    for param in PARAMS_SENS:
        eff_grads[param] = np.abs(grads[param]) / sigma_arr[:, np.newaxis]

    pheno_grads[name] = {
        "raw":   {p: np.abs(grads[p]) for p in PARAMS_SENS},
        "eff":   eff_grads,
        "center": center,
        "label": label,
        "color": pheno["color"],
    }
    print(f"  {name}: centre = f={center[0]:.2f}, Dex={center[1]:.2f}, "
          f"rmean={center[2]:.1f}, rsd={center[3]:.1f}")


# ── Fig A — Per-phenotype: 3 params × 2 (raw | effective) ─────────────────────
for name, data in pheno_grads.items():
    fig, axes = plt.subplots(len(PARAMS_SENS), 2,
                             figsize=(9, 2.8 * len(PARAMS_SENS)),
                             squeeze=False)

    for row, param in enumerate(PARAMS_SENS):
        raw = data["raw"][param]   # (N_TD, N_BVAL)
        eff = data["eff"][param]

        for col, (arr, title_suffix, cmap) in enumerate([
            (raw, "raw  |∂S/∂θ|",          "YlOrRd"),
            (eff, "effective  |∂S/∂θ|/σ",  "YlGnBu"),
        ]):
            ax   = axes[row, col]
            vmax = np.nanmax(arr) if np.nanmax(arr) > 0 else 1.0
            im   = ax.pcolormesh(b_edges, td_edges, arr,
                                 cmap=cmap, vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.85)

            # Mark maximum
            bi, ti = np.unravel_index(np.nanargmax(arr), arr.shape)
            ax.plot(bval_use[ti], TD_arr[bi],
                    marker="*", color="cyan", ms=9, zorder=5)

            ax.set_xticks(bval_use)
            ax.set_xticklabels([f"{b:.2f}" for b in bval_use],
                               rotation=45, ha="right", fontsize=6)
            ax.set_yticks(TD_arr)
            ax.set_yticklabels([f"{t:.0f}" for t in TD_arr], fontsize=7)
            ax.set_ylabel("TD (ms)", fontsize=8)
            if row == len(PARAMS_SENS) - 1:
                ax.set_xlabel("b (ms/µm²)", fontsize=8)
            if row == 0:
                ax.set_title(title_suffix, fontsize=9, fontweight="bold")
            ax.text(0.02, 0.97, PARAM_LABELS[param],
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle(f"Sensitivity maps — {data['label']}\n"
                 f"f={data['center'][0]:.2f}, Dex={data['center'][1]:.2f}, "
                 f"rmean={data['center'][2]:.1f} µm  |  "
                 f"T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f}",
                 fontsize=10, y=1.01)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"fig_sens_{name}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: fig_sens_{name}.png")


# ── Fig B — Summary: all phenotypes × all params, raw and effective ────────────
n_pheno = len(pheno_grads)
n_param = len(PARAMS_SENS)

for mode, key, cmap_base in [("raw", "raw", "YlOrRd"), ("eff", "eff", "YlGnBu")]:
    fig, axes = plt.subplots(n_param, n_pheno,
                             figsize=(2.8 * n_pheno, 2.6 * n_param),
                             squeeze=False)

    # Shared colour scale per row (per parameter)
    for row, param in enumerate(PARAMS_SENS):
        vmax_row = max(
            np.nanmax(data[key][param])
            for data in pheno_grads.values()
            if np.nanmax(data[key][param]) > 0
        )

        for col, (name, data) in enumerate(pheno_grads.items()):
            ax  = axes[row, col]
            arr = data[key][param]

            im = ax.pcolormesh(b_edges, td_edges, arr,
                               cmap=cmap_base, vmin=0, vmax=vmax_row)

            # Annotate values
            for ti, t in enumerate(TD_arr):
                for bi, b in enumerate(bval_use):
                    v = arr[ti, bi]
                    if np.isnan(v):
                        continue
                    color = "white" if v / vmax_row > 0.6 else "black"
                    ax.text(b, t, f"{v:.2f}", ha="center", va="center",
                            fontsize=4, color=color)

            # Mark maximum
            bi, ti = np.unravel_index(np.nanargmax(arr), arr.shape)
            ax.plot(bval_use[ti], TD_arr[bi],
                    marker="*", color="cyan", ms=7, zorder=5)

            ax.set_xticks(bval_use)
            ax.set_xticklabels([f"{b:.2f}" for b in bval_use],
                               rotation=45, ha="right", fontsize=5)
            ax.set_yticks(TD_arr)
            ax.set_yticklabels([f"{t:.0f}" for t in TD_arr], fontsize=5)

            if row == 0:
                ax.set_title(data["label"], fontsize=9, fontweight="bold",
                             color=data["color"])
            if col == 0:
                ax.set_ylabel(f"{PARAM_LABELS[param]}\nTD (ms)", fontsize=7)
            if row == n_param - 1:
                ax.set_xlabel("b (ms/µm²)", fontsize=7)

        # Shared colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.68 - row * 0.32, 0.012, 0.27])
        sm = plt.cm.ScalarMappable(
            cmap=cmap_base, norm=mcolors.Normalize(vmin=0, vmax=vmax_row))
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(PARAM_LABELS[param], fontsize=7)

    title_map = {"raw": "raw  |∂S/∂θ|", "eff": "effective  |∂S/∂θ| / σ_noise(TD)"}
    fig.suptitle(f"Sensitivity maps — {title_map[mode]}\n"
                 f"T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f}, δ={delta:.0f} ms  |  "
                 "cyan star = argmax per panel",
                 fontsize=10, y=1.01)
    fig.subplots_adjust(right=0.90, hspace=0.4, wspace=0.4)

    out = os.path.join(OUT_DIR, f"fig_sens_all_{mode}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: fig_sens_all_{mode}.png")

print(f"\nAll figures -> {os.path.abspath(OUT_DIR)}")
