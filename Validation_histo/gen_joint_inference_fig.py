"""
Standalone runner — publication figure for joint chi² inference
Generates fig for Patient 1/H&E2/ROI1/exp=3 and Patient 3/H&E1/ROI1/exp=3
Saves to ../Paper/Figures/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
from scipy.spatial import Delaunay
from scipy.stats import chi2 as chi2_dist

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from lut_utils import get_signal_curve_lut, extract_patient_curves

SAVE_DIR = os.path.join("..", "Paper", "Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── LUT ───────────────────────────────────────────────────────────────────────
with h5py.File("lookup_table_val.mat", "r") as _f:
    params_lut  = np.array(_f["params"]).T
    _sig_raw    = np.array(_f["signals_4D"])
    signals_lut = np.transpose(_sig_raw, (3, 2, 1, 0))
    _bval  = np.array(_f["sequence"]["bval"]).flatten()
    _TD    = np.array(_f["sequence"]["TD"]).flatten()
    _bvecs = np.array(_f["sequence"]["bvecs"]).T

sequence_lut = {
    "bval":  np.array([[_bval]]),
    "TD":    np.array([[_TD]]),
    "bvecs": np.array([[_bvecs]]),
}
tri = Delaunay(params_lut)
bval_lut_min = _bval.min()

# ── DWI data ──────────────────────────────────────────────────────────────────
data_p1 = loadmat("./Patient_1/combined/CON_0101SP_12072016_tumor_all.mat")
data_p3 = loadmat("./Patient_3/combined/CON_03_V01_tumor_all.mat")

df_p1 = extract_patient_curves(data_p1, slice_idx=36, center=(49, 49),
                                roi_size=3, roi_type="square")
df_p3 = extract_patient_curves(data_p3, slice_idx=31, center=(73, 28),
                                roi_size=3, roi_type="square")
dwi_map = {1: df_p1, 3: df_p3}

# ── Histology ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("Qupath_project", "text_data")

CASES = [
    {"pat_id": 1, "he": 2, "roi": "ROI1", "exp": 3},
    {"pat_id": 3, "he": 1, "roi": "ROI1", "exp": 3},
]

# ── Inference helpers ─────────────────────────────────────────────────────────
Dex_grid    = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms
k_det_grid  = np.linspace(0.05, 1.0, 50)


def _ci_bounds_interp(grid, profile, best_value, delta):
    """Same sub-grid CI interpolation as Average_signal.py's _ci_bounds_interp."""
    threshold = best_value + delta
    below = profile <= threshold
    if not np.any(below):
        best_idx = int(np.nanargmin(profile))
        return float(grid[best_idx]), float(grid[best_idx])

    idx_in = np.where(below)[0]
    i_lo, i_hi = int(idx_in[0]), int(idx_in[-1])

    if i_lo > 0 and np.isfinite(profile[i_lo - 1]):
        x0, x1 = grid[i_lo - 1], grid[i_lo]
        y0, y1 = profile[i_lo - 1], profile[i_lo]
        ci_low = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    else:
        ci_low = float(grid[i_lo])

    if i_hi < len(grid) - 1 and np.isfinite(profile[i_hi + 1]):
        x0, x1 = grid[i_hi], grid[i_hi + 1]
        y0, y1 = profile[i_hi], profile[i_hi + 1]
        ci_high = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    else:
        ci_high = float(grid[i_hi])

    return float(ci_low), float(ci_high)
k_r         = 1.0
bval_max    = 7.5
slice_w     = 4.0

def _dwi_arrays(pat_id, TD):
    df = dwi_map[pat_id]
    td_str = f"{TD}ms"
    df_td = df[df["TD"] == td_str].copy()
    df_td["bval_um2"] = df_td["b_value"] / 1000
    df_td = df_td[(df_td["bval_um2"] > 0) &
                  (df_td["bval_um2"] <= bval_max) &
                  (df_td["bval_um2"] >= bval_lut_min)].sort_values("bval_um2")
    bv  = df_td["bval_um2"].to_numpy()
    sm  = df_td["signal"].to_numpy()
    sig = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6, df_td["noise"].to_numpy())
    return bv, sm, sig


def run_inference(pat_id, he, roi_name, exp):
    cell_file  = os.path.join(DATA_DIR, f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt")
    annot_file = os.path.join(DATA_DIR, f"Patient_{pat_id}_H&E_{he}_annot.txt")
    df_cells = pd.read_csv(cell_file, sep="\t", engine="python")
    df_annot = pd.read_csv(annot_file, sep="\t", engine="python")

    df_roi = df_cells[df_cells["Parent"] == roi_name]
    roi_area = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()[0]

    max_cal = df_roi["Cell: Max caliper"].to_numpy()
    min_cal = df_roi["Cell: Min caliper"].to_numpy()
    areas   = df_roi["Cell: Area"].to_numpy()

    mean_max = max_cal.mean(); std_max = max_cal.std(ddof=1)
    mean_min = min_cal.mean(); std_min = min_cal.std(ddof=1)
    r2D   = np.sqrt(mean_max * mean_min / 4)
    r3D   = 1.27 * r2D / k_r
    std_r3D = 1.27 * (0.25 * np.sqrt((mean_max/mean_min)*std_min**2 +
                                       (mean_min/mean_max)*std_max**2)) / k_r
    N_v   = len(areas) / ((slice_w + 2 * r2D / k_r) * roi_area)
    f_vivo = N_v * (4/3) * np.pi * r3D**3

    bv_19, S_19, sig_19 = _dwi_arrays(pat_id, 19)
    bv_49, S_49, sig_49 = _dwi_arrays(pat_id, 49)

    chi2_map_19 = np.full((len(k_det_grid), len(Dex_grid)), np.nan)
    chi2_map_49 = np.full((len(k_det_grid), len(Dex_grid)), np.nan)

    for i, k in enumerate(k_det_grid):
        f_eff = f_vivo * k
        for j, Dex in enumerate(Dex_grid):
            for td, bv, S, sigma, store in [
                    (19, bv_19, S_19, sig_19, chi2_map_19),
                    (49, bv_49, S_49, sig_49, chi2_map_49)]:
                bv_c, sc = get_signal_curve_lut(
                    f_eff, Dex, r3D, std_r3D, td, tri, signals_lut, params_lut,
                    sequence_lut=sequence_lut)
                log_sc = np.log(np.clip(sc, 1e-10, None))
                sp = np.exp(np.interp(bv, bv_c, log_sc))
                if not np.any(np.isnan(sp)):
                    store[i, j] = np.sum(((S - sp) / sigma) ** 2)

    loss_19 = np.nanmin(chi2_map_19, axis=1)
    loss_49 = np.nanmin(chi2_map_49, axis=1)
    total   = loss_19 + loss_49
    i_best  = int(np.nanargmin(total))
    k_best  = k_det_grid[i_best]
    j19     = int(np.nanargmin(chi2_map_19[i_best]))
    j49     = int(np.nanargmin(chi2_map_49[i_best]))
    D19     = Dex_grid[j19]; D49 = Dex_grid[j49]
    chi2_tot  = float(chi2_map_19[i_best, j19] + chi2_map_49[i_best, j49])
    chi219_min = float(chi2_map_19[i_best, j19])
    chi249_min = float(chi2_map_49[i_best, j49])

    delta = chi2_dist.ppf(0.95, df=3)
    k_ci   = _ci_bounds_interp(k_det_grid, total, chi2_tot, delta)
    d19_ci = _ci_bounds_interp(Dex_grid, chi2_map_19[i_best], chi219_min, delta)
    d49_ci = _ci_bounds_interp(Dex_grid, chi2_map_49[i_best], chi249_min, delta)

    return {
        "k_best": k_best, "D19": D19, "D49": D49,
        "k_ci": k_ci, "d19_ci": d19_ci, "d49_ci": d49_ci,
        "chi2_tot": chi2_tot, "chi219_min": chi219_min, "chi249_min": chi249_min,
        "delta": delta, "f_eff": float(f_vivo * k_best),
        "total_per_k": total, "chi2_map_19": chi2_map_19, "chi2_map_49": chi2_map_49,
        "i_best": i_best,
    }


def plot_inference(res, label, pat_id, save_path):
    _pub_rc = {"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8}
    with plt.rc_context(_pub_rc):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        _ci_kw   = dict(color="#d62728", ls="--", lw=1.2, alpha=0.8)
        _best_kw = dict(color="#d62728", ls=":", lw=1.5)

        ax = axes[0]
        ax.plot(k_det_grid, res["total_per_k"] - res["chi2_tot"],
                color="#2166ac", lw=1.8)
        ax.axhline(res["delta"], label=f"95% CI ({res['delta']:.2f})", **_ci_kw)
        ax.axvline(res["k_best"], **_best_kw,
                   label=f"$k_{{det}}$ = {res['k_best']:.3f}")
        ax.set_xlabel("$k_{det}$")
        ax.set_ylabel("$\\Delta\\chi^2_{total}$")
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        ax = axes[1]
        ax.plot(Dex_grid, res["chi2_map_19"][res["i_best"]] - res["chi219_min"],
                color="#4393c3", lw=1.8)
        ax.axhline(res["delta"], **_ci_kw)
        ax.axvline(res["D19"], **_best_kw,
                   label=f"$D_{{ex,19}}$ = {res['D19']:.3f} µm²/ms")
        ax.set_xlabel("$D_{ex}$ (µm²/ms)")
        ax.set_ylabel("$\\Delta\\chi^2$  (TD = 19 ms)")
        ax.set_xlim(1.0, 3.0)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        ax = axes[2]
        ax.plot(Dex_grid, res["chi2_map_49"][res["i_best"]] - res["chi249_min"],
                color="#d6604d", lw=1.8)
        ax.axhline(res["delta"], **_ci_kw)
        ax.axvline(res["D49"], **_best_kw,
                   label=f"$D_{{ex,49}}$ = {res['D49']:.3f} µm²/ms")
        ax.set_xlabel("$D_{ex}$ (µm²/ms)")
        ax.set_ylabel("$\\Delta\\chi^2$  (TD = 49 ms)")
        ax.set_xlim(1.0, 3.0)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(C)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        fig.text(0.5, 1.01, label, ha="center", va="bottom",
                 fontsize=11, transform=fig.transFigure)
        fig.tight_layout(w_pad=2.0)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)


for case in CASES:
    pat_id   = case["pat_id"]
    he       = case["he"]
    roi      = case["roi"]
    exp      = case["exp"]
    label    = f"Patient {pat_id} | H&E {he} | {roi} | exp = {exp}"
    print(f"\nRunning inference: {label}")
    res = run_inference(pat_id, he, roi, exp)
    print(f"  k_det = {res['k_best']:.3f}  D19 = {res['D19']:.3f}  D49 = {res['D49']:.3f}")
    fname = f"fig_joint_inference_P{pat_id}_HE{he}_{roi}.png"
    plot_inference(res, label, pat_id, os.path.join(SAVE_DIR, fname))

print("\nDone.")
