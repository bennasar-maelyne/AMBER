# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.0
#   kernelspec:
#     display_name: .venv (3.13.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Libraries

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat as _loadmat
from scipy.spatial import Delaunay
import h5py as _h5py

from lut_utils import get_signal_curve_lut, extract_patient_curves

# %% [markdown]
# # Load LUT

# %%
with _h5py.File('lookup_table_val.mat', 'r') as _f:
    params_lut  = np.array(_f['params']).T
    _sig_raw    = np.array(_f['signals_4D'])
    signals_lut = np.transpose(_sig_raw, (3, 2, 1, 0))
    _seq   = _f['sequence']  # type: ignore[index]
    _bval  = np.array(_seq['bval']).flatten()   # type: ignore[index]
    _TD    = np.array(_seq['TD']).flatten()     # type: ignore[index]
    _bvecs = np.array(_seq['bvecs']).T          # type: ignore[index]

sequence_lut = {
    'bval':  np.array([[_bval]]),
    'TD':    np.array([[_TD]]),
    'bvecs': np.array([[_bvecs]]),
}
params_lut = np.array(params_lut)
print(sequence_lut["bval"])

# Patch lut_utils functions to use the local sequence_lut
import lut_utils as _lut
import functools

get_signal_curve_lut = functools.partial(_lut.get_signal_curve_lut,
                                         sequence_lut=sequence_lut)

tri = Delaunay(params_lut)

print("LUT loaded — params:", params_lut.shape, "| signals:", signals_lut.shape)

# %% [markdown]
# # Load inference results and patient data

# %%
# df_summary is produced by Average_signal.py and saved as CSV
df_summary = pd.read_csv("df_summary.csv")

Patient    = sorted(df_summary["Patient"].unique().tolist())
H_E        = sorted(df_summary["H&E"].unique().tolist())
TD_list    = [19, 49]
exp_merged = 3

full_path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Qupath_project", "text_data")

# DWI data
data_p1 = _loadmat("./Patient_1/Combined/CON_0101SP_12072016_tumor_all.mat")
data_p3 = _loadmat("./Patient_3/Combined/CON_03_V01_tumor_all.mat")
df_p1   = extract_patient_curves(data_p1, slice_idx=36, center=(49, 49),
                                  roi_size=3, roi_type='square')
df_p3   = extract_patient_curves(data_p3, slice_idx=31, center=(73, 28),
                                  roi_size=3, roi_type='square')
dwi_map = {1: df_p1, 3: df_p3}

# Spatial bootstrap JSON
with open("signal_results_spatial_bootstrap.json", "r", encoding="utf-8") as _f:
    _spatial_json = json.load(_f)

print("df_summary:", df_summary.shape)
print("Patients:", Patient, "| TDs:", TD_list)

# %% [markdown]
# # Helper functions

# %%
def _weighted_consensus(pat_id, TD_val, col):
    """Weighted average of `col` in df_summary for a given patient, weights = 1/chi2_min.

    k_det is shared across TDs — no TD filter applied.
    Dex is TD-dependent: pass col='Dex_19_best' or 'Dex_49_best' explicitly,
    or use the helper aliases below.
    """
    # Map old column names to new df_summary structure
    _col_map = {
        "Dex_best":    "Dex_19_best" if TD_val == 19 else "Dex_49_best",
        "TD (ms)":     None,   # not a column anymore
    }
    resolved_col = _col_map.get(col, col)

    sub = df_summary[df_summary["Patient"] == pat_id].copy()
    w = 1.0 / sub["chi2_min"].replace(0, np.nan).dropna()
    return float(np.average(sub.loc[w.index, resolved_col], weights=w.to_numpy()))


def _histo_per_he(pat_key, td_key, bvals_dwi, he_filter=None):
    """Return list of (he_label, mean, p_low, p_high) — one entry per H&E slide.

    Parameters
    ----------
    he_filter : list of str or None
        If provided, only include H&E keys in this list (e.g. ["H&E_2"]).
    """
    results = []
    for he_key, he_data in _spatial_json.get(pat_key, {}).items():
        if he_filter is not None and he_key not in he_filter:
            continue
        td_data = (he_data.get(f"exp_{exp_merged}", {})
                         .get("full_slide", {})
                         .get(td_key, {}))
        if not td_data:
            continue
        entry = next(iter(td_data.values()))
        bv    = np.array(entry["bval"])
        mask  = bv <= 7.5
        results.append((
            he_key,
            np.interp(bvals_dwi, bv[mask], np.array(entry["signals_mean"])[mask]),
            np.interp(bvals_dwi, bv[mask], np.array(entry["p_low"])[mask]),
            np.interp(bvals_dwi, bv[mask], np.array(entry["p_high"])[mask]),
        ))
    return results


# H&E slide to use per patient in merged plots and sensitivity plots
_he_filter = {1: ["H&E_2"], 3: None}   # Patient 1 → H&E_2 only; Patient 3 → all


def _dwi_td(pat_id, TD_val):
    """Return filtered DWI DataFrame (b > 0, b ≤ 7.5) for a given patient × TD."""
    td_str = f"{TD_val}ms"
    return (
        dwi_map[pat_id][dwi_map[pat_id]["TD"] == td_str]
        .assign(b=lambda d: d["b_value"] / 1000)
        .query("b > 0 and b <= 7.5")
        .sort_values("b")
    )


_he_colors  = ["tab:blue", "tab:cyan"]
_roi_colors = plt.colormaps["tab10"](np.linspace(0, 0.8, 8))

# %% [markdown]
# # Step 1 — Per-ROI: measured DWI vs histological simulation + bootstrap CI

# %%
# One figure per (Patient, H&E, ROI): two subplots TD=19 | TD=49.
# Central curve = best-fit from df_summary. CI = combined bootstrap on cells + Dex + k_det.

from lut_utils import get_signal_curve_lut as _gsc_raw

def _get_signal(f, Dex, r, rsd, TD):
    return get_signal_curve_lut(f, Dex, r, rsd, TD, tri, signals_lut, params_lut)


def compute_full_uncertainty_curve(df_roi, roi_area_um2, slice_width_um,
                                   Dex_ci_low, Dex_ci_high,
                                   bvals_um2, TD,
                                   k_ci=(1.0, 1.0), n_bootstrap=200):
    """Bootstrap CI over a list of b-values (histological variability + Dex + k_det)."""
    cell_areas   = df_roi["Cell: Area"].to_numpy()
    max_calipers = df_roi["Cell: Max caliper"].to_numpy()
    min_calipers = df_roi["Cell: Min caliper"].to_numpy()
    n_cells      = len(cell_areas)

    means, p_lows, p_highs = [], [], []
    for bval in bvals_um2:
        signals_boot = []
        for _ in range(n_bootstrap):
            idx       = np.random.choice(n_cells, size=n_cells, replace=True)
            max_cal_b = max_calipers[idx]
            min_cal_b = min_calipers[idx]
            mm = np.mean(max_cal_b); mn = np.mean(min_cal_b)
            sm = np.std(max_cal_b, ddof=1); sn = np.std(min_cal_b, ddof=1)
            r2D     = np.sqrt(mm * mn / 4)
            std_r2D = 0.25 * np.sqrt((mm / mn) * sn**2 + (mn / mm) * sm**2)
            r3D     = 1.27 * r2D
            std_r3D = 1.27 * std_r2D
            N_v     = n_cells / ((slice_width_um + 2 * r2D) * roi_area_um2)
            f_raw   = N_v * (4/3) * np.pi * r3D**3
            k_s     = np.random.uniform(k_ci[0], k_ci[1])
            f_boot  = f_raw * k_s
            Dex_s   = np.random.uniform(Dex_ci_low, Dex_ci_high)
            bv, sc  = _get_signal(f_boot, Dex_s, r3D, std_r3D, TD)
            log_sc  = np.log(np.clip(sc, 1e-10, None))
            sig     = float(np.exp(np.interp(bval, bv, log_sc)))
            if not np.isnan(sig):
                signals_boot.append(sig)
        sb = np.array(signals_boot)
        means.append(float(np.mean(sb)))
        p_lows.append(float(np.percentile(sb, 2.5)))
        p_highs.append(float(np.percentile(sb, 97.5)))
    return np.array(means), np.array(p_lows), np.array(p_highs)


exp_plot = 3

combos = (
    df_summary[["Patient", "H&E", "ROI"]]
    .drop_duplicates()
    .sort_values(["Patient", "H&E", "ROI"])
)

for _, combo in combos.iterrows():
    pat_id   = int(combo["Patient"])
    he_plot  = int(combo["H&E"])
    roi_name = str(combo["ROI"])

    cell_path  = os.path.join(full_path_dir,
                              f"Patient_{pat_id}_H&E_{he_plot}_exp_{exp_plot}.txt")
    annot_path = os.path.join(full_path_dir,
                              f"Patient_{pat_id}_H&E_{he_plot}_annot.txt")
    if not os.path.exists(cell_path) or not os.path.exists(annot_path):
        print(f"[skip] Patient {pat_id} H&E {he_plot}: missing files.")
        continue

    df_cells  = pd.read_csv(cell_path,  sep='\t', engine='python')
    df_annot  = pd.read_csv(annot_path, sep='\t', engine='python')
    df_roi    = df_cells[df_cells["Parent"] == roi_name]
    area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
    if len(area_vals) == 0 or len(df_roi) == 0:
        continue
    roi_area_um2 = area_vals[0]

    max_cal = df_roi["Cell: Max caliper"].to_numpy()
    min_cal = df_roi["Cell: Min caliper"].to_numpy()
    mm, mn  = np.mean(max_cal), np.mean(min_cal)
    sm, sn  = np.std(max_cal, ddof=1), np.std(min_cal, ddof=1)
    r3D_std_vivo = 1.27 * 0.25 * np.sqrt((mm / mn) * sn**2 + (mn / mm) * sm**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mask_row = (
        (df_summary["Patient"] == pat_id) &
        (df_summary["H&E"]     == he_plot) &
        (df_summary["ROI"]     == roi_name)
    )
    rows = df_summary[mask_row]
    if rows.empty:
        plt.close()
        continue
    row = rows.iloc[0]

    for ax, TD_val in zip(axes, TD_list):
        _dex_col    = "Dex_19_best"    if TD_val == 19 else "Dex_49_best"
        _dex_ci_lo  = "Dex_19_CI_low"  if TD_val == 19 else "Dex_49_CI_low"
        _dex_ci_hi  = "Dex_19_CI_high" if TD_val == 19 else "Dex_49_CI_high"

        Dex_best    = float(row[_dex_col])
        k_det_best  = float(row["k_det_best"])
        f_corrected = float(row["f_eff"])
        r3D_vivo    = float(row["r3D_vivo (µm)"])
        Dex_ci      = (float(row[_dex_ci_lo]), float(row[_dex_ci_hi]))
        k_det_ci    = (float(row["k_det_CI_low"]), float(row["k_det_CI_high"]))

        df_td     = _dwi_td(pat_id, TD_val)
        bvals_dwi = df_td["b"].to_numpy()

        bv_all, sig_all = _get_signal(f_corrected, Dex_best, r3D_vivo, r3D_std_vivo, TD_val)
        sig_plot = np.interp(bvals_dwi, bv_all, sig_all)

        _, p_lows, p_highs = compute_full_uncertainty_curve(
            df_roi, roi_area_um2, 4,
            Dex_ci[0], Dex_ci[1], bvals_dwi, TD_val,
            k_ci=k_det_ci, n_bootstrap=300,
        )

        ax.fill_between(df_td["b"],
                        df_td["signal"] - df_td["noise"],
                        df_td["signal"] + df_td["noise"],
                        color="tab:gray", alpha=0.3, label="DWI noise")
        ax.plot(df_td["b"], df_td["signal"],
                "o-", color="tab:gray", lw=1.5, ms=5, label="Measured DWI")
        ax.fill_between(bvals_dwi, p_lows, p_highs,
                        color="tab:blue", alpha=0.2, label="Bootstrap 95% CI")
        ax.plot(bvals_dwi, sig_plot, "o-", color="tab:blue", ms=5, lw=2, zorder=3,
                label=f"Histo — LUT  $D_{{ex}}$={Dex_best:.2f}, k_det={k_det_best:.2f}")
        ax.set_xlabel("b-value (ms/µm²)")
        ax.set_ylabel("Normalised signal S/S₀")
        ax.set_title(
            f"TD = {TD_val} ms\n"
            f"$D_{{ex}}$ = {Dex_best:.2f} [{Dex_ci[0]:.2f}, {Dex_ci[1]:.2f}]  |  "
            f"k_det = {k_det_best:.2f} [{k_det_ci[0]:.2f}, {k_det_ci[1]:.2f}]",
            fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Patient {pat_id} | H&E {he_plot} | {roi_name} | exp={exp_plot}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# # Merged signal per patient — all ROIs aggregated

# %%
_FIG_DIR_VAL = os.path.join(os.path.dirname(os.path.abspath(__file__))
                             if "__file__" in dir() else os.getcwd(),
                             "..", "Paper", "Figures")
os.makedirs(_FIG_DIR_VAL, exist_ok=True)

_PUB_RC_VAL = {"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8}
_he_selected = {1: "H&E_2", 3: "H&E_1"}
_panel_labels = ["A", "B", "C", "D"]

with plt.rc_context(_PUB_RC_VAL):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=False, sharey=False)

    for row_idx, pat_id in enumerate(Patient):
        pat_key  = f"Patient_{pat_id}"
        pat_rows = df_summary[df_summary["Patient"] == pat_id]
        he_key_target = _he_selected.get(pat_id)

        for col_idx, TD_val in enumerate(TD_list):
            ax = axes[row_idx, col_idx]
            panel_idx = row_idx * 2 + col_idx
            td_key = f"TD_{TD_val}"
            if pat_rows.empty:
                ax.set_visible(False)
                continue

            df_td     = _dwi_td(pat_id, TD_val)
            bvals_dwi = df_td["b"].to_numpy()

            ax.fill_between(df_td["b"],
                            df_td["signal"] - df_td["noise"],
                            df_td["signal"] + df_td["noise"],
                            color="#888888", alpha=0.25, label="DWI noise band")
            ax.plot(df_td["b"], df_td["signal"],
                    "o-", color="#444444", lw=1.8, ms=5,
                    label="Measured DWI", zorder=5)

            sig_means_all, p_lows_all, p_highs_all = [], [], []
            for he_key, he_data in _spatial_json.get(pat_key, {}).items():
                if he_key != he_key_target:
                    continue
                td_data = (he_data.get(f"exp_{exp_merged}", {})
                                  .get("full_slide", {}).get(td_key, {}))
                if not td_data:
                    continue
                entry = next(iter(td_data.values()))
                bv    = np.array(entry["bval"])
                mask  = bv <= 7.5
                sig_means_all.append(np.interp(bvals_dwi, bv[mask],
                                               np.array(entry["signals_mean"])[mask]))
                p_lows_all.append(np.interp(bvals_dwi, bv[mask],
                                            np.array(entry["p_low"])[mask]))
                p_highs_all.append(np.interp(bvals_dwi, bv[mask],
                                             np.array(entry["p_high"])[mask]))

            if not sig_means_all:
                continue

            sig_mean    = np.array(sig_means_all).mean(axis=0)
            p_low_mean  = np.array(p_lows_all).mean(axis=0)
            p_high_mean = np.array(p_highs_all).mean(axis=0)

            ax.fill_between(bvals_dwi, p_low_mean, p_high_mean,
                            color="#2166ac", alpha=0.18,
                            label="Bootstrap 95% CI")
            ax.plot(bvals_dwi, sig_mean, "o-", color="#2166ac",
                    lw=1.8, ms=5, zorder=4, label="Synthetic DWI (mean)")

            ax.set_xlabel("$b$-value (ms/µm²)")
            ax.set_ylabel("Normalised signal $S/S_0$")
            ax.set_title(f"Patient {pat_id} | TD = {TD_val} ms", fontsize=10.5)
            ax.grid(True, alpha=0.2, lw=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.text(0.03, 0.97, f'({_panel_labels[panel_idx]})', transform=ax.transAxes,
                    va="top", ha="left", fontsize=11, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fname = os.path.join(_FIG_DIR_VAL, "fig6_merged_signal_combined.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()

# %% [markdown]
# # Plot 1 — Influence of extreme geometries (rmean, f)

# %%
# Fixed: Dex = weighted consensus per (patient, TD)
# Varied: rmean ∈ {3, 6, 9} µm  ×  f ∈ {0.4, 0.8}  (rsd fixed = 0.5 µm)

_r_vals    = [3, 9]
_f_vals    = [0.05, 0.6]
_rsd_fixed = 0.5
_cmap_r    = {3: "tab:red", 9: "tab:green"}
_ls_f      = {0.05: "--", 0.3: "-.", 0.6: ":"}

for pat_id in Patient:
    pat_key = f"Patient_{pat_id}"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, TD_val in zip(axes, TD_list):
        td_key = f"TD_{TD_val}"
        td_rows_pat = df_summary[df_summary["Patient"] == pat_id]
        if td_rows_pat.empty:
            ax.set_visible(False)
            continue
        dex_cons = _weighted_consensus(pat_id, TD_val, "Dex_best")

        df_td     = _dwi_td(pat_id, TD_val)
        bvals_dwi = df_td["b"].to_numpy()

        ax.fill_between(df_td["b"],
                        df_td["signal"] - df_td["noise"],
                        df_td["signal"] + df_td["noise"],
                        color="tab:gray", alpha=0.3)
        ax.plot(df_td["b"], df_td["signal"],
                "o-", color="tab:gray", lw=1.5, ms=3, label="Measured DWI", zorder=5)

        for (he_lbl, sig_h, p_low_h, p_high_h), col_h in zip(
                _histo_per_he(pat_key, td_key, bvals_dwi,
                              he_filter=_he_filter.get(pat_id)), _he_colors):
            ax.errorbar(bvals_dwi, sig_h,
                        yerr=[sig_h - p_low_h, p_high_h - sig_h],
                        fmt="o-", color=col_h, lw=1.5, ms=3,
                        capsize=4, elinewidth=1.2, zorder=4,
                        label=f"Histo sim {he_lbl} (Dex={dex_cons:.2f})")

        for r_val in _r_vals:
            for f_val in _f_vals:
                bv, sc = _get_signal(f_val, dex_cons, r_val, _rsd_fixed, TD_val)
                ax.plot(bvals_dwi, np.interp(bvals_dwi, bv, sc),
                        _ls_f[f_val], color=_cmap_r[r_val], lw=1.4,
                        label=f"r={r_val}µm, f={f_val}")

        ax.set_xlabel("b-value (ms/µm²)")
        ax.set_ylabel("Normalised signal S/S₀")
        ax.set_title(f"TD = {TD_val} ms  |  Dex={dex_cons:.2f} µm²/ms  |  rsd={_rsd_fixed} µm",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Patient {pat_id} — Influence of geometry (rmean, f)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# # Plot 2 — Influence of Dex

# %%
# Fixed: consensus geometry (f_eff, rmean, rsd) from df_summary
# Varied: Dex ∈ {1.5, 2.0, 2.5, 3.0} µm²/ms

_dex_range = [1.5, 2.0, 2.5, 3.0]
_cmap_dex  = plt.colormaps["plasma"](np.linspace(0.15, 0.85, len(_dex_range)))

for pat_id in Patient:
    pat_key = f"Patient_{pat_id}"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, TD_val in zip(axes, TD_list):
        td_key = f"TD_{TD_val}"
        td_rows_pat = df_summary[df_summary["Patient"] == pat_id]
        if td_rows_pat.empty:
            ax.set_visible(False)
            continue

        dex_cons = _weighted_consensus(pat_id, TD_val, "Dex_best")
        f_cons   = _weighted_consensus(pat_id, TD_val, "f_eff")
        r_cons   = _weighted_consensus(pat_id, TD_val, "r3D_vivo (µm)")
        rsd_cons = (_weighted_consensus(pat_id, TD_val, "r3D_std_vivo")
                    if "r3D_std_vivo" in df_summary.columns else 0.5)

        df_td     = _dwi_td(pat_id, TD_val)
        bvals_dwi = df_td["b"].to_numpy()

        ax.fill_between(df_td["b"],
                        df_td["signal"] - df_td["noise"],
                        df_td["signal"] + df_td["noise"],
                        color="tab:gray", alpha=0.3)
        ax.plot(df_td["b"], df_td["signal"],
                "o-", color="tab:gray", lw=1, ms=3, label="Measured DWI", zorder=5)

        for (he_lbl, sig_h, p_low_h, p_high_h), col_h in zip(
                _histo_per_he(pat_key, td_key, bvals_dwi,
                              he_filter=_he_filter.get(pat_id)), _he_colors):
            ax.errorbar(bvals_dwi, sig_h,
                        yerr=[sig_h - p_low_h, p_high_h - sig_h],
                        fmt="o-", color=col_h, lw=1, ms=3,
                        capsize=4, elinewidth=1.2, zorder=4,
                        label=f"Histo sim {he_lbl} (Dex={dex_cons:.2f})")

        for dex_val, color in zip(_dex_range, _cmap_dex):
            bv, sc = _get_signal(f_cons, dex_val, r_cons, rsd_cons, TD_val)
            ax.plot(bvals_dwi, np.interp(bvals_dwi, bv, sc),
                    "--", color=color, lw=1.4, label=f"Dex={dex_val:.1f} µm²/ms")

        ax.set_xlabel("b-value (ms/µm²)")
        ax.set_ylabel("Normalised signal S/S₀")
        ax.set_title(f"TD = {TD_val} ms  |  f={f_cons:.2f}, r={r_cons:.1f}µm, rsd={rsd_cons:.2f}µm",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Patient {pat_id} — Influence of Dex",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# # Quantitative validation — normalised residuals (S_sim − S_DWI) / σ_noise

# %%
exp_val = 3

residual_rows = []

for _, row in df_summary.iterrows():
    pat_id = int(row["Patient"])
    he_r   = int(row["H&E"])
    roi_r  = str(row["ROI"])
    f_r    = float(row["f_eff"])
    r3D_r  = float(row["r3D_vivo (µm)"])

    cell_path = os.path.join(full_path_dir,
                             f"Patient_{pat_id}_H&E_{he_r}_exp_{exp_val}.txt")
    if not os.path.exists(cell_path):
        continue
    df_cells_r = pd.read_csv(cell_path, sep='\t', engine='python')
    df_roi_r   = df_cells_r[df_cells_r["Parent"] == roi_r]
    if len(df_roi_r) < 5:
        continue
    max_c = df_roi_r["Cell: Max caliper"].to_numpy()
    min_c = df_roi_r["Cell: Min caliper"].to_numpy()
    mm, mn = np.mean(max_c), np.mean(min_c)
    sm, sn = np.std(max_c, ddof=1), np.std(min_c, ddof=1)
    r3D_std_r = 1.27 * 0.25 * np.sqrt((mm / mn) * sn**2 + (mn / mm) * sm**2)

    for TD_val in TD_list:
        Dex_r = float(row["Dex_19_best"] if TD_val == 19 else row["Dex_49_best"])

        df_td     = _dwi_td(pat_id, TD_val)
        if len(df_td) < 3:
            continue
        bvals_dwi = df_td["b"].to_numpy()
        S_dwi     = df_td["signal"].to_numpy()
        sigma     = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6,
                             df_td["noise"].to_numpy())

        bv_all, sig_all = _get_signal(f_r, Dex_r, r3D_r, r3D_std_r, TD_val)
        S_sim      = np.interp(bvals_dwi, bv_all, sig_all)
        residuals  = (S_sim - S_dwi) / sigma
        rmse_norm  = float(np.sqrt(np.mean(residuals**2)))

        for b_i, r_i, s_sim_i, s_dwi_i, sig_i in zip(
                bvals_dwi, residuals, S_sim, S_dwi, sigma):
            residual_rows.append({
                "Patient":    pat_id,
                "H&E":        he_r,
                "ROI":        roi_r,
                "TD (ms)":    TD_val,
                "b (ms/µm²)": round(b_i, 3),
                "S_sim":      round(s_sim_i, 5),
                "S_DWI":      round(s_dwi_i, 5),
                "sigma":      round(sig_i, 5),
                "residual":   round(r_i, 4),
                "RMSE_norm":  round(rmse_norm, 4),
            })

df_resid = pd.DataFrame(residual_rows)
print(df_resid.groupby(["Patient", "TD (ms)", "H&E", "ROI"])["RMSE_norm"]
              .first().reset_index().to_string(index=False))

# %% [markdown]
# ## Plot A — Normalised residual profile per b-value

# %%
for pat_id in Patient:
    df_p = df_resid[df_resid["Patient"] == pat_id]
    if df_p.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, TD_val in zip(axes, TD_list):
        df_pt = df_p[df_p["TD (ms)"] == TD_val]
        if df_pt.empty:
            ax.set_visible(False)
            continue

        ax.axhspan(-1, 1, color="tab:gray", alpha=0.15, label="±1σ noise level")
        ax.axhline(0, color="black", lw=0.8, ls="--")

        combos = df_pt[["H&E", "ROI"]].drop_duplicates().sort_values(["H&E", "ROI"])
        for idx, (_, c) in enumerate(combos.iterrows()):
            sub = df_pt[(df_pt["H&E"] == c["H&E"]) & (df_pt["ROI"] == c["ROI"])]
            sub = sub.sort_values("b (ms/µm²)")
            ax.plot(sub["b (ms/µm²)"], sub["residual"],
                    "o-", color=_roi_colors[idx % len(_roi_colors)],
                    lw=1.5, ms=5, label=f"H&E {c['H&E']} {c['ROI']}")

        ax.set_xlabel("b-value (ms/µm²)")
        ax.set_ylabel("(S_sim - S_DWI) / σ")
        ax.set_title(f"TD = {TD_val} ms", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Patient {pat_id} — Normalised residuals  (|r| ≤ 1 → within noise)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Plot B — RMSE_norm summary per ROI

# %%
df_rmse = (df_resid.groupby(["Patient", "TD (ms)", "H&E", "ROI"])["RMSE_norm"]
           .first().reset_index())
df_rmse["label"] = "H&E " + df_rmse["H&E"].astype(str) + " " + df_rmse["ROI"].astype(str)

for pat_id in Patient:
    df_p = df_rmse[df_rmse["Patient"] == pat_id]
    if df_p.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, TD_val in zip(axes, TD_list):
        df_pt = df_p[df_p["TD (ms)"] == TD_val].sort_values("label")
        if df_pt.empty:
            ax.set_visible(False)
            continue

        colors = [_roi_colors[i % len(_roi_colors)] for i in range(len(df_pt))]
        bars   = ax.bar(df_pt["label"], df_pt["RMSE_norm"],
                        color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="black", lw=1.5, ls="--",
                   label="Noise threshold (RMSE_norm = 1)")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xlabel("H&E slide — ROI")
        ax.set_ylabel("RMSE_norm = √⟨(S_sim−S_DWI)²/σ²⟩")
        ax.set_title(f"TD = {TD_val} ms", fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Patient {pat_id} — RMSE normalised by noise  (< 1 → within noise level)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# # Specificity — chi² between DWI and extreme/histological geometries

# %%
# For each patient × TD, compute chi²_min for:
#   - histological simulation (best-fit from df_summary)
#   - extreme geometries: (r=3µm, f=0.1), (r=3µm, f=0.8), (r=9µm, f=0.1), (r=9µm, f=0.8)
# A lower chi² = better match to DWI. Histological sim should have the lowest chi².

_extreme_geoms = [
    (3,  0.05, "r=3µm, f=0.05"),
    (3,  0.6, "r=3µm, f=0.6"),
    (9,  0.05, "r=9µm, f=0.05"),
    (9,  0.6, "r=9µm, f=0.6"),
]
_rsd_spec = 0.5   # fixed rsd for extreme geometries

spec_rows = []

for pat_id in Patient:
    for TD_val in TD_list:
        if df_summary[df_summary["Patient"] == pat_id].empty:
            continue

        dex_cons = _weighted_consensus(pat_id, TD_val, "Dex_best")
        f_cons   = _weighted_consensus(pat_id, TD_val, "f_eff")
        r_cons   = _weighted_consensus(pat_id, TD_val, "r3D_vivo (µm)")
        rsd_cons = (_weighted_consensus(pat_id, TD_val, "r3D_std_vivo")
                    if "r3D_std_vivo" in df_summary.columns else 0.5)

        df_td     = _dwi_td(pat_id, TD_val)
        bvals_dwi = df_td["b"].to_numpy()
        S_dwi     = df_td["signal"].to_numpy()
        sigma     = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6,
                             df_td["noise"].to_numpy())

        # chi² histological (consensus)
        bv, sc = _get_signal(f_cons, dex_cons, r_cons, rsd_cons, TD_val)
        S_histo = np.interp(bvals_dwi, bv, sc)
        chi2_histo = float(np.sum(((S_histo - S_dwi) / sigma) ** 2))
        spec_rows.append({
            "Patient": pat_id, "TD (ms)": TD_val,
            "Geometry": "Histological (consensus)",
            "chi2": round(chi2_histo, 3),
        })

        # chi² extreme geometries (use consensus Dex)
        for r_ext, f_ext, label in _extreme_geoms:
            bv, sc = _get_signal(f_ext, dex_cons, r_ext, _rsd_spec, TD_val)
            S_ext  = np.interp(bvals_dwi, bv, sc)
            chi2_ext = float(np.sum(((S_ext - S_dwi) / sigma) ** 2))
            spec_rows.append({
                "Patient": pat_id, "TD (ms)": TD_val,
                "Geometry": label,
                "chi2": round(chi2_ext, 3),
            })

df_spec = pd.DataFrame(spec_rows)
print(df_spec.to_string(index=False))

# %% [markdown]
# ## Plot C — Specificity bar chart

# %%
# For each patient × TD: grouped bar chart of chi² values.
# Histological bar should be the lowest (best fit).

_geom_order   = ["Histological (consensus)"] + [g[2] for g in _extreme_geoms]
_geom_colors  = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]

for pat_id in Patient:
    df_sp = df_spec[df_spec["Patient"] == pat_id]
    if df_sp.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, TD_val in zip(axes, TD_list):
        df_spt = df_sp[df_sp["TD (ms)"] == TD_val]
        if df_spt.empty:
            ax.set_visible(False)
            continue

        df_spt = df_spt.set_index("Geometry").reindex(_geom_order).reset_index()
        bars   = ax.bar(df_spt["Geometry"], df_spt["chi2"],
                        color=_geom_colors, alpha=0.8, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_ylabel("χ² (lower = better match to DWI)")
        ax.set_title(f"TD = {TD_val} ms", fontsize=10)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"Patient {pat_id} — Specificity: χ² vs DWI for histological vs extreme geometries",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()

# %%
