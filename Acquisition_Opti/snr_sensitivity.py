"""
SNR0 sensitivity analysis for Level 2 optimisation.

Reuses the LHS signals already stored in opt_results.npz (no need to re-run
opt_engine.py) and sweeps SNR0 over a configurable range at fixed T2.

Generates:
  fig_snr0_heatmaps.pdf  — CNR heatmaps for each SNR0, one column per SNR0,
                           one row per phenotype pair
  fig_snr0_optimum.pdf   — optimal (TD*, b*) and peak CNR vs SNR0 for every pair,
                           with Level 1 (no noise) as dashed reference
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from opt_engine import discrimination_metrics_snr
from tumor_catalog import CATALOG

# ── Config ─────────────────────────────────────────────────────────────────────
SNR0_VALUES = [30, 40, 50, 60]   # values to sweep — Josh estimate: SNR_final~20 → SNR_0~35-50
T2_MS       = 100.0                             # fixed — literature value, glioma 3T

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH    = os.path.join(THIS_DIR, "results", "opt_results.npz")
OUT_DIR     = os.path.join(THIS_DIR, "Figures", "snr_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
data        = np.load(NPZ_PATH, allow_pickle=True)
bval_use    = data["bval_use"]
TD          = data["TD"]
delta       = float(data["delta"][0])
pheno_names = list(data["pheno_names"])
pairs       = list(itertools.combinations(pheno_names, 2))

label_map = {p["name"]: p["label"] for p in CATALOG}
color_map = {p["name"]: p["color"] for p in CATALOG}

# Level 1 signals and CNR (no noise, already in npz)
sigs = {}
cnr_l1 = {}
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    sigs[key] = {
        "sigs_A": data[f"{key}__sigs_A"],
        "sigs_B": data[f"{key}__sigs_B"],
    }
    cnr_l1[key] = data[f"{key}__CNR"]   # (N_TD, N_BVAL_USE) — Level 1

plt.rcParams.update({"font.size": 9})


# ── Helper: cell edges for pcolormesh ─────────────────────────────────────────
def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    edges = np.empty(len(c) + 1)
    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0]    = c[0]  - (c[1]  - c[0])  / 2
    edges[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return edges

b_edges  = cell_edges(bval_use)
td_edges = cell_edges(TD)


# ── Compute Level 2 metrics for every SNR0 ────────────────────────────────────
print(f"Computing metrics for {len(SNR0_VALUES)} SNR0 values x {len(pairs)} pairs...")
metrics = {}   # snr0 -> pair_key -> CNR array (N_TD, N_BVAL)
for snr0 in SNR0_VALUES:
    metrics[snr0] = {}
    for pA, pB in pairs:
        key = f"{pA}_vs_{pB}"
        CNR = discrimination_metrics_snr(
            sigs[key]["sigs_A"], sigs[key]["sigs_B"],
            TD, delta, T2_MS, snr0)
        metrics[snr0][key] = CNR
    print(f"  SNR0={snr0} done")


# =============================================================================
# Fig 1 — CNR heatmaps: rows = pairs, columns = SNR0 values
# =============================================================================
n_snr0  = len(SNR0_VALUES)
n_pairs = len(pairs)

vmax = max(np.nanmax(metrics[snr0][f"{pA}_vs_{pB}"])
           for snr0 in SNR0_VALUES for pA, pB in pairs)

fig, axes = plt.subplots(n_pairs, n_snr0,
                         figsize=(2.8 * n_snr0, 2.6 * n_pairs),
                         squeeze=False)

for row, (pA, pB) in enumerate(pairs):
    key = f"{pA}_vs_{pB}"
    for col, snr0 in enumerate(SNR0_VALUES):
        ax  = axes[row, col]
        CNR = metrics[snr0][key]

        im = ax.pcolormesh(b_edges, td_edges, CNR, cmap="hot", vmin=0, vmax=vmax)

        for ti, t in enumerate(TD):
            for bi, b in enumerate(bval_use):
                val = CNR[ti, bi]
                if np.isnan(val):
                    continue
                color = "white" if val / vmax < 0.55 else "black"
                ax.text(b, t, f"{val:.1f}", ha="center", va="center",
                        fontsize=4.5, color=color, fontweight="bold")

        best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        ax.plot(bval_use[best_bi], TD[best_ti],
                marker="*", color="cyan", ms=8, zorder=5)

        ax.set_xticks(bval_use)
        ax.set_xticklabels([f"{b:.2f}" for b in bval_use], rotation=45,
                           ha="right", fontsize=5)
        ax.set_yticks(TD)
        ax.set_yticklabels([f"{t:.0f}" for t in TD], fontsize=5)

        if row == 0:
            ax.set_title(f"SNR₀ = {snr0}", fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{label_map[pA]}\nvs {label_map[pB]}",
                          fontsize=7, labelpad=4)
        if row == n_pairs - 1:
            ax.set_xlabel("b (ms/µm²)", fontsize=7)

fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.35)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax, label="CNR")

fig.suptitle(f"CNR heatmaps vs SNR₀  (T₂={T2_MS:.0f} ms, δ={delta:.0f} ms)\n"
             "cyan star = optimal (b, TD)", fontsize=11, y=1.01)

out1 = os.path.join(OUT_DIR, "fig_snr0_heatmaps.png")
fig.savefig(out1, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nFig 1 saved -> {out1}")


# =============================================================================
# Fig 2 — Optimal (b*, TD*, peak CNR) as phenotype × phenotype matrices
#          One row per SNR0 value, one column per metric.
#          Each cell: value for the pair (row-pheno, col-pheno); diagonal = gray.
#          Last row: Level 1 reference (no noise, SNR0 → ∞).
# =============================================================================
n_pheno      = len(pheno_names)
short_labels = [label_map[n] for n in pheno_names]

# ── Build matrices for each SNR0 ──────────────────────────────────────────────
def make_matrices(snr0_key):
    """Return b*, TD*, CNR_peak matrices (n_pheno × n_pheno) for a given snr0_key."""
    b_mat   = np.full((n_pheno, n_pheno), np.nan)
    td_mat  = np.full((n_pheno, n_pheno), np.nan)
    cnr_mat = np.full((n_pheno, n_pheno), np.nan)
    for pA, pB in pairs:
        i = pheno_names.index(pA)
        j = pheno_names.index(pB)
        CNR = metrics[snr0_key][f"{pA}_vs_{pB}"]
        best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        b_mat[i, j]   = b_mat[j, i]   = bval_use[best_bi]
        td_mat[i, j]  = td_mat[j, i]  = TD[best_ti]
        cnr_mat[i, j] = cnr_mat[j, i] = CNR[best_ti, best_bi]
    return b_mat, td_mat, cnr_mat

# ── Build Level 1 matrices (no noise) ────────────────────────────────────────
def make_l1_matrices():
    b_mat   = np.full((n_pheno, n_pheno), np.nan)
    td_mat  = np.full((n_pheno, n_pheno), np.nan)
    cnr_mat = np.full((n_pheno, n_pheno), np.nan)
    for pA, pB in pairs:
        i = pheno_names.index(pA)
        j = pheno_names.index(pB)
        CNR = cnr_l1[f"{pA}_vs_{pB}"]
        best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        b_mat[i, j]   = b_mat[j, i]   = bval_use[best_bi]
        td_mat[i, j]  = td_mat[j, i]  = TD[best_ti]
        cnr_mat[i, j] = cnr_mat[j, i] = CNR[best_ti, best_bi]
    return b_mat, td_mat, cnr_mat

all_rows = [(snr0, *make_matrices(snr0)) for snr0 in SNR0_VALUES]
all_rows.append(("L1 (no noise)", *make_l1_matrices()))

# ── Global colour scales per metric (shared across all SNR0 rows) ─────────────
all_b   = [r[1] for r in all_rows]
all_td  = [r[2] for r in all_rows]
all_cnr = [r[3] for r in all_rows]

vmin_b,  vmax_b   = np.nanmin(all_b),  np.nanmax(all_b)
vmin_td, vmax_td  = np.nanmin(all_td), np.nanmax(all_td)
vmin_cnr = 0;      vmax_cnr = np.nanmax(all_cnr)

col_configs = [
    ("b*  (ms/µm²)", vmin_b,   vmax_b,   "Blues",   "{:.2f}"),
    ("TD* (ms)",     vmin_td,  vmax_td,  "Oranges", "{:.0f}"),
    ("Peak CNR",     vmin_cnr, vmax_cnr, "Greens",  "{:.2f}"),
]

# ── Draw figure ───────────────────────────────────────────────────────────────
n_rows_fig = len(all_rows)   # n_SNR0 + 1 (Level 1)
fig2, axes2 = plt.subplots(n_rows_fig, 3,
                            figsize=(11, 2.8 * n_rows_fig),
                            squeeze=False)

for row, (row_label, b_mat, td_mat, cnr_mat) in enumerate(all_rows):
    mats = [b_mat, td_mat, cnr_mat]
    is_l1 = (row == n_rows_fig - 1)

    for col, (col_title, vmin, vmax, cmap_name, fmt) in enumerate(col_configs):
        ax  = axes2[row, col]
        mat = mats[col]

        cmap_obj = plt.colormaps[cmap_name].copy()
        cmap_obj.set_bad(color="#d0d0d0")   # diagonal → gray
        masked = np.ma.masked_invalid(mat)

        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")

        # Annotate each off-diagonal cell
        for i in range(n_pheno):
            for j in range(n_pheno):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                txt_color  = "white" if brightness > 0.65 else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=7, color=txt_color, fontweight="bold")

        ax.set_xticks(range(n_pheno))
        ax.set_yticks(range(n_pheno))

        # x-labels only on last row
        if row == n_rows_fig - 1:
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])

        # y-labels only on first column
        if col == 0:
            ax.set_yticklabels(short_labels, fontsize=7)
            row_title = f"SNR₀ = {row_label}" if not is_l1 else row_label
            ax.set_ylabel(row_title, fontsize=8, fontweight="bold",
                          color="#222222" if not is_l1 else "#8b0000")
        else:
            ax.set_yticklabels([])

        # Column title only on first row
        if row == 0:
            ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)

        # Highlight Level 1 row with a box
        if is_l1:
            for spine in ax.spines.values():
                spine.set_edgecolor("#8b0000")
                spine.set_linewidth(1.8)

# Shared colourbars, one per metric column
fig2.suptitle(
    f"Optimal (b*, TD*, CNR) per phenotype pair — sensitivity to SNR₀\n"
    f"T₂ = {T2_MS:.0f} ms  |  δ = {delta:.0f} ms  "
    f"|  last row (red border) = Level 1 (no noise)",
    fontsize=11)
fig2.tight_layout(rect=(0, 0, 0.92, 0.97))

for col, (col_title, vmin, vmax, cmap_name, _) in enumerate(col_configs):
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        cmap=plt.colormaps[cmap_name])
    x0 = 0.93 + col * 0.025
    cbar_ax = fig2.add_axes([x0, 0.15, 0.012, 0.65])
    fig2.colorbar(sm, cax=cbar_ax, label=col_title)

out2 = os.path.join(OUT_DIR, "fig_snr0_optimum.png")
fig2.savefig(out2, bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"Fig 2 saved -> {out2}")

print(f"\nAll figures -> {os.path.abspath(OUT_DIR)}")
