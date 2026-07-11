"""
Publication figures for DWI acquisition optimisation.

Reads results/opt_results.npz produced by opt_engine.py and generates:
  fig1_signal_decay.pdf   — signal S/S0(b) mean ± std per phenotype, 3 TDs
  fig2_cnr_heatmaps.pdf   — CNR heatmap on (b, TD) grid for all 10 pairs
  fig3_violin_optimal.pdf — signal distributions at each pair's optimal (b, TD)
  fig4_summary_table.pdf  — ranked table: best CNR / AUC per pair
  summary_table.csv       — same table as CSV

Run after opt_engine.py.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from tumor_catalog import CATALOG

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "results", "opt_results.npz")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "opt_results")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})

# ── Load results ───────────────────────────────────────────────────────────────
data        = np.load(NPZ_PATH, allow_pickle=True)
bval_use    = data["bval_use"]                   # (N_BVAL_USE,)  b > 0
TD          = data["TD"]                         # (N_TD,)
pheno_names = list(data["pheno_names"])
pairs       = list(itertools.combinations(pheno_names, 2))

color_map = {p["name"]: p["color"] for p in CATALOG}
label_map = {p["name"]: p["label"] for p in CATALOG}

# Reconstruct per-pair metric dicts and per-phenotype signals
results = {}
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    results[key] = {
        "CNR":    data[f"{key}__CNR"],
        "sigs_A": data[f"{key}__sigs_A"],
        "sigs_B": data[f"{key}__sigs_B"],
    }

pheno_signals = {}
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    if pA not in pheno_signals:
        pheno_signals[pA] = data[f"{key}__sigs_A"]
    if pB not in pheno_signals:
        pheno_signals[pB] = data[f"{key}__sigs_B"]

# 3 TDs to show in signal decay (first, middle, last)
TD_show = [TD[0], TD[2], TD[-1]]
TD_idx  = [int(np.where(TD == t)[0][0]) for t in TD_show]


def cell_edges(centers):
    """Compute cell boundaries from cell centers for use with pcolormesh."""
    c = np.asarray(centers, dtype=float)
    edges = np.empty(len(c) + 1)
    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0]    = c[0]  - (c[1]  - c[0])  / 2
    edges[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return edges


def draw_heatmap(ax, CNR, bval_arr, TD_arr, vmax, cmap="hot", fontsize=6):
    """
    Draw a discrete CNR heatmap using pcolormesh so that each cell is centred
    on its (b-value, TD) coordinate, regardless of axis spacing.
    Returns the pcolormesh object for colorbar attachment.
    """
    b_edges  = cell_edges(bval_arr)
    td_edges = cell_edges(TD_arr)
    im = ax.pcolormesh(b_edges, td_edges, CNR, cmap=cmap, vmin=0, vmax=vmax)

    # Value annotations at cell centres
    for ti, t in enumerate(TD_arr):
        for bi, b in enumerate(bval_arr):
            val = CNR[ti, bi]
            if np.isnan(val):
                continue
            brightness = val / vmax if vmax > 0 else 0
            color = "white" if brightness < 0.55 else "black"
            ax.text(b, t, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")

    ax.set_xticks(bval_arr)
    ax.set_xticklabels([f"{b:.2f}" for b in bval_arr], rotation=45, ha="right")
    ax.set_yticks(TD_arr)
    ax.set_yticklabels([f"{t:.0f}" for t in TD_arr])
    return im


# =============================================================================
# Fig 1 — Signal decay per phenotype (mean ± std over 500 LHS samples)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes_flat = axes.flatten()

for ax_i, name in enumerate(pheno_names):
    ax   = axes_flat[ax_i]
    sigs = pheno_signals[name]            # (N_valid, N_TD, N_BVAL_USE)
    col  = color_map[name]

    for ti, (t, t_idx) in enumerate(zip(TD_show, TD_idx)):
        alpha_line  = 0.4 + 0.3 * ti     # darker for longer TD
        s_mean = np.nanmean(sigs[:, t_idx, :], axis=0)
        s_std  = np.nanstd(sigs[:, t_idx, :],  axis=0)
        ax.plot(bval_use, s_mean, color=col, alpha=alpha_line, lw=1.8,
                label=f"TD = {t:.0f} ms")
        ax.fill_between(bval_use, s_mean - s_std, s_mean + s_std,
                        alpha=0.10, color=col)

    ax.set_title(label_map[name])
    ax.set_xlabel("b-value (ms/µm²)")
    ax.set_ylabel("S/S₀")
    ax.set_xlim(0, bval_use[-1]);  ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

axes_flat[-1].set_visible(False)   # 5 phenotypes → 6th panel hidden
fig.suptitle("Signal decay S/S₀(b) per phenotype\n"
             "Mean ± std over 500 Latin Hypercube samples", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_signal_decay.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("Fig 1 saved  (signal decay)")


# =============================================================================
# Fig 2 — CNR heatmaps for all 10 pairs
# Layout: 2 rows × 5 cols
# =============================================================================
n_pairs  = len(pairs)
n_cols   = 5
n_rows   = (n_pairs + n_cols - 1) // n_cols    # = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))
axes_flat = axes.flatten()

vmax_cnr = max(np.nanmax(results[f"{pA}_vs_{pB}"]["CNR"])
               for pA, pB in pairs)

for ax_i, (pA, pB) in enumerate(pairs):
    ax  = axes_flat[ax_i]
    key = f"{pA}_vs_{pB}"
    CNR = results[key]["CNR"]             # (N_TD, N_BVAL_USE)

    im = draw_heatmap(ax, CNR, bval_use, TD, vmax_cnr)

    # Mark optimal point at cell centre
    best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
    ax.plot(bval_use[best_bi], TD[best_ti], marker="*", color="cyan", ms=10, zorder=5)

    ax.set_title(f"{label_map[pA]}\nvs {label_map[pB]}", fontsize=9)
    ax.set_xlabel("b (ms/µm²)", fontsize=8)
    ax.set_ylabel("TD (ms)", fontsize=8)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, shrink=0.85, label="CNR")

for ax_i in range(n_pairs, n_rows * n_cols):
    axes_flat[ax_i].set_visible(False)

fig.suptitle("CNR on (b-value, TD) grid — cyan star = optimal point", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_cnr_heatmaps.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("Fig 2 saved  (CNR heatmaps)")


# =============================================================================
# Fig 3 — Violin plots at each pair's optimal (b, TD)
# Layout: 2 rows × 5 cols
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8))
axes_flat = axes.flatten()

for ax_i, (pA, pB) in enumerate(pairs):
    ax  = axes_flat[ax_i]
    key = f"{pA}_vs_{pB}"
    CNR = results[key]["CNR"]
    sigs_A = results[key]["sigs_A"]
    sigs_B = results[key]["sigs_B"]

    best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
    a = sigs_A[:, best_ti, best_bi]
    b = sigs_B[:, best_ti, best_bi]

    vp = ax.violinplot([a, b], positions=[0, 1], showmedians=True, showextrema=False)
    vp["bodies"][0].set_facecolor(color_map[pA]);  vp["bodies"][0].set_alpha(0.7)
    vp["bodies"][1].set_facecolor(color_map[pB]);  vp["bodies"][1].set_alpha(0.7)
    vp["cmedians"].set_color("black")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_map[pA], label_map[pB]], fontsize=8, rotation=15)
    ax.set_ylabel("S/S₀", fontsize=8)
    ax.set_title(f"b={bval_use[best_bi]:.2f}, TD={TD[best_ti]:.0f} ms\n"
                 f"CNR={CNR[best_ti, best_bi]:.2f}", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

for ax_i in range(n_pairs, n_rows * n_cols):
    axes_flat[ax_i].set_visible(False)

fig.suptitle("Signal distributions at optimal (b, TD) per pair", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_violin_optimal.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("Fig 3 saved  (violin plots)")


# =============================================================================
# Fig 4 — Summary table: best (b, TD) per pair, CNR / Cohen's d / AUC
# =============================================================================
rows = []
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    CNR = results[key]["CNR"]

    best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
    rows.append({
        "Pair":             f"{label_map[pA]} / {label_map[pB]}",
        "b_opt (ms/µm²)":  float(bval_use[best_bi]),
        "TD_opt (ms)":      float(TD[best_ti]),
        "CNR":              float(CNR[best_ti, best_bi]),
    })

df = pd.DataFrame(rows).sort_values("CNR", ascending=False).reset_index(drop=True)

csv_path = os.path.join(OUT_DIR, "summary_table.csv")
df.to_csv(csv_path, index=False, float_format="%.3f")
print(f"Summary CSV saved -> {csv_path}")
print(df.to_string(index=False))

# Render as figure
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
col_labels = list(df.columns)
cell_vals  = [[f"{v:.3f}" if isinstance(v, float) else str(v)
               for v in row] for row in df.itertuples(index=False)]
tbl = ax.table(cellText=cell_vals, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.6)
# Colour header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# Colour CNR column
for i in range(1, len(df) + 1):
    tbl[i, 3].set_facecolor("#fce4d6")

fig.suptitle("Optimal (b, TD) per phenotype pair — sorted by CNR", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_summary_table.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("Fig 4 saved  (summary table)")

print(f"\nAll figures -> {os.path.abspath(OUT_DIR)}")


# =============================================================================
# Level 2 figures — T2 / SNR-corrected metrics (only if run in opt_engine.py)
# =============================================================================
snr_enabled = bool(data["snr__enabled"][0]) if "snr__enabled" in data.files else False

if snr_enabled:
    T2_MS = float(data["snr__T2_MS"][0])
    SNR0  = float(data["snr__SNR0"][0])
    delta = float(data["delta"][0])

    results_snr = {}
    for pA, pB in pairs:
        key = f"{pA}_vs_{pB}"
        results_snr[key] = {
            "CNR": data[f"snr__{key}__CNR"],
        }

    # ── Fig 5 — CNR heatmaps: Level 1 vs Level 2 ─────────────────────────────
    # Layout: 2 rows (L1 top, L2 bottom) × n_pairs columns.
    # Each column = one phenotype pair; rows separate the two analysis levels.
    vmax_global = max(
        max(np.nanmax(results[f"{pA}_vs_{pB}"]["CNR"]),
            np.nanmax(results_snr[f"{pA}_vs_{pB}"]["CNR"]))
        for pA, pB in pairs
    )

    row_specs = [
        ("Level 1 — diffusion only",
         lambda key: results[key]["CNR"]),
        (f"Level 2 — T2={T2_MS:.0f} ms, SNR₀={SNR0:.0f}",
         lambda key: results_snr[key]["CNR"]),
    ]

    fig, axes = plt.subplots(2, n_pairs, figsize=(3.5 * n_pairs, 8),
                             squeeze=False)

    for row_idx, (row_label, get_cnr) in enumerate(row_specs):
        for col_idx, (pA, pB) in enumerate(pairs):
            ax  = axes[row_idx, col_idx]
            key = f"{pA}_vs_{pB}"
            CNR = get_cnr(key)

            im = draw_heatmap(ax, CNR, bval_use, TD, vmax_global, fontsize=5)

            best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
            ax.plot(bval_use[best_bi], TD[best_ti],
                    marker="*", color="cyan", ms=10, zorder=5)

            if row_idx == 0:
                ax.set_title(f"{label_map[pA]}\nvs {label_map[pB]}", fontsize=8)
            ax.set_xlabel("b (ms/µm²)", fontsize=8)
            ax.set_ylabel("TD (ms)", fontsize=8)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, shrink=0.85, label="CNR")

        # Row label on the leftmost panel
        axes[row_idx, 0].set_ylabel(
            f"{row_label}\nTD (ms)", fontsize=8, labelpad=6)

    fig.suptitle("CNR heatmaps — Level 1 vs Level 2 (cyan star = optimal point)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_cnr_l1_vs_l2.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Fig 5 saved  (L1 vs L2 CNR heatmaps)")

    # ── Fig 6 — Summary table: optimal (b, TD) comparison L1 vs L2 ───────────
    rows6 = []
    for pA, pB in pairs:
        key  = f"{pA}_vs_{pB}"
        CNR1 = results[key]["CNR"]
        CNR2 = results_snr[key]["CNR"]

        ti1, bi1 = np.unravel_index(np.nanargmax(CNR1), CNR1.shape)
        ti2, bi2 = np.unravel_index(np.nanargmax(CNR2), CNR2.shape)

        rows6.append({
            "Pair":              f"{label_map[pA]} / {label_map[pB]}",
            "L1 b* (ms/µm²)":   float(bval_use[bi1]),
            "L1 TD* (ms)":       float(TD[ti1]),
            "L1 CNR":            float(CNR1[ti1, bi1]),
            "L2 b* (ms/µm²)":   float(bval_use[bi2]),
            "L2 TD* (ms)":       float(TD[ti2]),
            "L2 CNR":            float(CNR2[ti2, bi2]),
        })

    df6 = pd.DataFrame(rows6).sort_values("L1 CNR", ascending=False).reset_index(drop=True)
    csv6 = os.path.join(OUT_DIR, "summary_table_l1_vs_l2.csv")
    df6.to_csv(csv6, index=False, float_format="%.3f")
    print(f"Summary L1 vs L2 CSV saved -> {csv6}")

    fig6, ax6 = plt.subplots(figsize=(15, 4))
    ax6.axis("off")
    col_labels6 = list(df6.columns)
    cell_vals6  = [[f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in row] for row in df6.itertuples(index=False)]
    tbl6 = ax6.table(cellText=cell_vals6, colLabels=col_labels6,
                     loc="center", cellLoc="center")
    tbl6.auto_set_font_size(False)
    tbl6.set_fontsize(8)
    tbl6.scale(1.0, 1.6)
    # Header style
    for j in range(len(col_labels6)):
        tbl6[0, j].set_facecolor("#2c3e50")
        tbl6[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight L1 columns (cols 1-3) and L2 columns (cols 4-6)
    for i in range(1, len(df6) + 1):
        for j in [1, 2, 3]:
            tbl6[i, j].set_facecolor("#dce9f7")   # light blue — Level 1
        for j in [4, 5, 6]:
            tbl6[i, j].set_facecolor("#fce4d6")   # light orange — Level 2

    fig6.suptitle(
        f"Optimal (b, TD) — Level 1 (diffusion only) vs "
        f"Level 2 (T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f})",
        fontsize=11, y=0.98)
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUT_DIR, "fig6_summary_l1_vs_l2.png"), bbox_inches="tight", dpi=150)
    plt.close(fig6)
    print("Fig 6 saved  (L1 vs L2 summary table)")

    print(f"\nLevel 2 figures -> {os.path.abspath(OUT_DIR)}")
else:
    print("\nLevel 2 figures skipped (ENABLE_T2_SNR = False in opt_engine.py)")
