"""
Reduced CNR heatmap figure for the manuscript (2 representative pairs
instead of all 10), to replace the full fig2_cnr_heatmaps grid in the
main text. Full grid stays available as Supplementary (fig2_cnr_heatmaps.png).

Pairs chosen from summary_table.csv ranking:
  - Cyst / Fibrosis        -> highest CNR  (most discriminable pair)
  - Small cells / Fibrosis -> lowest CNR   (least discriminable pair)

Run after opt_engine.py / opt_figures.py (reads results/opt_results.npz).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tumor_catalog import CATALOG

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "results", "opt_results.npz")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "opt_results")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})

# Pairs to keep, as (phenotype_A, phenotype_B) matching CATALOG "name" keys
PAIRS_TO_SHOW = [
    ("cyst", "fibrosis"),         # most discriminable
    ("small_cells", "fibrosis"),  # least discriminable
]

data     = np.load(NPZ_PATH, allow_pickle=True)
bval_use = data["bval_use"]
TD       = data["TD"]

label_map = {p["name"]: p["label"] for p in CATALOG}


def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    edges = np.empty(len(c) + 1)
    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0]    = c[0]  - (c[1]  - c[0])  / 2
    edges[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return edges


def draw_heatmap(ax, CNR, bval_arr, TD_arr, vmax, cmap="hot"):
    b_edges  = cell_edges(bval_arr)
    td_edges = cell_edges(TD_arr)
    im = ax.pcolormesh(b_edges, td_edges, CNR, cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(bval_arr)
    ax.set_xticklabels([f"{b:.2f}" for b in bval_arr], rotation=45, ha="right")
    ax.set_yticks(TD_arr)
    ax.set_yticklabels([f"{t:.0f}" for t in TD_arr])
    return im


def load_cnr(pA, pB):
    key = f"{pA}_vs_{pB}"
    if f"{key}__CNR" in data.files:
        return data[f"{key}__CNR"]
    key_rev = f"{pB}_vs_{pA}"
    return data[f"{key_rev}__CNR"]


vmax_cnr = max(np.nanmax(load_cnr(pA, pB)) for pA, pB in PAIRS_TO_SHOW)

fig, axes = plt.subplots(1, len(PAIRS_TO_SHOW), figsize=(6 * len(PAIRS_TO_SHOW) + 1, 5))
if len(PAIRS_TO_SHOW) == 1:
    axes = [axes]

for ax, (pA, pB) in zip(axes, PAIRS_TO_SHOW):
    CNR = load_cnr(pA, pB)
    im = draw_heatmap(ax, CNR, bval_use, TD, vmax_cnr)

    best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
    ax.plot(bval_use[best_bi], TD[best_ti], marker="*", color="cyan", ms=14, zorder=5)

    ax.set_title(f"{label_map[pA]} vs {label_map[pB]}", fontsize=11)
    ax.set_xlabel("b (ms/µm²)")
    ax.set_ylabel("TD (ms)")
    plt.colorbar(im, ax=ax, shrink=0.85, label="CNR")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_cnr_heatmaps_2pairs.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved -> {os.path.join(OUT_DIR, 'fig2_cnr_heatmaps_2pairs.png')}")
