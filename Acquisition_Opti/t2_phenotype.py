"""
Option B optimisation: phenotype-specific T2 values.

Option A (t2_sensitivity.py / snr_sensitivity.py): single T2 for all phenotypes,
swept as a sensitivity parameter. Isolates the TE penalty without assuming
tissue-specific T2 priors.

Option B (this script): each phenotype is assigned its own literature T2.
The noise floor for phenotype X at diffusion time TD is:
    σ_noise_X(TD) = (1/SNR0) × exp((TE(TD) − TE_ref) / T2_X)
This models the scenario where phenotype-specific T2 priors are available
(e.g., from a prior T2 mapping scan), allowing a phenotype-aware optimisation.

T2 values assigned in tumor_catalog.py (3T, literature):
    edema       : 400 ms  — vasogenic edema     (Hattingen et al. 2009)
    cyst        : 1000 ms — protein-rich cyst
    large_cells : 146 ms  — LGG proxy            (Gu et al. 2021, DOI: 10.21037/qims-20-916)
    small_cells : 124 ms  — HGG proxy            (Gu et al. 2021, DOI: 10.21037/qims-20-916)
    fibrosis    :  80 ms  — fibrous meningioma   (Kamada et al. 2005)

Generates:
    fig_optB_heatmaps.png     — CNR heatmaps per pair (Option B)
    fig_optB_vs_A.png         — 3-way comparison: Level 1 / Option A / Option B
                                  matrices showing b*, TD*, peak CNR
    t2pheno_summary.csv       — table: b*, TD*, peak CNR per pair × scenario
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opt_engine import discrimination_metrics_snr, discrimination_metrics_snr_t2
from tumor_catalog import CATALOG

# ── Config ─────────────────────────────────────────────────────────────────────
SNR0     = 40.0    # conservative — calibrated from validation noise maps
T2_OPT_A = 100.0  # ms — single T2 used in Option A (Gu et al. 2021 intermediate)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "results", "opt_results.npz")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "t2_phenotype")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9})

# ── Load ───────────────────────────────────────────────────────────────────────
data        = np.load(NPZ_PATH, allow_pickle=True)
bval_use    = data["bval_use"]
TD          = data["TD"]
delta       = float(data["delta"][0])
pheno_names = list(data["pheno_names"])
pairs       = list(itertools.combinations(pheno_names, 2))

label_map = {p["name"]: p["label"] for p in CATALOG}
color_map = {p["name"]: p["color"] for p in CATALOG}
t2_map    = {p["name"]: p["t2"]   for p in CATALOG}

sigs   = {}
cnr_l1 = {}
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    sigs[key]   = {"sigs_A": data[f"{key}__sigs_A"],
                   "sigs_B": data[f"{key}__sigs_B"]}
    cnr_l1[key] = data[f"{key}__CNR"]

print("Phenotype T2 values used in Option B:")
for name in pheno_names:
    print(f"  {label_map[name]:15s} : T2 = {t2_map[name]:.0f} ms")
print(f"\nTD grid  : {TD} ms  |  delta = {delta} ms  |  SNR0 = {SNR0}")


# ── Compute CNR grids ──────────────────────────────────────────────────────────
print("\nComputing CNR grids...")

cnr_optA = {}   # Option A — single T2=100ms, same noise floor for both phenotypes
cnr_optB = {}   # Option B — per-phenotype T2

for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    sA, sB = sigs[key]["sigs_A"], sigs[key]["sigs_B"]

    cnr_optA[key] = discrimination_metrics_snr(
        sA, sB, TD, delta, T2_OPT_A, SNR0)

    cnr_optB[key] = discrimination_metrics_snr_t2(
        sA, sB, TD, delta,
        t2_A=t2_map[pA], t2_B=t2_map[pB],
        snr0=SNR0)

    ti_A, bi_A = np.unravel_index(np.nanargmax(cnr_optA[key]), cnr_optA[key].shape)
    ti_B, bi_B = np.unravel_index(np.nanargmax(cnr_optB[key]), cnr_optB[key].shape)
    print(f"  {label_map[pA]:12s} vs {label_map[pB]:12s} | "
          f"Opt A: b={bval_use[bi_A]:.2f} TD={TD[ti_A]:.0f}ms CNR={cnr_optA[key][ti_A,bi_A]:.3f} | "
          f"Opt B: b={bval_use[bi_B]:.2f} TD={TD[ti_B]:.0f}ms CNR={cnr_optB[key][ti_B,bi_B]:.3f}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    e = np.empty(len(c) + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0]    = c[0]  - (c[1]  - c[0])  / 2
    e[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return e


def draw_heatmap(ax, CNR, bval_arr, TD_arr, vmax, fontsize=5.5):
    b_edges  = cell_edges(bval_arr)
    td_edges = cell_edges(TD_arr)
    im = ax.pcolormesh(b_edges, td_edges, CNR, cmap="hot", vmin=0, vmax=vmax)
    for ti, t in enumerate(TD_arr):
        for bi, b in enumerate(bval_arr):
            val = CNR[ti, bi]
            if np.isnan(val):
                continue
            color = "white" if val / vmax < 0.55 else "black"
            ax.text(b, t, f"{val:.2f}", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")
    ax.set_xticks(bval_arr)
    ax.set_xticklabels([f"{b:.2f}" for b in bval_arr], rotation=45, ha="right")
    ax.set_yticks(TD_arr)
    ax.set_yticklabels([f"{t:.0f}" for t in TD_arr])
    return im


# =============================================================================
# Fig 1 — CNR heatmaps: Level 1 / Option A / Option B  (3 rows × n_pairs cols)
# =============================================================================
n_pairs = len(pairs)
vmax_global = max(
    max(np.nanmax(cnr_l1[f"{pA}_vs_{pB}"]),
        np.nanmax(cnr_optA[f"{pA}_vs_{pB}"]),
        np.nanmax(cnr_optB[f"{pA}_vs_{pB}"]))
    for pA, pB in pairs)

row_specs = [
    ("Level 1 — diffusion only",
     lambda key: cnr_l1[key]),
    (f"Option A — T₂ = {T2_OPT_A:.0f} ms (single, all phenotypes)  SNR₀={SNR0:.0f}",
     lambda key: cnr_optA[key]),
    (f"Option B — phenotype-specific T₂  SNR₀={SNR0:.0f}",
     lambda key: cnr_optB[key]),
]

fig1, axes1 = plt.subplots(3, n_pairs,
                            figsize=(3.2 * n_pairs, 9),
                            squeeze=False)

for row_idx, (row_label, get_cnr) in enumerate(row_specs):
    for col_idx, (pA, pB) in enumerate(pairs):
        ax  = axes1[row_idx, col_idx]
        key = f"{pA}_vs_{pB}"
        CNR = get_cnr(key)

        im = draw_heatmap(ax, CNR, bval_use, TD, vmax_global)

        best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        ax.plot(bval_use[best_bi], TD[best_ti],
                marker="*", color="cyan", ms=9, zorder=5)

        if row_idx == 0:
            ax.set_title(f"{label_map[pA]}\nvs {label_map[pB]}", fontsize=8)
        ax.set_xlabel("b (ms/µm²)", fontsize=7)
        ax.set_ylabel("TD (ms)", fontsize=7)
        ax.tick_params(labelsize=5.5)
        plt.colorbar(im, ax=ax, shrink=0.82, label="CNR")

    axes1[row_idx, 0].set_ylabel(f"{row_label}\nTD (ms)", fontsize=7, labelpad=5)

# Add T2 legend for Option B
t2_legend = "  ".join(f"{label_map[n]} T₂={t2_map[n]:.0f}ms"
                       for n in pheno_names)
fig1.text(0.5, 0.01, f"Option B T₂ values: {t2_legend}",
          ha="center", fontsize=7, style="italic")

fig1.suptitle("CNR heatmaps — Level 1 / Option A / Option B\n"
              "Cyan star = optimal (b*, TD*)", fontsize=11)
fig1.tight_layout(rect=(0, 0.03, 1, 0.97))
out1 = os.path.join(OUT_DIR, "fig_optB_heatmaps.png")
fig1.savefig(out1, bbox_inches="tight", dpi=150)
plt.close(fig1)
print(f"\nFig 1 saved -> {out1}")


# =============================================================================
# Fig 2 — 3-way comparison matrix: b*, TD*, peak CNR
#          Rows: Level 1 / Option A / Option B
#          Columns: b* / TD* / peak CNR
#          Each cell: pheno × pheno matrix
# =============================================================================
n_pheno      = len(pheno_names)
short_labels = [label_map[n] for n in pheno_names]

def build_summary_matrices(cnr_dict):
    b_mat   = np.full((n_pheno, n_pheno), np.nan)
    td_mat  = np.full((n_pheno, n_pheno), np.nan)
    cnr_mat = np.full((n_pheno, n_pheno), np.nan)
    for pA, pB in pairs:
        i = pheno_names.index(pA)
        j = pheno_names.index(pB)
        CNR = cnr_dict[f"{pA}_vs_{pB}"]
        best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        b_mat[i, j]   = b_mat[j, i]   = bval_use[best_bi]
        td_mat[i, j]  = td_mat[j, i]  = TD[best_ti]
        cnr_mat[i, j] = cnr_mat[j, i] = CNR[best_ti, best_bi]
    return b_mat, td_mat, cnr_mat

rows_data = [
    ("Level 1\n(no noise)", *build_summary_matrices(cnr_l1)),
    (f"Option A\nT₂={T2_OPT_A:.0f}ms all",  *build_summary_matrices(cnr_optA)),
    ("Option B\nper-phenotype T₂",           *build_summary_matrices(cnr_optB)),
]

# Shared colour scales across all rows
all_b   = [r[1] for r in rows_data]
all_td  = [r[2] for r in rows_data]
all_cnr = [r[3] for r in rows_data]
vmax_b    = max(np.nanmax(m) for m in all_b)
vmax_td   = max(np.nanmax(m) for m in all_td)
vmax_cnr_m = max(np.nanmax(m) for m in all_cnr)

col_specs = [
    ("b*\n(ms/µm²)", 0, vmax_b,     "Blues",   "{:.2f}"),
    ("TD*\n(ms)",    0, vmax_td,    "Oranges", "{:.0f}"),
    ("Peak CNR",     0, vmax_cnr_m, "Greens",  "{:.2f}"),
]

is_optB = [False, False, True]   # highlight Option B row

fig2, axes2 = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)

for row_idx, (row_label, b_mat, td_mat, cnr_mat) in enumerate(rows_data):
    mats = [b_mat, td_mat, cnr_mat]

    for col_idx, (col_title, vmin, vmax, cmap_name, fmt) in enumerate(col_specs):
        ax  = axes2[row_idx, col_idx]
        mat = mats[col_idx]

        cmap_obj = plt.colormaps[cmap_name].copy()
        cmap_obj.set_bad(color="#d0d0d0")
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")

        for i in range(n_pheno):
            for j in range(n_pheno):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                txt_color  = "white" if brightness > 0.65 else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

        ax.set_xticks(range(n_pheno))
        ax.set_yticks(range(n_pheno))

        if row_idx == 2:
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])

        if col_idx == 0:
            ax.set_yticklabels(short_labels, fontsize=7)
            ax.set_ylabel(row_label, fontsize=9, fontweight="bold",
                          color="#006400" if is_optB[row_idx] else "#222222",
                          labelpad=6)
        else:
            ax.set_yticklabels([])

        if row_idx == 0:
            ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)

        # Green border on Option B row
        if is_optB[row_idx]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#006400")
                spine.set_linewidth(2.0)

        plt.colorbar(im, ax=ax, shrink=0.80)

# T2 legend below
t2_lines = [f"{label_map[n]}: T₂ = {t2_map[n]:.0f} ms"
            for n in pheno_names]
fig2.text(0.5, 0.01,
          "Option B T₂ priors:  " + "   |   ".join(t2_lines),
          ha="center", fontsize=7.5, style="italic",
          bbox=dict(boxstyle="round,pad=0.3", fc="#f0fff0", ec="#006400", lw=1))

fig2.suptitle(
    "Optimal (b*, TD*, peak CNR) — Level 1 / Option A / Option B\n"
    f"Shared colour scales across rows  |  SNR₀={SNR0:.0f}  |  δ={delta:.0f} ms  "
    "|  diagonal = gray  |  green border = Option B",
    fontsize=11)
fig2.tight_layout(rect=(0, 0.05, 1, 0.97))
out2 = os.path.join(OUT_DIR, "fig_optB_vs_A.png")
fig2.savefig(out2, bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"Fig 2 saved -> {out2}")


# =============================================================================
# Summary CSV
# =============================================================================
rows = []
for pA, pB in pairs:
    key  = f"{pA}_vs_{pB}"
    pair_label = f"{label_map[pA]} / {label_map[pB]}"

    for scenario, cnr_dict in [("Level 1", cnr_l1),
                                ("Option A", cnr_optA),
                                ("Option B", cnr_optB)]:
        CNR = cnr_dict[key]
        ti, bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        t2_info = (f"T2_A={t2_map[pA]:.0f}, T2_B={t2_map[pB]:.0f}"
                   if scenario == "Option B"
                   else (f"T2={T2_OPT_A:.0f}" if scenario == "Option A" else "—"))
        rows.append({
            "Pair":            pair_label,
            "Scenario":        scenario,
            "T2 (ms)":         t2_info,
            "b* (ms/µm²)":    float(bval_use[bi]),
            "TD* (ms)":        float(TD[ti]),
            "Peak CNR":        float(CNR[ti, bi]),
        })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "t2pheno_summary.csv")
df.to_csv(csv_path, index=False, float_format="%.3f")
print(f"CSV saved  -> {csv_path}")

print(f"\nAll outputs -> {os.path.abspath(OUT_DIR)}")
