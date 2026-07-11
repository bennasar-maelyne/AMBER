"""
Fixed-TD cost analysis — answers Josh's question (2026-06-20):
  "Could you fix TD = min allowable TD and check how much worse the CNR is
  compared to the optimized TD results for each T2?"

For each T2 scenario (same sweep as t2_sensitivity.py) and for Level 1
(no noise), computes:
  - CNR_opt     : peak CNR when both b and TD are co-optimized
  - CNR_fixed   : peak CNR when TD is fixed to TD_min, only b is optimized
  - % CNR loss  : (CNR_opt - CNR_fixed) / CNR_opt × 100
  - b_opt       : optimal b-value at (b*, TD*)
  - b_fixed     : optimal b-value at fixed TD_min

Generates:
  fig_fixed_td_bars.png        — grouped bar chart: CNR_opt vs CNR_fixed per pair,
                                  one subplot per T2 (+ Level 1 reference)
  fig_fixed_td_matrices.png    — pheno × pheno matrices showing b_opt, b_fixed,
                                  and % CNR loss, one row per T2 (+ Level 1)
  fixed_td_summary.csv         — full table: pair × T2, all metrics
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opt_engine import discrimination_metrics_snr
from tumor_catalog import CATALOG

# ── Config ─────────────────────────────────────────────────────────────────────
T2_VALUES  = [100, 200, 500, 1000]   # ms — same sweep as t2_sensitivity.py
SNR0       = 40.0                    # fixed — calibrated from validation noise maps
TD_MIN_IDX = 3                       # index of fixed TD in the LUT grid (TD=60ms, as suggested by Josh)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(THIS_DIR, "results", "opt_results.npz")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "fixed_td_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9})

# ── Load pre-computed LHS signals ──────────────────────────────────────────────
data        = np.load(NPZ_PATH, allow_pickle=True)
bval_use    = data["bval_use"]                   # (N_BVAL_USE,)  b > 0
TD          = data["TD"]                         # (N_TD,)
delta       = float(data["delta"][0])            # gradient pulse width (ms)
pheno_names = list(data["pheno_names"])
pairs       = list(itertools.combinations(pheno_names, 2))

label_map = {p["name"]: p["label"] for p in CATALOG}

sigs   = {}
cnr_l1 = {}
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    sigs[key]   = {"sigs_A": data[f"{key}__sigs_A"],
                   "sigs_B": data[f"{key}__sigs_B"]}
    cnr_l1[key] = data[f"{key}__CNR"]   # (N_TD, N_BVAL_USE) — Level 1

TD_min = float(TD[TD_MIN_IDX])
print(f"TD grid      : {TD} ms")
print(f"Fixed TD     : {TD_min:.0f} ms  (index {TD_MIN_IDX})")
print(f"T2 sweep     : {T2_VALUES} ms")
print(f"SNR0 (fixed) : {SNR0}")
print(f"delta        : {delta} ms")


# ── Helper ─────────────────────────────────────────────────────────────────────
def analyse_pair(CNR_grid):
    """
    Given a (N_TD, N_BVAL) CNR grid, return a dict with:
      CNR_opt   — peak CNR over full (b, TD) grid
      b_opt     — b-value at (b*, TD*)
      td_opt    — TD at (b*, TD*)
      CNR_fixed — peak CNR when TD is locked to TD_min, b optimized
      b_fixed   — optimal b-value at TD_min
      pct_loss  — % CNR reduction from fixing TD (0 if TD_min is already optimal)
    """
    best_ti, best_bi = np.unravel_index(np.nanargmax(CNR_grid), CNR_grid.shape)
    CNR_opt  = float(CNR_grid[best_ti, best_bi])
    b_opt    = float(bval_use[best_bi])
    td_opt   = float(TD[best_ti])

    row_fixed = CNR_grid[TD_MIN_IDX, :]
    best_bi_f = int(np.nanargmax(row_fixed))
    CNR_fixed = float(row_fixed[best_bi_f])
    b_fixed   = float(bval_use[best_bi_f])

    pct_loss = max(0.0, 100.0 * (CNR_opt - CNR_fixed) / (CNR_opt + 1e-12))
    return {
        "CNR_opt":  CNR_opt,  "b_opt":   b_opt,   "td_opt":  td_opt,
        "CNR_fixed": CNR_fixed, "b_fixed": b_fixed, "pct_loss": pct_loss,
    }


# ── Compute metrics ────────────────────────────────────────────────────────────
print("\nComputing metrics...")

metrics_l1 = {f"{pA}_vs_{pB}": analyse_pair(cnr_l1[f"{pA}_vs_{pB}"])
              for pA, pB in pairs}

metrics = {}
for t2 in T2_VALUES:
    metrics[t2] = {}
    for pA, pB in pairs:
        key = f"{pA}_vs_{pB}"
        CNR = discrimination_metrics_snr(
            sigs[key]["sigs_A"], sigs[key]["sigs_B"],
            TD, delta, t2, SNR0)
        metrics[t2][key] = analyse_pair(CNR)
    print(f"  T2={t2} ms done")


# ── Summary CSV ────────────────────────────────────────────────────────────────
rows = []
for t2 in T2_VALUES:
    for pA, pB in pairs:
        key = f"{pA}_vs_{pB}"
        r = metrics[t2][key]
        rows.append({"T2 (ms)": t2, "Pair": f"{label_map[pA]} / {label_map[pB]}",
                     **r})
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    r = metrics_l1[key]
    rows.append({"T2 (ms)": "L1", "Pair": f"{label_map[pA]} / {label_map[pB]}",
                 **r})

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "fixed_td_summary.csv")
df.to_csv(csv_path, index=False, float_format="%.3f")
print(f"\nSummary CSV saved -> {csv_path}")


# =============================================================================
# Fig 1 — Grouped bar chart: CNR_opt vs CNR_fixed per pair, one row per T2
# =============================================================================
all_scenarios = T2_VALUES + ["L1"]
pair_labels   = [f"{label_map[pA]}\nvs {label_map[pB]}" for pA, pB in pairs]
x     = np.arange(len(pairs))
width = 0.35

fig1, axes1 = plt.subplots(len(all_scenarios), 1,
                            figsize=(max(12, 1.8 * len(pairs)), 3.5 * len(all_scenarios)),
                            squeeze=False)

for row_idx, t2 in enumerate(all_scenarios):
    ax  = axes1[row_idx, 0]
    met = metrics[t2] if t2 != "L1" else metrics_l1

    cnr_opts  = [met[f"{pA}_vs_{pB}"]["CNR_opt"]   for pA, pB in pairs]
    cnr_fixed = [met[f"{pA}_vs_{pB}"]["CNR_fixed"]  for pA, pB in pairs]
    pct_losses = [met[f"{pA}_vs_{pB}"]["pct_loss"]  for pA, pB in pairs]

    ax.bar(x - width / 2, cnr_opts,  width, label="CNR optimal (b*, TD*)",
           color="#4878d0", alpha=0.85)
    ax.bar(x + width / 2, cnr_fixed, width,
           label=f"CNR at TD = {TD_min:.0f} ms (b* only)",
           color="#ee854a", alpha=0.85)

    for xi, (val, loss) in enumerate(zip(cnr_fixed, pct_losses)):
        ax.text(xi + width / 2, val + 0.003 * max(cnr_opts),
                f"−{loss:.1f}%", ha="center", va="bottom",
                fontsize=7, color="#b84a0a", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=7)
    ax.set_ylabel("CNR")
    ax.set_ylim(0, max(cnr_opts) * 1.28)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    if t2 == "L1":
        title = "Level 1 — diffusion only (no T2/noise penalty)"
    else:
        title = f"T2 = {t2} ms  |  SNR₀ = {SNR0:.0f}"
    ax.set_title(title, fontsize=9, fontweight="bold")

fig1.suptitle(
    f"CNR cost of fixing TD = {TD_min:.0f} ms vs co-optimized (b*, TD*)\n"
    f"Orange annotation = % CNR loss (0% means TD_min was already optimal)",
    fontsize=11)
fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "fig_fixed_td_bars.png")
fig1.savefig(out1, bbox_inches="tight", dpi=150)
plt.close(fig1)
print(f"Fig 1 saved -> {out1}")


# =============================================================================
# Fig 2 — Phenotype × phenotype matrices: b_opt, b_fixed, % CNR loss
#          One row per T2 scenario (+ Level 1 reference)
# =============================================================================
n_pheno      = len(pheno_names)
short_labels = [label_map[n] for n in pheno_names]

def build_matrices(met):
    b_opt_mat   = np.full((n_pheno, n_pheno), np.nan)
    b_fixed_mat = np.full((n_pheno, n_pheno), np.nan)
    loss_mat    = np.full((n_pheno, n_pheno), np.nan)
    for pA, pB in pairs:
        i = pheno_names.index(pA)
        j = pheno_names.index(pB)
        r = met[f"{pA}_vs_{pB}"]
        b_opt_mat[i, j]   = b_opt_mat[j, i]   = r["b_opt"]
        b_fixed_mat[i, j] = b_fixed_mat[j, i] = r["b_fixed"]
        loss_mat[i, j]    = loss_mat[j, i]    = r["pct_loss"]
    return b_opt_mat, b_fixed_mat, loss_mat

# Pre-compute global colour scales (consistent across all T2 rows)
all_b_opt   = []
all_b_fixed = []
all_loss    = []
for t2 in all_scenarios:
    met = metrics[t2] if t2 != "L1" else metrics_l1
    bm, bf, lm = build_matrices(met)
    all_b_opt.append(bm); all_b_fixed.append(bf); all_loss.append(lm)

vmax_b    = max(np.nanmax(m) for m in all_b_opt + all_b_fixed)
vmax_loss = max(np.nanmax(m) for m in all_loss)

col_specs = [
    ("b* — free TD\n(ms/µm²)",      0, vmax_b,    "Blues",   "{:.2f}"),
    ("b* — fixed TD\n(ms/µm²)",     0, vmax_b,    "Blues",   "{:.2f}"),
    ("% CNR loss\nfixing TD",        0, vmax_loss, "Oranges", "{:.1f}"),
]

n_rows_fig = len(all_scenarios)
fig2, axes2 = plt.subplots(n_rows_fig, 3,
                            figsize=(4.5 * 3, 3.2 * n_rows_fig),
                            squeeze=False)

for row_idx, (t2, b_opt_mat, b_fixed_mat, loss_mat) in enumerate(
        zip(all_scenarios, all_b_opt, all_b_fixed, all_loss)):
    mats = [b_opt_mat, b_fixed_mat, loss_mat]
    is_l1 = (t2 == "L1")

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

        if row_idx == n_rows_fig - 1:
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])

        if col_idx == 0:
            ax.set_yticklabels(short_labels, fontsize=7)
            row_lbl = f"T2 = {t2} ms" if not is_l1 else "Level 1 (no noise)"
            ax.set_ylabel(row_lbl, fontsize=9, fontweight="bold",
                          color="#222222" if not is_l1 else "#8b0000")
        else:
            ax.set_yticklabels([])

        if row_idx == 0:
            ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)

        if is_l1:
            for spine in ax.spines.values():
                spine.set_edgecolor("#8b0000")
                spine.set_linewidth(1.8)

        plt.colorbar(im, ax=ax, shrink=0.75)

fig2.suptitle(
    f"b-value stability and CNR cost — fixed TD = {TD_min:.0f} ms vs optimal TD\n"
    f"SNR₀ = {SNR0:.0f}  |  δ = {delta:.0f} ms  |  colour scales shared across rows  "
    f"|  red border = Level 1 (no noise)",
    fontsize=11)
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "fig_fixed_td_matrices.png")
fig2.savefig(out2, bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"Fig 2 saved -> {out2}")


# =============================================================================
# Fig 3 — Pair x T2 pivot table of % CNR loss (manuscript Figure 12)
#          Rows = phenotype pairs, sorted by descending loss at T2 = 100 ms.
#          Columns = T2 scenarios. Dagger marks cells where the fixed-TD
#          optimal b-value differs from the co-optimized b* (shift in b*).
#          No title — caption is added directly in the manuscript.
# =============================================================================
short_label = {
    "edema": "Edema", "cyst": "Cyst", "large_cells": "Lg. cells",
    "small_cells": "Sm. cells", "fibrosis": "Fibrosis",
}

pair_rows = []
for pA, pB in pairs:
    key = f"{pA}_vs_{pB}"
    label = f"{short_label[pA]} / {short_label[pB]}"
    losses = [metrics[t2][key]["pct_loss"] for t2 in T2_VALUES]
    shifts = [metrics[t2][key]["b_fixed"] != metrics[t2][key]["b_opt"] for t2 in T2_VALUES]
    pair_rows.append((label, losses, shifts))

pair_rows.sort(key=lambda r: r[1][0], reverse=True)

loss_mat   = np.array([losses for _, losses, _ in pair_rows])
shift_mat  = np.array([shifts for _, _, shifts in pair_rows])
row_labels = [label for label, _, _ in pair_rows]
col_labels = [f"T2 = {t2} ms" for t2 in T2_VALUES]

fig3, ax3 = plt.subplots(figsize=(7, 0.55 * len(pair_rows) + 1))
im3 = ax3.imshow(loss_mat, cmap="Oranges", vmin=0, vmax=np.nanmax(loss_mat), aspect="auto")

for i in range(loss_mat.shape[0]):
    for j in range(loss_mat.shape[1]):
        val = loss_mat[i, j]
        brightness = val / np.nanmax(loss_mat)
        txt_color = "white" if brightness > 0.6 else "black"
        label = f"{val:.1f}%" + (" †" if shift_mat[i, j] else "")
        ax3.text(j, i, label, ha="center", va="center",
                 fontsize=9, color=txt_color, fontweight="bold")

ax3.set_xticks(range(len(col_labels)))
ax3.set_xticklabels(col_labels, fontsize=9)
ax3.set_yticks(range(len(row_labels)))
ax3.set_yticklabels(row_labels, fontsize=9)
plt.colorbar(im3, ax=ax3, shrink=0.85, label="CNR loss (%)")
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "fig_fixed_td_loss_table.png")
fig3.savefig(out3, bbox_inches="tight", dpi=150)
plt.close(fig3)
print(f"Fig 3 saved -> {out3}")

print(f"\nAll outputs -> {os.path.abspath(OUT_DIR)}")
