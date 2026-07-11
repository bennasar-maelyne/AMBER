"""
CRLB (Cramér-Rao Lower Bound) maps — Approach C.

For each phenotype (operating point = centre of its bounds), computes
the Fisher Information Matrix (FIM) and derives the CRLB for f, Dex, rmean.

Theory
------
For Gaussian noise with std σ(TD), the Fisher information of a single
measurement at (b, TD) on parameter θ is:

    I(θ; b, TD) = (∂S/∂θ)² / σ²(TD)

For a multi-b acquisition at a given TD (b-values independent):

    I(θ; TD)    = Σ_b (∂S/∂θ)² / σ²(TD)       [scalar, per parameter]

The CRLB gives the minimum variance of any unbiased estimator:

    Var(θ̂) ≥ CRLB(θ; TD) = 1 / I(θ; TD)

For joint estimation of θ = (f, Dex, rmean), the Fisher Information Matrix is:

    FIM_ij(TD) = (1/σ²(TD)) × Σ_b (∂S/∂θ_i)(∂S/∂θ_j)

The joint CRLB accounts for inter-parameter correlations:

    CRLB_joint(θ_i; TD) = [FIM⁻¹]_ii  ≥  CRLB_marginal(θ_i; TD)

sqrt(CRLB) is the theoretical minimum std of θ̂:
  small → good precision, the parameter is well-identifiable at this (b, TD).

Two quantities are computed:
  marginal  CRLB(θ_i; TD) = σ²(TD) / Σ_b (∂S/∂θ_i)²    [ignores correlations]
  joint     CRLB(θ_i; TD) = [FIM⁻¹]_ii                   [accounts for correlations]

Generates
---------
  Figures/crlb/fig_crlb_heatmap.pdf      — single-b CRLB(θ; b, TD) heatmaps
  Figures/crlb/fig_crlb_vs_td.pdf        — multi-b: marginal + joint CRLB vs TD
  Figures/crlb/fig_crlb_correlation.pdf  — FIM correlation matrix per phenotype
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt_engine import load_lut, build_tri, interpolate_batch
from tumor_catalog import CATALOG

# ── Config ─────────────────────────────────────────────────────────────────────
T2_MS = 100.0
SNR0  = 40.0
STEPS = {"f": 0.04, "Dex": 0.12, "rmean": 0.8, "rsd": 0.4}
PARAMS_EST  = ["f", "Dex", "rmean"]   # parameters to estimate
PARAM_IDX   = {"f": 0, "Dex": 1, "rmean": 2, "rsd": 3}
PARAM_UNITS = {"f": "", "Dex": " (µm²/ms)", "rmean": " (µm)"}
PARAM_LABELS = {
    "f":     r"$f$",
    "Dex":   r"$D_{ex}$ (µm²/ms)",
    "rmean": r"$r_{mean}$ (µm)",
}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LUT_PATH = os.path.join(THIS_DIR, "..", "Validation", "lookup_table_optim.mat")
OUT_DIR  = os.path.join(THIS_DIR, "Figures", "crlb")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9})

# ── Load LUT ───────────────────────────────────────────────────────────────────
print("Loading LUT...")
params_lut, sig_norm, bval, TD_arr, delta = load_lut(LUT_PATH)
tri = build_tri(params_lut)

b0_idx   = int(np.where(bval == 0)[0][0])
bval_use = np.delete(bval, b0_idx)
b_use_idx = [i for i in range(len(bval)) if i != b0_idx]
N_BVAL   = len(bval_use)
N_TD     = len(TD_arr)
N_PARAMS = len(PARAMS_EST)

te_ref = TD_arr[0] + delta
print(f"  b-values: {bval_use}")
print(f"  TDs: {TD_arr} ms  |  delta: {delta} ms  |  TE_ref: {te_ref} ms\n")

# ── Noise model ────────────────────────────────────────────────────────────────
sigma_arr = np.array([(1/SNR0) * np.exp(((td + delta) - te_ref) / T2_MS)
                      for td in TD_arr])   # (N_TD,)

# ── Signal interpolation ───────────────────────────────────────────────────────
def signal_at(params_1d):
    out = interpolate_batch(params_1d.reshape(1, 4), sig_norm, tri)
    out_use = out[0][:, b_use_idx]   # (N_TD, N_BVAL)
    return None if np.isnan(out_use).any() else out_use

# ── Gradient computation (reused from sensitivity_maps) ───────────────────────
def compute_gradients(center_4d):
    """
    Returns grads: dict param -> (N_TD, N_BVAL), or (None, False) if outside hull.
    """
    grads = {}
    for param in PARAMS_EST:
        step = STEPS[param]
        idx  = PARAM_IDX[param]
        p_p  = center_4d.copy(); p_p[idx] += step
        p_m  = center_4d.copy(); p_m[idx] -= step
        sp   = signal_at(p_p)
        sm   = signal_at(p_m)
        if sp is None or sm is None:
            return None, False
        grads[param] = (sp - sm) / (2.0 * step)   # (N_TD, N_BVAL)
    return grads, True

# ── CRLB computation ───────────────────────────────────────────────────────────
def compute_crlb(grads):
    """
    From gradients (N_TD, N_BVAL) per parameter, compute:

    single_b  : (N_TD, N_BVAL) per param — CRLB for one (b, TD) measurement
    marginal  : (N_TD,) per param         — CRLB summing over all b at each TD
    joint     : (N_TD,) per param         — CRLB from full FIM⁻¹ at each TD
    fim_norm  : (N_TD, N_PARAMS, N_PARAMS) — normalised FIM correlation matrix
    """
    # ── Single-b CRLB ─────────────────────────────────────────────────────────
    # σ²(TD) broadcast over b axis
    sigma2_td = sigma_arr ** 2   # (N_TD,)
    single_b  = {}
    for param in PARAMS_EST:
        dS = grads[param]                              # (N_TD, N_BVAL)
        dS2 = dS ** 2
        # Avoid division by zero where gradient is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            crlb = np.where(dS2 > 1e-20,
                            sigma2_td[:, np.newaxis] / dS2,
                            np.nan)
        single_b[param] = crlb   # (N_TD, N_BVAL)

    # ── Multi-b FIM: Σ_b (∂S/∂θ_i)(∂S/∂θ_j) / σ²(TD) ───────────────────────
    # Stack gradients: G shape (N_PARAMS, N_TD, N_BVAL)
    G = np.stack([grads[p] for p in PARAMS_EST], axis=0)

    # FIM(TD)_ij = (1/σ²(TD)) × Σ_b G_i(b) × G_j(b)
    # Sum over b: einsum over last axis
    FIM = np.einsum('itb,jtb->ijt', G, G)   # (N_PARAMS, N_PARAMS, N_TD)
    FIM /= sigma2_td[np.newaxis, np.newaxis, :]   # divide by σ²(TD)

    # Marginal CRLB: only diagonal, ignore off-diagonal
    marginal = {}
    for k, param in enumerate(PARAMS_EST):
        with np.errstate(divide='ignore', invalid='ignore'):
            marginal[param] = np.where(FIM[k, k, :] > 1e-30,
                                       1.0 / FIM[k, k, :],
                                       np.nan)   # (N_TD,)

    # Joint CRLB: full FIM inversion per TD
    joint = {p: np.full(N_TD, np.nan) for p in PARAMS_EST}
    fim_norm = np.full((N_TD, N_PARAMS, N_PARAMS), np.nan)

    for ti in range(N_TD):
        F = FIM[:, :, ti]   # (N_PARAMS, N_PARAMS)
        if np.any(np.isnan(F)) or np.linalg.matrix_rank(F) < N_PARAMS:
            continue
        try:
            Finv = np.linalg.inv(F)
            for k, param in enumerate(PARAMS_EST):
                joint[param][ti] = Finv[k, k]
            # Normalised FIM: correlation matrix C_ij = FIM_ij / sqrt(FIM_ii * FIM_jj)
            diag = np.sqrt(np.diag(F))
            fim_norm[ti] = F / np.outer(diag, diag)
        except np.linalg.LinAlgError:
            pass

    return single_b, marginal, joint, FIM, fim_norm


# ── Helper: cell edges ─────────────────────────────────────────────────────────
def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    e = np.empty(len(c) + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0]  = c[0]  - (c[1]  - c[0])  / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e

b_edges  = cell_edges(bval_use)
td_edges = cell_edges(TD_arr)


# ── Main loop — compute per phenotype ─────────────────────────────────────────
print("Computing CRLB per phenotype...")
pheno_results = {}

for pheno in CATALOG:
    name   = pheno["name"]
    label  = pheno["label"]
    bounds = pheno["bounds"]

    center = np.array([
        (bounds["f"][0]     + bounds["f"][1])     / 2,
        (bounds["Dex"][0]   + bounds["Dex"][1])   / 2,
        (bounds["rmean"][0] + bounds["rmean"][1]) / 2,
        (bounds["rsd"][0]   + bounds["rsd"][1])   / 2,
    ])

    grads, ok = compute_gradients(center)
    if not ok:
        print(f"  {name}: outside hull — skipped")
        continue

    single_b, marginal, joint, FIM, fim_norm = compute_crlb(grads)
    pheno_results[name] = {
        "label":    label,
        "color":    pheno["color"],
        "center":   center,
        "single_b": single_b,
        "marginal": marginal,
        "joint":    joint,
        "FIM":      FIM,
        "fim_norm": fim_norm,
    }
    print(f"  {name}: done  (f={center[0]:.2f}, Dex={center[1]:.2f}, rmean={center[2]:.1f})")


# =============================================================================
# Fig 1 — Single-b CRLB heatmaps  (all phenotypes × all params)
# Displayed as sqrt(CRLB) = minimum std of estimator, same units as θ
# =============================================================================
print("\nGenerating Fig 1 — single-b CRLB heatmaps...")

n_pheno = len(pheno_results)
n_param = N_PARAMS

fig, axes = plt.subplots(n_param, n_pheno,
                         figsize=(2.8 * n_pheno, 2.6 * n_param),
                         squeeze=False)

for row, param in enumerate(PARAMS_EST):
    # Shared colour scale: clamp to 95th percentile to avoid extreme values
    all_vals = np.concatenate([
        np.sqrt(data["single_b"][param]).ravel()
        for data in pheno_results.values()
        if not np.all(np.isnan(data["single_b"][param]))
    ])
    vmax_row = np.nanpercentile(all_vals, 95)
    vmin_row = 0

    for col, (name, data) in enumerate(pheno_results.items()):
        ax   = axes[row, col]
        crlb = np.sqrt(data["single_b"][param])   # std units

        im = ax.pcolormesh(b_edges, td_edges, crlb,
                           cmap="YlOrRd_r", vmin=vmin_row, vmax=vmax_row)
        plt.colorbar(im, ax=ax, shrink=0.85)

        # Mark minimum (best precision)
        valid = ~np.isnan(crlb)
        if valid.any():
            best = np.unravel_index(np.nanargmin(crlb), crlb.shape)
            ax.plot(bval_use[best[1]], TD_arr[best[0]],
                    marker="*", color="cyan", ms=9, zorder=5)

        ax.set_xticks(bval_use)
        ax.set_xticklabels([f"{b:.2f}" for b in bval_use],
                           rotation=45, ha="right", fontsize=5)
        ax.set_yticks(TD_arr)
        ax.set_yticklabels([f"{t:.0f}" for t in TD_arr], fontsize=7)
        ax.set_ylabel("TD (ms)", fontsize=7)

        if row == 0:
            ax.set_title(data["label"], fontsize=9, fontweight="bold",
                         color=data["color"])
        if col == 0:
            ax.set_ylabel(f"{PARAM_LABELS[param]}\nTD (ms)", fontsize=7)
        if row == n_param - 1:
            ax.set_xlabel("b (ms/µm²)", fontsize=7)

        ax.text(0.02, 0.97, PARAM_LABELS[param],
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle(
    f"Single-b CRLB  (= min std of estimator) — colour: sqrt(CRLB)\n"
    f"cyan star = argmin per panel  |  T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f}\n"
    "Note: single-b CRLB optimal ≡ sensitivity optimal (1/effective sensitivity²)",
    fontsize=9, y=1.02)
fig.subplots_adjust(hspace=0.45, wspace=0.45)
fig.savefig(os.path.join(OUT_DIR, "fig_crlb_heatmap.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: fig_crlb_heatmap.pdf")


# =============================================================================
# Fig 2 — Multi-b CRLB vs TD  (marginal vs joint), normalised by param range
#
# To compare across parameters with incompatible units we express precision as
# a fraction of the biologically plausible range of each parameter:
#   rel_CRLB(θ) = sqrt(CRLB(θ)) / range(θ)    [dimensionless, 0–1]
# range: f → 0.70 (0.05–0.75),  Dex → 2.5 µm²/ms,  rmean → 17 µm (3–20)
#
# Three panels per phenotype:
#   (a) Relative marginal CRLB — each param estimated independently
#   (b) Relative joint CRLB    — simultaneous estimation (FIM⁻¹)
#   (c) Joint/marginal ratio   — cost of simultaneous estimation per param
# =============================================================================
print("Generating Fig 2 — multi-b CRLB vs TD (relative precision)...")

PARAM_RANGE  = {"f": 0.70, "Dex": 2.5, "rmean": 17.0}   # plausible bio range
PARAM_COLORS = {"f": "#e6194b", "Dex": "#3cb44b", "rmean": "#4363d8"}

fig, axes = plt.subplots(3, n_pheno, figsize=(4.2 * n_pheno, 10), squeeze=False)

for col, (name, data) in enumerate(pheno_results.items()):
    ax_marg  = axes[0, col]
    ax_joint = axes[1, col]
    ax_ratio = axes[2, col]

    for param in PARAMS_EST:
        c     = PARAM_COLORS[param]
        rng   = PARAM_RANGE[param]
        label = PARAM_LABELS[param]

        marg_rel  = np.sqrt(data["marginal"][param]) / rng
        joint_rel = np.sqrt(data["joint"][param])    / rng
        ratio     = joint_rel / marg_rel              # always ≥ 1

        ax_marg.plot(TD_arr, marg_rel,  color=c, marker="o", ms=5, label=label)
        ax_joint.plot(TD_arr, joint_rel, color=c, marker="s", ms=5, label=label)
        ax_ratio.plot(TD_arr, ratio,     color=c, marker="^", ms=5, label=label)

        # Mark optimal TD per parameter (marginal)
        if not np.all(np.isnan(marg_rel)):
            best_ti = np.nanargmin(marg_rel)
            for ax in [ax_marg, ax_joint]:
                ax.axvline(TD_arr[best_ti], color=c, alpha=0.2, lw=2)

    for ax in [ax_marg, ax_joint, ax_ratio]:
        ax.set_xticks(TD_arr)
        ax.set_xlabel("TD (ms)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    ax_marg.set_ylabel("sqrt(CRLB) / range(θ)\n[marginal — params indep.]", fontsize=7)
    ax_joint.set_ylabel("sqrt(CRLB) / range(θ)\n[joint — simultaneous est.]", fontsize=7)
    ax_ratio.set_ylabel("joint / marginal ratio\n[cost of correlations]", fontsize=7)
    ax_ratio.axhline(1, color="gray", ls=":", lw=1)
    ax_ratio.set_yscale("log")

    ax_marg.set_title(f"{data['label']}\n"
                      f"f={data['center'][0]:.2f}, Dex={data['center'][1]:.2f},"
                      f" rmean={data['center'][2]:.1f}",
                      fontsize=9, fontweight="bold", color=data["color"])

    if col == 0:
        axes[0, 0].text(-0.25, 0.5, "(a)", transform=axes[0,0].transAxes,
                        fontsize=11, fontweight="bold", va="center")
        axes[1, 0].text(-0.25, 0.5, "(b)", transform=axes[1,0].transAxes,
                        fontsize=11, fontweight="bold", va="center")
        axes[2, 0].text(-0.25, 0.5, "(c)", transform=axes[2,0].transAxes,
                        fontsize=11, fontweight="bold", va="center")

fig.suptitle(
    "Multi-b CRLB vs TD  —  relative precision = sqrt(CRLB) / param_range\n"
    "(a) Marginal: each param estimated independently\n"
    "(b) Joint: simultaneous estimation via FIM⁻¹\n"
    "(c) Joint/marginal ratio (log scale) — penalty from inter-param correlations\n"
    f"T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f}  |  vertical bars = optimal TD (marginal)",
    fontsize=9, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_crlb_vs_td.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: fig_crlb_vs_td.pdf")


# =============================================================================
# Fig 3 — FIM correlation matrix at each TD, per phenotype
# Shows inter-parameter identifiability: are f, Dex, rmean jointly estimable?
# =============================================================================
print("Generating Fig 3 — FIM correlation matrices...")

tick_labels = [r"$f$", r"$D_{ex}$", r"$r_{mean}$"]
fig, axes = plt.subplots(n_pheno, N_TD,
                         figsize=(2.0 * N_TD, 2.2 * n_pheno),
                         squeeze=False)

for row, (name, data) in enumerate(pheno_results.items()):
    for col, ti in enumerate(range(N_TD)):
        ax = axes[row, col]
        C  = data["fim_norm"][ti]   # (N_PARAMS, N_PARAMS)

        if np.any(np.isnan(C)):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            im = ax.imshow(np.abs(C), cmap="Reds", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(N_PARAMS)); ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_yticks(range(N_PARAMS)); ax.set_yticklabels(tick_labels, fontsize=7)
            for i in range(N_PARAMS):
                for j in range(N_PARAMS):
                    ax.text(j, i, f"{abs(C[i, j]):.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(C[i, j]) > 0.6 else "black")
            if row == 0:
                ax.set_title(f"TD={int(TD_arr[col])}ms", fontsize=8)
            if col == N_TD - 1:
                plt.colorbar(im, ax=ax, shrink=0.8)

        if col == 0:
            ax.set_ylabel(data["label"], fontsize=8, color=data["color"])

fig.suptitle(
    "FIM normalised correlation matrix  |C_ij| = FIM_ij / sqrt(FIM_ii · FIM_jj)\n"
    "Off-diagonal = parameter correlations: 1 → fully correlated (not jointly identifiable)\n"
    f"T₂={T2_MS:.0f} ms, SNR₀={SNR0:.0f}",
    fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_crlb_correlation.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: fig_crlb_correlation.pdf")

print(f"\nAll figures -> {os.path.abspath(OUT_DIR)}")
