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
# # Identifiability analysis — FIM + MCMC
#
# Checks whether the 3-parameter model (k_det shared, Dex_19, Dex_49) is
# identifiable given the DWI data.
#
# For each ROI in df_summary:
#   1. Fisher Information Matrix (FIM) -> eigenvalues, condition number
#   2. MCMC with emcee -> posterior distributions, pairwise correlations
#
# A well-identified model shows:
#   - All FIM eigenvalues >> 0 (no near-zero directions)
#   - Narrow, unimodal MCMC posteriors
#   - No strong pairwise correlations between parameters

# %% [markdown]
# # Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as _h5py
import os
import emcee
import corner
from functools import partial
from scipy.spatial import Delaunay
from scipy.io import loadmat

from lut_utils import get_signal_curve_lut, extract_patient_curves

# %% [markdown]
# # Load LUT

# %%
os.chdir(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

with _h5py.File('lookup_table_val.mat', 'r') as _f:
    params_lut  = np.array(_f['params']).T                      # (n_params, 4)
    _sig_raw    = np.array(_f['signals_4D'])                    # (64, 13, 2, n_params)
    signals_lut = np.transpose(_sig_raw, (3, 2, 1, 0))         # (n_params, 2, 13, 64)
    _seq   = _f['sequence']
    _bval  = np.array(_seq['bval']).flatten()                   # type: ignore[index]
    _TD    = np.array(_seq['TD']).flatten()                     # type: ignore[index]
    _bvecs = np.array(_seq['bvecs']).T                          # type: ignore[index]

sequence_lut = {
    'bval':  np.array([[_bval]]),
    'TD':    np.array([[_TD]]),
    'bvecs': np.array([[_bvecs]]),
}
params_lut = np.array(params_lut)
tri        = Delaunay(params_lut)

_get_signal = partial(get_signal_curve_lut,
                      tri=tri,
                      signals_lookup=signals_lut,
                      params_lut=params_lut,
                      sequence_lut=sequence_lut)

bval_lut_min = _bval.min()

print(f"LUT loaded: {params_lut.shape[0]} entries | "
      f"b-values: {_bval.tolist()} ms/um2 | TDs: {_TD.tolist()} ms")

# %% [markdown]
# # Load patient DWI data

# %%
data_p1 = loadmat("./Patient_1/Combined/CON_0101SP_12072016_tumor_all.mat")
data_p3 = loadmat("./Patient_3/Combined/CON_03_V01_tumor_all.mat")

df_p1 = extract_patient_curves(data_p1, slice_idx=36, center=(49, 49),
                                roi_size=3, roi_type='square')
df_p3 = extract_patient_curves(data_p3, slice_idx=31, center=(73, 28),
                                roi_size=3, roi_type='square')

# Rename b_value column to match Average_signal.py conventions (s/mm2 -> ms/um2)
for df in [df_p1, df_p3]:
    df["b_value"] = df["b_value"]   # already in s/mm2 from extract_patient_curves

dwi_map = {1: df_p1, 3: df_p3}

def _get_dwi_arrays(pat_id, TD, bval_max_um2=7.5):
    """Return (bvals_um2, S_meas, sigma) for a given patient and TD."""
    df    = dwi_map[pat_id]
    td_str = f"{TD}ms"
    df_td = df[df["TD"] == td_str].copy()
    df_td["bval_um2"] = df_td["b_value"] / 1000
    df_td = df_td[(df_td["bval_um2"] > 0) &
                  (df_td["bval_um2"] <= bval_max_um2) &
                  (df_td["bval_um2"] >= bval_lut_min)].sort_values("bval_um2")
    bv  = df_td["bval_um2"].to_numpy()
    sm  = df_td["signal"].to_numpy()
    sig = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6,
                   df_td["noise"].to_numpy())
    return bv, sm, sig

# %% [markdown]
# # Load df_summary

# %%
df_summary = pd.read_csv("df_summary.csv")
print(f"df_summary loaded: {len(df_summary)} ROIs")
print(df_summary.to_string(index=False))

# %% [markdown]
# # Signal model
#
# For a given ROI, the predicted signal at b-values bv and diffusion time TD is:
#   S_pred(b, TD | k_det, Dex_TD) = LUT(f_vivo * k_det, Dex_TD, r3D, r3D_std, TD)
# interpolated at the DWI b-values.

# %%
def predict_signal(k_det, Dex, f_vivo, r3D, r3D_std, TD, bv_target):
    """Return predicted S/S0 at bv_target (ms/um2) for given parameters."""
    f_eff = f_vivo * k_det
    bv_c, sc = _get_signal(f_eff, Dex, r3D, r3D_std, TD)
    log_sc   = np.log(np.clip(sc, 1e-10, None))
    return np.exp(np.interp(bv_target, bv_c, log_sc))


def chi2_total(k_det, Dex_19, Dex_49,
               f_vivo, r3D, r3D_std,
               bv_19, S_19, sigma_19,
               bv_49, S_49, sigma_49):
    """Total chi2 across both TDs."""
    sp_19 = predict_signal(k_det, Dex_19, f_vivo, r3D, r3D_std, 19, bv_19)
    sp_49 = predict_signal(k_det, Dex_49, f_vivo, r3D, r3D_std, 49, bv_49)
    c19   = np.sum(((S_19 - sp_19) / sigma_19) ** 2)
    c49   = np.sum(((S_49 - sp_49) / sigma_49) ** 2)
    return c19 + c49

# %% [markdown]
# # Fisher Information Matrix (FIM)
#
# FIM = J^T Sigma^{-1} J  where J is the Jacobian of S w.r.t. (k_det, Dex_19, Dex_49).
# Computed numerically by finite differences.
# Near-zero eigenvalues -> non-identifiable directions.

# %%
def compute_fim(k_det, Dex_19, Dex_49,
                f_vivo, r3D, r3D_std,
                bv_19, sigma_19,
                bv_49, sigma_49,
                eps_rel=1e-4):
    """
    Numerical FIM for parameters theta = (k_det, Dex_19, Dex_49).

    Returns
    -------
    FIM       : (3, 3) array
    eigvals   : sorted eigenvalues (ascending)
    cond      : condition number = max/min eigenvalue
    """
    theta = np.array([k_det, Dex_19, Dex_49])
    eps   = np.abs(theta) * eps_rel + 1e-6

    # S concatenated over both TDs: shape (n_19 + n_49,)
    # sigma concatenated: shape (n_19 + n_49,)
    sigma_all = np.concatenate([sigma_19, sigma_49])

    def S_all(th):
        k, D19, D49 = th
        sp19 = predict_signal(k, D19, f_vivo, r3D, r3D_std, 19, bv_19)
        sp49 = predict_signal(k, D49, f_vivo, r3D, r3D_std, 49, bv_49)
        return np.concatenate([sp19, sp49])

    S0 = S_all(theta)

    # Jacobian: (n_pts, 3)
    J = np.zeros((len(S0), 3))
    for i in range(3):
        dth       = np.zeros(3)
        dth[i]    = eps[i]
        S_fwd     = S_all(theta + dth)
        S_bwd     = S_all(theta - dth)
        J[:, i]   = (S_fwd - S_bwd) / (2 * eps[i])

    # FIM = J^T Sigma^{-2} J  (diagonal noise)
    w   = 1.0 / sigma_all**2
    FIM = J.T @ (w[:, None] * J)

    eigvals = np.linalg.eigvalsh(FIM)
    cond    = float(eigvals[-1] / max(eigvals[0], 1e-30))
    return FIM, eigvals, cond

# %% [markdown]
# # MCMC log-probability

# %%
# Parameter bounds for uniform priors
_BOUNDS = {
    "k_det":  (0.05, 1.0),
    "Dex_19": (0.5,  3.5),
    "Dex_49": (0.5,  3.5),
}

def log_prior(theta):
    k, D19, D49 = theta
    if (_BOUNDS["k_det"][0]  <= k   <= _BOUNDS["k_det"][1]  and
        _BOUNDS["Dex_19"][0] <= D19 <= _BOUNDS["Dex_19"][1] and
        _BOUNDS["Dex_49"][0] <= D49 <= _BOUNDS["Dex_49"][1]):
        return 0.0
    return -np.inf


def log_likelihood(theta, f_vivo, r3D, r3D_std,
                   bv_19, S_19, sigma_19,
                   bv_49, S_49, sigma_49):
    k, D19, D49 = theta
    try:
        c = chi2_total(k, D19, D49, f_vivo, r3D, r3D_std,
                       bv_19, S_19, sigma_19,
                       bv_49, S_49, sigma_49)
    except Exception:
        return -np.inf
    if not np.isfinite(c):
        return -np.inf
    return -0.5 * c


def log_prob(theta, f_vivo, r3D, r3D_std,
             bv_19, S_19, sigma_19,
             bv_49, S_49, sigma_49):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, f_vivo, r3D, r3D_std,
                                bv_19, S_19, sigma_19,
                                bv_49, S_49, sigma_49)

# %% [markdown]
# # Run FIM + MCMC for all ROIs

# %%
N_WALKERS  = 32
N_STEPS    = 3000
N_BURNIN   = 500
PARAM_NAMES = ["k_det", "Dex_19", "Dex_49"]

fim_rows  = []
mcmc_rows = []

for _, row in df_summary.iterrows():
    pat_id   = int(row["Patient"])
    he       = int(row["H&E"])
    roi_name = str(row["ROI"])
    f_vivo   = float(row["f_vivo"])
    r3D      = float(row["r3D_vivo (µm)"])
    r3D_std  = float(row["r3D_std_vivo"]) if "r3D_std_vivo" in row else 0.5
    k0       = float(row["k_det_best"])
    D19_0    = float(row["Dex_19_best"])
    D49_0    = float(row["Dex_49_best"])

    label = f"P{pat_id}|H&E{he}|{roi_name}"
    print(f"\n{'='*60}")
    print(f"  {label}")

    try:
        bv_19, S_19, sigma_19 = _get_dwi_arrays(pat_id, 19)
        bv_49, S_49, sigma_49 = _get_dwi_arrays(pat_id, 49)
    except Exception as e:
        print(f"  [skip] DWI data error: {e}")
        continue

    # ── FIM ──────────────────────────────────────────────────────────────
    FIM, eigvals, cond = compute_fim(
        k0, D19_0, D49_0,
        f_vivo, r3D, r3D_std,
        bv_19, sigma_19,
        bv_49, sigma_49,
    )
    print(f"  FIM eigenvalues: {eigvals}")
    print(f"  Condition number: {cond:.1f}")
    identif = ("well-identified" if cond < 1e4
               else "poorly-identified" if cond > 1e6
               else "marginal")
    print(f"  -> {identif}")

    fim_rows.append({
        "label":    label,
        "Patient":  pat_id,
        "H&E":      he,
        "ROI":      roi_name,
        "eig_min":  float(eigvals[0]),
        "eig_mid":  float(eigvals[1]),
        "eig_max":  float(eigvals[2]),
        "cond":     float(cond),
        "status":   identif,
    })

    # ── MCMC ─────────────────────────────────────────────────────────────
    theta0 = np.array([k0, D19_0, D49_0])
    # Initial positions: small ball around the best-fit
    spread = np.array([0.05, 0.05, 0.05])
    pos    = theta0 + spread * np.random.randn(N_WALKERS, 3)
    # Clip to prior bounds
    for i, (lo, hi) in enumerate([_BOUNDS["k_det"],
                                   _BOUNDS["Dex_19"],
                                   _BOUNDS["Dex_49"]]):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-4, hi - 1e-4)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, 3, log_prob,
        args=(f_vivo, r3D, r3D_std,
              bv_19, S_19, sigma_19,
              bv_49, S_49, sigma_49),
    )

    print(f"  Running MCMC ({N_WALKERS} walkers × {N_STEPS} steps) ...")
    sampler.run_mcmc(pos, N_STEPS, progress=True)

    # Flatten chain after burn-in
    flat_samples = sampler.get_chain(discard=N_BURNIN, thin=10, flat=True)
    print(f"  Posterior samples: {flat_samples.shape[0]}")

    # Acceptance fraction
    acc = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction: {acc:.3f}"
          + ("  OK" if 0.2 < acc < 0.5 else "  <- check (ideal: 0.2–0.5)"))

    # Marginal statistics
    for i, name in enumerate(PARAM_NAMES):
        med  = float(np.percentile(flat_samples[:, i], 50))
        lo   = float(np.percentile(flat_samples[:, i], 2.5))
        hi   = float(np.percentile(flat_samples[:, i], 97.5))
        print(f"    {name}: {med:.3f}  [{lo:.3f}, {hi:.3f}]")
        mcmc_rows.append({
            "label":   label,
            "Patient": pat_id,
            "H&E":     he,
            "ROI":     roi_name,
            "param":   name,
            "median":  med,
            "p2.5":    lo,
            "p97.5":   hi,
            "width_95": round(hi - lo, 4),
        })

    # ── Pairwise correlations ────────────────────────────────────────────
    corr_mat = np.corrcoef(flat_samples.T)   # (3, 3)
    print("  Pairwise correlations (posterior samples):")
    for i in range(3):
        for j in range(i+1, 3):
            r = corr_mat[i, j]
            strength = ("strong" if abs(r) > 0.7
                        else "moderate" if abs(r) > 0.4
                        else "weak")
            print(f"    {PARAM_NAMES[i]} vs {PARAM_NAMES[j]}: r={r:+.3f}  ({strength})")
    fim_rows[-1]["r_kdet_D19"] = round(float(corr_mat[0, 1]), 3)
    fim_rows[-1]["r_kdet_D49"] = round(float(corr_mat[0, 2]), 3)
    fim_rows[-1]["r_D19_D49"]  = round(float(corr_mat[1, 2]), 3)

    # ── Corner plot ──────────────────────────────────────────────────────
    fig = corner.corner(
        flat_samples,
        labels=PARAM_NAMES,
        truths=[k0, D19_0, D49_0],
        truth_color="red",
        quantiles=[0.025, 0.5, 0.975],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
    )
    fig.suptitle(
        f"{label}  |  cond={cond:.0f}  ({identif})\n"
        f"r(k_det,D19)={corr_mat[0,1]:+.2f}  "
        f"r(k_det,D49)={corr_mat[0,2]:+.2f}  "
        f"r(D19,D49)={corr_mat[1,2]:+.2f}",
        fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Summary tables

# %%
df_fim  = pd.DataFrame(fim_rows)
df_mcmc = pd.DataFrame(mcmc_rows)

print("\n" + "="*70)
print("FIM SUMMARY")
print("="*70)
cols = ["label", "eig_min", "eig_mid", "eig_max", "cond", "status"]
if "r_kdet_D19" in df_fim.columns:
    cols += ["r_kdet_D19", "r_kdet_D49", "r_D19_D49"]
print(df_fim[cols].to_string(index=False))

print("\n" + "="*70)
print("MCMC SUMMARY — 95% CI width per parameter")
print("="*70)
pivot = df_mcmc.pivot_table(index="label", columns="param",
                             values="width_95", aggfunc="first")
print(pivot.to_string())

# %% [markdown]
# # FIM condition number — bar chart

# %%
fig, ax = plt.subplots(figsize=(max(8, len(df_fim) * 0.8), 5))
colors = ["tab:green" if c < 1e4 else "tab:orange" if c < 1e6 else "tab:red"
          for c in df_fim["cond"]]
bars = ax.bar(df_fim["label"], np.log10(df_fim["cond"]),
              color=colors, alpha=0.8, edgecolor="white")
ax.axhline(4, color="orange", ls="--", lw=1.5, label="Marginal (1e4)")
ax.axhline(6, color="red",    ls="--", lw=1.5, label="Poor (1e6)")
ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
ax.set_ylabel("log₁₀(condition number)")
ax.set_title("FIM condition number per ROI\n"
             "Green < 1e4 (well-identified) | Orange 1e4–1e6 | Red > 1e6 (poor)")
ax.tick_params(axis='x', rotation=35)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# # MCMC CI widths — bar chart per parameter

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for ax, param in zip(axes, PARAM_NAMES):
    sub = df_mcmc[df_mcmc["param"] == param].sort_values("label")
    colors = [f"C{i}" for i in range(len(sub))]
    bars = ax.bar(sub["label"], sub["width_95"],
                  color=colors, alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel("95% CI width")
    ax.set_title(f"{param} — posterior width")
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle("MCMC posterior 95% CI width per ROI\n"
             "Narrow CI -> well-identified | Wide CI -> poorly-identified",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # Part 2 — Full LUT inverse problem: identifiability of (f, Dex, rmean, rsd)
#
# This section answers the question raised by the supervisor:
# "When the only input is a DWI signal curve, can we recover the biological
# parameters (f, Dex, rmean, rsd) uniquely?"
#
# This is the full inverse problem of the LUT, without any histological
# constraint. f, rmean, and rsd are treated as unknown (not fixed from
# QuPath), and a single TD is used (19 ms) to match typical clinical
# acquisition.
#
# Reference point: per-patient consensus values from df_summary are used
# as the "true" microstructure to generate synthetic DWI, then FIM and
# MCMC explore whether the 4 parameters can be recovered from that signal.

# %% [markdown]
# ## Signal model — 4 free parameters

# %%
PARAM_NAMES_4 = ["f", "Dex", "rmean", "rsd"]

# Prior bounds for the full inverse problem
_BOUNDS_4 = {
    "f":     (0.01, 0.99),
    "Dex":   (0.5,  3.5),
    "rmean": (1.0,  20.0),
    "rsd":   (0.0,  5.0),
}


def predict_signal_4(f, Dex, rmean, rsd, TD, bv_target):
    """Predicted S/S0 at bv_target for uncorrected (f, Dex, rmean, rsd)."""
    bv_c, sc = _get_signal(f, Dex, rmean, rsd, TD)
    log_sc   = np.log(np.clip(sc, 1e-10, None))
    return np.exp(np.interp(bv_target, bv_c, log_sc))


def log_prior_4(theta):
    f, Dex, rmean, rsd = theta
    if (_BOUNDS_4["f"][0]     <= f     <= _BOUNDS_4["f"][1]     and
        _BOUNDS_4["Dex"][0]   <= Dex   <= _BOUNDS_4["Dex"][1]   and
        _BOUNDS_4["rmean"][0] <= rmean <= _BOUNDS_4["rmean"][1]  and
        _BOUNDS_4["rsd"][0]   <= rsd   <= _BOUNDS_4["rsd"][1]):
        return 0.0
    return -np.inf


def log_prob_4(theta, bv, S_meas, sigma, TD):
    lp = log_prior_4(theta)
    if not np.isfinite(lp):
        return -np.inf
    f, Dex, rmean, rsd = theta
    try:
        S_pred = predict_signal_4(f, Dex, rmean, rsd, TD, bv)
    except Exception:
        return -np.inf
    if not np.isfinite(S_pred).all():
        return -np.inf
    return lp - 0.5 * np.sum(((S_meas - S_pred) / sigma) ** 2)


def compute_fim_4(f, Dex, rmean, rsd, TD, bv, sigma, eps_rel=1e-3):
    """Numerical FIM for theta = (f, Dex, rmean, rsd) at a single TD."""
    theta = np.array([f, Dex, rmean, rsd])
    eps   = np.abs(theta) * eps_rel + 1e-6

    def S(th):
        return predict_signal_4(th[0], th[1], th[2], th[3], TD, bv)

    S0 = S(theta)
    J  = np.zeros((len(S0), 4))
    for i in range(4):
        dth    = np.zeros(4)
        dth[i] = eps[i]
        J[:, i] = (S(theta + dth) - S(theta - dth)) / (2 * eps[i])

    w   = 1.0 / sigma**2
    FIM = J.T @ (w[:, None] * J)
    eigvals = np.linalg.eigvalsh(FIM)
    cond    = float(eigvals[-1] / max(eigvals[0], 1e-30))
    return FIM, eigvals, cond

# %% [markdown]
# ## Run FIM + MCMC — full inverse problem
#
# One analysis per patient (using consensus microstructure as reference point).
# The synthetic DWI signal is generated from the consensus parameters and
# used as the "measured" signal — this isolates identifiability from data
# quality issues.

# %%
N_WALKERS_4 = 32
N_STEPS_4   = 3000
N_BURNIN_4  = 500

fim_rows_4  = []
mcmc_rows_4 = []

for pat_id in [1, 3]:
    sub = df_summary[df_summary["Patient"] == pat_id]
    if sub.empty:
        continue

    w      = 1.0 / sub["chi2_min"].replace(0, np.nan).dropna()
    sub_w  = sub.loc[w.index]
    w_arr  = w.to_numpy()

    # Consensus microstructure (reference point)
    f0     = float(np.average(sub_w["f_eff"],          weights=w_arr))
    D19_0  = float(np.average(sub_w["Dex_19_best"],    weights=w_arr))
    r0     = float(np.average(sub_w["r3D_vivo (µm)"],  weights=w_arr))
    rsd0   = float(np.average(sub_w["r3D_std_vivo"],   weights=w_arr)
                   if "r3D_std_vivo" in sub_w.columns else 0.5)

    label4 = f"Patient_{pat_id} (consensus)"
    print(f"\n{'='*60}")
    print(f"  {label4}")
    print(f"  Reference: f={f0:.3f}, Dex={D19_0:.3f}, rmean={r0:.2f}, rsd={rsd0:.2f}")

    # Generate synthetic DWI from consensus + real noise level
    bv_19, S_real, sigma_19 = _get_dwi_arrays(pat_id, 19)
    S_synth = predict_signal_4(f0, D19_0, r0, rsd0, 19, bv_19)
    # Use real noise sigma so the identifiability reflects actual SNR
    sigma_use = sigma_19

    # ── FIM ──────────────────────────────────────────────────────────────
    FIM4, eigvals4, cond4 = compute_fim_4(
        f0, D19_0, r0, rsd0, 19, bv_19, sigma_use)
    print(f"  FIM eigenvalues: {np.round(eigvals4, 2)}")
    print(f"  Condition number: {cond4:.1f}")
    identif4 = ("well-identified" if cond4 < 1e4
                else "poorly-identified" if cond4 > 1e6
                else "marginal")
    print(f"  -> {identif4}")

    fim_rows_4.append({
        "label":   label4,
        "Patient": pat_id,
        "eig_min": float(eigvals4[0]),
        "eig_mid1":float(eigvals4[1]),
        "eig_mid2":float(eigvals4[2]),
        "eig_max": float(eigvals4[3]),
        "cond":    float(cond4),
        "status":  identif4,
    })

    # ── MCMC ─────────────────────────────────────────────────────────────
    theta0_4 = np.array([f0, D19_0, r0, rsd0])
    spread4  = np.array([0.02, 0.05, 0.5, 0.2])
    pos4     = theta0_4 + spread4 * np.random.randn(N_WALKERS_4, 4)
    for i, (lo, hi) in enumerate([_BOUNDS_4["f"],    _BOUNDS_4["Dex"],
                                   _BOUNDS_4["rmean"], _BOUNDS_4["rsd"]]):
        pos4[:, i] = np.clip(pos4[:, i], lo + 1e-4, hi - 1e-4)

    sampler4 = emcee.EnsembleSampler(
        N_WALKERS_4, 4, log_prob_4,
        args=(bv_19, S_synth, sigma_use, 19),
    )
    print(f"  Running MCMC ({N_WALKERS_4} walkers × {N_STEPS_4} steps) ...")
    sampler4.run_mcmc(pos4, N_STEPS_4, progress=True)

    flat4 = sampler4.get_chain(discard=N_BURNIN_4, thin=10, flat=True)
    acc4  = np.mean(sampler4.acceptance_fraction)
    print(f"  Acceptance fraction: {acc4:.3f}")

    # Marginal statistics
    for i, name in enumerate(PARAM_NAMES_4):
        med = float(np.percentile(flat4[:, i], 50))
        lo  = float(np.percentile(flat4[:, i], 2.5))
        hi  = float(np.percentile(flat4[:, i], 97.5))
        print(f"    {name}: {med:.3f}  [{lo:.3f}, {hi:.3f}]")
        mcmc_rows_4.append({
            "label":    label4,
            "Patient":  pat_id,
            "param":    name,
            "median":   med,
            "p2.5":     lo,
            "p97.5":    hi,
            "width_95": round(hi - lo, 4),
            "truth":    round(theta0_4[i], 4),
        })

    # Pairwise correlations
    corr4 = np.corrcoef(flat4.T)
    print("  Pairwise correlations:")
    for i in range(4):
        for j in range(i+1, 4):
            r = corr4[i, j]
            strength = ("strong" if abs(r) > 0.7
                        else "moderate" if abs(r) > 0.4
                        else "weak")
            print(f"    {PARAM_NAMES_4[i]} vs {PARAM_NAMES_4[j]}: "
                  f"r={r:+.3f}  ({strength})")

    # Corner plot
    fig4 = corner.corner(
        flat4,
        labels=PARAM_NAMES_4,
        truths=theta0_4.tolist(),
        truth_color="red",
        quantiles=[0.025, 0.5, 0.975],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 9},
    )
    corr_str = (f"r(f,Dex)={corr4[0,1]:+.2f}  r(f,r)={corr4[0,2]:+.2f}  "
                f"r(Dex,r)={corr4[1,2]:+.2f}  r(f,rsd)={corr4[0,3]:+.2f}")
    fig4.suptitle(
        f"{label4}  |  cond={cond4:.0f}  ({identif4})\n{corr_str}",
        fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary — full inverse problem

# %%
df_fim_4  = pd.DataFrame(fim_rows_4)
df_mcmc_4 = pd.DataFrame(mcmc_rows_4)

print("\n" + "="*70)
print("FIM SUMMARY — full inverse problem (f, Dex, rmean, rsd)")
print("="*70)
print(df_fim_4[["label", "eig_min", "eig_mid1", "eig_mid2",
                "eig_max", "cond", "status"]].to_string(index=False))

print("\n" + "="*70)
print("MCMC SUMMARY — 95% CI width  (red = truth value)")
print("="*70)
for _, row in df_mcmc_4.iterrows():
    pct = (row["width_95"] /
           (_BOUNDS_4[row["param"]][1] - _BOUNDS_4[row["param"]][0])) * 100
    print(f"  {row['label']} | {row['param']:6s}: "
          f"truth={row['truth']:.3f}  "
          f"median={row['median']:.3f}  "
          f"CI=[{row['p2.5']:.3f}, {row['p97.5']:.3f}]  "
          f"width={row['width_95']:.4f} ({pct:.1f}% of prior)")

# %% [markdown]
# ## Comparison: 3-param vs 4-param identifiability

# %%
print("\n" + "="*70)
print("IDENTIFIABILITY COMPARISON")
print("="*70)
print("\nPart 1 — Inference problem (k_det, Dex_19, Dex_49 | histology fixed):")
print("  -> Parameters fixed from histology: f_vivo, rmean, rsd")
print("  -> Parameters inferred: k_det (detection correction), Dex_19, Dex_49")
if len(df_fim) > 0:
    print(f"  FIM condition numbers: {df_fim['cond'].min():.0f} – {df_fim['cond'].max():.0f}")
    print(f"  MCMC CI widths:  k_det {df_mcmc[df_mcmc['param']=='k_det']['width_95'].mean():.3f} "
          f"| Dex_19 {df_mcmc[df_mcmc['param']=='Dex_19']['width_95'].mean():.3f} "
          f"| Dex_49 {df_mcmc[df_mcmc['param']=='Dex_49']['width_95'].mean():.3f} (mean)")

print("\nPart 2 — Full inverse problem (f, Dex, rmean, rsd | DWI signal only):")
print("  -> Parameters fixed: none")
print("  -> Parameters inferred: all 4 biological parameters")
if len(df_fim_4) > 0:
    print(f"  FIM condition numbers: {df_fim_4['cond'].min():.0f} – {df_fim_4['cond'].max():.0f}")
    for param in PARAM_NAMES_4:
        sub = df_mcmc_4[df_mcmc_4["param"] == param]
        pct = (sub["width_95"].mean() /
               (_BOUNDS_4[param][1] - _BOUNDS_4[param][0])) * 100
        print(f"  MCMC CI width  {param:6s}: {sub['width_95'].mean():.3f} "
              f"({pct:.1f}% of prior range)")

# %% [markdown]
# ## Save results

# %%
df_fim.to_csv("identifiability_fim_part1.csv", index=False)
df_mcmc.to_csv("identifiability_mcmc_part1.csv", index=False)
df_fim_4.to_csv("identifiability_fim_part2.csv", index=False)
df_mcmc_4.to_csv("identifiability_mcmc_part2.csv", index=False)
print("\nResults saved:")
print("  identifiability_fim_part1.csv  — FIM (k_det, Dex_19, Dex_49)")
print("  identifiability_mcmc_part1.csv — MCMC posteriors Part 1")
print("  identifiability_fim_part2.csv  — FIM (f, Dex, rmean, rsd)")
print("  identifiability_mcmc_part2.csv — MCMC posteriors Part 2")
