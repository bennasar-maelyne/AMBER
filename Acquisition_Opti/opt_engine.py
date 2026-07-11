"""
Optimization engine for DWI acquisition parameters.

For each pair of tumor phenotypes (from tumor_catalog.py), computes
CNR / Cohen's d / AUC on the full (b-value, TD) grid of lookup_table_optim.mat,
using Latin Hypercube Sampling to cover each phenotype's parameter space.

Usage:
    python opt_engine.py

Output:
    Acquisition_Opti/results/opt_results.npz
    Contains per-pair metrics on the (N_TD × N_BVAL) grid + sampled signals.
"""

import os
import itertools
import numpy as np
import scipy.io as sio
from scipy.spatial import Delaunay
from scipy.stats.qmc import LatinHypercube

from tumor_catalog import CATALOG

# ── Config — Level 1 (diffusion only) ─────────────────────────────────────────
N_SAMPLES = 500       # LHS samples per phenotype
PARAMS    = ["f", "Dex", "rmean", "rsd"]   # column order in LUT params

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config — Level 2 (T2 attenuation + thermal noise) ─────────────────────────
# Set ENABLE_T2_SNR = True to compute SNR-corrected metrics alongside Level 1.
#
# Physical model:
#   S_measured(b, TD) = S_LUT(b, TD) × exp(−TE / T2)   with TE = TD + delta
#   Noise floor: σ_noise = 1 / SNR0  (normalised to S0 at b=0)
#
# T2_MS calibration
#   The validation acquisitions use a fixed TE (independent of TD): the ratio
#   S0(TD=19 ms) / S0(TD=49 ms) ≈ 1.00–1.01 across both patients and both
#   b-shells, whereas T2=100 ms would predict a ratio of 1.35.  T2 cannot
#   therefore be estimated from these data.  T2=100 ms is a round literature
#   value for glioma at 3T — VERIFY and add proper citation before publication.
#   Level 2 models the scenario where the recommended protocol is implemented
#   with a standard PGSE sequence (TE = TD + delta), i.e. longer TD incurs a
#   real T2 penalty.  If a STEAM sequence is used (TE << TD), Level 1 applies.
#
# SNR0 calibration (measured from validation SNR maps, tumour mask)
#   Computed over both patients (P1, P3), both protocols (qb32, qb64) and both
#   TDs (D19, D49), weighted by number of directions (32 or 64):
#     P1 qb32 D19=47.7 | P1 qb32 D49=47.8 | P1 qb64 D19=54.4 | P1 qb64 D49=59.7
#     P3 qb32 D19=36.9 | P3 qb32 D49=36.0 | P3 qb64 D19=41.1 | P3 qb64 D49=46.6
#   Weighted mean (by N_dir): P1=53.9 | P3=41.4 | global=47.7
#   Range: 36–60  |  Weighted mean: 47.7  |  Conservative (25th pct): ~39
ENABLE_T2_SNR = True
T2_MS         = 100.0   # ms — literature value, glioma at 3T (VERIFY: Gu 2021 HGG=127ms, LGG=164ms)
SNR0          = 40.0    # conservative (25th percentile over both patients, both TDs, both protocols)


# ── LUT loading ────────────────────────────────────────────────────────────────

def load_lut(lut_path):
    """
    Load lookup_table_optim.mat and return normalised signals + acquisition grid.

    Returns
    -------
    params_lut  : (N_lut, 4)       — [f, Dex, rmean, rsd]
    sig_norm    : (N_lut, N_TD, N_BVAL)  — S/S0, direction-averaged
    bval        : (N_BVAL,)        — ms/µm²
    TD          : (N_TD,)          — ms
    """
    lut = sio.loadmat(lut_path, squeeze_me=True)
    params_lut  = lut["params"]          # (N, 4)
    signals_4D  = lut["signals_4D"]      # (N, N_TD, N_BVAL, N_DIR) — raw counts
    seq         = lut["sequence"]

    # scipy.io returns struct fields as numpy record arrays when squeeze_me=True
    bval  = np.asarray(seq["bval"].item()).flatten()
    TD    = np.asarray(seq["TD"].item()).flatten()
    delta = float(np.asarray(seq["delta"].item()).flatten()[0])   # gradient pulse width (ms)

    # Normalise: direction average, then divide by S0 (b=0)
    sig_avg  = signals_4D.mean(axis=-1)           # (N, N_TD, N_BVAL)
    b0_idx   = int(np.where(bval == 0)[0][0])
    S0       = sig_avg[:, :, b0_idx]              # (N, N_TD)
    sig_norm = sig_avg / S0[:, :, np.newaxis]     # (N, N_TD, N_BVAL)

    return params_lut, sig_norm, bval, TD, delta


# ── Delaunay interpolation ─────────────────────────────────────────────────────

def build_tri(params_lut):
    return Delaunay(params_lut)


def interpolate_batch(params_samples, sig_norm_lut, tri, f_min=0.007):
    """
    Interpolate signals for a batch of parameter samples on the full (TD, bval) grid.

    Parameters
    ----------
    params_samples : (N, 4) — [f, Dex, rmean, rsd]
    sig_norm_lut   : (N_lut, N_TD, N_BVAL)
    tri            : Delaunay triangulation of params_lut

    Returns
    -------
    sig_out : (N, N_TD, N_BVAL)  — NaN for points outside the convex hull
    """
    N       = params_samples.shape[0]
    N_TD, N_BVAL = sig_norm_lut.shape[1], sig_norm_lut.shape[2]
    sig_out = np.full((N, N_TD, N_BVAL), np.nan)

    # Empty voxels (f very low) → pure free diffusion, handled externally
    mask_empty = params_samples[:, 0] < f_min
    # (left as NaN; opt_engine will drop them from metrics)

    # Find simplices for all non-empty points at once
    pts    = params_samples[~mask_empty]
    s_idx  = tri.find_simplex(pts)

    inside = s_idx != -1
    n_out  = int((~inside).sum())
    if n_out > 0:
        print(f"  [WARNING] {n_out}/{N} samples outside LUT convex hull — set to NaN")

    # Barycentric interpolation for each inside point
    global_idx = np.where(~mask_empty)[0]
    for k, (g_idx, si) in enumerate(zip(global_idx, s_idx)):
        if si == -1:
            continue
        verts = tri.simplices[si]          # (n_verts,) = 5 for 4D
        coords = tri.points[verts]         # (5, 4)
        v = pts[k] - coords[0]
        T = coords[1:] - coords[0]        # (4, 4)
        try:
            bc_tail = np.linalg.solve(T.T, v)
        except np.linalg.LinAlgError:
            bc_tail, *_ = np.linalg.lstsq(T.T, v, rcond=None)
        bc    = np.empty(verts.shape[0])
        bc[1:] = bc_tail
        bc[0]  = 1.0 - bc_tail.sum()

        # Weighted sum of LUT signals at vertices: (N_TD, N_BVAL)
        sig_out[g_idx] = np.tensordot(bc, sig_norm_lut[verts], axes=(0, 0))

    return sig_out


# ── Latin Hypercube Sampling ───────────────────────────────────────────────────

def sample_phenotype(phenotype, n=N_SAMPLES, seed=42):
    """
    Draw n LHS samples from a phenotype's parameter bounds.

    Returns
    -------
    samples : (n, 4)  — [f, Dex, rmean, rsd]
    """
    bounds = phenotype["bounds"]
    lows   = np.array([bounds[p][0] for p in PARAMS])
    highs  = np.array([bounds[p][1] for p in PARAMS])

    sampler = LatinHypercube(d=4, seed=seed)
    unit    = sampler.random(n=n)                  # (n, 4) in [0, 1]
    return lows + unit * (highs - lows)


# ── Discrimination metrics ─────────────────────────────────────────────────────

def discrimination_metrics(sigs_A, sigs_B):
    """
    Compute CNR at every (TD, bval) grid point.

    Parameters
    ----------
    sigs_A, sigs_B : (N_samples, N_TD, N_BVAL) — NaN-free expected

    Returns
    -------
    CNR : (N_TD, N_BVAL)
    """
    N_TD, N_BVAL = sigs_A.shape[1], sigs_A.shape[2]
    CNR = np.full((N_TD, N_BVAL), np.nan)

    for ti in range(N_TD):
        for bi in range(N_BVAL):
            a = sigs_A[:, ti, bi]
            b = sigs_B[:, ti, bi]
            if np.isnan(a).any() or np.isnan(b).any():
                continue

            mu_a, mu_b = a.mean(), b.mean()
            std_a, std_b = a.std(), b.std()

            denom = np.sqrt(std_a**2 + std_b**2 + 1e-12)
            if denom < 1e-10:
                continue

            CNR[ti, bi] = abs(mu_a - mu_b) / denom

    return CNR


# ── Level 2: T2 + SNR correction ──────────────────────────────────────────────

def discrimination_metrics_snr(sigs_A, sigs_B, TD_arr, delta_ms, t2_ms, snr0):
    """
    Level 2: CNR / Cohen's d / AUC corrected for T2 attenuation and thermal noise.

    Physical model
    --------------
    LUT signals are S(b,TD)/S0(b=0,TD): pure diffusion ratios, T2-independent
    (T2 cancels in the ratio since numerator and denominator share the same TE).

    Thermal noise σ_thermal is constant in absolute MRI units.  Relative to
    S0(TD) ∝ exp(−TE/T2), it grows with TD:

        σ_noise(TD) = (1/snr0) × exp((TE(TD) − TE_ref) / T2)
                      with TE = TD + delta,  TE_ref = TD_arr[0] + delta

    Biological variance σ_bio is T2-independent (it reflects microstructural
    spread, not T2).  Total variance:

        σ_total_X(TD)² = σ_bio_X² + σ_noise(TD)²

    This is algebraically equivalent to the "T2 on signal" formulation
    S_meas = S_LUT × exp(−TE/T2) with σ_noise_abs = exp(−TE_ref/T2)/snr0,
    but more natural given the per-TD normalisation of the LUT.

    Two usage modes (controlled externally via t2_ms):
      - Single T2: one value for all phenotypes → isolates the TE penalty
      - Per-pathology T2: pass the phenotype-specific T2 to account for
        differences in T2 between tissue types

    Parameters
    ----------
    sigs_A, sigs_B : (N_samples, N_TD, N_BVAL)  — Level 1 signals (S/S0)
    TD_arr         : (N_TD,)                     — diffusion times (ms)
    delta_ms       : float                       — gradient pulse width (ms)
    t2_ms          : float                       — tissue T2 (ms); single value
                                                   or the mean T2 for this pair
    snr0           : float                       — SNR at b=0, shortest TD

    Returns
    -------
    CNR : (N_TD, N_BVAL)
    """
    N_TD, N_BVAL = sigs_A.shape[1], sigs_A.shape[2]
    CNR = np.full((N_TD, N_BVAL), np.nan)

    # Reference TE: shortest TD in the grid (SNR0 is calibrated here)
    te_ref = TD_arr[0] + delta_ms

    for ti in range(N_TD):
        te = TD_arr[ti] + delta_ms
        # Thermal noise in per-TD normalised units grows with TD because
        # S0(TD) ∝ exp(−TE/T2) while σ_thermal stays constant.
        sigma_noise = (1.0 / snr0) * np.exp((te - te_ref) / t2_ms)

        for bi in range(N_BVAL):
            a = sigs_A[:, ti, bi]
            b = sigs_B[:, ti, bi]
            if np.isnan(a).any() or np.isnan(b).any():
                continue

            mu_a, mu_b = a.mean(), b.mean()

            # Total variance: biological variability (T2-independent) + TD-dependent noise
            var_a = a.std() ** 2 + sigma_noise ** 2
            var_b = b.std() ** 2 + sigma_noise ** 2

            denom = np.sqrt(var_a + var_b)
            if denom < 1e-10:
                continue

            CNR[ti, bi] = abs(mu_a - mu_b) / denom

    return CNR


# ── Level 3: per-phenotype T2 ──────────────────────────────────────────────────

def discrimination_metrics_snr_t2(sigs_A, sigs_B, TD_arr, delta_ms,
                                   t2_A, t2_B, snr0):
    """
    Level 3: CNR with phenotype-specific T2 values (Option B).

    Each tissue type has its own T2, so the T2/TE noise penalty differs
    between phenotype A and phenotype B.

        σ_noise_X(TD) = (1/snr0) × exp((TE(TD) − TE_ref) / T2_X)

    This reflects a scenario where phenotype-specific T2 priors are assumed
    (e.g. from a T2 map or literature values) to optimise the acquisition.
    T2-dependent noise is otherwise identical to Level 2.

    Parameters
    ----------
    sigs_A, sigs_B : (N_samples, N_TD, N_BVAL)  — Level 1 signals (S/S0)
    TD_arr         : (N_TD,)
    delta_ms       : float
    t2_A, t2_B    : float  — T2 (ms) for phenotype A and B respectively
    snr0           : float

    Returns
    -------
    CNR : (N_TD, N_BVAL)
    """
    N_TD, N_BVAL = sigs_A.shape[1], sigs_A.shape[2]
    CNR   = np.full((N_TD, N_BVAL), np.nan)
    te_ref = TD_arr[0] + delta_ms

    for ti in range(N_TD):
        te = TD_arr[ti] + delta_ms
        sigma_A = (1.0 / snr0) * np.exp((te - te_ref) / t2_A)
        sigma_B = (1.0 / snr0) * np.exp((te - te_ref) / t2_B)

        for bi in range(N_BVAL):
            a = sigs_A[:, ti, bi]
            b = sigs_B[:, ti, bi]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            mu_a, mu_b = a.mean(), b.mean()
            var_a = a.std() ** 2 + sigma_A ** 2
            var_b = b.std() ** 2 + sigma_B ** 2
            denom = np.sqrt(var_a + var_b)
            if denom < 1e-10:
                continue
            CNR[ti, bi] = abs(mu_a - mu_b) / denom

    return CNR


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    # All phenotypes use the same LUT (optim)
    lut_path = CATALOG[0]["lut"]
    print(f"Loading LUT: {os.path.basename(lut_path)}")
    params_lut, sig_norm, bval, TD, delta = load_lut(lut_path)
    print(f"  LUT: {params_lut.shape[0]} entries  |  "
          f"b-values: {bval}  |  TD: {TD} ms  |  delta: {delta} ms")

    tri = build_tri(params_lut)

    # Exclude b=0 from discrimination grid (signal ≈ 1 everywhere)
    b0_idx    = int(np.where(bval == 0)[0][0])
    bval_use  = np.delete(bval, b0_idx)
    b_use_idx = [i for i in range(len(bval)) if i != b0_idx]

    # ── Sample each phenotype ──────────────────────────────────────────────────
    print(f"\nSampling {N_SAMPLES} LHS points per phenotype...")
    pheno_signals = {}   # name → (N_SAMPLES, N_TD, N_BVAL_USE)
    pheno_samples = {}   # name → (N_SAMPLES, 4)

    for pheno in CATALOG:
        name    = pheno["name"]
        samples = sample_phenotype(pheno, n=N_SAMPLES)
        pheno_samples[name] = samples

        sigs_full = interpolate_batch(samples, sig_norm, tri)   # (N, N_TD, N_BVAL)
        sigs_use  = sigs_full[:, :, b_use_idx]                  # drop b=0

        # Drop samples with NaN (outside hull)
        valid = ~np.isnan(sigs_use).any(axis=(1, 2))
        n_dropped = int((~valid).sum())
        if n_dropped:
            print(f"  {name}: {n_dropped} samples outside hull dropped "
                  f"({100*n_dropped/N_SAMPLES:.1f}%)")
        pheno_signals[name] = sigs_use[valid]
        print(f"  {name}: {valid.sum()} valid samples")

    # ── Compute metrics for all pairs ─────────────────────────────────────────
    print("\nComputing discrimination metrics for all pairs...")
    pairs   = list(itertools.combinations([p["name"] for p in CATALOG], 2))
    results = {}

    for name_A, name_B in pairs:
        sigs_A = pheno_signals[name_A]
        sigs_B = pheno_signals[name_B]
        CNR = discrimination_metrics(sigs_A, sigs_B)
        key = f"{name_A}_vs_{name_B}"
        results[key] = {
            "CNR":    CNR,      # (N_TD, N_BVAL_USE)
            "sigs_A": sigs_A,   # (N_valid, N_TD, N_BVAL_USE)
            "sigs_B": sigs_B,
        }
        best_td, best_b = np.unravel_index(np.nanargmax(CNR), CNR.shape)
        print(f"  {name_A} vs {name_B}: best CNR={CNR[best_td, best_b]:.3f} "
              f"@ b={bval_use[best_b]:.2f} ms/µm², TD={TD[best_td]:.0f} ms")

    # ── Level 2: T2 / SNR-corrected metrics ───────────────────────────────────
    results_snr = {}
    if ENABLE_T2_SNR:
        print(f"\n-- Level 2: T2={T2_MS} ms  SNR0={SNR0}  delta={delta} ms --")
        for name_A, name_B in pairs:
            sigs_A = pheno_signals[name_A]
            sigs_B = pheno_signals[name_B]
            CNR = discrimination_metrics_snr(sigs_A, sigs_B, TD, delta, T2_MS, SNR0)
            key = f"{name_A}_vs_{name_B}"
            results_snr[key] = {"CNR": CNR}
            best_td, best_b = np.unravel_index(np.nanargmax(CNR), CNR.shape)
            print(f"  {name_A} vs {name_B}: best CNR={CNR[best_td, best_b]:.3f} "
                  f"@ b={bval_use[best_b]:.2f} ms/µm², TD={TD[best_td]:.0f} ms")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "opt_results.npz")
    save_dict = {
        "bval_use":    bval_use,
        "TD":          TD,
        "delta":       np.array([delta]),
        "pheno_names": np.array([p["name"] for p in CATALOG]),
        # Level 2 metadata
        "snr__enabled": np.array([ENABLE_T2_SNR]),
        "snr__T2_MS":   np.array([T2_MS]),
        "snr__SNR0":    np.array([SNR0]),
    }
    # Level 1 results — CNR + signals (no cohen_d/AUC)
    for key, val in results.items():
        save_dict[f"{key}__CNR"]    = val["CNR"]
        save_dict[f"{key}__sigs_A"] = val["sigs_A"]
        save_dict[f"{key}__sigs_B"] = val["sigs_B"]
    # Level 2 results (snr__ prefix) — CNR only
    for key, val in results_snr.items():
        save_dict[f"snr__{key}__CNR"] = val["CNR"]

    np.savez(out_path, **save_dict)
    print(f"\nResults saved -> {out_path}")

    return results, results_snr, bval_use, TD


if __name__ == "__main__":
    run()
