"""
LHS sample-size convergence check — justifies N_SAMPLES = 500 in opt_engine.py.

For each candidate N, draws several independent LHS realisations (different
seeds) per phenotype, computes the Level 1 CNR grid for every phenotype pair,
and measures how much the peak CNR value and its argmax (b*, TD*) location
vary across seeds. If N=500 sits on the convergence plateau (low seed-to-seed
variability, stable argmax), it is an adequate sample size.

Usage:
    python n_convergence_check.py

Output:
    Printed summary table (N -> CNR variability, argmax agreement)
    Acquisition_Opti/results/n_convergence.npz
"""

import os
import itertools
import numpy as np

from opt_engine import load_lut, build_tri, sample_phenotype, interpolate_batch, discrimination_metrics
from tumor_catalog import CATALOG

N_VALUES = [50, 100, 250, 500, 1000, 2000]
N_SEEDS  = 10
SEEDS    = list(range(N_SEEDS))

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    lut_path = CATALOG[0]["lut"]
    params_lut, sig_norm, bval, TD, delta = load_lut(lut_path)
    tri = build_tri(params_lut)

    b0_idx    = int(np.where(bval == 0)[0][0])
    b_use_idx = [i for i in range(len(bval)) if i != b0_idx]

    pairs = list(itertools.combinations([p["name"] for p in CATALOG], 2))

    # peak_cnr[N][seed][pair] and argmax_bt[N][seed][pair] = (b_idx, td_idx)
    peak_cnr  = {n: {s: {} for s in SEEDS} for n in N_VALUES}
    argmax_bt = {n: {s: {} for s in SEEDS} for n in N_VALUES}

    for n in N_VALUES:
        print(f"N = {n}")
        for seed in SEEDS:
            sigs = {}
            for pheno in CATALOG:
                name = pheno["name"]
                samples = sample_phenotype(pheno, n=n, seed=seed)
                sigs_full = interpolate_batch(samples, sig_norm, tri)
                sigs_use = sigs_full[:, :, b_use_idx]
                valid = ~np.isnan(sigs_use).any(axis=(1, 2))
                sigs[name] = sigs_use[valid]

            for name_A, name_B in pairs:
                CNR = discrimination_metrics(sigs[name_A], sigs[name_B])
                best_ti, best_bi = np.unravel_index(np.nanargmax(CNR), CNR.shape)
                key = f"{name_A}_vs_{name_B}"
                peak_cnr[n][seed][key]  = CNR[best_ti, best_bi]
                argmax_bt[n][seed][key] = (best_bi, best_ti)

    # ── Summary: for each N, aggregate over pairs ──────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'N':>6} | {'mean CV(%) peak CNR':>20} | {'max CV(%) peak CNR':>19} | "
          f"{'argmax agreement (%)':>21}")
    print("-" * 78)

    summary = {}
    for n in N_VALUES:
        cvs = []
        agreements = []
        for name_A, name_B in pairs:
            key = f"{name_A}_vs_{name_B}"
            vals = np.array([peak_cnr[n][s][key] for s in SEEDS])
            cv = 100 * vals.std() / vals.mean()
            cvs.append(cv)

            bts = [argmax_bt[n][s][key] for s in SEEDS]
            # mode = most frequent (b_idx, td_idx)
            uniq, counts = np.unique(bts, axis=0, return_counts=True)
            mode_count = counts.max()
            agreements.append(100 * mode_count / N_SEEDS)

        mean_cv = np.mean(cvs)
        max_cv  = np.max(cvs)
        mean_agree = np.mean(agreements)
        summary[n] = (mean_cv, max_cv, mean_agree)
        print(f"{n:>6} | {mean_cv:>20.2f} | {max_cv:>19.2f} | {mean_agree:>21.1f}")

    print("=" * 78)

    out_path = os.path.join(RESULTS_DIR, "n_convergence.npz")
    np.savez(out_path,
              N_values=np.array(N_VALUES),
              summary=np.array([summary[n] for n in N_VALUES]))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
