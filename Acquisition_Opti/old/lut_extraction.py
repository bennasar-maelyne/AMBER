"""
LUT extraction for AMBER-opti simulation results.

Matches the structure of LUT_extraction.m exactly:
  signals_4D : (N_spheres, N_TD, N_bval, N_bvec) — raw counts (not normalised)
  params     : (N_spheres, 4)                     — [f, Dex, rmean, rsd]
  sequence   : struct  bval, TD, delta, kappa, n_bval, n_bvec, n_del

Reads sig_diffusion.txt from each sphere result folder and assembles
params using sobol_array.txt + extra_points + kappa from sim_para.txt.
Does NOT require the input sphere files (sphere_data_pre_opti.zip).
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.abspath(os.path.join(THIS_DIR, ".."))
LUT_DIR   = os.path.join(
    ROOT, "monte-carlo-simulation-sphere-PGSE-main", "data", "AMBER-opti",
    "lognorm_population_RealisticTests2_V3",
)
RESULTS_DIR = os.path.join(LUT_DIR, "results_2026-05_maelyne_optim-grid")
SOBOL_FILE  = os.path.join(LUT_DIR, "sobol_array.txt")
OUTPUT_FILE = os.path.join(ROOT, "Validation", "lookup_table_optim.mat")

# ── Acquisition parameters — must match the MATLAB generation script ───────────
BVAL        = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
TD          = np.array([20, 30, 45, 60, 80])   # ms
DELTA       = 10.0                              # pulse width (ms)
N_DIR       = 6
N_TD        = len(TD)
N_BVAL      = len(BVAL)
N_SIGNALS   = N_DIR * N_BVAL * N_TD            # 330 values per sphere

# ── Extra points — copied verbatim from the MATLAB generation script ───────────
# Order in MATLAB loop: dex, rmean, rsd, f
_extra = []
for dex   in [0.5, 3.0]:
    for rmean in [1.0, 20.0]:
        for rsd in [0.0, 12.0]:
            for f in [0.001, 0.2, 0.6, 0.7, 0.8]:
                _extra.append([dex, rmean, rsd, f])
EXTRA_POINTS = np.array(_extra)   # (40, 4)  [Dex, rmean, rsd, f]

# ── Build base parameter array from Sobol + extra points ──────────────────────
sobol_params    = np.loadtxt(SOBOL_FILE)                       # (1026, 4) [Dex, rmean, rsd, f]
params_dexfirst = np.vstack([sobol_params, EXTRA_POINTS])      # (1066, 4)

# Re-order columns: [Dex, rmean, rsd, f] → [f, Dex, rmean, rsd]
col_Dex, col_rmean, col_rsd, col_f = 0, 1, 2, 3
base_params = params_dexfirst[:, [col_f, col_Dex, col_rmean, col_rsd]]  # (1066, 4)

N_SPHERES = len(base_params)
print(f"Total parameter sets : {N_SPHERES}")

# ── Verify sphere count ────────────────────────────────────────────────────────
available = [
    d for d in os.listdir(RESULTS_DIR)
    if d.startswith("sphere_") and os.path.isdir(os.path.join(RESULTS_DIR, d))
]
print(f"Sphere folders found : {len(available)}")
if len(available) != N_SPHERES:
    print(f"  [WARNING] Expected {N_SPHERES}, found {len(available)}")

# ── Load signals and kappa ─────────────────────────────────────────────────────
#
# sig_diffusion.txt layout (MATLAB column-major):
#   reshape(sig, [N_DIR, N_BVAL, N_TD])  →  (6, 11, 5)
#   permute([3, 2, 1])                   →  (5, 11, 6) = (TD, bval, dir)
#
# Python equivalent (Fortran / column-major order):
#   sig.reshape(N_DIR, N_BVAL, N_TD, order='F').transpose(2, 1, 0)
#   → shape (N_TD, N_BVAL, N_DIR)
#
# sim_para.txt layout (0-indexed):
#   0: dt   1: NPar_total   2: NParICS   3: Din   4: Dex   5: kappa

signals_4D = np.full((N_SPHERES, N_TD, N_BVAL, N_DIR), np.nan)
missing    = []

# Read kappa once from sphere_0001 (constant across all spheres)
_para0 = np.loadtxt(os.path.join(RESULTS_DIR, "sphere_0001", "sim_para.txt"))
KAPPA  = float(_para0[5])   # line 6 (1-indexed)
print(f"kappa (fixed) : {KAPPA}")

for ii in range(1, N_SPHERES + 1):
    sphere_dir = os.path.join(RESULTS_DIR, f"sphere_{ii:04d}")

    # --- signals from sig_diffusion.txt ---
    sig_file = os.path.join(sphere_dir, "sig_diffusion.txt")
    if not os.path.isfile(sig_file):
        missing.append(ii)
        continue

    raw = np.loadtxt(sig_file)

    if len(raw) != N_SIGNALS:
        print(f"  [WARNING] sphere_{ii:04d}: length {len(raw)}, expected {N_SIGNALS}")
        missing.append(ii)
        continue

    # Clip negative / near-zero before storing (matches MATLAB: sig(sig < 1e-6) = 1e-3)
    raw[raw < 1e-6] = 1e-3

    # Replicate MATLAB reshape→permute: (N_DIR, N_BVAL, N_TD) → (N_TD, N_BVAL, N_DIR)
    signals_4D[ii - 1] = raw.reshape(N_DIR, N_BVAL, N_TD, order='F').transpose(2, 1, 0)

if missing:
    print(f"\n[WARNING] {len(missing)} missing/invalid spheres: "
          f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
else:
    print("All spheres loaded successfully.")

# ── params: [f, Dex, rmean, rsd] — kappa is fixed, stored in sequence ─────────
all_params = base_params   # (N, 4) — columns: 0=f, 1=Dex, 2=rmean, 3=rsd

# ── Sanity checks ──────────────────────────────────────────────────────────────
b0_idx  = int(np.where(BVAL == 0)[0][0])
b0_raw  = signals_4D[:, :, b0_idx, :]          # (N_spheres, N_TD, N_DIR) — raw counts at b=0
b0_mean = np.nanmean(b0_raw)
b0_std  = np.nanstd(b0_raw)

print(f"\nS(b=0) raw counts — mean: {b0_mean:.1f}  std: {b0_std:.1f}  "
      f"min: {np.nanmin(b0_raw):.1f}  max: {np.nanmax(b0_raw):.1f}")
print(f"NaN count : {int(np.sum(np.isnan(signals_4D)))}")

# ── sequence struct ────────────────────────────────────────────────────────────
sequence = {
    "bval":   BVAL,
    "TD":     TD,
    "delta":  np.array([DELTA]),
    "kappa":  np.array([KAPPA]),
    "n_bval": np.array([N_BVAL]),
    "n_bvec": np.array([N_DIR]),
    "n_del":  np.array([N_TD]),
}

# ── Save ───────────────────────────────────────────────────────────────────────
sio.savemat(OUTPUT_FILE, {
    "signals_4D": signals_4D,   # (N_spheres, N_TD, N_bval, N_bvec)
    "params":     all_params,   # (N_spheres, 5) — [f, Dex, rmean, rsd, kappa]
    "sequence":   sequence,
})
print(f"\nLUT saved  →  {os.path.abspath(OUTPUT_FILE)}")
print(f"signals_4D shape : {signals_4D.shape}")
print(f"params     shape : {all_params.shape}")

# ── Normalise for plots ────────────────────────────────────────────────────────
sig_avg  = signals_4D.mean(axis=-1)            # (N, N_TD, N_BVAL) — avg over directions
S0       = sig_avg[:, :, b0_idx]               # (N, N_TD)
sig_norm = sig_avg / S0[:, :, np.newaxis]      # (N, N_TD, N_BVAL)

from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata

OUT_DIR = os.path.join(THIS_DIR, "Figures", "lut_optim_check")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 11})

# 3 representative TDs for display (first, middle, last)
TDs_show   = [TD[0], TD[2], TD[-1]]       # [20, 45, 80] ms
TD_indices = [int(np.where(TD == t)[0][0]) for t in TDs_show]
TD_COLORS  = {t: c for t, c in zip(TDs_show, ["#1f77b4", "#2ca02c", "#d62728"])}


def _fill_nan(stat, xg, yg):
    xc = 0.5 * (xg[:-1] + xg[1:])
    yc = 0.5 * (yg[:-1] + yg[1:])
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    valid = ~np.isnan(stat)
    if valid.any() and (~valid).any():
        filled = stat.copy()
        filled[~valid] = griddata(
            (XX[valid], YY[valid]), stat[valid],
            (XX[~valid], YY[~valid]), method='nearest')
        return filled
    return stat


# =============================================================================
# Fig 1 — 2D scatter: 3 parameter pairs × 3 TDs at b = 0.5 ms/µm²
# =============================================================================
b_scatter     = 0.5
b_idx_scatter = int(np.argmin(np.abs(BVAL - b_scatter)))

pairs = [
    (0, 1, 'f', '$D_{ex}$ (µm²/ms)'),
    (0, 2, 'f', '$r_{mean}$ (µm)'),
    (1, 2, '$D_{ex}$ (µm²/ms)', '$r_{mean}$ (µm)'),
]

sigs_scatter = {t: sig_norm[:, i, b_idx_scatter] for t, i in zip(TDs_show, TD_indices)}
vmax_scatter = max(s.max() for s in sigs_scatter.values())

fig, axes = plt.subplots(3, 3, figsize=(14, 13))
for row, (xi, yi, xlabel, ylabel) in enumerate(pairs):
    for col, (t, t_idx) in enumerate(zip(TDs_show, TD_indices)):
        ax  = axes[row, col]
        sc  = ax.scatter(all_params[:, xi], all_params[:, yi],
                         c=sigs_scatter[t], cmap='viridis',
                         vmin=0, vmax=vmax_scatter, s=12, alpha=0.6, linewidths=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'TD = {t} ms')
        plt.colorbar(sc, ax=ax, label='S/S₀', shrink=0.9)
fig.suptitle(f'Signal S/S₀ — {N_SPHERES} LUT points,  b = {b_scatter} ms/µm²', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_scatter2D.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Fig 1 saved  (2D scatter)')


# =============================================================================
# Fig 2 — Signal decay: median Dex curve + Dex-range error bars, per TD
# =============================================================================
dex_vals    = all_params[:, 1]
dex_lo_mask = dex_vals <= np.percentile(dex_vals, 20)
dex_mid_mask= (dex_vals >= np.percentile(dex_vals, 40)) & (dex_vals <= np.percentile(dex_vals, 60))
dex_hi_mask = dex_vals >= np.percentile(dex_vals, 80)

fig, ax = plt.subplots(figsize=(8, 5))
for t, t_idx in zip(TDs_show, TD_indices):
    col = TD_COLORS[t]
    s_lo  = np.nanmean(sig_norm[dex_lo_mask,  t_idx, :], axis=0)
    s_mid = np.nanmean(sig_norm[dex_mid_mask, t_idx, :], axis=0)
    s_hi  = np.nanmean(sig_norm[dex_hi_mask,  t_idx, :], axis=0)
    ye_lo = np.maximum(0, s_mid - s_hi)
    ye_hi = np.maximum(0, s_lo  - s_mid)
    ax.errorbar(BVAL, s_mid, yerr=[ye_lo, ye_hi], fmt='o-', color=col,
                label=f'TD = {t} ms', capsize=4, elinewidth=1.2, markersize=4, lw=1.5)
ax.set_xlabel('b-value (ms/µm²)')
ax.set_ylabel('S/S₀')
ax.set_xlim(0, BVAL[-1]);  ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3);  ax.legend()
fig.suptitle('Signal decay — curve at median $D_{ex}$\n'
             'Error bars: $D_{ex}$ = low (top) to high (bottom)\n'
             'Averaged over all (f, $r_{mean}$, $r_{sd}$)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_decay.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Fig 2 saved  (signal decay)')


# =============================================================================
# Fig 3 — Marginalised heatmaps (f×Dex and f×rmean) at b = 1.5 ms/µm²
# =============================================================================
n_bins   = 20
b_hm     = 1.5
b_idx_hm = int(np.argmin(np.abs(BVAL - b_hm)))
f_g      = np.linspace(all_params[:, 0].min(), all_params[:, 0].max(), n_bins + 1)
Dex_g    = np.linspace(all_params[:, 1].min(), all_params[:, 1].max(), n_bins + 1)
rmean_g  = np.linspace(all_params[:, 2].min(), all_params[:, 2].max(), n_bins + 1)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, (t, t_idx) in enumerate(zip(TDs_show, TD_indices)):
    sig = sig_norm[:, t_idx, b_idx_hm]

    stat_fd, _, _, _ = binned_statistic_2d(
        all_params[:, 0], all_params[:, 1], sig, statistic='mean', bins=n_bins,
        range=[[f_g[0], f_g[-1]], [Dex_g[0], Dex_g[-1]]])
    stat_fd = _fill_nan(stat_fd, f_g, Dex_g)
    im0 = axes[0, col].imshow(stat_fd.T, origin='lower', aspect='auto', cmap='RdYlBu_r',
                               extent=[f_g[0], f_g[-1], Dex_g[0], Dex_g[-1]], vmin=0, vmax=1)
    axes[0, col].set_xlabel('f');  axes[0, col].set_ylabel('$D_{ex}$ (µm²/ms)')
    axes[0, col].set_title(f'S(f, $D_{{ex}}$) — TD = {t} ms')
    plt.colorbar(im0, ax=axes[0, col], label='Mean S/S₀', shrink=0.9)

    stat_fr, _, _, _ = binned_statistic_2d(
        all_params[:, 0], all_params[:, 2], sig, statistic='mean', bins=n_bins,
        range=[[f_g[0], f_g[-1]], [rmean_g[0], rmean_g[-1]]])
    stat_fr = _fill_nan(stat_fr, f_g, rmean_g)
    im1 = axes[1, col].imshow(stat_fr.T, origin='lower', aspect='auto', cmap='RdYlBu_r',
                               extent=[f_g[0], f_g[-1], rmean_g[0], rmean_g[-1]], vmin=0, vmax=1)
    axes[1, col].set_xlabel('f');  axes[1, col].set_ylabel('$r_{mean}$ (µm)')
    axes[1, col].set_title(f'S(f, $r_{{mean}}$) — TD = {t} ms')
    plt.colorbar(im1, ax=axes[1, col], label='Mean S/S₀', shrink=0.9)
fig.suptitle(f'Marginalised heatmaps at b = {b_hm} ms/µm²  '
             f'(mean over $r_{{sd}}$ — empty bins: nearest interp.)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_heatmaps.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Fig 3 saved  (heatmaps)')


# =============================================================================
# Fig 4 — Pearson correlation matrix
# =============================================================================
rows = []
for t_idx, t_val in enumerate(TD):
    for b_i, bv in enumerate(BVAL):
        s = sig_norm[:, t_idx, b_i]
        rows.append(np.column_stack([all_params, np.full(N_SPHERES, bv),
                                     np.full(N_SPHERES, t_val), s]))
data_all = np.vstack(rows)
labels   = ['f', '$D_{ex}$', '$r_{mean}$', '$r_{sd}$', 'b-value', 'TD', 'Signal']
corr     = np.corrcoef(data_all.T)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)));  ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticks(range(len(labels)));  ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        c = corr[i, j]
        ax.text(j, i, f'{c:.2f}', ha='center', va='center',
                fontsize=9, color='white' if abs(c) > 0.6 else 'black')
plt.colorbar(im, ax=ax, shrink=0.85, label='Pearson r')
ax.set_title('Pearson correlation — LUT optim')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_correlation.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Fig 4 saved  (correlation matrix)')

print(f'\nDone. Figures in {os.path.abspath(OUT_DIR)}')
