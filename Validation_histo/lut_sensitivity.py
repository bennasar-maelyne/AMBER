"""
Section 3.1 — LUT sensitivity analysis  (lookup_table_val.mat, b < 7.5 ms/µm²)

Figures:
  fig1_scatter2D.png      : 2D scatter grid, 3 parameter pairs × 2 TDs
  fig2_decay.png          : S/S₀(b) decay — median Dex curve + Dex extremes as error bars
  fig3_heatmaps.png       : Marginalised heatmaps (binned + nearest interpolation for empty bins)
  fig4_correlation.png    : Pearson correlation matrix
  fig6_corr_vs_bvalue.png : Pearson r(parameter, signal) vs b-value, one line per parameter
  figS_kdet_sensitivity.png : Supplementary — k_det effect (1D curves)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import h5py

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join('..', 'Paper', 'Figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── Load LUT ──────────────────────────────────────────────────────────────────
with h5py.File('lookup_table_val.mat', 'r') as _f:
    params_lut  = np.array(_f['params']).T                   # (N, 4): f, Dex, rmean, rsd
    signals_lut = np.transpose(np.array(_f['signals_4D']),
                                (3, 2, 1, 0))                # (N, 2, 13, 64)
    bvals_lut   = np.array(_f['sequence']['bval']).flatten() # ms/µm²
    TDs_lut     = np.array(_f['sequence']['TD']).flatten()   # [19., 49.] ms

b_mask      = bvals_lut < 7.5
bvals_use   = bvals_lut[b_mask]
signals_use = signals_lut[:, :, b_mask, :]                  # (N, 2, 11, 64)

tri = Delaunay(params_lut)
N   = params_lut.shape[0]

print(f"LUT: {N} entries")
print(f"b-values (< 7.5 ms/µm²): {bvals_use}")
print(f"TDs: {TDs_lut} ms")
print(f"Dex range: [{params_lut[:,1].min():.2f}, {params_lut[:,1].max():.2f}] µm²/ms")

# ── Style ─────────────────────────────────────────────────────────────────────
TD_LABELS = {19: 'TD = 19 ms', 49: 'TD = 49 ms'}
TDs_show  = [19, 49]
TD_COLORS = {19: '#1f77b4', 49: '#ff7f0e'}

b_scatter     = 0.5
b_idx_scatter = int(np.argmin(np.abs(bvals_use - b_scatter)))

f_REF, Dex_REF, rmean_REF, rsd_REF = 0.30, 2.0, 7.0, 2.0


def sig_at(TD, b_idx):
    TD_idx = int(np.argmin(np.abs(TDs_lut - TD)))
    return np.mean(signals_use[:, TD_idx, b_idx, :], axis=-1)


def _interp_curve(q, TD_idx):
    s = tri.find_simplex(q)
    if s == -1:
        idx = np.argmin(np.linalg.norm(params_lut - q, axis=1))
        return np.mean(signals_use[idx, TD_idx, :, :], axis=-1)
    vidx  = tri.simplices[s]
    verts = params_lut[vidx]
    try:
        bc = np.linalg.solve((verts[1:] - verts[0]).T, q - verts[0])
        bc = np.append(1 - bc.sum(), bc)
        return np.mean(np.tensordot(bc, signals_use[vidx], axes=(0, 0))[TD_idx, :, :], axis=-1)
    except np.linalg.LinAlgError:
        idx = vidx[np.argmin(np.linalg.norm(verts - q, axis=1))]
        return np.mean(signals_use[idx, TD_idx, :, :], axis=-1)

def signal_curve(f, Dex, rmean, rsd, TD):
    TD_idx = int(np.argmin(np.abs(TDs_lut - TD)))
    if f < 0.007:
        return np.exp(-bvals_use * Dex)
    return _interp_curve(np.array([f, Dex, rmean, rsd]), TD_idx)


def _fill_nan(stat, xg, yg):
    """Fill NaN bins with nearest-neighbour interpolation from non-empty bins."""
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
# Fig 1 — 2D scatter grid: 3 parameter pairs × 2 TDs at b = 0.5 ms/µm²
# Colormap scale: global max across all 6 subplots
# =============================================================================
pairs = [
    (0, 1, 'f', '$D_{ex}$ (µm²/ms)'),
    (0, 2, 'f', '$r_{mean}$ (µm)'),
    (1, 2, '$D_{ex}$ (µm²/ms)', '$r_{mean}$ (µm)'),
]

# Pre-compute signals and global vmax
sigs_scatter = {TD: sig_at(TD, b_idx_scatter) for TD in TDs_show}
vmax_scatter = max(s.max() for s in sigs_scatter.values())
print(f"Scatter vmax: {vmax_scatter:.4f}")

_panel_labels = iter("ABCDEF")
fig, axes = plt.subplots(3, 2, figsize=(9, 12))

for row, (xi, yi, xlabel, ylabel) in enumerate(pairs):
    for col, TD in enumerate(TDs_show):
        sig = sigs_scatter[TD]
        ax  = axes[row, col]
        sc  = ax.scatter(params_lut[:, xi], params_lut[:, yi], c=sig,
                         cmap='viridis', vmin=0, vmax=vmax_scatter,
                         s=10, alpha=0.6, linewidths=0, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(TD_LABELS[TD], fontsize=11)
        cb = plt.colorbar(sc, ax=ax, label='$S/S_0$', shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        pan = next(_panel_labels)
        ax.text(0.03, 0.97, f'({pan})', transform=ax.transAxes,
                va='top', ha='left', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
fig.savefig(os.path.join(OUT_DIR, 'fig2_scatter2D.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Fig 2 saved  (2D scatter, b = {b_scatter} ms/µm², vmax = {vmax_scatter:.3f})')


# =============================================================================
# Fig 2 — Signal decay: central curve at median Dex, error bars = Dex extremes
# Dex bins: low ≈ 1 µm²/ms, mid ≈ 2 µm²/ms, high ≈ 3 µm²/ms
# At each b-value, the signal is averaged over all (f, rmean, rsd) entries
# within the Dex bin → error bars reflect Dex sensitivity only
# =============================================================================
dex_vals  = params_lut[:, 1]
dex_lo_mask  = dex_vals <= np.percentile(dex_vals, 20)   # Dex ≈ 1 µm²/ms
dex_mid_mask = (dex_vals >= np.percentile(dex_vals, 40)) & \
               (dex_vals <= np.percentile(dex_vals, 60)) # Dex ≈ 2 µm²/ms
dex_hi_mask  = dex_vals >= np.percentile(dex_vals, 80)   # Dex ≈ 3 µm²/ms

print(f"Dex bins — low: {dex_vals[dex_lo_mask].mean():.2f}  "
      f"mid: {dex_vals[dex_mid_mask].mean():.2f}  "
      f"high: {dex_vals[dex_hi_mask].mean():.2f} µm²/ms")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for td_i, TD in enumerate(TDs_show):
    TD_idx = int(np.argmin(np.abs(TDs_lut - TD)))
    col = TD_COLORS[TD]

    def mean_sig(mask, _TD_idx=TD_idx):
        return np.mean(signals_use[mask, _TD_idx, :, :], axis=(0, -1))

    s_lo  = mean_sig(dex_lo_mask)
    s_mid = mean_sig(dex_mid_mask)
    s_hi  = mean_sig(dex_hi_mask)

    ye_lo = np.maximum(0, s_mid - s_hi)
    ye_hi = np.maximum(0, s_lo  - s_mid)

    ax.errorbar(bvals_use, s_mid, yerr=[ye_lo, ye_hi],
                fmt='o-', color=col, label=TD_LABELS[TD],
                capsize=3.5, elinewidth=1.0, markersize=4, lw=1.5)

ax.set_xlabel('$b$-value (ms/µm²)')
ax.set_ylabel('$S/S_0$')
ax.set_xlim(0, bvals_use[-1])
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2, lw=0.6)
ax.legend(frameon=False)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_decay.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig 3 saved  (signal decay with Dex error bars)')


# =============================================================================
# Fig 3 — Marginalised heatmaps (mean over all rmean, rsd at each (f, Dex/rmean) bin)
# Empty bins filled by nearest-neighbour interpolation
# =============================================================================
n_bins    = 20
f_g       = np.linspace(params_lut[:, 0].min(), params_lut[:, 0].max(), n_bins + 1)
Dex_g     = np.linspace(params_lut[:, 1].min(), params_lut[:, 1].max(), n_bins + 1)
rmean_g   = np.linspace(params_lut[:, 2].min(), params_lut[:, 2].max(), n_bins + 1)
b_heatmap = 1.3
b_idx_hm  = int(np.argmin(np.abs(bvals_use - b_heatmap)))

_hm_panels = iter("ABCD")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for j, TD in enumerate(TDs_show):
    sig = sig_at(TD, b_idx_hm)

    stat_fd, _, _, _ = binned_statistic_2d(
        params_lut[:, 0], params_lut[:, 1], sig, statistic='mean', bins=n_bins,
        range=[[f_g[0], f_g[-1]], [Dex_g[0], Dex_g[-1]]])
    stat_fd = _fill_nan(stat_fd, f_g, Dex_g)
    im0 = axes[0, j].imshow(stat_fd.T, origin='lower', aspect='auto',
                             cmap='RdYlBu_r', extent=[f_g[0], f_g[-1], Dex_g[0], Dex_g[-1]],
                             vmin=0, vmax=1)
    axes[0, j].set_xlabel('$f$')
    axes[0, j].set_ylabel('$D_{ex}$ (µm²/ms)')
    axes[0, j].set_title(TD_LABELS[TD])
    cb0 = plt.colorbar(im0, ax=axes[0, j], label='Mean $S/S_0$', shrink=0.88)
    cb0.ax.tick_params(labelsize=9)
    axes[0, j].text(0.03, 0.97, f'({next(_hm_panels)})', transform=axes[0, j].transAxes,
                    va='top', ha='left', fontsize=11, fontweight='bold', color='black')

    stat_fr, _, _, _ = binned_statistic_2d(
        params_lut[:, 0], params_lut[:, 2], sig, statistic='mean', bins=n_bins,
        range=[[f_g[0], f_g[-1]], [rmean_g[0], rmean_g[-1]]])
    stat_fr = _fill_nan(stat_fr, f_g, rmean_g)
    im1 = axes[1, j].imshow(stat_fr.T, origin='lower', aspect='auto',
                             cmap='RdYlBu_r', extent=[f_g[0], f_g[-1], rmean_g[0], rmean_g[-1]],
                             vmin=0, vmax=1)
    axes[1, j].set_xlabel('$f$')
    axes[1, j].set_ylabel('$r_{mean}$ (µm)')
    axes[1, j].set_title(TD_LABELS[TD])
    cb1 = plt.colorbar(im1, ax=axes[1, j], label='Mean $S/S_0$', shrink=0.88)
    cb1.ax.tick_params(labelsize=9)
    axes[1, j].text(0.03, 0.97, f'({next(_hm_panels)})', transform=axes[1, j].transAxes,
                    va='top', ha='left', fontsize=11, fontweight='bold', color='black')

fig.tight_layout(h_pad=1.5, w_pad=1.5)
fig.savefig(os.path.join(OUT_DIR, 'fig4_heatmaps.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig 4 saved  (heatmaps, NaN filled)')


# =============================================================================
# Fig 4 — Pearson correlation matrix
# =============================================================================
rows = []
for td_i, TD_val in enumerate(TDs_lut):
    for b_i, bv in enumerate(bvals_use):
        s = np.mean(signals_use[:, td_i, b_i, :], axis=-1)
        rows.append(np.column_stack([params_lut, np.full(N, bv), np.full(N, TD_val), s]))
data_all = np.vstack(rows)
labels   = ['f', '$D_{ex}$', '$r_{mean}$', '$r_{sd}$', 'b-value', 'TD', 'Signal']
corr     = np.corrcoef(data_all.T)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
for i in range(len(labels)):
    for j in range(len(labels)):
        c = corr[i, j]
        ax.text(j, i, f'{c:.2f}', ha='center', va='center',
                fontsize=8.5, color='white' if abs(c) > 0.65 else 'black')
cb = plt.colorbar(im, ax=ax, shrink=0.82, label='Pearson $r$')
cb.ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_correlation.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig 5 saved  (correlation matrix)')


# =============================================================================
# Fig 6 — Pearson r(parameter, signal) as a function of b-value
# One line per biological parameter (f, Dex, rmean, rsd), pooled over TD
# (marginalised over the other 3 parameters at each b-value)
# =============================================================================
param_names  = ['f', '$D_{ex}$', '$r_{mean}$', '$r_{sd}$']
param_colors = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']

corr_vs_b = np.zeros((len(bvals_use), 4))
for b_i in range(len(bvals_use)):
    s_TD     = np.mean(signals_use[:, :, b_i, :], axis=-1)   # (N, 2) over gradient dirs
    s_pooled = s_TD.T.flatten()                               # (2N,) TD-major
    p_pooled = np.tile(params_lut, (2, 1))                    # (2N, 4)
    for p_i in range(4):
        corr_vs_b[b_i, p_i] = np.corrcoef(p_pooled[:, p_i], s_pooled)[0, 1]

for p_i, name in enumerate(param_names):
    print(f"  r({name}, S) vs b: " +
          ", ".join(f"{bv:.1f}->{corr_vs_b[b_i, p_i]:+.2f}"
                    for b_i, bv in enumerate(bvals_use)))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for p_i, (name, col) in enumerate(zip(param_names, param_colors)):
    ax.plot(bvals_use, corr_vs_b[:, p_i], 'o-', color=col, label=name,
            markersize=4, lw=1.5)
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel('$b$-value (ms/µm²)')
ax.set_ylabel('Pearson $r$ (parameter, $S/S_0$)')
ax.set_xlim(0, bvals_use[-1])
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.2, lw=0.6)
ax.legend(frameon=False, fontsize=9)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig6_corr_vs_bvalue.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig 6 saved  (correlation vs b-value)')


# =============================================================================
# Supplementary — k_det sensitivity
# All parameters from ONE ROI: Patient 3, H&E 1, ROI 3 (median k_det in df_summary)
#   f_vivo  = 0.509  (QuPath)
#   Dex_19  = 1.715 µm²/ms,  Dex_49 = 1.753 µm²/ms  (chi² inference)
#   rmean   = 6.66 µm,  rsd = 0.797 µm  (QuPath, Abercrombie-corrected)
#   k_det_best = 0.321
# =============================================================================
# Reference ROI parameters (all consistent, same biological sample)
f_ROI    = 0.509
Dex_ROI  = {19: 1.715, 49: 1.753}
rmean_ROI = 6.66
rsd_ROI   = 0.797
kdet_best_ROI = 0.321   # inferred value for this ROI

k_det_values = [0.13, 0.20, 0.30, 0.40]
linestyles   = [':', '-.', '--', '-']
colors_kdet  = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

fig, axes = plt.subplots(2, 1, figsize=(6.5, 8), sharex=True)
for pan_idx, (ax, TD) in enumerate(zip(axes, TDs_show)):
    Dex = Dex_ROI[TD]

    sig_nofix = signal_curve(f_ROI, Dex, rmean_ROI, rsd_ROI, TD)
    ax.plot(bvals_use, sig_nofix, color='black', ls='-', lw=2.0,
            label=f'No correction  ($f$ = {f_ROI:.3f})')

    for k_det, ls, col in zip(k_det_values, linestyles, colors_kdet):
        f_eff = k_det * f_ROI
        sig   = signal_curve(f_eff, Dex, rmean_ROI, rsd_ROI, TD)
        lbl   = f'$k_{{det}}$ = {k_det:.2f}  ($f_{{eff}}$ = {f_eff:.3f})'
        lw    = 2.2 if abs(k_det - kdet_best_ROI) < 0.05 else 1.6
        ax.plot(bvals_use, sig, color=col, ls=ls, lw=lw, label=lbl)

    ax.set_ylabel('$S/S_0$')
    ax.set_xlim(0, bvals_use[-1])
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{TD_LABELS[TD]}  ($D_{{ex}}$ = {Dex:.3f} µm²/ms)')
    ax.grid(True, alpha=0.2, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.03, 0.97, f'({"AB"[pan_idx]})', transform=ax.transAxes,
            va='top', ha='left', fontsize=11, fontweight='bold')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel('$b$-value (ms/µm²)')
fig.tight_layout(h_pad=1.2)
fig.savefig(os.path.join(OUT_DIR, 'figS1_kdet_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig S1 saved  (k_det sensitivity)')

print(f'\nDone. Figures in {os.path.abspath(OUT_DIR)}')
