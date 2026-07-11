# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # LUT utilities — shared functions for Average_signal.py and validation.py

# %%
import numpy as np
import pandas as pd


def interpolate_signal(f, Dex, rmean, rsd, bval, TD, tri, signals_lookup, params_lut,
                       sequence_lut=None):
    """Barycentric Delaunay interpolation of LUT signal at a single (b-value, TD)."""
    if f < 0.007:
        return 1.0

    query_params = np.array([f, Dex, rmean, rsd])

    bvals      = sequence_lut['bval'][0][0].flatten()
    TDs        = sequence_lut['TD'][0][0].flatten()
    TD_index   = int(np.argmin(np.abs(TDs   - TD)))
    bval_index = int(np.argmin(np.abs(bvals - bval)))

    simplex_index = tri.find_simplex(query_params)

    if simplex_index == -1:
        distances   = np.linalg.norm(params_lut - query_params, axis=1)
        nearest_idx = np.argmin(distances)
        return np.mean(signals_lookup[nearest_idx][TD_index, bval_index, :])

    vertex_index = tri.simplices[simplex_index]
    vertices     = params_lut[vertex_index]

    try:
        T           = vertices[1:] - vertices[0]
        v           = query_params - vertices[0]
        bary_coords = np.linalg.solve(T.T, v)
        bary_coords = np.append(1 - np.sum(bary_coords), bary_coords)
        signals_subset      = signals_lookup[vertex_index]
        interpolated_signal = np.tensordot(bary_coords, signals_subset, axes=(0, 0))
        return np.mean(interpolated_signal[TD_index, bval_index, :])
    except Exception:
        distances     = np.linalg.norm(vertices - query_params, axis=1)
        nearest_v_idx = vertex_index[np.argmin(distances)]
        return np.mean(signals_lookup[nearest_v_idx][TD_index, bval_index, :])


def get_signal_curve_lut(f, Dex, rmean, rsd, TD, tri, signals_lookup, params_lut,
                         sequence_lut=None):
    """
    Return the predicted signal at ALL LUT b-values for given (f, Dex, rmean, rsd, TD).

    Returns
    -------
    bvals_lut : 1-D array, LUT b-values in ms/µm²
    signals   : 1-D array, predicted S/S0 at each LUT b-value
    """
    bvals_lut = np.array(sequence_lut['bval'][0][0]).flatten()
    TDs       = np.array(sequence_lut['TD'][0][0]).flatten()
    matches   = np.where(TDs == TD)[0]
    if len(matches) == 0:
        raise ValueError(f"TD={TD} ms not found in LUT. Available TDs: {TDs.tolist()}")
    TD_index = matches[0]

    if f < 0.007:
        return bvals_lut, np.exp(-bvals_lut * Dex)

    query_params  = np.array([f, Dex, rmean, rsd])
    simplex_index = tri.find_simplex(query_params)

    if simplex_index == -1:
        nearest_idx = np.argmin(np.linalg.norm(params_lut - query_params, axis=1))
        signals = np.mean(signals_lookup[nearest_idx][TD_index, :, :], axis=-1)
    else:
        vertex_index = tri.simplices[simplex_index]
        vertices     = params_lut[vertex_index]
        try:
            T           = vertices[1:] - vertices[0]
            v           = query_params - vertices[0]
            bary_coords = np.linalg.solve(T.T, v)
            bary_coords = np.append(1 - np.sum(bary_coords), bary_coords)
            interpolated = np.tensordot(bary_coords, signals_lookup[vertex_index],
                                        axes=(0, 0))
            signals = np.mean(interpolated[TD_index, :, :], axis=-1)
        except Exception:
            nearest_v_idx = vertex_index[
                np.argmin(np.linalg.norm(vertices - query_params, axis=1))]
            signals = np.mean(signals_lookup[nearest_v_idx][TD_index, :, :], axis=-1)

    return bvals_lut, signals


def extract_patient_curves(data_dict, slice_idx, center, roi_size, roi_type='square'):
    """Extract DWI curves from a .mat file for a given patient/slice, with noise envelope."""
    full_brain   = np.transpose(data_dict["fullbrain_all"], (0, 2, 1, 3))
    signal_slice = full_brain[:, :, slice_idx, :]

    noise_data  = np.transpose(data_dict["noisemap_all"], (0, 2, 1, 3))
    noise_slice = noise_data[:, :, slice_idx, :]

    H, W, num_b_volumes = signal_slice.shape

    b_values     = data_dict["bv_all"].flatten()
    zero_idx     = np.where(b_values == 0)[0]
    split_points = np.concatenate(([-1], zero_idx, [len(b_values)]))

    blocks = []
    for i in range(len(split_points) - 1):
        block = b_values[split_points[i] + 1: split_points[i + 1]]
        if len(block) > 0:
            blocks.append(block)

    final_bvalues = [b for block in blocks for b in block]
    ind_mid       = len(blocks[0]) + len(blocks[1])

    Y, X = np.ogrid[:H, :W]
    if roi_type == 'square':
        mask = (
            (Y >= center[0] - roi_size) & (Y <= center[0] + roi_size) &
            (X >= center[1] - roi_size) & (X <= center[1] + roi_size)
        )
    else:
        mask = np.sqrt((Y - center[0])**2 + (X - center[1])**2) <= roi_size

    n_valid = np.sum(~np.isnan(noise_slice[:, :, 0][mask]))
    # 1/sqrt(32): averaging over 32 gradient directions
    # 1/sqrt(n_valid): averaging over ROI voxels
    noise_factor = 1 / np.sqrt(32 * max(n_valid, 1))
    mean_signals, std_signals, mean_noise = [], [], []
    for i in range(num_b_volumes):
        mean_signals.append(np.nanmean(signal_slice[:, :, i][mask]))
        std_signals.append(np.nanstd(signal_slice[:, :, i][mask]))
        mean_noise.append(np.nanmean(noise_slice[:, :, i][mask]) * noise_factor)

    df = pd.DataFrame({
        'b_value': final_bvalues,
        'signal':  mean_signals,
        'std':     std_signals,
        'noise':   mean_noise,
    })
    td_col = ['19ms'] * ind_mid + ['49ms'] * (len(final_bvalues) - ind_mid)
    df['TD'] = td_col
    return df
