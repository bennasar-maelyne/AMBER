# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.0
#   kernelspec:
#     display_name: .venv (3.13.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Libraries

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
from scipy.spatial import Delaunay

# %% [markdown]
# # Extraction of parameters from file

# %%
H_E=[1, 2]
Patient=[1, 3]
expansion=[3, 3.5, 4]

dir=os.getcwd()
full_path_dir=os.path.join(dir, "Qupath_project", "text_data")

for pat_id in Patient:
    cell_area_pat=0
    cell_max_cal_pat=0
    cell_min_cal_pat=0
    r2D_pat=0
    r3D_pat=0
    den=0
    for he in H_E:
        for coef in expansion:
            text_file=f"Patient_{pat_id}_H&E_{he}_exp_{coef}.txt"
            file_path=os.path.join(full_path_dir, text_file)
            if not os.path.exists(file_path):
                print(f"No file named {text_file} in {full_path_dir}")
                continue
            den+=1
            df = pd.read_csv(file_path, sep='\t', engine='python')
            cell_area=df["Cell: Area"].to_numpy()
            cell_max_cal=df["Cell: Max caliper"].to_numpy()
            cell_min_cal=df["Cell: Min caliper"].to_numpy()

            # Plot histograms for cell area, max caliper, and min caliper
            '''
            plt.figure(figsize=(15, 4))

            plt.subplot(1, 3, 1)
            plt.hist(cell_area, bins=30, color='skyblue', edgecolor='black')
            plt.title(f"Cell Area patient {pat_id} H&E {he} cell expansion of {coef} µm")
            plt.xlabel("µm²")
            plt.ylabel("Number of cells")

            plt.subplot(1, 3, 2)
            plt.hist(cell_max_cal, bins=30, color='lightgreen', edgecolor='black')
            plt.title(f"Cell Max Caliper patient {pat_id} H&E {he} cell expansion of {coef} µm")
            plt.xlabel("µm")

            plt.subplot(1, 3, 3)
            plt.hist(cell_min_cal, bins=30, color='salmon', edgecolor='black')
            plt.title(f"Cell Min Caliper patient {pat_id} H&E {he} cell expansion of {coef} µm")
            plt.xlabel("µm")

            plt.tight_layout()
            plt.show()
            '''

            mean_cell_area=np.mean(cell_area)
            mean_cell_max_cal=np.mean(cell_max_cal)
            mean_cell_min_cal=np.mean(cell_min_cal)

            std_cell_area=np.std(cell_area)
            std_cell_max_cal=np.std(cell_max_cal)
            std_cell_min_cal=np.std(cell_min_cal)

            r2D=np.sqrt(mean_cell_min_cal*mean_cell_max_cal/4)
            std_r2D = 0.25 * np.sqrt((mean_cell_max_cal / mean_cell_min_cal) * std_cell_min_cal**2+ (mean_cell_min_cal / mean_cell_max_cal) * std_cell_max_cal**2)
            r3D=1.27*r2D
            std_r3D=1.27*std_r2D

            print("---------Statistics---------")
            print(f"Mean cell area = {mean_cell_area:.2f} µm²")
            print(f"Mean max caliper = {mean_cell_max_cal:.2f} µm")
            print(f"Mean min caliper = {mean_cell_min_cal:.2f} µm")
            print(f"Equivalent average 2D radius {r2D:.2f} µm and 2D standard deviation {std_r2D:.2f} µm")
            print(f"Equivalent average 3D radius: {r3D:.2f} µm and 3D standard deviation {std_r3D:.2f} µm")

            cell_area_pat+=mean_cell_area
            cell_max_cal_pat+=mean_cell_max_cal
            cell_min_cal_pat+=mean_cell_min_cal
            r2D_pat+=r2D
            r3D_pat+=r3D


    cell_area_pat=cell_area_pat/den
    cell_max_cal_pat=cell_max_cal_pat/den
    cell_min_cal_pat=cell_min_cal_pat/den
    r2D_pat=r2D_pat/den
    r3D_pat=r3D_pat/den

    print(f"---------Statistics {pat_id}---------")
    print(f"Mean cell area = {cell_area_pat:.2f} µm²")
    print(f"Mean max caliper = {cell_max_cal_pat:.2f} µm")
    print(f"Mean min caliper = {cell_min_cal_pat:.2f} µm")
    print(f"Equivalent average 2D radius {r2D_pat:.2f} µm")
    print(f"Equivalent average 3D radius: {r3D_pat:.2f} µm")




# %% [markdown]
# # STD

# %%
H_E=[1, 2]
Patient=[1, 3]
expansion=[3, 3.5, 4]

dir=os.getcwd()
full_path_dir=os.path.join(dir, "Qupath_project", "text_data")

for pat_id in Patient:
    all_cell_area = []
    all_cell_max_cal = []
    all_cell_min_cal = []

    for he in H_E:
        for coef in expansion:
            text_file=f"Patient_{pat_id}_H&E_{he}_exp_{coef}.txt"
            file_path=os.path.join(full_path_dir, text_file)

            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path, sep='\t', engine='python')

            all_cell_area.extend(df["Cell: Area"].to_numpy())
            all_cell_max_cal.extend(df["Cell: Max caliper"].to_numpy())
            all_cell_min_cal.extend(df["Cell: Min caliper"].to_numpy())

    all_cell_area = np.array(all_cell_area)
    all_cell_max_cal = np.array(all_cell_max_cal)
    all_cell_min_cal = np.array(all_cell_min_cal)

    mean_cell_area_pat = np.mean(all_cell_area)
    std_cell_area_pat  = np.std(all_cell_area, ddof=1)

    mean_cell_max_cal_pat = np.mean(all_cell_max_cal)
    std_cell_max_cal_pat  = np.std(all_cell_max_cal, ddof=1)

    mean_cell_min_cal_pat = np.mean(all_cell_min_cal)
    std_cell_min_cal_pat  = np.std(all_cell_min_cal, ddof=1)

    r2D_pat = np.sqrt(mean_cell_min_cal_pat * mean_cell_max_cal_pat / 4)

    std_r2D_pat = 0.25 * np.sqrt(
        (mean_cell_max_cal_pat / mean_cell_min_cal_pat) * std_cell_min_cal_pat**2 +
        (mean_cell_min_cal_pat / mean_cell_max_cal_pat) * std_cell_max_cal_pat**2
    )

    r3D_pat = 1.27 * r2D_pat
    std_r3D_pat = 1.27 * std_r2D_pat

    print(f"==== PATIENT STATISTICS {pat_id} ====")
    print(f"Mean area : {mean_cell_area_pat:.2f} ± {std_cell_area_pat:.2f} µm²")
    print(f"Mean max caliper : {mean_cell_max_cal_pat:.2f} ± {std_cell_max_cal_pat:.2f} µm")
    print(f"Mean min caliper : {mean_cell_min_cal_pat:.2f} ± {std_cell_min_cal_pat:.2f} µm")
    print(f"Radius 2D : {r2D_pat:.2f} ± {std_r2D_pat:.2f} µm")
    print(f"Radius 3D : {r3D_pat:.2f} ± {std_r3D_pat:.2f} µm")


# %% [markdown]
# # Generation of the JSON file with the parameters for each ROI

# %%
if not os.path.exists("roi_results.json"):
    H_E = [1, 2]
    Patient = [1, 3]
    expansion = [3, 3.5, 4]

    allowed_ROI = {"ROI1", "ROI2", "ROI3", "ROI4"}

    results_json = {}

    dir = os.getcwd()
    full_path_dir = os.path.join(dir, "Qupath_project", "text_data")

    for pat_id in Patient:

        results_json[f"Patient_{pat_id}"] = {}

        for he in H_E:
            results_json[f"Patient_{pat_id}"][f"H&E_{he}"] = {}

            for coef in expansion:

                results_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{coef}"] = {}

                text_file = f"Patient_{pat_id}_H&E_{he}_exp_{coef}.txt"
                file_path = os.path.join(full_path_dir, text_file)

                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path, sep='\t', engine='python')

                df = df[df["Parent"].isin(allowed_ROI)]

                for roi in sorted(df["Parent"].unique()):

                    df_roi = df[df["Parent"] == roi]

                    area = df_roi["Cell: Area"].to_numpy()
                    max_cal = df_roi["Cell: Max caliper"].to_numpy()
                    min_cal = df_roi["Cell: Min caliper"].to_numpy()

                    mean_area = np.mean(area)
                    std_area = np.std(area, ddof=1)

                    mean_max = np.mean(max_cal)
                    std_max = np.std(max_cal, ddof=1)

                    mean_min = np.mean(min_cal)
                    std_min = np.std(min_cal, ddof=1)

                    r2D = np.sqrt(mean_min * mean_max / 4)

                    std_r2D = 0.25 * np.sqrt(
                        (mean_max / mean_min) * std_min**2 +
                        (mean_min / mean_max) * std_max**2
                    )

                    # Correction factor for 3D radius
                    r3D = 1.27 * r2D
                    std_r3D = 1.27 * std_r2D

                    results_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{coef}"][roi] = {
                        "mean_area": mean_area,
                        "std_area": std_area,
                        "mean_max_caliper": mean_max,
                        "std_max_caliper": std_max,
                        "mean_min_caliper": mean_min,
                        "std_min_caliper": std_min,
                        "mean_r2D": r2D,
                        "std_r2D": std_r2D,
                        "mean_r3D": r3D,
                        "std_r3D": std_r3D
                    }

    # Saving results
    with open("roi_results.json", "w") as f:
        json.dump(results_json, f, indent=4)

    print("JSON generated : roi_results.json")

else:
    with open("roi_results.json", 'r', encoding='utf-8') as f:
        results_json = json.load(f)

    print("ROI results loaded from roi_results.json")


# %% [markdown]
# # Cell density with Abercrombie

# %%
if not os.path.exists("density_results.json"):
    slice_width=4 #4 um

    density_json={}

    for pat_id in Patient:
        print(f"----------Statistics for patient {pat_id}----------")
        density_json[f"Patient_{pat_id}"] = {}
        for he in H_E:
            print(f"=== H&E {he} ===")
            density_json[f"Patient_{pat_id}"][f"H&E_{he}"] = {}
            for exp in expansion:
                density_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]={}
                annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
                annot_path = os.path.join(full_path_dir, annot_file)
                if not os.path.exists(annot_path):
                    print(f"No file named {annot_file} in {full_path_dir}")
                    continue

                df_annot = pd.read_csv(annot_path, sep='\t', engine='python')

                # ROIs data
                df_annot_sorted = df_annot.sort_values(by="Name")
                names = df_annot_sorted["Name"].to_numpy()
                areas = df_annot_sorted["Area µm^2"].to_numpy()
                
                # Cell data
                patient_file = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
                patient_path = os.path.join(full_path_dir, patient_file)
                if not os.path.exists(patient_path):
                    print(f"No file named {patient_file} in {full_path_dir}")
                    continue

                df_patient = pd.read_csv(patient_path, sep='\t', engine='python')

                # Quantity calculations
                preliminary_file = "roi_results.json"
                with open(preliminary_file, 'r', encoding='utf-8') as f:
                    preliminary_results=json.load(f)

                for i in range(len(names)):
                    roi_name = names[i]
                    df_roi = df_patient[df_patient["Parent"] == roi_name]
                    cells_area = df_roi["Cell: Area"].to_numpy()
                    if len(cells_area) == 0:
                        print(f"ROI {roi_name} (H&E {he}, exp {exp}) : no cell found")
                        continue

                    density_2d = sum(cells_area) / areas[i]

                    n = len(df_roi)
                    r2D = preliminary_results[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name]["mean_r2D"]
                    sigma_r2D = preliminary_results[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name]["std_r2D"]

                    r3D = preliminary_results[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name]["mean_r3D"]
                    sigma_r3D = preliminary_results[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name]["std_r3D"]

                    # N_v et V_cell
                    N_v = n / ((slice_width + 2*r2D) * areas[i])
                    V_cell = 4*np.pi*(r3D**3)/3
                    density_3d = N_v * V_cell

                    print(f"patient {pat_id} | roi {roi_name:<20} | roi area: {areas[i]} | number of_cells: {n} | N_v: {N_v:.4f} cells/µm³ | V_cell: {V_cell:.2f} µm³ | 3D fraction: {density_3d:.2f}")

                    # Uncertainty propagation
                    dN_dr2D = -2 * n / ((slice_width + 2*r2D)**2 * areas[i])
                    dV_dr3D = 4 * np.pi * (r3D**2)

                    sigma_Nv = abs(dN_dr2D) * sigma_r2D
                    sigma_Vcell = abs(dV_dr3D) * sigma_r3D

                    sigma_density3D = np.sqrt((sigma_Nv * V_cell)**2 + (N_v * sigma_Vcell)**2)

                    density_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name] = {
                        "density_2d": round(density_2d, 2),
                        "density_3d": round(density_3d, 2),
                        "sigma_density_3d": round(sigma_density3D, 2)
                    }

                    print(f"{roi_name:<20} | 3D density: {density_3d:>10.2f} ± {sigma_density3D:>6.2f} | 2D density: {density_2d:>10.2f} | Expansion: {exp}")


    # Saving results JSON
    with open("density_results.json", "w") as f:
        json.dump(density_json, f, indent=4)

    print("JSON generated : density_results.json")

else:
    with open("density_results.json", 'r', encoding='utf-8') as f:
        density_json = json.load(f)

    print("Density results loaded from density_results.json")


# %% [markdown]
# # Load the LUT

# %%
import h5py as _h5py

with _h5py.File('lookup_table_val.mat', 'r') as _f:
    # params: stored as (4, n_params) in HDF5 → transpose to (n_params, 4)
    params_lut  = np.array(_f['params']).T                            # (1090, 4)

    # signals_4D: stored as (n_bvec, n_bval, n_del, n_params) in HDF5
    # → transpose to (n_params, n_del, n_bval, n_bvec) for compatibility
    _sig_raw    = np.array(_f['signals_4D'])                          # (64, 13, 2, 1090)
    signals_lut = np.transpose(_sig_raw, (3, 2, 1, 0))               # (1090, 2, 13, 64)

    # Signals are already NPar-normalised (values in [0, 1]) — no /1e5 needed

    # Rebuild sequence_lut in scipy-compatible format (nested [0][0] indexing)
    _seq   = _f['sequence']
    _bval  = np.array(_seq['bval']).flatten()    # ms/µm², no b=0  # type: ignore[index]
    _TD    = np.array(_seq['TD']).flatten()      # [19., 49.]       # type: ignore[index]
    _bvecs = np.array(_seq['bvecs']).T           # (64, 3)          # type: ignore[index]

sequence_lut = {
    'bval':  np.array([[_bval]]),    # shape (1, 1, n_bval) → [0][0] = _bval
    'TD':    np.array([[_TD]]),      # shape (1, 1, n_del)  → [0][0] = _TD
    'bvecs': np.array([[_bvecs]]),   # shape (1, 1, 64, 3)  → [0][0] = _bvecs
}

params_lut = np.array(params_lut)

# %% [markdown]
# # Checking LUT

# %%
bval = np.array(sequence_lut['bval'][0][0]).flatten()   # shape: (n_bval,)
TD   = np.array(sequence_lut['TD'][0][0]).flatten()     # shape: (n_del,)
bvecs = np.array(sequence_lut['bvecs'][0][0])           # shape: (n_bvec, 3)
print("params shape:", params_lut.shape)
print("signals shape:", signals_lut.shape)
print("bval:", bval)
print("TD:", TD[:5])
print("bvecs shape:", bvecs.shape)


# %% [markdown]
# # Delaunay interpolation

# %%
def interpolate_signal(f, Dex, rmean, rsd, bval, TD, tri, signals_lookup, params_lut):
    # Security check for zero density
    if f < 0.007:
        return 1.0

    query_params = np.array([f, Dex, rmean, rsd])
    
    # Get indices for TD and bval — nearest-neighbour match to handle
    # cases where the query value does not exactly match the LUT grid
    # (e.g. DWI b-values differ between patients / from LUT b-values).
    bvals = sequence_lut['bval'][0][0].flatten()
    TDs   = sequence_lut['TD'][0][0].flatten()
    TD_index   = int(np.argmin(np.abs(TDs   - TD)))
    bval_index = int(np.argmin(np.abs(bvals - bval)))

    # Find the simplex containing the query point
    simplex_index = tri.find_simplex(query_params)

    # OUT OF HULL : Fallback on nearest neighbor
    if simplex_index == -1:
        distances = np.linalg.norm(params_lut - query_params, axis=1)
        nearest_idx = np.argmin(distances)
        signal_nearest = signals_lookup[nearest_idx]
        return np.mean(signal_nearest[TD_index, bval_index, :])
    # IN HULL : Barycentric interpolation
    vertex_index = tri.simplices[simplex_index]
    vertices = params_lut[vertex_index]

    try:
        T = vertices[1:] - vertices[0]
        v = query_params - vertices[0]
        bary_coords = np.linalg.solve(T.T, v)
        bary_coords = np.append(1 - np.sum(bary_coords), bary_coords)
        
        signals_subset = signals_lookup[vertex_index]
        interpolated_signal = np.tensordot(bary_coords, signals_subset, axes=(0, 0))
        return np.mean(interpolated_signal[TD_index, bval_index, :])
    except:
        # Fallback to nearest neighbor in case of numerical issues
        distances = np.linalg.norm(vertices - query_params, axis=1)
        nearest_v_idx = vertex_index[np.argmin(distances)]
        return np.mean(signals_lookup[nearest_v_idx][TD_index, bval_index, :])
# %% [markdown]
# # Dex inference via chi² grid

# %%
def get_signal_curve_lut(f, Dex, rmean, rsd, TD, tri, signals_lookup, params_lut):
    """
    Return the predicted signal at ALL LUT b-values for a given (f, Dex, rmean, rsd, TD).
    Used to allow b-value interpolation when DWI and LUT b-grids don't match.

    Returns
    -------
    bvals_lut : 1-D array, LUT b-values in ms/µm²
    signals   : 1-D array, predicted S/S0 at each LUT b-value
    """
    bvals_lut = np.array(sequence_lut['bval'][0][0]).flatten()
    TDs       = np.array(sequence_lut['TD'][0][0]).flatten()
    matches = np.where(TDs == TD)[0]
    if len(matches) == 0:
        raise ValueError(f"TD={TD} ms not found in LUT. Available TDs: {TDs.tolist()}")
    TD_index = matches[0]

    if f < 0.007:
        # Free diffusion (extracellular only): S(b) = exp(-b * Dex)
        return bvals_lut, np.exp(-bvals_lut * Dex)

    query_params  = np.array([f, Dex, rmean, rsd])
    simplex_index = tri.find_simplex(query_params)

    if simplex_index == -1:
        # Nearest-neighbour fallback
        nearest_idx = np.argmin(np.linalg.norm(params_lut - query_params, axis=1))
        signals = np.mean(signals_lookup[nearest_idx][TD_index, :, :], axis=-1)
    else:
        vertex_index = tri.simplices[simplex_index]
        vertices     = params_lut[vertex_index]
        try:
            T = vertices[1:] - vertices[0]
            v = query_params - vertices[0]
            bary_coords = np.linalg.solve(T.T, v)
            bary_coords = np.append(1 - np.sum(bary_coords), bary_coords)
            interpolated = np.tensordot(bary_coords, signals_lookup[vertex_index], axes=(0, 0))
            signals = np.mean(interpolated[TD_index, :, :], axis=-1)
        except Exception:
            nearest_v_idx = vertex_index[np.argmin(np.linalg.norm(vertices - query_params, axis=1))]
            signals = np.mean(signals_lookup[nearest_v_idx][TD_index, :, :], axis=-1)
    return bvals_lut, signals


def _ci_bounds_interp(grid, profile, best_value, delta, warn_label=None):
    """
    CI bounds where the chi²/NLL profile crosses best_value + delta, refined
    by linear interpolation between the two grid points bracketing each
    crossing (rather than snapping to the nearest grid point).

    Without this, a CI is reported as a single grid point whenever the
    profile is steep enough that already the first neighbouring point
    exceeds the threshold — an artifact of grid resolution, not a real
    zero-width interval. Interpolating recovers the true sub-grid crossing.

    If a crossing falls outside the grid (CI touches the edge), the bound
    is clamped to the grid edge and a warning is printed, since that means
    the grid does not fully bracket the CI and should be widened.

    Parameters
    ----------
    grid        : 1-D array of parameter candidates, ascending
    profile     : chi²/NLL values at each grid point (same shape as grid)
    best_value  : chi²/NLL at the minimum (profile.min())
    delta       : threshold offset (e.g. Δχ²=1 for 68%, 3.84 for 1-param 95%)
    warn_label  : optional string included in the edge-touching warning

    Returns
    -------
    (ci_low, ci_high) : tuple of floats
    """
    grid    = np.asarray(grid)
    profile = np.asarray(profile)
    threshold = best_value + delta
    best_idx  = int(np.nanargmin(profile))

    below = profile <= threshold
    if not np.any(below):
        return float(grid[best_idx]), float(grid[best_idx])

    idx_in = np.where(below)[0]
    i_lo, i_hi = int(idx_in[0]), int(idx_in[-1])

    if i_lo > 0 and np.isfinite(profile[i_lo - 1]):
        x0, x1 = grid[i_lo - 1], grid[i_lo]
        y0, y1 = profile[i_lo - 1], profile[i_lo]
        ci_low = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    else:
        ci_low = float(grid[i_lo])
        tag = f" ({warn_label})" if warn_label else ""
        print(f"  [warn] CI lower bound touches grid edge {grid[0]:.3f}{tag} "
              f"— widen the grid.")

    if i_hi < len(grid) - 1 and np.isfinite(profile[i_hi + 1]):
        x0, x1 = grid[i_hi], grid[i_hi + 1]
        y0, y1 = profile[i_hi], profile[i_hi + 1]
        ci_high = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    else:
        ci_high = float(grid[i_hi])
        tag = f" ({warn_label})" if warn_label else ""
        print(f"  [warn] CI upper bound touches grid edge {grid[-1]:.3f}{tag} "
              f"— widen the grid.")

    return float(ci_low), float(ci_high)

def infer_dex_chi2(df_dwi, cell_areas, max_calipers, min_calipers, roi_area_um2,
                   Dex_grid, TD, tri, signals_lut, params_lut,
                   slice_width_um=4, bval_max_um2=7.5):
    """
    Infer Dex by chi² minimisation on a grid.

    The LUT and DWI b-values do not need to match: the LUT signal curve is
    interpolated (np.interp, log-linear) to the DWI b-values automatically.
    DWI points outside the LUT b-value range are discarded with a warning.

    Parameters
    ----------
    df_dwi        : DataFrame ['b_value' (s/mm²), 'signal', 'noise', 'TD']
                    as returned by extract_patient_curves
    cell_areas, max_calipers, min_calipers : QuPath cell arrays for one ROI
    roi_area_um2  : ROI area in µm²
    Dex_grid      : 1-D array of Dex candidates (µm²/ms)
    TD            : diffusion time in ms (19 or 49)
    tri           : Delaunay triangulation of params_lut
    signals_lut, params_lut : LUT arrays
    slice_width_um : histological slice thickness (default 4 µm)
    bval_max_um2  : max b-value to use, in ms/µm² (default 7.5)

    Returns
    -------
    dict with keys:
        Dex_best, Dex_ci_low, Dex_ci_high  – inferred value and 68% CI (Δχ²=1)
        chi2_grid, Dex_grid                 – full chi² profile
        chi2_min, f_mean, r3D_mean, r3D_std – intermediate quantities
        bvals_used                           – DWI b-values actually used (ms/µm²)
    """
    # --- 1. Histological point estimate (f, r) ---
    mean_max = np.mean(max_calipers)
    mean_min = np.mean(min_calipers)
    std_max  = np.std(max_calipers, ddof=1)
    std_min  = np.std(min_calipers, ddof=1)

    r2D     = np.sqrt(mean_max * mean_min / 4)
    std_r2D = 0.25 * np.sqrt(
        (mean_max / mean_min) * std_min**2 +
        (mean_min / mean_max) * std_max**2
    )
    r3D     = 1.27 * r2D
    std_r3D = 1.27 * std_r2D

    n_cells = len(cell_areas)
    N_v     = n_cells / ((slice_width_um + 2 * r2D) * roi_area_um2)
    f_mean  = N_v * (4/3) * np.pi * r3D**3

    # --- 2. Measured DWI points for this TD ---
    bvals_lut_um2 = np.array(sequence_lut['bval'][0][0]).flatten()
    bval_lut_min, bval_lut_max = bvals_lut_um2.min(), bvals_lut_um2.max()

    td_str = f"{TD}ms"
    df_td  = (df_dwi[df_dwi["TD"] == td_str]
              .copy()
              .assign(bval_um2=lambda d: d["b_value"] / 1000)
              .query("bval_um2 > 0 and bval_um2 <= @bval_max_um2")
              .sort_values("bval_um2"))

    # Discard DWI points outside the LUT b-value range
    out_of_range = df_td["bval_um2"] < bval_lut_min
    if out_of_range.any():
        print(f"[infer_dex_chi2] Warning: {out_of_range.sum()} DWI b-value(s) below LUT "
              f"range [{bval_lut_min:.3f}, {bval_lut_max:.3f}] ms/µm² — discarded.")
        df_td = df_td[~out_of_range]

    bvals_um2 = df_td["bval_um2"].to_numpy()
    S_meas    = df_td["signal"].to_numpy()
    sigma     = df_td["noise"].to_numpy()
    sigma     = np.where(sigma < 1e-6, 1e-6, sigma)

    if len(bvals_um2) == 0:
        raise ValueError("No usable DWI b-values after filtering — check bval_max_um2 and LUT range.")

    # --- 3. Chi² profile (with LUT → DWI b-value interpolation) ---
    chi2_vals = np.zeros(len(Dex_grid))
    for k, Dex in enumerate(Dex_grid):
        bvals_curve, sig_curve = get_signal_curve_lut(
            f_mean, Dex, r3D, std_r3D, TD, tri, signals_lut, params_lut)
        # Log-linear interpolation for smoother decay curves
        log_sig_curve = np.log(np.clip(sig_curve, 1e-10, None))
        log_S_pred    = np.interp(bvals_um2, bvals_curve, log_sig_curve)
        S_pred        = np.exp(log_S_pred)
        chi2_vals[k]  = np.sum(((S_meas - S_pred) / sigma) ** 2)

    # --- 4. Best estimate and 68% CI (Δχ² = 1, one free parameter) ---
    best_idx  = np.argmin(chi2_vals)
    chi2_min  = chi2_vals[best_idx]
    Dex_best  = Dex_grid[best_idx]

    Dex_ci_low, Dex_ci_high = _ci_bounds_interp(
        Dex_grid, chi2_vals, chi2_min, 1.0, warn_label=f"infer_dex_chi2 TD={TD}")

    return {
        "Dex_best":    float(Dex_best),
        "Dex_ci_low":  Dex_ci_low,
        "Dex_ci_high": Dex_ci_high,
        "chi2_grid":   chi2_vals.tolist(),
        "Dex_grid":    Dex_grid.tolist(),
        "chi2_min":    float(chi2_min),
        "f_mean":      float(f_mean),
        "r3D_mean":    float(r3D),
        "r3D_std":     float(std_r3D),
        "bvals_used":  bvals_um2.tolist(),
    }

def plot_dex_inference(result, ax=None, title=""):
    """Display the chi² profile with the best Dex and its 68% CI."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    Dex_grid    = np.array(result["Dex_grid"])
    chi2        = np.array(result["chi2_grid"])
    chi2_min    = result["chi2_min"]
    Dex_best    = result["Dex_best"]
    Dex_ci_low  = result["Dex_ci_low"]
    Dex_ci_high = result["Dex_ci_high"]

    ax.plot(Dex_grid, chi2, "k-", linewidth=1.5, label=r"$\chi^2(D_{ex})$")
    ax.axhline(chi2_min + 1, color="gray", linestyle="--", alpha=0.7,
               label=r"$\chi^2_{\min} + 1$")
    ax.axvline(Dex_best, color="tab:red", linestyle="-",
               label=f"$D_{{ex}}^*$ = {Dex_best:.2f} µm²/ms")
    ax.axvspan(Dex_ci_low, Dex_ci_high, alpha=0.15, color="tab:red",
               label=f"CI 68% [{Dex_ci_low:.2f}, {Dex_ci_high:.2f}] µm²/ms")

    ax.set_xlabel(r"$D_{ex}$ (µm²/ms)")
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title(title or "Dex inference — chi² grid")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# %%
# ── Load DWI data (mirrored from combined.ipynb) ────────────────────────────

def extract_patient_curves(data_dict, slice_idx, center, roi_size, roi_type='square'):
    """Extraction of the DWI curves for a given patient and slice, with noise envelope."""
    full_brain  = np.transpose(data_dict["fullbrain_all"], (0, 2, 1, 3))
    signal_slice = full_brain[:, :, slice_idx, :]

    noise_data  = np.transpose(data_dict["noisemap_all"], (0, 2, 1, 3))
    noise_slice = noise_data[:, :, slice_idx, :]

    H, W, num_b_volumes = signal_slice.shape

    b_values     = data_dict["bv_all"].flatten()
    zero_idx     = np.where(b_values == 0)[0]
    split_points = np.concatenate(([-1], zero_idx, [len(b_values)]))

    blocks = []
    for i in range(len(split_points) - 1):
        block = b_values[split_points[i] + 1 : split_points[i + 1]]
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

    noise_factor = 1 / np.sqrt(32)
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


# Load .mat files
from scipy.io import loadmat as _loadmat
data_p1 = _loadmat("./Patient_1/Combined/CON_0101SP_12072016_tumor_all.mat")
data_p3 = _loadmat("./Patient_3/Combined/CON_03_V01_tumor_all.mat")

# Extract DWI curves (same ROI centres as combined.ipynb)
df_p1 = extract_patient_curves(data_p1, slice_idx=36, center=(49, 49), roi_size=3, roi_type='square')
df_p3 = extract_patient_curves(data_p3, slice_idx=31, center=(73, 28), roi_size=3, roi_type='square')

print("df_p1:", df_p1.shape, "| TDs:", df_p1['TD'].unique())
print("df_p3:", df_p3.shape, "| TDs:", df_p3['TD'].unique())

# %%
# ── Run Dex inference on one ROI ────────────────────────────────────────────

tri      = Delaunay(params_lut)
Dex_grid = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms

# Parameters to change as needed
pat_id   = 3
he       = 1
exp      = 3
roi_name = "ROI3"
TD_val   = 19
df_dwi   = df_p3    # df_p1 for Patient 1, df_p3 for Patient 3

cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
df_cells   = pd.read_csv(os.path.join(full_path_dir, cell_file),  sep='\t', engine='python')
df_annot   = pd.read_csv(os.path.join(full_path_dir, annot_file), sep='\t', engine='python')

df_roi = df_cells[df_cells["Parent"] == roi_name]
_area_vals = df_annot.loc[df_annot["Name"] == roi_name, "Area µm^2"].to_numpy()
if len(_area_vals) == 0:
    raise ValueError(f"ROI '{roi_name}' not found in annotation file.")
roi_area_um2 = _area_vals[0]

result = infer_dex_chi2(
    df_dwi       = df_dwi,
    cell_areas   = df_roi["Cell: Area"].to_numpy(),
    max_calipers = df_roi["Cell: Max caliper"].to_numpy(),
    min_calipers = df_roi["Cell: Min caliper"].to_numpy(),
    roi_area_um2 = roi_area_um2,
    Dex_grid     = Dex_grid,
    TD           = TD_val,
    tri          = tri,
    signals_lut  = signals_lut,
    params_lut   = params_lut,
)

print(f"Dex_best  = {result['Dex_best']:.3f} µm²/ms")
print(f"CI 68%    = [{result['Dex_ci_low']:.3f}, {result['Dex_ci_high']:.3f}] µm²/ms")
print(f"f_mean    = {result['f_mean']:.4f}")
print(f"r3D_mean  = {result['r3D_mean']:.2f} µm")

plot_dex_inference(result, title=f"Patient {pat_id} | {roi_name} | TD={TD_val} ms")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Infer Dex shrinkage

# %%
def infer_Dex_shrinkage_chi2(df_dwi, cell_areas, max_calipers, min_calipers,
                              roi_area_um2, TD, tri, signals_lut, params_lut,
                              slice_width_um=4, bval_max_um2=7.5,
                              Dex_grid=None, k_det_grid=None, k_r=1.0,
                              use_rician=True):
    """
    Joint inference of Dex and k_det via a 2D grid (chi² or Rician NLL).

    Two separate corrections are applied:

    1. k_r  — geometric shrinkage (FIXED, not inferred):
       r_vivo = r_hist / k_r
       Accounts for the physical shrinkage of tissue during fixation.
       Applied once before the grid loop. Default k_r=0.75 (Reynaud 2017).

    2. k_det — detection calibration factor (INFERRED):
       f_corrected = f_vivo * k_det,  k_det ∈ ]0, 1]
       Empirical correction for QuPath over-segmentation and the expansion
       coefficient inflating apparent cell size. This is NOT a geometric
       shrinkage — it calibrates the effective volume fraction to a
       physiologically realistic range (~0.05–0.30 for glioma).

    The LUT signal is log-linearly interpolated to DWI b-values.

    Parameters
    ----------
    df_dwi          : DataFrame ['b_value' (s/mm²), 'signal', 'noise', 'TD']
    cell_areas, max_calipers, min_calipers : QuPath arrays for one ROI
    roi_area_um2    : ROI area in µm²
    TD              : diffusion time in ms (19 or 49)
    slice_width_um  : slice thickness (µm, default 4)
    bval_max_um2    : max b-value used (ms/µm², default 7.5)
    Dex_grid        : Dex candidates (µm²/ms), default linspace(1.0, 3.0, 400)
    k_det_grid      : k_det candidates, default linspace(0.05, 1.0, 40)
    k_r             : fixed geometric shrinkage factor, default 0.75
    use_rician      : if True, minimise Rician negative log-likelihood instead
                      of chi² (rigorous noise model for magnitude MRI).
                      The 95% CI threshold is still derived from the likelihood
                      ratio statistic: -2·ΔNLL ~ χ²(2) → threshold = 5.99/2.

    Returns
    -------
    dict with: Dex_best, k_det_best, Dex_ci, k_det_ci,
               loss_map (chi2_map or rician_nll_map), loss_name,
               Dex_grid, k_det_grid, k_r, use_rician,
               f_raw, f_vivo, f_corrected,
               r3D_hist, r3D_vivo, r3D_std_vivo, bvals_used
    """
    from scipy.stats import chi2 as chi2_dist
    from scipy.special import i0e as bessel_i0e

    if Dex_grid is None:
        Dex_grid = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms
    if k_det_grid is None:
        k_det_grid = np.linspace(0.05, 1.0, 40)

    # --- 1. Raw histological parameters (QuPath, no correction) ---
    mean_max = np.mean(max_calipers)
    mean_min = np.mean(min_calipers)
    std_max  = np.std(max_calipers, ddof=1)
    std_min  = np.std(min_calipers, ddof=1)

    r2D_hist = np.sqrt(mean_max * mean_min / 4)
    std_r2D_hist = 0.25 * np.sqrt(
        (mean_max / mean_min) * std_min**2 +
        (mean_min / mean_max) * std_max**2
    )
    n_cells = len(cell_areas)

    # f_raw: Abercrombie at k_r=1, for reporting only
    r3D_hist = 1.27 * r2D_hist
    N_v_hist = n_cells / ((slice_width_um + 2 * r2D_hist) * roi_area_um2)
    f_raw    = N_v_hist * (4/3) * np.pi * r3D_hist**3

    # --- 2. Apply fixed geometric shrinkage k_r on r (computed once) ---
    r2D_vivo     = r2D_hist     / k_r
    std_r2D_vivo = std_r2D_hist / k_r
    r3D_vivo     = 1.27 * r2D_vivo
    std_r3D_vivo = 1.27 * std_r2D_vivo
    N_v_vivo     = n_cells / ((slice_width_um + 2 * r2D_vivo) * roi_area_um2)
    f_vivo       = N_v_vivo * (4/3) * np.pi * r3D_vivo**3  # ≈ f_raw (conserved)

    print(f"  [infer_Dex_shrinkage_chi2] k_r={k_r} fixed | "
          f"r3D_hist={r3D_hist:.2f} µm → r3D_vivo={r3D_vivo:.2f} µm | "
          f"f_raw={f_raw:.3f} → f_vivo={f_vivo:.3f}")

    # --- 3. Measured DWI for this TD ---
    bvals_lut_um2 = np.array(sequence_lut['bval'][0][0]).flatten()
    bval_lut_min  = bvals_lut_um2.min()

    td_str = f"{TD}ms"
    df_td  = (df_dwi[df_dwi["TD"] == td_str]
              .copy()
              .assign(bval_um2=lambda d: d["b_value"] / 1000)
              .query("bval_um2 > 0 and bval_um2 <= @bval_max_um2")
              .sort_values("bval_um2"))

    out_of_range = df_td["bval_um2"] < bval_lut_min
    if out_of_range.any():
        print(f"  [infer_Dex_shrinkage_chi2] {out_of_range.sum()} point(s) outside LUT range — discarded.")
        df_td = df_td[~out_of_range]

    bvals_um2 = df_td["bval_um2"].to_numpy()
    S_meas    = df_td["signal"].to_numpy()
    sigma     = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6, df_td["noise"].to_numpy())
    n_pts     = len(bvals_um2)

    # --- 4. 2D grid over (k_det, Dex) — chi² or Rician NLL ---
    # k_det calibrates f_vivo to account for QuPath over-segmentation
    # (expansion coefficient inflating apparent cell size).
    # r3D_vivo and std_r3D_vivo are fixed (k_r already applied above).
    loss_map = np.full((len(k_det_grid), len(Dex_grid)), np.nan)
    loss_name = "Rician NLL" if use_rician else "χ²"

    for i, k_det in enumerate(k_det_grid):
        f_corrected = f_vivo * k_det   # effective volume fraction fed to LUT

        for j, Dex in enumerate(Dex_grid):
            bvals_curve, sig_curve = get_signal_curve_lut(
                f_corrected, Dex, r3D_vivo, std_r3D_vivo, TD, tri, signals_lut, params_lut)
            log_sig = np.log(np.clip(sig_curve, 1e-10, None))
            S_pred  = np.exp(np.interp(bvals_um2, bvals_curve, log_sig))
            if np.any(np.isnan(S_pred)):
                continue

            if use_rician:
                # Rician NLL: -log p(S_meas | S_pred, sigma) for magnitude MRI
                # p(s|ν,σ) = (s/σ²) · I₀(sν/σ²) · exp(-(s²+ν²)/(2σ²))
                # Using i0e(x) = I₀(x)·exp(-x) for numerical stability:
                # -log p = (S²+ν²)/(2σ²) - log(i0e(Sν/σ²)) - Sν/σ²
                x = S_meas * S_pred / sigma**2
                loss_map[i, j] = float(np.sum(
                    (S_meas**2 + S_pred**2) / (2 * sigma**2)
                    - np.log(np.clip(bessel_i0e(x), 1e-300, None))
                    - x
                ))
            else:
                loss_map[i, j] = np.sum(((S_meas - S_pred) / sigma) ** 2)

    # --- 5. Global minimum and joint 95% CI ---
    flat_idx            = np.nanargmin(loss_map)
    i_best, j_best      = np.unravel_index(flat_idx, loss_map.shape)
    k_det_best          = k_det_grid[i_best]
    Dex_best            = Dex_grid[j_best]
    loss_min            = loss_map[i_best, j_best]
    f_corrected_best    = float(f_vivo * k_det_best)

    # CI threshold: Δχ²=5.99 for chi², ΔNLL=5.99/2=3.0 for Rician
    # (LR statistic -2·ΔNLL ~ χ²(2) → same 95% percentile)
    chi2_threshold = chi2_dist.ppf(0.95, df=2)          # ≈ 5.99
    delta = chi2_threshold / 2 if use_rician else chi2_threshold

    # Marginal profiles (min over the other axis), used both for the CI and
    # for the marginal-profile plot below.
    loss_dex   = np.nanmin(loss_map, axis=0)
    loss_k_det = np.nanmin(loss_map, axis=1)

    Dex_ci   = _ci_bounds_interp(Dex_grid, loss_dex, loss_min, delta,
                                  warn_label="infer_Dex_shrinkage_chi2 Dex")
    k_det_ci = _ci_bounds_interp(k_det_grid, loss_k_det, loss_min, delta,
                                  warn_label="infer_Dex_shrinkage_chi2 k_det")

    # --- 6. Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.contourf(Dex_grid, k_det_grid, loss_map, levels=30, cmap='viridis_r')
    ax.contour(Dex_grid, k_det_grid, loss_map,
               levels=[loss_min + delta], colors='red', linewidths=1.5, linestyles='--')
    ax.scatter(Dex_best, k_det_best, color='red', s=80, zorder=5,
               label=f'Min: Dex={Dex_best:.2f}, k_det={k_det_best:.2f}')
    plt.colorbar(im, ax=ax, label=loss_name)
    ax.set_xlabel('Dex (µm²/ms)')
    ax.set_ylabel('Detection factor k_det')
    ax.set_title(f'{loss_name}(Dex, k_det) — k_r={k_r} fixed\nRed contour = joint 95% CI')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Dex_grid,   loss_dex   - loss_min, color='tab:blue',   lw=2, label='Dex profile (min over k_det)')
    ax2 = ax.twinx()
    ax2.plot(k_det_grid, loss_k_det - loss_min, color='tab:orange', lw=2, label='k_det profile (min over Dex)')
    ax.axhline(delta, color='red', ls='--', lw=1, label=f'95% CI threshold = {delta:.2f}')
    ax.set_xlabel('Dex (µm²/ms)  /  k_det')
    ylabel_loss = "ΔNLL" if use_rician else "Δχ²"
    ax.set_ylabel(f'{ylabel_loss} (Dex profile)', color='tab:blue')
    ax2.set_ylabel(f'{ylabel_loss} (k_det profile)', color='tab:orange')
    ax.set_title(f'Marginal {ylabel_loss} profiles')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    method_label = "Rician NLL" if use_rician else "χ²"
    plt.suptitle(
        f'Joint Dex / k_det inference [{method_label}]  (k_r={k_r} fixed)\n'
        f'Dex = {Dex_best:.3f} [{Dex_ci[0]:.3f}, {Dex_ci[1]:.3f}] µm²/ms  |  '
        f'k_det = {k_det_best:.3f} [{k_det_ci[0]:.3f}, {k_det_ci[1]:.3f}]  |  '
        f'f_eff = {f_corrected_best:.3f}',
        fontsize=11)
    plt.tight_layout()
    plt.show()

    if use_rician:
        print(f"  NLL_min = {loss_min:.3f}  (N={n_pts} points, Rician likelihood)")
    else:
        print(f"  χ²_min / (N-2) = {loss_min / max(n_pts - 2, 1):.3f}  "
              f"(N={n_pts} points, 2 free parameters)")
    print(f"  Dex_best={Dex_best:.3f} µm²/ms | k_det_best={k_det_best:.3f} | "
          f"f_eff={f_corrected_best:.3f} (f_vivo={f_vivo:.3f})")

    return {
        "Dex_best":       float(Dex_best),
        "k_det_best":     float(k_det_best),
        "Dex_ci":         Dex_ci,
        "k_det_ci":       k_det_ci,
        "loss_map":       loss_map,
        "chi2_map":       loss_map,   # alias for backwards compatibility
        "loss_name":      loss_name,
        "use_rician":     use_rician,
        "Dex_grid":       Dex_grid,
        "k_det_grid":     k_det_grid,
        "k_r":            float(k_r),
        "loss_min":       float(loss_min),
        "chi2_min":       float(loss_min),  # alias for backwards compatibility
        "f_raw":          float(f_raw),             # Abercrombie, no correction
        "f_vivo":         float(f_vivo),            # after k_r on r (≈ f_raw)
        "f_corrected":    f_corrected_best,         # f_vivo * k_det_best
        "r3D_hist":       float(r3D_hist),
        "r3D_vivo":       float(r3D_vivo),
        "r3D_std_vivo":   float(std_r3D_vivo),
        "bvals_used":     bvals_um2.tolist(),
    }


# %%
def infer_joint_kdet(df_dwi, cell_areas, max_calipers, min_calipers,
                     roi_area_um2, tri, signals_lut, params_lut,
                     slice_width_um=4, bval_max_um2=7.5,
                     Dex_grid=None, k_det_grid=None, k_r=1.0,
                     save_dir=None, fig_label=None):
    """
    Joint inference of (k_det, Dex_19, Dex_49) with a SHARED k_det.

    Minimises:
        chi2_total(k_det, Dex_19, Dex_49) = chi2(TD=19, Dex_19, k_det)
                                           + chi2(TD=49, Dex_49, k_det)

    Because Dex_19 and Dex_49 are independent given k_det, the optimisation
    factorises:
        For each k_det:
            loss_19(k_det) = min_{Dex_19} chi2(TD=19, Dex_19, k_det)
            loss_49(k_det) = min_{Dex_49} chi2(TD=49, Dex_49, k_det)
            total(k_det)   = loss_19(k_det) + loss_49(k_det)
        k_det_best = argmin total(k_det)

    This reduces the grid from n_k × n_D² to n_k × 2×n_D evaluations.

    CI uses the joint 3-parameter threshold: Δχ²(3, 0.95) = 7.815.
    Marginal CIs for each parameter are obtained by projecting the joint
    3D CI region onto each axis.

    Parameters
    ----------
    df_dwi         : DataFrame ['b_value' (s/mm²), 'signal', 'noise', 'TD']
                     Must contain rows for both TD='19ms' and TD='49ms'.
    cell_areas, max_calipers, min_calipers : QuPath arrays for one ROI
    roi_area_um2   : ROI area in µm²
    tri            : Delaunay triangulation of params_lut
    signals_lut    : LUT signal array
    params_lut     : LUT parameter array
    slice_width_um : slice thickness (µm, default 4)
    bval_max_um2   : max b-value used (ms/µm², default 7.5)
    Dex_grid       : Dex candidates (µm²/ms), default linspace(1.0, 3.0, 400)
    k_det_grid     : k_det candidates, default linspace(0.05, 1.0, 50)
    k_r            : fixed geometric shrinkage factor, default 1.0

    Returns
    -------
    dict with: k_det_best, Dex_19_best, Dex_49_best,
               k_det_ci, Dex_19_ci, Dex_49_ci,
               chi2_min, chi2_19_min, chi2_49_min,
               f_vivo, f_eff, r3D_vivo, r3D_std_vivo,
               k_det_grid, Dex_grid,
               loss_19_per_k (min chi2_19 for each k_det),
               loss_49_per_k (min chi2_49 for each k_det)
    """
    from scipy.stats import chi2 as chi2_dist

    if Dex_grid is None:
        Dex_grid = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms
    if k_det_grid is None:
        k_det_grid = np.linspace(0.05, 1.0, 50)

    # --- 1. Histological parameters ---
    mean_max = np.mean(max_calipers)
    mean_min = np.mean(min_calipers)
    std_max  = np.std(max_calipers, ddof=1)
    std_min  = np.std(min_calipers, ddof=1)

    r2D_hist     = np.sqrt(mean_max * mean_min / 4)
    std_r2D_hist = 0.25 * np.sqrt(
        (mean_max / mean_min) * std_min**2 +
        (mean_min / mean_max) * std_max**2
    )
    r2D_vivo     = r2D_hist     / k_r
    std_r2D_vivo = std_r2D_hist / k_r
    r3D_vivo     = 1.27 * r2D_vivo
    std_r3D_vivo = 1.27 * std_r2D_vivo

    n_cells  = len(cell_areas)
    N_v_vivo = n_cells / ((slice_width_um + 2 * r2D_vivo) * roi_area_um2)
    f_vivo   = N_v_vivo * (4/3) * np.pi * r3D_vivo**3

    print(f"  [infer_joint_kdet] k_r={k_r} | r3D_vivo={r3D_vivo:.2f} µm | f_vivo={f_vivo:.3f}")

    # --- 2. DWI data for each TD ---
    bvals_lut_um2 = np.array(sequence_lut['bval'][0][0]).flatten()
    bval_lut_min  = bvals_lut_um2.min()

    def _get_dwi(TD):
        td_str = f"{TD}ms"
        df_td  = df_dwi[df_dwi["TD"] == td_str].copy()
        df_td["bval_um2"] = df_td["b_value"] / 1000
        df_td  = df_td[(df_td["bval_um2"] > 0) &
                       (df_td["bval_um2"] <= bval_max_um2) &
                       (df_td["bval_um2"] >= bval_lut_min)].sort_values("bval_um2")
        bv     = df_td["bval_um2"].to_numpy()
        sm     = df_td["signal"].to_numpy()
        sig    = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6,
                          df_td["noise"].to_numpy())
        return bv, sm, sig

    bv_19, S_19, sigma_19 = _get_dwi(19)
    bv_49, S_49, sigma_49 = _get_dwi(49)

    if len(bv_19) < 3 or len(bv_49) < 3:
        raise ValueError("Not enough DWI points for one or both TDs.")

    # --- 3. Precompute chi2 maps per TD ---
    # chi2_map_19[i, j] = chi2 for k_det[i], Dex[j], TD=19
    # chi2_map_49[i, j] = chi2 for k_det[i], Dex[j], TD=49
    chi2_map_19 = np.full((len(k_det_grid), len(Dex_grid)), np.nan)
    chi2_map_49 = np.full((len(k_det_grid), len(Dex_grid)), np.nan)

    for i, k_det in enumerate(k_det_grid):
        f_eff = f_vivo * k_det
        for j, Dex in enumerate(Dex_grid):
            bv_c, sc = get_signal_curve_lut(
                f_eff, Dex, r3D_vivo, std_r3D_vivo, 19, tri, signals_lut, params_lut)
            log_sc = np.log(np.clip(sc, 1e-10, None))
            sp_19  = np.exp(np.interp(bv_19, bv_c, log_sc))
            if not np.any(np.isnan(sp_19)):
                chi2_map_19[i, j] = np.sum(((S_19 - sp_19) / sigma_19) ** 2)

            bv_c, sc = get_signal_curve_lut(
                f_eff, Dex, r3D_vivo, std_r3D_vivo, 49, tri, signals_lut, params_lut)
            log_sc = np.log(np.clip(sc, 1e-10, None))
            sp_49  = np.exp(np.interp(bv_49, bv_c, log_sc))
            if not np.any(np.isnan(sp_49)):
                chi2_map_49[i, j] = np.sum(((S_49 - sp_49) / sigma_49) ** 2)

    # --- 4. Factorised optimisation ---
    # For each k_det: best Dex_19 and Dex_49 are independent
    loss_19_per_k = np.nanmin(chi2_map_19, axis=1)   # shape (n_k,)
    loss_49_per_k = np.nanmin(chi2_map_49, axis=1)   # shape (n_k,)
    total_per_k   = loss_19_per_k + loss_49_per_k

    i_best       = int(np.nanargmin(total_per_k))
    k_det_best   = k_det_grid[i_best]
    j_best_19    = int(np.nanargmin(chi2_map_19[i_best]))
    j_best_49    = int(np.nanargmin(chi2_map_49[i_best]))
    Dex_19_best  = Dex_grid[j_best_19]
    Dex_49_best  = Dex_grid[j_best_49]
    chi2_19_min  = float(chi2_map_19[i_best, j_best_19])
    chi2_49_min  = float(chi2_map_49[i_best, j_best_49])
    chi2_total   = chi2_19_min + chi2_49_min
    f_eff_best   = float(f_vivo * k_det_best)

    # --- 5. Joint CI (3 parameters → Δχ²(3, 0.95) = 7.815) ---
    # 3D CI mask: (k_det, Dex_19, Dex_49) where
    #   chi2_19[k,j19] + chi2_49[k,j49] < chi2_total + delta
    delta_3 = chi2_dist.ppf(0.95, df=3)   # ≈ 7.815

    # Marginal k_det CI: project 3D CI onto k_det axis
    # total_per_k is the minimum total loss at each k_det
    k_det_ci = _ci_bounds_interp(k_det_grid, total_per_k, chi2_total, delta_3,
                                  warn_label=f"{fig_label or ''} k_det")

    # Marginal Dex_19 / Dex_49 CI at best k_det
    Dex_19_ci = _ci_bounds_interp(Dex_grid, chi2_map_19[i_best], chi2_19_min, delta_3,
                                   warn_label=f"{fig_label or ''} Dex_19")
    Dex_49_ci = _ci_bounds_interp(Dex_grid, chi2_map_49[i_best], chi2_49_min, delta_3,
                                   warn_label=f"{fig_label or ''} Dex_49")

    # --- 6. Visualisation (publication style) ---
    _pub_rc = {"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8}
    with plt.rc_context(_pub_rc):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

        _ci_kw  = dict(color="#d62728", ls="--", lw=1.2, alpha=0.8)
        _best_kw = dict(color="#d62728", ls=":",  lw=1.5)

        ax = axes[0]
        ax.plot(k_det_grid, total_per_k - chi2_total, color="#2166ac", lw=1.8)
        ax.axhline(delta_3, label=f"95% CI threshold = {delta_3:.2f}", **_ci_kw)
        ax.axvline(k_det_best, **_best_kw,
                   label=f"$k_{{det}}$ = {k_det_best:.3f}")
        ax.set_xlabel("$k_{det}$")
        ax.set_ylabel("$\\Delta\\chi^2_{total}$")
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        ax = axes[1]
        ax.plot(Dex_grid, chi2_map_19[i_best] - chi2_19_min,
                color="#4393c3", lw=1.8)
        ax.axhline(delta_3, **_ci_kw)
        ax.axvline(Dex_19_best, **_best_kw,
                   label=f"$D_{{ex,19}}$ = {Dex_19_best:.3f} µm²/ms")
        ax.set_xlabel("$D_{ex}$ (µm²/ms)")
        ax.set_ylabel("$\\Delta\\chi^2$  (TD = 19 ms)")
        ax.set_xlim(1.0, 3.0)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        ax = axes[2]
        ax.plot(Dex_grid, chi2_map_49[i_best] - chi2_49_min,
                color="#d6604d", lw=1.8)
        ax.axhline(delta_3, **_ci_kw)
        ax.axvline(Dex_49_best, **_best_kw,
                   label=f"$D_{{ex,49}}$ = {Dex_49_best:.3f} µm²/ms")
        ax.set_xlabel("$D_{ex}$ (µm²/ms)")
        ax.set_ylabel("$\\Delta\\chi^2$  (TD = 49 ms)")
        ax.set_xlim(1.0, 3.0)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, alpha=0.2, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.03, 0.97, "(C)", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, fontweight="bold")

        if fig_label:
            fig.text(0.5, 1.01, fig_label, ha="center", va="bottom",
                     fontsize=11, transform=fig.transFigure)

        fig.tight_layout(w_pad=2.0)

        if save_dir is not None:
            import os as _os
            _os.makedirs(save_dir, exist_ok=True)
            _sfx = fig_label.replace(" ", "_").replace("|", "").replace("/", "-") \
                   if fig_label else "ROI"
            _fpath = _os.path.join(save_dir, f"fig_joint_inference_{_sfx}.png")
            fig.savefig(_fpath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {_fpath}")

        plt.show()

    n_pts = len(bv_19) + len(bv_49)
    print(f"  χ²_total_min = {chi2_total:.3f}  (N={n_pts} points, 3 free params)")
    print(f"  k_det={k_det_best:.3f} | Dex_19={Dex_19_best:.3f} | Dex_49={Dex_49_best:.3f} µm²/ms"
          f"  | f_eff={f_eff_best:.3f}")

    return {
        "k_det_best":   float(k_det_best),
        "Dex_19_best":  float(Dex_19_best),
        "Dex_49_best":  float(Dex_49_best),
        "k_det_ci":     k_det_ci,
        "Dex_19_ci":    Dex_19_ci,
        "Dex_49_ci":    Dex_49_ci,
        "chi2_min":     float(chi2_total),
        "chi2_19_min":  float(chi2_19_min),
        "chi2_49_min":  float(chi2_49_min),
        "f_vivo":       float(f_vivo),
        "f_eff":        float(f_eff_best),
        "r3D_vivo":     float(r3D_vivo),
        "r3D_std_vivo": float(std_r3D_vivo),
        "k_det_grid":   k_det_grid,
        "Dex_grid":     Dex_grid,
        "loss_19_per_k": loss_19_per_k.tolist(),
        "loss_49_per_k": loss_49_per_k.tolist(),
    }


# %%
def infer_dex_fixed_kdet(df_dwi, cell_areas, max_calipers, min_calipers,
                         roi_area_um2, tri, signals_lut, params_lut,
                         k_det_fixed,
                         slice_width_um=4, bval_max_um2=7.5,
                         Dex_grid=None, k_r=1.0):
    """
    Strategy C: infer (Dex_19, Dex_49) with k_det fixed externally.

    k_det is not a free parameter — it is provided as k_det_fixed.
    For each TD, a 1D grid search over Dex is run independently.
    CI uses chi²(1, 0.95) = 3.84 per TD (1 free parameter each).

    Parameters
    ----------
    k_det_fixed : float
        Fixed detection correction factor (e.g. consensus from strategy A/B,
        or a literature-based value).

    Returns
    -------
    dict with: Dex_19_best, Dex_49_best, Dex_19_ci, Dex_49_ci,
               chi2_19_min, chi2_49_min, chi2_min,
               f_vivo, f_eff, r3D_vivo, r3D_std_vivo, k_det_fixed
    """
    from scipy.stats import chi2 as chi2_dist

    if Dex_grid is None:
        Dex_grid = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms

    # --- 1. Histological parameters ---
    mean_max = np.mean(max_calipers);  mean_min = np.mean(min_calipers)
    std_max  = np.std(max_calipers, ddof=1);  std_min = np.std(min_calipers, ddof=1)
    r2D_hist     = np.sqrt(mean_max * mean_min / 4)
    std_r2D_hist = 0.25 * np.sqrt(
        (mean_max / mean_min) * std_min**2 + (mean_min / mean_max) * std_max**2)
    r2D_vivo     = r2D_hist / k_r;  std_r2D_vivo = std_r2D_hist / k_r
    r3D_vivo     = 1.27 * r2D_vivo;  std_r3D_vivo = 1.27 * std_r2D_vivo
    n_cells  = len(cell_areas)
    N_v_vivo = n_cells / ((slice_width_um + 2 * r2D_vivo) * roi_area_um2)
    f_vivo   = N_v_vivo * (4/3) * np.pi * r3D_vivo**3
    f_eff    = f_vivo * k_det_fixed

    print(f"  [infer_dex_fixed_kdet] k_det={k_det_fixed:.3f} fixed | "
          f"f_vivo={f_vivo:.3f} | f_eff={f_eff:.3f} | r3D={r3D_vivo:.2f} µm")

    # --- 2. DWI data for each TD ---
    bvals_lut_um2 = np.array(sequence_lut['bval'][0][0]).flatten()
    bval_lut_min  = bvals_lut_um2.min()

    def _get_dwi(TD):
        td_str = f"{TD}ms"
        df_td  = df_dwi[df_dwi["TD"] == td_str].copy()
        df_td["bval_um2"] = df_td["b_value"] / 1000
        df_td  = df_td[(df_td["bval_um2"] > 0) &
                       (df_td["bval_um2"] <= bval_max_um2) &
                       (df_td["bval_um2"] >= bval_lut_min)].sort_values("bval_um2")
        bv  = df_td["bval_um2"].to_numpy()
        sm  = df_td["signal"].to_numpy()
        sig = np.where(df_td["noise"].to_numpy() < 1e-6, 1e-6,
                       df_td["noise"].to_numpy())
        return bv, sm, sig

    bv_19, S_19, sigma_19 = _get_dwi(19)
    bv_49, S_49, sigma_49 = _get_dwi(49)
    if len(bv_19) < 3 or len(bv_49) < 3:
        raise ValueError("Not enough DWI points for one or both TDs.")

    # --- 3. 1D grid per TD ---
    delta_1 = chi2_dist.ppf(0.95, df=1)   # ≈ 3.84

    def _fit_td(bv, S_meas, sigma, TD):
        loss = np.full(len(Dex_grid), np.nan)
        for j, Dex in enumerate(Dex_grid):
            bv_c, sc = get_signal_curve_lut(
                f_eff, Dex, r3D_vivo, std_r3D_vivo, TD, tri, signals_lut, params_lut)
            log_sc = np.log(np.clip(sc, 1e-10, None))
            sp = np.exp(np.interp(bv, bv_c, log_sc))
            if not np.any(np.isnan(sp)):
                loss[j] = np.sum(((S_meas - sp) / sigma) ** 2)
        j_best  = int(np.nanargmin(loss))
        Dex_best = float(Dex_grid[j_best])
        loss_min = float(loss[j_best])
        ci = _ci_bounds_interp(Dex_grid, loss, loss_min, delta_1,
                                warn_label=f"infer_dex_fixed_kdet TD={TD}")
        return Dex_best, loss_min, ci, loss

    Dex_19_best, chi2_19, Dex_19_ci, loss_19 = _fit_td(bv_19, S_19, sigma_19, 19)
    Dex_49_best, chi2_49, Dex_49_ci, loss_49 = _fit_td(bv_49, S_49, sigma_49, 49)

    # --- 4. Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, Dex_best, loss, _, chi2_min_td, td_label in [
        (axes[0], Dex_19_best, loss_19, Dex_19_ci, chi2_19, "TD=19 ms"),
        (axes[1], Dex_49_best, loss_49, Dex_49_ci, chi2_49, "TD=49 ms"),
    ]:
        ax.plot(Dex_grid, loss - chi2_min_td, color='navy', lw=2)
        ax.axhline(delta_1, color='red', ls='--', lw=1.5,
                   label=f'95% CI threshold = {delta_1:.2f}')
        ax.axvline(Dex_best, color='red', lw=1.5, ls=':',
                   label=f'Dex = {Dex_best:.3f}')
        ax.set_xlabel('Dex (µm²/ms)');  ax.set_ylabel('Δχ²')
        ax.set_title(f'{td_label}  (k_det={k_det_fixed:.3f} fixed)')
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Strategy C — fixed k_det={k_det_fixed:.3f}  |  k_r={k_r} fixed\n'
        f'Dex_19={Dex_19_best:.3f} [{Dex_19_ci[0]:.3f}, {Dex_19_ci[1]:.3f}]  |  '
        f'Dex_49={Dex_49_best:.3f} [{Dex_49_ci[0]:.3f}, {Dex_49_ci[1]:.3f}]',
        fontsize=10)
    plt.tight_layout()
    plt.show()

    print(f"  Dex_19={Dex_19_best:.3f} | Dex_49={Dex_49_best:.3f} µm²/ms "
          f"| f_eff={f_eff:.3f}")

    return {
        "Dex_19_best":  float(Dex_19_best),
        "Dex_49_best":  float(Dex_49_best),
        "Dex_19_ci":    Dex_19_ci,
        "Dex_49_ci":    Dex_49_ci,
        "chi2_19_min":  float(chi2_19),
        "chi2_49_min":  float(chi2_49),
        "chi2_min":     float(chi2_19 + chi2_49),
        "f_vivo":       float(f_vivo),
        "f_eff":        float(f_eff),
        "r3D_vivo":     float(r3D_vivo),
        "r3D_std_vivo": float(std_r3D_vivo),
        "k_det_fixed":  float(k_det_fixed),
    }


def plot_degeneracy_valley(chi2_map, Dex_grid, k_grid, chi2_min, delta):
    """
    Visualise the degeneracy valley in the (Dex, k) space.

    If the 95% CI contour forms a diagonal band, Dex and k are coupled:
    multiple (Dex, k) pairs explain the data equally well — the problem
    is partially or totally non-identifiable.

    Parameters
    ----------
    chi2_map  : array (n_k, n_Dex) from infer_Dex_shrinkage_chi2
    Dex_grid  : 1-D array of tested Dex values
    k_grid    : 1-D array of tested k values
    chi2_min  : global minimal chi²
    delta     : CI threshold (chi2_dist.ppf(0.95, df=2) ≈ 5.99)
    """
    from scipy.stats import chi2 as chi2_dist

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: chi² map with 95% CI contour ──────────────────────
    ax = axes[0]
    im = ax.contourf(Dex_grid, k_grid, chi2_map, levels=50, cmap='viridis_r')
    ax.contour(Dex_grid, k_grid, chi2_map,
               levels=[chi2_min + delta], colors='red', linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax, label='χ²(Dex, k)')
    ax.set_xlabel('Dex (µm²/ms)')
    ax.set_ylabel('Shrinkage factor k')
    ax.set_title('χ² map — red contour = 95% CI\n'
                 'A diagonal band → non-identifiable problem')

    # ── Panel 2: cloud of compatible pairs ────────────────────────
    ax = axes[1]
    ci_mask          = chi2_map < (chi2_min + delta)
    K_mesh, D_mesh   = np.meshgrid(k_grid, Dex_grid, indexing='ij')
    sc = ax.scatter(D_mesh[ci_mask], K_mesh[ci_mask],
                    c=chi2_map[ci_mask], cmap='viridis_r', s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='χ²')
    ax.set_xlabel('Dex (µm²/ms)')
    ax.set_ylabel('k (shrinkage)')
    ax.set_title('Compatible (Dex, k) pairs — 95% CI\n'
                 'each point is a plausible solution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Coupling quantification ───────────────────────────────────────
    K_vals   = K_mesh[ci_mask]
    Dex_vals = D_mesh[ci_mask]
    if len(Dex_vals) > 1:
        corr = np.corrcoef(Dex_vals, K_vals)[0, 1]
        niveau = ("strong" if abs(corr) > 0.7 else "moderate")
        identif = ("non-identifiable" if abs(corr) > 0.9 else "partially identifiable")
        print(f"Dex–k correlation in the 95% CI: r = {corr:.3f}")
        print(f"→ {niveau} coupling — problem is {identif}")
    else:
        print("95% CI reduced to a single point: well-identified parameters.")


# %%
# ── Joint (k_det shared, Dex_19, Dex_49) inference — all ROIs summary ───────
#
# For each (patient, H&E, ROI), calls infer_joint_kdet which fits a single
# k_det shared across both TDs and two independent Dex (one per TD).
# This ensures k_det is treated as a purely histological correction factor,
# independent of diffusion time.

from scipy.stats import chi2 as chi2_dist

k_r_fixed      = 1.0
exp_infer      = 3
TD_list        = [19, 49]
Dex_grid_inf   = np.linspace(1.0, 3.0, 400)  # LUT support is [1, 3] µm²/ms
k_det_grid_inf = np.linspace(0.05, 1.0, 50)
slice_width_um = 4

# Map patient id → DWI dataframe
dwi_map = {1: df_p1, 3: df_p3}

summary_rows = []

for pat_id in Patient:
    df_dwi_pat = dwi_map.get(pat_id)
    if df_dwi_pat is None:
        print(f"[skip] No DWI data for Patient {pat_id}")
        continue

    for he in H_E:
        cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp_infer}.txt"
        annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
        cell_path  = os.path.join(full_path_dir, cell_file)
        annot_path = os.path.join(full_path_dir, annot_file)

        if not os.path.exists(cell_path) or not os.path.exists(annot_path):
            continue

        df_cells = pd.read_csv(cell_path,  sep='\t', engine='python')
        df_annot = pd.read_csv(annot_path, sep='\t', engine='python')

        roi_names = [
            r for r in df_annot["Name"].unique()
            if r in {"ROI1", "ROI2", "ROI3", "ROI4"}
        ]

        for roi_name in sorted(roi_names):
            df_roi = df_cells[df_cells["Parent"] == roi_name]
            if len(df_roi) < 10:
                continue

            area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
            if len(area_vals) == 0:
                print(f"  [skip] ROI '{roi_name}' missing in annotation file.")
                continue
            roi_area_um2 = area_vals[0]

            print(f"\nPatient {pat_id} | H&E {he} | exp={exp_infer} | {roi_name}")

            res = infer_joint_kdet(
                df_dwi         = df_dwi_pat,
                cell_areas     = df_roi["Cell: Area"].to_numpy(),
                max_calipers   = df_roi["Cell: Max caliper"].to_numpy(),
                min_calipers   = df_roi["Cell: Min caliper"].to_numpy(),
                roi_area_um2   = roi_area_um2,
                tri            = tri,
                signals_lut    = signals_lut,
                params_lut     = params_lut,
                slice_width_um = slice_width_um,
                bval_max_um2   = 7.5,
                Dex_grid       = Dex_grid_inf,
                k_det_grid     = k_det_grid_inf,
                k_r            = k_r_fixed,
            )

            summary_rows.append({
                "Patient":          pat_id,
                "H&E":              he,
                "ROI":              roi_name,
                "k_det_best":       round(res["k_det_best"],    3),
                "Dex_19_best":      round(res["Dex_19_best"],   3),
                "Dex_49_best":      round(res["Dex_49_best"],   3),
                "f_vivo":           round(res["f_vivo"],        3),
                "f_eff":            round(res["f_eff"],         3),
                "r3D_vivo (µm)":    round(res["r3D_vivo"],      2),
                "r3D_std_vivo":     round(res["r3D_std_vivo"],  3),
                "chi2_min":         round(res["chi2_min"],      2),
                "chi2_19_min":      round(res["chi2_19_min"],   2),
                "chi2_49_min":      round(res["chi2_49_min"],   2),
                "k_det_CI_low":     round(res["k_det_ci"][0],   3),
                "k_det_CI_high":    round(res["k_det_ci"][1],   3),
                "Dex_19_CI_low":    round(res["Dex_19_ci"][0],  3),
                "Dex_19_CI_high":   round(res["Dex_19_ci"][1],  3),
                "Dex_49_CI_low":    round(res["Dex_49_ci"][0],  3),
                "Dex_49_CI_high":   round(res["Dex_49_ci"][1],  3),
            })

# ── Summary table ─────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)

print("\n" + "="*80)
print(f"SUMMARY — Joint inference (shared k_det) | exp={exp_infer} | k_r={k_r_fixed} fixed")
print("="*80)
print(df_summary.to_string(index=False))

# Per-patient consensus (median over ROIs and H&E slides)
# k_det is shared across TDs — one value per patient
# Dex is TD-dependent — separate consensus for Dex_19 and Dex_49
print("\n── Per-patient consensus (median over ROIs) ──")
for pat_id in Patient:
    sub = df_summary[df_summary["Patient"] == pat_id]
    if sub.empty:
        continue
    print(f"  Patient {pat_id} : "
          f"k_det={sub['k_det_best'].median():.3f} "
          f"[{sub['k_det_best'].min():.3f}–{sub['k_det_best'].max():.3f}] | "
          f"Dex_19={sub['Dex_19_best'].median():.3f} "
          f"[{sub['Dex_19_best'].min():.3f}–{sub['Dex_19_best'].max():.3f}] µm²/ms | "
          f"Dex_49={sub['Dex_49_best'].median():.3f} "
          f"[{sub['Dex_49_best'].min():.3f}–{sub['Dex_49_best'].max():.3f}] µm²/ms | "
          f"f_eff={sub['f_eff'].median():.3f} | "
          f"r3D={sub['r3D_vivo (µm)'].median():.2f} µm")

# %%
# ── Chi² vs Rician comparison — side-by-side inference ───────────────────
#
# Runs both methods on all ROIs and compares Dex_best and k_det_best.
# Helps justify which noise model is more appropriate for these data.
# The two methods should agree when Gaussian approximation is valid (high SNR).

from scipy.special import i0e as bessel_i0e  # noqa: F811 (imported in function too)

comp_rows = []

for pat_id in Patient:
    df_dwi_pat = dwi_map.get(pat_id)
    if df_dwi_pat is None:
        continue

    for he in H_E:
        cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp_infer}.txt"
        annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
        cell_path  = os.path.join(full_path_dir, cell_file)
        annot_path = os.path.join(full_path_dir, annot_file)
        if not os.path.exists(cell_path) or not os.path.exists(annot_path):
            continue

        df_cells = pd.read_csv(cell_path,  sep='\t', engine='python')
        df_annot = pd.read_csv(annot_path, sep='\t', engine='python')
        roi_names = [r for r in df_annot["Name"].unique()
                     if r in {"ROI1", "ROI2", "ROI3", "ROI4"}]

        for roi_name in sorted(roi_names):
            df_roi = df_cells[df_cells["Parent"] == roi_name]
            if len(df_roi) < 10:
                continue
            area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
            if len(area_vals) == 0:
                print(f"  [skip] ROI '{roi_name}' missing in annotation file.")
                continue
            roi_area_um2 = area_vals[0]

            for TD in TD_list:
                _call_args = dict(
                    df_dwi         = df_dwi_pat,
                    cell_areas     = df_roi["Cell: Area"].to_numpy(),
                    max_calipers   = df_roi["Cell: Max caliper"].to_numpy(),
                    min_calipers   = df_roi["Cell: Min caliper"].to_numpy(),
                    roi_area_um2   = float(roi_area_um2),
                    TD             = int(TD),
                    tri            = tri,
                    signals_lut    = signals_lut,
                    params_lut     = params_lut,
                    slice_width_um = int(slice_width_um),
                    bval_max_um2   = 7.5,
                    Dex_grid       = Dex_grid_inf,
                    k_det_grid     = k_det_grid_inf,
                    k_r            = float(k_r_fixed),
                )

                r_chi2   = infer_Dex_shrinkage_chi2(**_call_args, use_rician=False)  # type: ignore[arg-type]
                r_rician = infer_Dex_shrinkage_chi2(**_call_args, use_rician=True)   # type: ignore[arg-type]

                comp_rows.append({
                    "Patient":          pat_id,
                    "H&E":              he,
                    "ROI":              roi_name,
                    "TD (ms)":          TD,
                    "Dex_chi2":         round(r_chi2["Dex_best"],    3),
                    "Dex_rician":       round(r_rician["Dex_best"],  3),
                    "ΔDex":             round(abs(r_chi2["Dex_best"] - r_rician["Dex_best"]), 3),
                    "k_det_chi2":       round(r_chi2["k_det_best"],  3),
                    "k_det_rician":     round(r_rician["k_det_best"],3),
                    "Δk_det":           round(abs(r_chi2["k_det_best"] - r_rician["k_det_best"]), 3),
                })

df_comp = pd.DataFrame(comp_rows)
print("\n" + "="*90)
print(f"COMPARISON chi² vs Rician | exp={exp_infer} | k_r={k_r_fixed}")
print("="*90)
print(df_comp.to_string(index=False))

mean_dDex   = df_comp["ΔDex"].mean()
mean_dk_det = df_comp["Δk_det"].mean()
print(f"\nMean |ΔDex| = {mean_dDex:.4f} µm²/ms  |  Mean |Δk_det| = {mean_dk_det:.4f}")
if mean_dDex < 0.1 and mean_dk_det < 0.05:
    print("→ Gaussian and Rician estimates agree: chi² is a valid approximation here.")
else:
    print("→ Non-negligible discrepancy: Rician model is preferable (Rician bias at high b).")

# %%
# ── Normalised residual analysis ─────────────────────────────────────────
#
# For each (Patient, H&E, ROI, TD), compute:
#   residual_i = (S_meas_i - S_pred_i) / sigma_i
# at all available b-values, using the best-fit (Dex_best, k_det_best).
#
# If chi² is appropriate (Gaussian noise), residuals ~ N(0, 1):
#   - Histogram should match N(0,1)
#   - Q-Q plot should be linear
#   - Residuals vs b-value should show no systematic trend
#   A positive trend at high b-values would indicate Rician bias.

from scipy import stats as scipy_stats

dwi_map    = {1: df_p1, 3: df_p3}
all_resid  = []          # all normalised residuals pooled
bval_resid = []          # corresponding b-values
label_resid = []         # "PatX|H&EY|ROIZ|TDw" for colour coding

for _, row in df_summary.iterrows():
    pat_id   = int(row["Patient"])
    he       = int(row["H&E"])
    roi_name = row["ROI"]
    TD       = int(row["TD (ms)"])
    Dex_b    = row["Dex_best"]
    k_det_b  = row["k_det_best"]
    r3D_v    = row["r3D_vivo (µm)"]
    k_r      = k_r_fixed   # from the inference cell above

    # Reload cell data to get std_r3D_vivo
    cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp_infer}.txt"
    annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
    df_cells   = pd.read_csv(os.path.join(full_path_dir, cell_file),  sep='\t', engine='python')
    df_annot   = pd.read_csv(os.path.join(full_path_dir, annot_file), sep='\t', engine='python')
    df_roi     = df_cells[df_cells["Parent"] == roi_name]

    max_cal = df_roi["Cell: Max caliper"].to_numpy()
    min_cal = df_roi["Cell: Min caliper"].to_numpy()
    mean_max, mean_min = np.mean(max_cal), np.mean(min_cal)
    std_max,  std_min  = np.std(max_cal, ddof=1), np.std(min_cal, ddof=1)
    r2D_hist   = np.sqrt(mean_max * mean_min / 4)
    std_r2D_h  = 0.25 * np.sqrt(
        (mean_max / mean_min) * std_min**2 + (mean_min / mean_max) * std_max**2)
    std_r3D_v  = 1.27 * std_r2D_h / k_r

    # f_eff used during inference
    area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
    if len(area_vals) == 0:
        raise ValueError(f"ROI '{roi_name}' not found in annotation file for residuals.")
    roi_area       = area_vals[0]
    slice_width_um = 4  # µm — defined locally in case this cell runs independently
    r2D_vivo   = r2D_hist / k_r
    N_v_vivo   = len(df_roi) / ((slice_width_um + 2 * r2D_vivo) * roi_area)
    f_vivo     = N_v_vivo * (4/3) * np.pi * r3D_v**3
    f_eff      = f_vivo * k_det_b

    # Predicted signal curve at best-fit parameters
    bvals_curve, sig_curve = get_signal_curve_lut(
        f_eff, Dex_b, r3D_v, std_r3D_v, TD, tri, signals_lut, params_lut)
    log_sig = np.log(np.clip(sig_curve, 1e-10, None))

    # Measured DWI at this TD
    df_dwi_pat = dwi_map.get(pat_id)
    if df_dwi_pat is None:
        continue
    td_str = f"{TD}ms"
    df_td  = (df_dwi_pat[df_dwi_pat["TD"] == td_str]
                  .copy()
                  .assign(bval_um2=lambda d: d["b_value"] / 1000)
                  .query("bval_um2 > 0 and bval_um2 <= 7.5")
                  .sort_values("bval_um2"))
    bvals_um2  = df_td["bval_um2"].to_numpy()
    S_meas     = df_td["signal"].to_numpy()
    sigma      = np.where(df_td["noise"].to_numpy() < 1e-6,
                          1e-6, df_td["noise"].to_numpy())

    S_pred = np.exp(np.interp(bvals_um2, bvals_curve, log_sig))
    resid  = (S_meas - S_pred) / sigma

    all_resid.extend(resid.tolist())
    bval_resid.extend(bvals_um2.tolist())
    label_resid.extend([f"P{pat_id}|H&E{he}|{roi_name}|TD{TD}"] * len(resid))

all_resid  = np.array(all_resid)
bval_resid = np.array(bval_resid)

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Histogram vs N(0,1)
ax = axes[0]
ax.hist(all_resid, bins=20, density=True, color="steelblue",
        alpha=0.7, edgecolor="white", label="Residuals")
x_gauss = np.linspace(all_resid.min() - 0.5, all_resid.max() + 0.5, 200)
ax.plot(x_gauss, scipy_stats.norm.pdf(x_gauss), "r-", lw=2, label="N(0,1)")
ax.set_xlabel("Normalised residual  (S_meas − S_pred) / σ")
ax.set_ylabel("Density")
ax.set_title(f"Histogram of residuals\n"
             f"mean={all_resid.mean():.3f}, std={all_resid.std():.3f}, n={len(all_resid)}")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Q-Q plot
ax = axes[1]
if len(all_resid) < 3:
    ax.set_title("Q-Q plot — not enough residuals")
else:
    n_res    = len(all_resid)
    osm_arr  = scipy_stats.norm.ppf((np.arange(1, n_res + 1) - 0.5) / n_res)
    osr_arr  = np.sort(all_resid).astype(np.float64)
    coeffs   = np.polyfit(osm_arr, osr_arr, 1)
    slope_qq = float(coeffs[0])
    inter_qq = float(coeffs[1])
    r_corr   = float(np.corrcoef(osm_arr, osr_arr)[0, 1])
    ax.scatter(osm_arr, osr_arr, s=20, color="steelblue", alpha=0.7, label="Data")
    ax.plot(osm_arr, slope_qq * osm_arr + inter_qq, "r-", lw=2, label=f"Fit (r={r_corr:.3f})")
    ax.set_xlabel("Theoretical quantiles N(0,1)")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

# 3. Residuals vs b-value (Rician bias check)
ax = axes[2]
unique_labels = list(dict.fromkeys(label_resid))
cmap = plt.get_cmap("tab10")
for idx, lbl in enumerate(unique_labels):
    mask = np.array(label_resid) == lbl
    ax.scatter(bval_resid[mask], all_resid[mask],
               s=25, alpha=0.8, color=cmap(idx % 10), label=lbl)
ax.axhline(0, color="black", lw=1, ls="--")
ax.axhline(+1.96, color="red", lw=1, ls=":", alpha=0.7, label="±1.96")
ax.axhline(-1.96, color="red", lw=1, ls=":", alpha=0.7)
ax.set_xlabel("b-value (ms/µm²)")
ax.set_ylabel("Normalised residual")
ax.set_title("Residuals vs b-value\n(positive trend at high b → Rician bias)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Normalised residual analysis — exp={exp_infer}, k_r={k_r_fixed}\n"
    f"All ROIs × patients × TD pooled  (N={len(all_resid)} points)",
    fontsize=12)
fig.tight_layout()
plt.show()

# ── Shapiro-Wilk normality test ──────────────────────────────────────────
if len(all_resid) <= 5000:
    stat, p_val = scipy_stats.shapiro(all_resid)
    print(f"Shapiro-Wilk test : W={stat:.4f}, p={p_val:.4f}")
    if p_val > 0.05:
        print("  → p > 0.05: cannot reject Gaussianity — chi² assumption is justified.")
    else:
        print("  → p < 0.05: residuals deviate from Gaussian. "
              "Check Rician bias at high b-values or model mis-specification.")
else:
    stat, p_val = scipy_stats.kstest(all_resid, "norm")
    print(f"Kolmogorov-Smirnov test (n>{5000}): D={stat:.4f}, p={p_val:.4f}")

# ── Systematic trend check: mean residual per b-value bin ────────────────
print("\nMean normalised residual per b-value bin:")
bins = np.percentile(bval_resid, [0, 25, 50, 75, 100])
for i in range(len(bins) - 1):
    mask = (bval_resid >= bins[i]) & (bval_resid < bins[i+1])
    if mask.sum() == 0:
        continue
    mean_r = all_resid[mask].mean()
    print(f"  b ∈ [{bins[i]:.2f}, {bins[i+1]:.2f}] ms/µm² : "
          f"mean residual = {mean_r:+.3f}  (n={mask.sum()})"
          f"{'  ← possible Rician bias' if mean_r > 0.5 else ''}")

# %%
# ── Signal insensitivity to k_det ──────────────────────────────────────────
# Loss profile along the k_det axis at fixed Dex_best.
# Uses the same ROI settings as the simple Dex inference cell above
# (pat_id, he, roi_name, TD_val must be set).

result_shrink = infer_Dex_shrinkage_chi2(
    df_dwi       = df_dwi,
    cell_areas   = df_roi["Cell: Area"].to_numpy(),
    max_calipers = df_roi["Cell: Max caliper"].to_numpy(),
    min_calipers = df_roi["Cell: Min caliper"].to_numpy(),
    roi_area_um2 = roi_area_um2,
    TD           = TD_val,
    tri          = tri,
    signals_lut  = signals_lut,
    params_lut   = params_lut,
    k_r          = 1.0,
)

loss_map_s  = np.array(result_shrink["loss_map"])
Dex_grid_s  = np.array(result_shrink["Dex_grid"])
k_det_grid  = np.array(result_shrink["k_det_grid"])
j_best_s    = np.argmin(np.abs(Dex_grid_s - result_shrink["Dex_best"]))
loss_min_s  = result_shrink["loss_min"]

loss_k_profile = loss_map_s[:, j_best_s]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_det_grid, loss_k_profile - loss_k_profile.min(), color='navy', lw=2)
ax.axhline(3.84, color='red', ls='--', lw=1.5, label='95% CI threshold (Δ = 3.84)')
ax.fill_between(k_det_grid, 0, loss_k_profile - loss_k_profile.min(),
                alpha=0.15, color='navy')
ax.set_xlabel('Detection factor k_det')
ax.set_ylabel(f"Δ{result_shrink['loss_name']}(k_det)  [Dex fixed at optimum]")
ax.set_title(f'Signal sensitivity to k_det  [{result_shrink["loss_name"]}]\n'
             f'Patient {pat_id} | {roi_name} | TD={TD_val} ms')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Dex and k_det consensus per patient (using df_summary)

# %%
# Weighted consensus (1/chi2_min) per patient.
# k_det: single value (shared across TDs) — one consensus per patient.
# Dex_19 and Dex_49: separate consensus per TD.

all_results = []

for pat_id in Patient:
    pat_label = f"Patient_{pat_id}"
    sub = df_summary[df_summary["Patient"] == pat_id].copy()

    if sub.empty:
        print(f"[skip] {pat_label} : absent de df_summary.")
        continue

    sub["roi_label"] = (
        "P" + sub["Patient"].astype(str) + "|H&E" +
        sub["H&E"].astype(str) + "|" + sub["ROI"]
    )

    # Weights inversely proportional to chi2_min
    weights_raw = 1.0 / sub["chi2_min"].replace(0, np.nan)
    valid       = weights_raw.dropna()
    if len(valid) == 0:
        print(f"[skip] {pat_label} : tous les chi2_min sont nuls.")
        continue
    sub_w = sub.loc[valid.index]
    w_arr = valid.to_numpy()

    kdet_cons      = float(np.average(sub_w["k_det_best"],    weights=w_arr))
    kdet_ci_low    = float(np.average(sub_w["k_det_CI_low"],  weights=w_arr))
    kdet_ci_high   = float(np.average(sub_w["k_det_CI_high"], weights=w_arr))
    Dex19_cons     = float(np.average(sub_w["Dex_19_best"],   weights=w_arr))
    Dex19_ci_low   = float(np.average(sub_w["Dex_19_CI_low"], weights=w_arr))
    Dex19_ci_high  = float(np.average(sub_w["Dex_19_CI_high"],weights=w_arr))
    Dex49_cons     = float(np.average(sub_w["Dex_49_best"],   weights=w_arr))
    Dex49_ci_low   = float(np.average(sub_w["Dex_49_CI_low"], weights=w_arr))
    Dex49_ci_high  = float(np.average(sub_w["Dex_49_CI_high"],weights=w_arr))

    print(f"\n── {pat_label} ────────────────────────────────────────────")
    print(f"  k_det  = {kdet_cons:.3f} [{kdet_ci_low:.3f}, {kdet_ci_high:.3f}]")
    print(f"  Dex_19 = {Dex19_cons:.3f} [{Dex19_ci_low:.3f}, {Dex19_ci_high:.3f}] µm²/ms")
    print(f"  Dex_49 = {Dex49_cons:.3f} [{Dex49_ci_low:.3f}, {Dex49_ci_high:.3f}] µm²/ms")
    print(f"  ({len(sub_w)} ROIs × H&E)")

    # ── Barh plots ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, max(3, 0.45 * len(sub_w))))

    for ax, col, ci_low_col, ci_high_col, cons, ci_l, ci_h, xlabel, title_suffix in [
        (axes[0], "k_det_best",  "k_det_CI_low",  "k_det_CI_high",
         kdet_cons,  kdet_ci_low,  kdet_ci_high,  "k_det",        "k_det"),
        (axes[1], "Dex_19_best", "Dex_19_CI_low", "Dex_19_CI_high",
         Dex19_cons, Dex19_ci_low, Dex19_ci_high, "Dex (µm²/ms)", "Dex — TD=19 ms"),
        (axes[2], "Dex_49_best", "Dex_49_CI_low", "Dex_49_CI_high",
         Dex49_cons, Dex49_ci_low, Dex49_ci_high, "Dex (µm²/ms)", "Dex — TD=49 ms"),
    ]:
        ax.barh(
            sub_w["roi_label"], sub_w[col],
            xerr=[sub_w[col] - sub_w[ci_low_col],
                  sub_w[ci_high_col] - sub_w[col]],
            capsize=4, color="steelblue", alpha=0.7,
        )
        ax.axvline(cons, color="red", lw=2, ls="--",
                   label=f"Consensus = {cons:.3f}")
        ax.axvspan(ci_l, ci_h, alpha=0.1, color="red", label="IC 95% consensus")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{pat_label} — {title_suffix} per ROI (1/χ²)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()

    all_results.append({
        "patient":      pat_label,
        "k_det_cons":   kdet_cons,
        "Dex_19_cons":  Dex19_cons,
        "Dex_49_cons":  Dex49_cons,
    })

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY — Consensus per patient (shared k_det)")
print(f"{'='*60}")
for r in all_results:
    print(f"{r['patient']} : k_det={r['k_det_cons']:.3f} | "
          f"Dex_19={r['Dex_19_cons']:.3f} | Dex_49={r['Dex_49_cons']:.3f} µm²/ms")

# Keep Strategy B result as df_summary_B
df_summary_B = df_summary.copy()

# %% [markdown]
# # Three-strategy inference comparison
#
# Strategy A : separate (k_det_19, Dex_19) and (k_det_49, Dex_49) per TD
#              → 2 free parameters per TD, k_det can vary between TDs
# Strategy B : shared k_det + (Dex_19, Dex_49)
#              → 3 free parameters total, k_det is TD-independent
#              (already computed above as df_summary_B)
# Strategy C : fixed k_det (from Strategy A TD=19 consensus) + (Dex_19, Dex_49)
#              → 2 free parameters, k_det fixed to TD=19 value (less exchange bias)
#
# Goal: compare Dex_19 and Dex_49 across strategies, especially whether
# the physically expected Dex_49 < Dex_19 is recovered.

# %%
# ── Strategy A — separate k_det per TD ───────────────────────────────────
print("\n" + "="*80)
print("STRATEGY A — separate (k_det, Dex) per TD")
print("="*80)

rows_A = []

for pat_id in Patient:
    df_dwi_pat = dwi_map.get(pat_id)
    if df_dwi_pat is None:
        continue
    for he in H_E:
        cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp_infer}.txt"
        annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
        cell_path  = os.path.join(full_path_dir, cell_file)
        annot_path = os.path.join(full_path_dir, annot_file)
        if not os.path.exists(cell_path) or not os.path.exists(annot_path):
            continue
        df_cells = pd.read_csv(cell_path,  sep='\t', engine='python')
        df_annot = pd.read_csv(annot_path, sep='\t', engine='python')
        roi_names = [r for r in df_annot["Name"].unique()
                     if r in {"ROI1", "ROI2", "ROI3", "ROI4"}]

        for roi_name in sorted(roi_names):
            df_roi = df_cells[df_cells["Parent"] == roi_name]
            if len(df_roi) < 10:
                continue
            area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
            if len(area_vals) == 0:
                continue
            roi_area_um2 = area_vals[0]

            print(f"\nPatient {pat_id} | H&E {he} | {roi_name}")
            res_19 = infer_Dex_shrinkage_chi2(
                df_dwi=df_dwi_pat,
                cell_areas=df_roi["Cell: Area"].to_numpy(),
                max_calipers=df_roi["Cell: Max caliper"].to_numpy(),
                min_calipers=df_roi["Cell: Min caliper"].to_numpy(),
                roi_area_um2=roi_area_um2, TD=19,
                tri=tri, signals_lut=signals_lut, params_lut=params_lut,
                slice_width_um=slice_width_um, bval_max_um2=7.5,
                Dex_grid=Dex_grid_inf, k_det_grid=k_det_grid_inf,
                k_r=k_r_fixed, use_rician=False,
            )
            res_49 = infer_Dex_shrinkage_chi2(
                df_dwi=df_dwi_pat,
                cell_areas=df_roi["Cell: Area"].to_numpy(),
                max_calipers=df_roi["Cell: Max caliper"].to_numpy(),
                min_calipers=df_roi["Cell: Min caliper"].to_numpy(),
                roi_area_um2=roi_area_um2, TD=49,
                tri=tri, signals_lut=signals_lut, params_lut=params_lut,
                slice_width_um=slice_width_um, bval_max_um2=7.5,
                Dex_grid=Dex_grid_inf, k_det_grid=k_det_grid_inf,
                k_r=k_r_fixed, use_rician=False,
            )
            rows_A.append({
                "Patient":       pat_id, "H&E": he, "ROI": roi_name,
                "k_det_19":      round(res_19["k_det_best"], 3),
                "k_det_49":      round(res_49["k_det_best"], 3),
                "Dex_19_best":   round(res_19["Dex_best"],   3),
                "Dex_49_best":   round(res_49["Dex_best"],   3),
                "f_vivo":        round(res_19["f_vivo"],      3),
                "f_eff_19":      round(res_19["f_corrected"], 3),
                "f_eff_49":      round(res_49["f_corrected"], 3),
                "r3D_vivo (µm)": round(res_19["r3D_vivo"],    2),
                "r3D_std_vivo":  round(res_19["r3D_std_vivo"],3),
                "chi2_19_min":   round(res_19["chi2_min"],    2),
                "chi2_49_min":   round(res_49["chi2_min"],    2),
                "chi2_min":      round(res_19["chi2_min"] + res_49["chi2_min"], 2),
                "Dex_19_CI_low":  round(res_19["Dex_ci"][0],  3),
                "Dex_19_CI_high": round(res_19["Dex_ci"][1],  3),
                "Dex_49_CI_low":  round(res_49["Dex_ci"][0],  3),
                "Dex_49_CI_high": round(res_49["Dex_ci"][1],  3),
                "k_det_CI_low":   round(res_19["k_det_ci"][0],3),
                "k_det_CI_high":  round(res_19["k_det_ci"][1],3),
            })

df_summary_A = pd.DataFrame(rows_A)
print("\n" + "="*80)
print("STRATEGY A — results")
print("="*80)
print(df_summary_A[["Patient","H&E","ROI",
                     "k_det_19","k_det_49",
                     "Dex_19_best","Dex_49_best"]].to_string(index=False))

# %%
# ── Strategy C — fixed k_det ─────────────────────────────────────────────
# k_det_fixed: per-patient weighted average of k_det_19 and k_det_49 from
# Strategy A, weighted by 1/chi²_min for each TD.
# Rationale: uses all available k_det estimates from Strategy A (both TDs),
# with better-fitting ROIs contributing more to the consensus.
print("\n" + "="*80)
print("STRATEGY C — fixed k_det (weighted mean of k_det_19 and k_det_49 from Strategy A)")
print("="*80)

def _kdet_fixed_for_patient(pat_id):
    sub = df_summary_A[df_summary_A["Patient"] == pat_id]
    # Pool k_det_19 and k_det_49 from Strategy A, weighted by 1/chi² each
    k_vals = np.concatenate([sub["k_det_19"].to_numpy(), sub["k_det_49"].to_numpy()])
    chi2_19 = sub["chi2_19_min"].replace(0, np.nan).to_numpy().astype(float)
    chi2_49 = sub["chi2_49_min"].replace(0, np.nan).to_numpy().astype(float)
    w_vals  = np.concatenate([1.0 / chi2_19, 1.0 / chi2_49])
    mask = np.isfinite(w_vals)
    return float(np.average(k_vals[mask], weights=w_vals[mask]))

rows_C = []

for pat_id in Patient:
    df_dwi_pat = dwi_map.get(pat_id)
    if df_dwi_pat is None:
        continue
    k_fixed = _kdet_fixed_for_patient(pat_id)
    print(f"\n  Patient {pat_id} — k_det fixed = {k_fixed:.3f}")

    for he in H_E:
        cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{exp_infer}.txt"
        annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
        cell_path  = os.path.join(full_path_dir, cell_file)
        annot_path = os.path.join(full_path_dir, annot_file)
        if not os.path.exists(cell_path) or not os.path.exists(annot_path):
            continue
        df_cells = pd.read_csv(cell_path,  sep='\t', engine='python')
        df_annot = pd.read_csv(annot_path, sep='\t', engine='python')
        roi_names = [r for r in df_annot["Name"].unique()
                     if r in {"ROI1", "ROI2", "ROI3", "ROI4"}]

        for roi_name in sorted(roi_names):
            df_roi = df_cells[df_cells["Parent"] == roi_name]
            if len(df_roi) < 10:
                continue
            area_vals = df_annot[df_annot["Name"] == roi_name]["Area µm^2"].to_numpy()
            if len(area_vals) == 0:
                continue
            roi_area_um2 = area_vals[0]

            print(f"\nPatient {pat_id} | H&E {he} | {roi_name}")
            res_C = infer_dex_fixed_kdet(
                df_dwi=df_dwi_pat,
                cell_areas=df_roi["Cell: Area"].to_numpy(),
                max_calipers=df_roi["Cell: Max caliper"].to_numpy(),
                min_calipers=df_roi["Cell: Min caliper"].to_numpy(),
                roi_area_um2=roi_area_um2,
                tri=tri, signals_lut=signals_lut, params_lut=params_lut,
                k_det_fixed=k_fixed,
                slice_width_um=slice_width_um, bval_max_um2=7.5,
                Dex_grid=Dex_grid_inf, k_r=k_r_fixed,
            )
            rows_C.append({
                "Patient":        pat_id, "H&E": he, "ROI": roi_name,
                "k_det_fixed":    round(k_fixed,                   3),
                "Dex_19_best":    round(res_C["Dex_19_best"],      3),
                "Dex_49_best":    round(res_C["Dex_49_best"],      3),
                "f_vivo":         round(res_C["f_vivo"],           3),
                "f_eff":          round(res_C["f_eff"],            3),
                "r3D_vivo (µm)":  round(res_C["r3D_vivo"],         2),
                "r3D_std_vivo":   round(res_C["r3D_std_vivo"],     3),
                "chi2_min":       round(res_C["chi2_min"],         2),
                "chi2_19_min":    round(res_C["chi2_19_min"],      2),
                "chi2_49_min":    round(res_C["chi2_49_min"],      2),
                "Dex_19_CI_low":  round(res_C["Dex_19_ci"][0],    3),
                "Dex_19_CI_high": round(res_C["Dex_19_ci"][1],    3),
                "Dex_49_CI_low":  round(res_C["Dex_49_ci"][0],    3),
                "Dex_49_CI_high": round(res_C["Dex_49_ci"][1],    3),
            })

df_summary_C = pd.DataFrame(rows_C)
print("\n" + "="*80)
print("STRATEGY C — results")
print("="*80)
print(df_summary_C[["Patient","H&E","ROI",
                     "k_det_fixed","Dex_19_best","Dex_49_best"]].to_string(index=False))

# %%
# ── Three-strategy comparison table ──────────────────────────────────────
print("\n" + "="*80)
print("COMPARISON — Dex_19, Dex_49 and chi² across three strategies")
print("="*80)
print(f"{'ROI':<22} {'Strategy A':^42} {'Strategy B':^42} {'Strategy C':^42}")
print(f"{'':22} {'k19|k49 | D19|D49 | X²19|X²49':^42} {'k_sh | D19|D49 | X²19|X²49':^42} {'k_fix | D19|D49 | X²19|X²49':^42}")
print("-"*150)

for _, rA in df_summary_A.iterrows():
    key = (rA["Patient"], rA["H&E"], rA["ROI"])
    rB  = df_summary_B[(df_summary_B["Patient"] == key[0]) &
                       (df_summary_B["H&E"]     == key[1]) &
                       (df_summary_B["ROI"]     == key[2])]
    rC  = df_summary_C[(df_summary_C["Patient"] == key[0]) &
                       (df_summary_C["H&E"]     == key[1]) &
                       (df_summary_C["ROI"]     == key[2])]
    lbl = f"P{key[0]}|H&E{key[1]}|{key[2]}"

    sA = (f"k={rA['k_det_19']:.2f}/{rA['k_det_49']:.2f} "
          f"D={rA['Dex_19_best']:.3f}/{rA['Dex_49_best']:.3f} "
          f"X²={rA['chi2_19_min']:.1f}/{rA['chi2_49_min']:.1f}")
    sB = (f"k={rB['k_det_best'].values[0]:.2f} "
          f"D={rB['Dex_19_best'].values[0]:.3f}/{rB['Dex_49_best'].values[0]:.3f} "
          f"X²={rB['chi2_19_min'].values[0]:.1f}/{rB['chi2_49_min'].values[0]:.1f}"
          if len(rB) else "—")
    sC = (f"k={rC['k_det_fixed'].values[0]:.2f} "
          f"D={rC['Dex_19_best'].values[0]:.3f}/{rC['Dex_49_best'].values[0]:.3f} "
          f"X²={rC['chi2_19_min'].values[0]:.1f}/{rC['chi2_49_min'].values[0]:.1f}"
          if len(rC) else "—")

    # Flag physically inconsistent results (Dex_49 > Dex_19)
    flag_A = " ⚠" if rA["Dex_49_best"] > rA["Dex_19_best"] else " ✓"
    flag_B = (" ⚠" if len(rB) and rB["Dex_49_best"].values[0] > rB["Dex_19_best"].values[0]
              else " ✓")
    flag_C = (" ⚠" if len(rC) and rC["Dex_49_best"].values[0] > rC["Dex_19_best"].values[0]
              else " ✓")

    print(f"{lbl:<22} {sA+flag_A:<44} {sB+flag_B:<44} {sC+flag_C:<44}")

# %%
# ── Strategy A — bar plots (Dex, chi², k_det per ROI) ────────────────────
_colors = {"TD=19": "#2196F3", "TD=49": "#FF5722"}

for pat_id in df_summary_A["Patient"].unique():
    sub = df_summary_A[df_summary_A["Patient"] == pat_id].copy()
    sub["label"] = sub["H&E"].astype(str) + "|" + sub["ROI"].astype(str)
    x = np.arange(len(sub))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Strategy A — Patient {pat_id}", fontsize=12, fontweight="bold")

    # — Dex_19 / Dex_49 —
    ax = axes[0]
    ax.bar(x - w/2, sub["Dex_19_best"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["Dex_49_best"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dex (µm²/ms)"); ax.set_title("Dex_19 vs Dex_49")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # — chi²_19 / chi²_49 —
    ax = axes[1]
    ax.bar(x - w/2, sub["chi2_19_min"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["chi2_49_min"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("χ² min"); ax.set_title("χ²_19 vs χ²_49")
    ax.axhline(3.84, color="gray", linestyle="--", linewidth=0.8, label="χ²(1, 0.95)=3.84")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # — k_det_19 / k_det_49 —
    ax = axes[2]
    ax.bar(x - w/2, sub["k_det_19"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["k_det_49"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("k_det"); ax.set_title("k_det_19 vs k_det_49")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── Strategy B — bar plots (Dex, chi² per ROI) ───────────────────────────
for pat_id in df_summary_B["Patient"].unique():
    sub = df_summary_B[df_summary_B["Patient"] == pat_id].copy()
    sub["label"] = sub["H&E"].astype(str) + "|" + sub["ROI"].astype(str)
    x = np.arange(len(sub))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Strategy B — Patient {pat_id}  (shared k_det)",
                 fontsize=12, fontweight="bold")

    # — Dex_19 / Dex_49 —
    ax = axes[0]
    ax.bar(x - w/2, sub["Dex_19_best"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["Dex_49_best"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dex (µm²/ms)"); ax.set_title("Dex_19 vs Dex_49")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # — chi²_19 / chi²_49 —
    ax = axes[1]
    ax.bar(x - w/2, sub["chi2_19_min"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["chi2_49_min"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("χ² min"); ax.set_title("χ²_19 vs χ²_49")
    ax.axhline(3.84, color="gray", linestyle="--", linewidth=0.8, label="χ²(1, 0.95)=3.84")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # — k_det (shared) —
    ax = axes[2]
    ax.bar(x, sub["k_det_best"], color="#4CAF50", label="k_det (shared)")
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("k_det"); ax.set_title("k_det shared")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── Strategy C — bar plots (Dex, chi² per ROI) ───────────────────────────
for pat_id in df_summary_C["Patient"].unique():
    sub = df_summary_C[df_summary_C["Patient"] == pat_id].copy()
    sub["label"] = sub["H&E"].astype(str) + "|" + sub["ROI"].astype(str)
    x = np.arange(len(sub))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Strategy C — Patient {pat_id}  (k_det fixed = {sub['k_det_fixed'].iloc[0]:.3f})",
                 fontsize=12, fontweight="bold")

    # — Dex_19 / Dex_49 —
    ax = axes[0]
    ax.bar(x - w/2, sub["Dex_19_best"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["Dex_49_best"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dex (µm²/ms)"); ax.set_title("Dex_19 vs Dex_49")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # — chi²_19 / chi²_49 —
    ax = axes[1]
    ax.bar(x - w/2, sub["chi2_19_min"], w, label="TD=19", color=_colors["TD=19"])
    ax.bar(x + w/2, sub["chi2_49_min"], w, label="TD=49", color=_colors["TD=49"])
    ax.set_xticks(x); ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("χ² min"); ax.set_title("χ²_19 vs χ²_49")
    ax.axhline(3.84, color="gray", linestyle="--", linewidth=0.8, label="χ²(1, 0.95)=3.84")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
# ── Active df_summary for downstream (bootstrap, validation) ─────────────
# Change this to df_summary_A or df_summary_C once strategy is validated.
# Strategy A requires updating _dex_consensus / _kdet_consensus in the
# bootstrap section (it has TD (ms) column like the old format).
ACTIVE_STRATEGY = "B"   # "A", "B", or "C"

if ACTIVE_STRATEGY == "A":
    # Strategy A has a different column structure (TD-specific k_det)
    # Build a B-compatible df_summary from A by duplicating rows per TD
    _rows_A_compat = []
    for _, r in df_summary_A.iterrows():
        for td, k_col, d_col, ci_lo, ci_hi in [
            (19, "k_det_19", "Dex_19_best", "Dex_19_CI_low", "Dex_19_CI_high"),
            (49, "k_det_49", "Dex_49_best", "Dex_49_CI_low", "Dex_49_CI_high"),
        ]:
            _rows_A_compat.append({
                "Patient": r["Patient"], "H&E": r["H&E"], "ROI": r["ROI"],
                "k_det_best":   r[k_col],
                "Dex_19_best":  r["Dex_19_best"],
                "Dex_49_best":  r["Dex_49_best"],
                "f_vivo":       r["f_vivo"],
                "f_eff":        r[f"f_eff_{td // 10 * 10}"] if f"f_eff_{td}" in r.index else r["f_vivo"] * r[k_col],
                "r3D_vivo (µm)":r["r3D_vivo (µm)"],
                "r3D_std_vivo": r["r3D_std_vivo"],
                "chi2_min":     r["chi2_min"],
                "chi2_19_min":  r["chi2_19_min"],
                "chi2_49_min":  r["chi2_49_min"],
                "k_det_CI_low": r["k_det_CI_low"],
                "k_det_CI_high":r["k_det_CI_high"],
                "Dex_19_CI_low": r["Dex_19_CI_low"],
                "Dex_19_CI_high":r["Dex_19_CI_high"],
                "Dex_49_CI_low": r["Dex_49_CI_low"],
                "Dex_49_CI_high":r["Dex_49_CI_high"],
            })
    df_summary = pd.DataFrame(_rows_A_compat)
elif ACTIVE_STRATEGY == "C":
    # Strategy C: no k_det_best column — add k_det_fixed as k_det_best
    df_summary = df_summary_C.rename(columns={"k_det_fixed": "k_det_best"}).copy()
    df_summary["k_det_CI_low"]  = df_summary["k_det_best"]
    df_summary["k_det_CI_high"] = df_summary["k_det_best"]
else:
    df_summary = df_summary_B.copy()  # default: Strategy B

print(f"\nActive strategy for downstream: {ACTIVE_STRATEGY}")
df_summary.to_csv("df_summary.csv", index=False)
print("df_summary.csv saved.")

# # Select the parameters for the bootstrap

# %%
def analyze_bootstrap_convergence(cell_areas, max_calipers, min_calipers,
                                   roi_area_um2, slice_width_um,
                                   Dex_central, bval, TD,
                                   tri, signals_lut, params_lut,
                                   n_max=1000,
                                   tolerance=0.01):
    """
    Computes bootstrap statistics for increasing values of n,
    and identifies the minimal n at which convergence is reached.

    tolerance : maximum relative variation accepted on the CI width
                between two consecutive steps (e.g. 0.01 = 1%)
    """
    # Steps to test — log-spaced to cover small n values
    n_values = np.unique(np.logspace(
        np.log10(20), np.log10(n_max), num=30
    ).astype(int))

    # Generate ALL samples at once, then subsample
    # → avoids re-running the random generator at each step
    n_cells = len(cell_areas)
    all_indices = [
        np.random.choice(n_cells, size=n_cells, replace=True)
        for _ in range(n_max)
    ]

    # Pre-compute signals for each replicate
    all_signals = []
    for idx in all_indices:
        max_cal_b = max_calipers[idx]
        min_cal_b = min_calipers[idx]

        mean_max = np.mean(max_cal_b)
        mean_min = np.mean(min_cal_b)
        std_max  = np.std(max_cal_b, ddof=1)
        std_min  = np.std(min_cal_b, ddof=1)

        r2D     = np.sqrt(mean_max * mean_min / 4)
        std_r2D = 0.25 * np.sqrt(
            (mean_max / mean_min) * std_min**2 +
            (mean_min / mean_max) * std_max**2
        )
        r3D     = 1.27 * r2D
        std_r3D = 1.27 * std_r2D

        N_v    = n_cells / ((slice_width_um + 2 * r2D) * roi_area_um2)
        V_cell = (4/3) * np.pi * r3D**3
        f_boot = N_v * V_cell

        sig = interpolate_signal(f_boot, Dex_central, r3D, std_r3D,
                                 bval, TD, tri, signals_lut, params_lut)
        if not np.isnan(sig):
            all_signals.append(sig)

    all_signals = np.array(all_signals)

    # Compute statistics for each step n
    results = {"n": [], "mean": [], "std": [], "p2.5": [], "p97.5": [], "ic_width": []}

    for n in n_values:
        if n > len(all_signals):
            break
        subset = all_signals[:n]
        p_low  = np.percentile(subset, 2.5)
        p_high = np.percentile(subset, 97.5)
        results["n"].append(n)
        results["mean"].append(np.mean(subset))
        results["std"].append(np.std(subset))
        results["p2.5"].append(p_low)
        results["p97.5"].append(p_high)
        results["ic_width"].append(p_high - p_low)

    # ── Convergence criterion ────────────────────────────────────────────────
    # Relative variation of the CI width between consecutive steps
    ic_widths   = np.array(results["ic_width"])
    rel_changes = np.abs(np.diff(ic_widths) / ic_widths[:-1])

    n_converged = None
    for k, rel_change in enumerate(rel_changes):
        if rel_change < tolerance:
            n_converged = results["n"][k + 1]
            break

    # ── Visualisation ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: percentile evolution
    ax = axes[0]
    ax.plot(results["n"], results["p2.5"],  color='steelblue',
            label='P2.5', lw=1.5)
    ax.plot(results["n"], results["p97.5"], color='coral',
            label='P97.5', lw=1.5)
    ax.fill_between(results["n"], results["p2.5"], results["p97.5"],
                    alpha=0.15, color='steelblue')
    if n_converged:
        ax.axvline(n_converged, color='red', ls='--', lw=1.2,
                   label=f'Convergence: n={n_converged}')
    ax.set_xlabel('Number of bootstrap replicates')
    ax.set_ylabel('Normalized signal')
    ax.set_title('Stability of percentiles 2.5 / 97.5')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: CI width
    ax = axes[1]
    ax.plot(results["n"], ic_widths, color='navy', lw=1.5)
    if n_converged:
        ax.axvline(n_converged, color='red', ls='--', lw=1.2,
                   label=f'n={n_converged}')
        ax.axhline(ic_widths[-1], color='gray', ls=':', lw=1,
                   label='Asymptotic width')
    ax.set_xlabel('Number of bootstrap replicates')
    ax.set_ylabel('CI 95% width (P97.5 - P2.5)')
    ax.set_title("Convergence of the CI width")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: relative variation between steps (criterion)
    ax = axes[2]
    ax.plot(results["n"][1:], rel_changes * 100,
            color='darkorange', lw=1.5)
    ax.axhline(tolerance * 100, color='red', ls='--', lw=1.2,
               label=f'Tolerance threshold = {tolerance*100:.1f}%')
    if n_converged:
        ax.axvline(n_converged, color='red', ls='--', lw=1.2,
                   label=f'n={n_converged}')
    ax.set_xlabel('Number of bootstrap replicates')
    ax.set_ylabel('Relative variation of the CI width (%)')
    ax.set_title('Convergence criterion')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Bootstrap convergence analysis — b={bval} ms/µm², TD={TD} ms',
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    if n_converged:
        print(f"Convergence reached at n = {n_converged} "
              f"(tolerance {tolerance*100:.1f}% on the CI width)")
    else:
        print(f"Convergence not reached before n={n_max} — "
              f"increase n_max or tolerance")

    return n_converged, results

# Run once to adjust bootstrap parameters : worst case = patient 1, H&E 2, expansion 3, ROI2
pat_id  = 1       
he      = 2
exp     = 3
roi_name = "ROI2"

cell_file = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"

df_cells = pd.read_csv(os.path.join(full_path_dir, cell_file), sep='\t', engine='python')
df_annot = pd.read_csv(os.path.join(full_path_dir, annot_file), sep='\t', engine='python')

df_roi       = df_cells[df_cells["Parent"] == roi_name]
area_vals = df_annot.loc[df_annot["Name"] == roi_name, "Area µm^2"].to_numpy()
if len(area_vals) == 0:
    raise ValueError(f"ROI '{roi_name}' not found in annotation file.")
roi_area_um2 = area_vals[0]

n_opt, conv_results = analyze_bootstrap_convergence(
    cell_areas   = df_roi["Cell: Area"].to_numpy(),
    max_calipers = df_roi["Cell: Max caliper"].to_numpy(),
    min_calipers = df_roi["Cell: Min caliper"].to_numpy(),
    roi_area_um2 = roi_area_um2,
    slice_width_um = 4,
    Dex_central  = 2.2,
    bval         = 1.5,   # intermediate b-value, sensitive to parameters
    TD           = 19,
    tri = Delaunay(params_lut), signals_lut=signals_lut, params_lut=params_lut,
    n_max        = 1000,
    tolerance    = 0.01   # 1% relative variation = stability criterion
)

# %% [markdown]
# # Boostrapping on cells variability

# %%
def bootstrap_signal_from_cells(cell_areas, max_calipers, min_calipers, roi_area_um2, slice_width_um, Dex_central, bval, TD, tri, signals_lut, params_lut, n_bootstrap=300):
    """Bootstrap on cell variability: at each replicate, cells are resampled with replacement"""
    n_cells = len(cell_areas)
    signals_boot = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_cells, size=n_cells, replace=True)
        max_cal_b = max_calipers[idx]
        min_cal_b = min_calipers[idx]

        mean_max = np.mean(max_cal_b)
        mean_min = np.mean(min_cal_b)
        std_max  = np.std(max_cal_b, ddof=1)
        std_min  = np.std(min_cal_b, ddof=1)

        r2D = np.sqrt(mean_max * mean_min / 4)

        # Uncertainty propagation for r2D
        std_r2D = 0.25 * np.sqrt(
            (mean_max / mean_min) * std_min**2 +
            (mean_min / mean_max) * std_max**2
        )

        r3D     = 1.27 * r2D
        std_r3D = 1.27 * std_r2D

        # Abercrombie's 3D density estimation
        N_v    = n_cells / ((slice_width_um + 2 * r2D) * roi_area_um2)
        V_cell = (4/3) * np.pi * r3D**3
        f_boot = N_v * V_cell

        sig = interpolate_signal(f_boot, Dex_central, r3D, std_r3D,
                                 bval, TD, tri, signals_lut, params_lut)
        if not np.isnan(sig):
            signals_boot.append(sig)

    signals_boot = np.array(signals_boot)
    return {
        "mean":   np.mean(signals_boot),
        "median": np.median(signals_boot),
        "std":    np.std(signals_boot),
        "p2.5":   np.percentile(signals_boot, 2.5),
        "p97.5":  np.percentile(signals_boot, 97.5),
    }



# %% [markdown]
# # Bootstrapping on density intra ROI

# %%
def bootstrap_signal_from_density(df_roi, roi_area_um2, slice_width_um, Dex_central, bval, TD, tri, signals_lut, params_lut, n_bootstrap=200):
    """
    Bootstrap on cell density within the ROI.
    
    Each replicate resamples the cells with replacement → the effective
    number of cells varies → N_v and f vary → signal varies.
    
    Cell size (r3D, std_r3D) is recalculated at each replicate
    but will vary little (as expected with a fixed experiment).
    """
    cell_areas   = df_roi["Cell: Area"].to_numpy()
    max_calipers = df_roi["Cell: Max caliper"].to_numpy()
    min_calipers = df_roi["Cell: Min caliper"].to_numpy()
    n_cells      = len(cell_areas)

    signals_boot = []
    f_boot_list  = []   # to inspect the f distribution

    for _ in range(n_bootstrap):
        # Resampling with replacement → some cells counted
        # multiple times, others absent → variable effective density
        idx      = np.random.choice(n_cells, size=n_cells, replace=True)

        max_cal_b = max_calipers[idx]
        min_cal_b = min_calipers[idx]

        mean_max  = np.mean(max_cal_b)
        mean_min  = np.mean(min_cal_b)
        std_max   = np.std(max_cal_b, ddof=1)
        std_min   = np.std(min_cal_b, ddof=1)

        r2D     = np.sqrt(mean_max * mean_min / 4)
        std_r2D = 0.25 * np.sqrt(
            (mean_max / mean_min) * std_min**2 +
            (mean_min / mean_max) * std_max**2
        )
        r3D     = 1.27 * r2D
        std_r3D = 1.27 * std_r2D

        # 3D density — this is where bootstrap variability comes into play
        # n_cells in the numerator varies with each draw
        N_v    = n_cells / ((slice_width_um + 2 * r2D) * roi_area_um2)
        V_cell = (4/3) * np.pi * r3D**3
        f_boot = N_v * V_cell

        f_boot_list.append(f_boot)

        sig = interpolate_signal(f_boot, Dex_central, r3D, std_r3D,
                                 bval, TD, tri, signals_lut, params_lut)
        if not np.isnan(sig):
            signals_boot.append(sig)

    signals_boot = np.array(signals_boot)
    f_boot_arr   = np.array(f_boot_list)

    return {
        "mean":    np.mean(signals_boot),
        "median":  np.median(signals_boot),
        "std":     np.std(signals_boot),
        "p2.5":    np.percentile(signals_boot, 2.5),
        "p97.5":   np.percentile(signals_boot, 97.5),
        # f distribution for diagnostics
        "f_mean":  np.mean(f_boot_arr),
        "f_std":   np.std(f_boot_arr),
        "f_p2.5":  np.percentile(f_boot_arr, 2.5),
        "f_p97.5": np.percentile(f_boot_arr, 97.5),
    }


# %% [markdown]
# # Bootstrapping on density inter ROI

# %%
def bootstrap_signal_interROI(df_all_rois, roi_names, roi_areas,
                               slice_width_um, Dex_central,
                               bval, TD, tri, signals_lut, params_lut,
                               n_bootstrap=200):
    """
    Inter-ROI bootstrap: at each replicate, one ROI is drawn at random
    with replacement to estimate the variability due to the choice of
    histological sampling area.

    df_all_rois : dict {roi_name: DataFrame} of cells per ROI
    roi_areas   : dict {roi_name: area in µm²}
    """
    signals_boot = []
    roi_list     = list(roi_names)

    for _ in range(n_bootstrap):
        # Sample one ROI at random with replacement
        roi_sampled  = np.random.choice(roi_list)
        df_roi       = df_all_rois[roi_sampled]
        roi_area_um2 = roi_areas[roi_sampled]

        res = bootstrap_signal_from_density(
            df_roi, roi_area_um2, slice_width_um,
            Dex_central, bval, TD,
            tri, signals_lut, params_lut,
            n_bootstrap=50   # reduced intra bootstrap to limit computation time
        )
        signals_boot.append(res["mean"])

    signals_boot = np.array(signals_boot)
    return {
        "mean":   np.mean(signals_boot),
        "median": np.median(signals_boot),
        "p2.5":   np.percentile(signals_boot, 2.5),
        "p97.5":  np.percentile(signals_boot, 97.5),
        "std":    np.std(signals_boot),
    }


# %% [markdown]
# # Diagnosis of bootstrapping results
#
# Compares two sources of uncertainty at a representative (b-value, TD):
#
# 1. **Intra-ROI bootstrap** (`bootstrap_signal_from_density`):
#    Resamples cells within each ROI individually.
#    Captures measurement noise and cell-size variability inside one ROI.
#
# 2. **Inter-ROI bootstrap** (`bootstrap_signal_interROI`):
#    Resamples ROIs across a whole H&E slide.
#    Captures spatial heterogeneity between anatomical regions.
#
# The CI width ratio (inter / intra) indicates whether spatial variability
# between ROIs dominates over intra-ROI cell measurement variability.

# %%
# ── Parameters ────────────────────────────────────────────────────────────
diag_exp      = 3       # expansion coefficient
diag_bval     = 1.5     # representative b-value (ms/µm²)
diag_TD       = 19      # diffusion time (ms)
n_boot_diag   = 300
slice_width_um = 4
Dex_per_patient = {1: 2.4, 3: 2.4}   # best-estimate Dex per patient

tri = Delaunay(params_lut)

intra_rows = []
inter_rows = []

for pat_id in Patient:
    dex_central = Dex_per_patient.get(pat_id, 2.2)

    for he in H_E:
        cell_file  = f"Patient_{pat_id}_H&E_{he}_exp_{diag_exp}.txt"
        annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
        cell_path  = os.path.join(full_path_dir, cell_file)
        annot_path = os.path.join(full_path_dir, annot_file)

        if not os.path.exists(cell_path) or not os.path.exists(annot_path):
            continue

        df_cells = pd.read_csv(cell_path,  sep='\t', engine='python')
        df_annot = pd.read_csv(annot_path, sep='\t', engine='python')

        df_all_rois = {}
        roi_areas   = {}

        for roi in ["ROI1", "ROI2", "ROI3", "ROI4"]:
            df_r = df_cells[df_cells["Parent"] == roi]
            if len(df_r) == 0:
                continue
            area_vals = df_annot[df_annot["Name"] == roi]["Area µm^2"].to_numpy()
            if len(area_vals) == 0:
                continue
            area = area_vals[0]

            # ── Intra-ROI bootstrap ──────────────────────────────────────
            res_intra = bootstrap_signal_from_density(
                df_r, area, slice_width_um=slice_width_um,
                Dex_central=dex_central,
                bval=diag_bval, TD=diag_TD,
                tri=tri, signals_lut=signals_lut, params_lut=params_lut,
                n_bootstrap=n_boot_diag,
            )
            intra_rows.append({
                "Patient":       pat_id,
                "H&E":           he,
                "ROI":           roi,
                "n_cells":       len(df_r),
                "f_mean":        round(res_intra["f_mean"],  3),
                "f_IC95_low":    round(res_intra["f_p2.5"],  3),
                "f_IC95_high":   round(res_intra["f_p97.5"], 3),
                "S_mean":        round(res_intra["mean"],    4),
                "S_IC95_low":    round(res_intra["p2.5"],    4),
                "S_IC95_high":   round(res_intra["p97.5"],   4),
                "S_CI_width":    round(res_intra["p97.5"] - res_intra["p2.5"], 5),
            })

            df_all_rois[roi] = df_r
            roi_areas[roi]   = area

        # ── Inter-ROI bootstrap (≥2 ROIs required) ──────────────────────
        if len(df_all_rois) < 2:
            inter_rows.append({
                "Patient": pat_id, "H&E": he,
                "n_ROIs": len(df_all_rois),
                "S_mean": None, "S_IC95_low": None, "S_IC95_high": None,
                "S_CI_width": None, "note": "skipped (<2 ROIs)",
            })
            continue

        res_inter = bootstrap_signal_interROI(
            df_all_rois, list(df_all_rois.keys()), roi_areas,
            slice_width_um=slice_width_um, Dex_central=dex_central,
            bval=diag_bval, TD=diag_TD,
            tri=tri, signals_lut=signals_lut, params_lut=params_lut,
            n_bootstrap=n_boot_diag,
        )
        inter_rows.append({
            "Patient":     pat_id,
            "H&E":         he,
            "n_ROIs":      len(df_all_rois),
            "S_mean":      round(res_inter["mean"],    4),
            "S_IC95_low":  round(res_inter["p2.5"],    4),
            "S_IC95_high": round(res_inter["p97.5"],   4),
            "S_CI_width":  round(res_inter["p97.5"] - res_inter["p2.5"], 5),
            "note":        "",
        })

# ── Display ───────────────────────────────────────────────────────────────
df_intra = pd.DataFrame(intra_rows)
df_inter = pd.DataFrame(inter_rows)

print("=" * 75)
print(f"INTRA-ROI BOOTSTRAP  (b={diag_bval} ms/µm², TD={diag_TD} ms, exp={diag_exp})")
print("=" * 75)
print(df_intra.to_string(index=False))

print("\n" + "=" * 75)
print(f"INTER-ROI BOOTSTRAP  (b={diag_bval} ms/µm², TD={diag_TD} ms, exp={diag_exp})")
print("=" * 75)
print(df_inter.to_string(index=False))

# ── CI width comparison ───────────────────────────────────────────────────
print("\n── CI width comparison (inter / intra median) ──")
for pat_id in Patient:
    for he in H_E:
        intra_sub = df_intra[
            (df_intra["Patient"] == pat_id) & (df_intra["H&E"] == he)
        ]
        inter_sub = df_inter[
            (df_inter["Patient"] == pat_id) & (df_inter["H&E"] == he)
        ]
        if intra_sub.empty or inter_sub.empty:
            continue
        inter_row = inter_sub.iloc[0]
        if inter_row["S_CI_width"] is None:
            continue
        intra_med = intra_sub["S_CI_width"].median()
        ratio = inter_row["S_CI_width"] / intra_med if intra_med > 0 else float("nan")
        print(f"  Patient {pat_id} | H&E {he} : "
              f"intra median CI={intra_med:.5f}  "
              f"inter CI={inter_row['S_CI_width']:.5f}  "
              f"ratio={ratio:.2f}  "
              f"({'inter > intra: spatial variability dominates' if ratio > 1 else 'intra > inter: cell measurement variability dominates'})")

# %% [markdown]
# # Spatial bootstrap — random windows over the full histological slide

# %%
def bootstrap_spatial_roi(df_cells_full, df_annot_full,
                           roi_size_um, slice_width_um,
                           Dex_best, bval, TD,
                           tri, signals_lut, params_lut,
                           n_bootstrap=200,
                           allowed_rois=None,
                           k_shrinkage=1.0,
                           k_det=1.0):
    """
    Spatial bootstrap: draws random sub-regions (moving windows) over the
    tissue, constrained to the bounding boxes of the annotated ROIs.

    This avoids the selection bias introduced by sampling from the full
    slide bounding box, which includes background/non-tissue areas and
    causes only the densest windows to be accepted.

    Parameters
    ----------
    df_cells_full : QuPath DataFrame of ALL cells (must include 'Parent'
                    column to identify which ROI each cell belongs to,
                    and 'Centroid X µm' / 'Centroid Y µm' coordinates).
    df_annot_full : QuPath annotation DataFrame (used for ROI area info;
                    the actual bounding boxes are derived from cell centroids).
    roi_size_um   : side length of the square sampling window in µm.
                    Should be comparable to the smallest annotated ROI.
    allowed_rois  : list of ROI names to use as valid tissue regions
                    (defaults to all ROIs found in df_cells_full["Parent"]).

    Notes on the f > 0.8 bug in the previous version
    -------------------------------------------------
    The old version sampled windows from the full bounding box of all cells
    on the slide. Windows outside the tissue were sparse (< 10 cells) and
    discarded, so only the densest windows survived — a strong selection
    bias. Here, windows are placed only within existing annotated ROI
    bounding boxes, which represent validated tissue regions, and the
    empty-window threshold is removed (sparse windows are biologically
    informative too).
    """
    if "Centroid X µm" not in df_cells_full.columns:
        raise ValueError("Column 'Centroid X µm' missing — check QuPath export.")
    if "Parent" not in df_cells_full.columns:
        raise ValueError("Column 'Parent' missing — re-export with ROI parent info.")

    x_all = df_cells_full["Centroid X µm"].to_numpy()
    y_all = df_cells_full["Centroid Y µm"].to_numpy()

    # ── Build per-ROI sampling zones from annotation Rectangle geometry ──
    # The annotations are QuPath Rectangles drawn entirely within tissue.
    # We reconstruct the exact rectangle bounds from the centroid + area +
    # perimeter exported by QuPath (available in df_annot_full), instead of
    # using the cell-centroid bounding box.
    #
    # Why not use cell centroids?
    # Cell centroids do not reach the annotation boundary — there is always
    # a margin of ~r_cell between the last centroid and the rectangle edge.
    # Using the centroid bounding box therefore restricts window placement to
    # a smaller, denser sub-region, causing f to be systematically overestimated.
    #
    # Using the annotation rectangle gives exactly the same denominator area
    # as density_results.json (which uses "Area µm^2" from QuPath directly),
    # consistent with Abercrombie: N_v = n / ((t + 2r) * A_annotation).
    if allowed_rois is None:
        roi_names_all = df_cells_full["Parent"].unique()
    else:
        roi_names_all = allowed_rois

    roi_boxes = []
    for roi_name in roi_names_all:
        mask_roi = df_cells_full["Parent"] == roi_name
        if mask_roi.sum() < 5:
            continue

        # Retrieve annotation geometry from df_annot_full
        row = df_annot_full[df_annot_full["Name"] == roi_name]
        if len(row) == 0:
            print(f"  [bootstrap_spatial_roi] '{roi_name}' not found in annotation file — skipped.")
            continue

        cx       = float(row["Centroid X µm"].values[0])
        cy       = float(row["Centroid Y µm"].values[0])
        area     = float(row["Area µm^2"].values[0])
        perim    = float(row["Perimeter µm"].values[0])

        # Rectangle dimensions from area and perimeter:
        #   W + H = P/2,  W * H = A  →  quadratic t^2 - (P/2)*t + A = 0
        half_p = perim / 2
        disc   = half_p**2 - 4 * area
        if disc >= 0:
            sqrt_d = np.sqrt(disc)
            W = (half_p + sqrt_d) / 2
            H = (half_p - sqrt_d) / 2
        else:
            # Fallback: assume square
            W = H = np.sqrt(area)

        x_min = cx - W / 2
        x_max = cx + W / 2
        y_min = cy - H / 2
        y_max = cy + H / 2

        print(f"  [annotation bounds] {roi_name}: "
              f"centre=({cx:.0f},{cy:.0f}) | {W:.0f}×{H:.0f}µm | area={area:.0f}µm²")

        if (x_max - x_min) < roi_size_um or (y_max - y_min) < roi_size_um:
            print(f"  [bootstrap_spatial_roi] ROI '{roi_name}' smaller than "
                  f"window ({roi_size_um:.0f}µm) — skipped.")
            continue

        roi_boxes.append({
            "name":  roi_name,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "placeable_area": (x_max - x_min - roi_size_um) * (y_max - y_min - roi_size_um),
        })

    if not roi_boxes:
        raise ValueError(
            f"No annotated ROI is large enough for a {roi_size_um:.0f} µm window. "
            "Reduce roi_size_um or annotate larger ROIs."
        )

    # Weight ROIs proportionally to their 'placeable area' so that larger
    # ROIs contribute more replicates (unbiased spatial coverage).
    placeable_areas = np.array([r["placeable_area"] for r in roi_boxes])
    weights = placeable_areas / placeable_areas.sum()

    roi_area_um2 = roi_size_um ** 2
    signals_boot = []
    f_boot_list  = []
    attempts     = 0
    max_attempts = n_bootstrap * 30  # safety cap

    while len(f_boot_list) < n_bootstrap and attempts < max_attempts:
        attempts += 1

        # 1. Pick a ROI proportional to its placeable area
        chosen = roi_boxes[np.random.choice(len(roi_boxes), p=weights)]

        # 2. Place window fully inside that ROI's bounding box
        x0 = np.random.uniform(chosen["x_min"], chosen["x_max"] - roi_size_um)
        y0 = np.random.uniform(chosen["y_min"], chosen["y_max"] - roi_size_um)

        # 3. Count cells within the window (across all ROIs, not just the
        #    chosen one, so boundary cells are not artificially excluded)
        mask = (
            (x_all >= x0) & (x_all <= x0 + roi_size_um) &
            (y_all >= y0) & (y_all <= y0 + roi_size_um)
        )
        df_window = df_cells_full[mask]
        n_cells = len(df_window)

        # Accept even empty windows — they represent low-density tissue regions.
        # A minimum of 2 cells is kept only to avoid degenerate std estimates.
        if n_cells < 2:
            f_boot_list.append(0.0)
            signals_boot.append(1.0)   # S/S0 → 1 at f ≈ 0
            continue

        # 4. Estimate biological parameters from the window cells
        max_cal = df_window["Cell: Max caliper"].to_numpy()
        min_cal = df_window["Cell: Min caliper"].to_numpy()

        mean_max = np.mean(max_cal)
        mean_min = np.mean(min_cal)
        std_max  = np.std(max_cal,  ddof=1) if n_cells > 1 else 0.0
        std_min  = np.std(min_cal,  ddof=1) if n_cells > 1 else 0.0

        r2D = np.sqrt(mean_max * mean_min / 4)
        std_r2D = 0.25 * np.sqrt(
            (mean_max / mean_min) * std_min**2 +
            (mean_min / mean_max) * std_max**2
        ) if n_cells > 1 else 0.0

        # Shrinkage correction on radii: histology shrinks tissue by factor k,
        # so the in-vivo radius is r_hist / k (physically larger cells in vivo).
        # Volume fraction f is conserved under uniform isotropic shrinkage,
        # but the Abercrombie denominator (slice_width + 2*r) changes with k.
        r2D_vivo    = r2D    / k_shrinkage
        std_r2D_vivo = std_r2D / k_shrinkage
        r3D     = 1.27 * r2D_vivo
        std_r3D = 1.27 * std_r2D_vivo

        # 5. Abercrombie 3D density and volume fraction
        # Same formula as density_results.json:
        #   N_v = n / ((slice_width + 2*r2D_vivo) * A)
        # where A = roi_size_um² = area of the sampling window.
        # Windows are fully inside the annotation rectangle, so A is correct.
        N_v    = n_cells / ((slice_width_um + 2 * r2D_vivo) * roi_area_um2)
        V_cell = (4/3) * np.pi * r3D**3
        f_boot = N_v * V_cell
        f_boot_list.append(f_boot)

        # Apply detection correction factor: f_eff = f_vivo * k_det
        # k_det accounts for the systematic underestimation of cell density
        # by QuPath (detection threshold, 2D→3D projection artefacts).
        # Must be consistent with how f is used in infer_Dex_shrinkage_chi2.
        f_eff = f_boot * k_det

        if f_boot > 1.0:
            print(f"  [WARNING f>1] ROI={chosen['name']} | "
                  f"window=({x0:.1f},{y0:.1f})+{roi_size_um:.1f}µm | "
                  f"n_cells={n_cells} | r2D={r2D:.2f}µm | "
                  f"roi_area={roi_area_um2:.0f}µm² | "
                  f"N_v={N_v:.5f} | f_vivo={f_boot:.3f} | f_eff={f_eff:.3f}")

        sig = interpolate_signal(
            f_eff, Dex_best, r3D, std_r3D,
            bval, TD, tri, signals_lut, params_lut
        )
        signals_boot.append(sig if not np.isnan(sig) else np.nan)

    signals_boot = np.array(signals_boot)
    f_boot_arr   = np.array(f_boot_list)

    # Drop NaN signals for statistics (keep f distribution intact)
    valid_mask   = ~np.isnan(signals_boot)
    signals_valid = signals_boot[valid_mask]

    print(f"Valid replicates: {len(signals_valid)} / {n_bootstrap} "
          f"(attempts: {attempts})")
    print(f"ROIs used: {[r['name'] for r in roi_boxes]}")
    print(f"f_boot: {np.mean(f_boot_arr):.4f} ± {np.std(f_boot_arr):.4f} "
          f"[{np.percentile(f_boot_arr, 2.5):.4f}, "
          f"{np.percentile(f_boot_arr, 97.5):.4f}]")

    if len(signals_valid) == 0:
        raise RuntimeError("No valid signal replicates — check LUT coverage.")

    return {
        "mean":   float(np.mean(signals_valid)),
        "p2.5":   float(np.percentile(signals_valid, 2.5)),
        "p97.5":  float(np.percentile(signals_valid, 97.5)),
        "std":    float(np.std(signals_valid)),
        "f_mean": float(np.mean(f_boot_arr)),
        "f_std":  float(np.std(f_boot_arr)),
        "f_p2.5": float(np.percentile(f_boot_arr, 2.5)),
        "f_p97.5":float(np.percentile(f_boot_arr, 97.5)),
        "n_valid": int(len(signals_valid)),
    }


def bootstrap_signal_poisson(f_central, r3D, std_r3D, n_cells_obs,
                              roi_area_um2, slice_width_um, r2D,
                              Dex_central, bval, TD,
                              tri, signals_lut, params_lut,
                              n_bootstrap=300):
    """
    Poisson bootstrap for cell density uncertainty.

    Models the cell count in a fixed volume as a Poisson process:
      n ~ Poisson(lambda),  lambda = N_v * V_sample
    The MLE is lambda = n_cells_obs, so we draw counts from Poisson(n_cells_obs).

    This is the standard parametric approach when the ROIs are too small or
    too few to support spatial resampling. It gives a theoretically motivated
    SE(N_v) = N_v / sqrt(n_cells_obs), i.e. ~10% for n=100 cells.

    The cell geometry (r3D, std_r3D) is kept fixed at the point estimate
    because it is dominated by the systematic expansion-factor bias rather
    than sampling noise.

    Parameters
    ----------
    f_central    : central estimate of f from Abercrombie + QuPath
    r3D, std_r3D : 3D radius and std at the point estimate (fixed across replicates)
    n_cells_obs  : observed cell count in the ROI
    roi_area_um2 : ROI area in µm²
    slice_width_um : histological section thickness (µm)
    r2D          : 2D radius (used only in the Abercrombie denominator)
    """
    V_sample = (slice_width_um + 2 * r2D) * roi_area_um2
    V_cell   = (4/3) * np.pi * r3D**3

    signals_boot = []
    f_boot_list  = []

    for _ in range(n_bootstrap):
        n_boot = np.random.poisson(n_cells_obs)
        if n_boot == 0:
            f_boot_list.append(0.0)
            signals_boot.append(1.0)
            continue
        N_v_boot = n_boot / V_sample
        f_boot   = N_v_boot * V_cell
        f_boot_list.append(f_boot)

        sig = interpolate_signal(f_boot, Dex_central, r3D, std_r3D,
                                 bval, TD, tri, signals_lut, params_lut)
        signals_boot.append(sig if not np.isnan(sig) else np.nan)

    signals_boot = np.array(signals_boot)
    f_boot_arr   = np.array(f_boot_list)
    valid         = ~np.isnan(signals_boot)
    signals_valid = signals_boot[valid]

    se_f_theoretical = f_central / np.sqrt(n_cells_obs)
    print(f"Poisson bootstrap — n_cells={n_cells_obs}, "
          f"SE(f) theoretical={se_f_theoretical:.4f}, "
          f"SE(f) empirical={np.std(f_boot_arr):.4f}")

    return {
        "mean":   float(np.mean(signals_valid)),
        "p2.5":   float(np.percentile(signals_valid, 2.5)),
        "p97.5":  float(np.percentile(signals_valid, 97.5)),
        "std":    float(np.std(signals_valid)),
        "f_mean": float(np.mean(f_boot_arr)),
        "f_std":  float(np.std(f_boot_arr)),
        "f_p2.5": float(np.percentile(f_boot_arr, 2.5)),
        "f_p97.5":float(np.percentile(f_boot_arr, 97.5)),
        "n_valid": int(len(signals_valid)),
    }


# %% [markdown]
# # Run spatial bootstrap and save results in JSON

# %%
SPATIAL_BOOTSTRAP_FILE = "signal_results_spatial_bootstrap.json"

_RUN_BOOTSTRAP = False  # set True only after FIM/MCMC validates the inference model

if not os.path.exists(SPATIAL_BOOTSTRAP_FILE) and _RUN_BOOTSTRAP:

    # Consensus pondéré par 1/chi2_min depuis df_summary (nouvelle structure).
    # k_det est partagé entre TDs → pas de filtre sur TD.
    # Dex est TD-dépendant → utilise Dex_19_best ou Dex_49_best selon TD.
    def _weighted_consensus_pat(pat_id: int, col: str) -> float:
        sub = df_summary[df_summary["Patient"] == pat_id].copy()
        if sub.empty:
            raise ValueError(f"Aucune ligne dans df_summary pour Patient={pat_id}")
        w = 1.0 / sub["chi2_min"].replace(0, np.nan).dropna()
        return float(np.average(sub.loc[w.index, col], weights=w.to_numpy()))

    def _dex_consensus(pat_id: int, TD: int) -> float:
        col = "Dex_19_best" if TD == 19 else "Dex_49_best"
        return _weighted_consensus_pat(pat_id, col)

    def _kdet_consensus(pat_id: int, TD: int) -> float:  # TD unused, kept for API compat
        return _weighted_consensus_pat(pat_id, "k_det_best")

    bval_options = [0.1,  0.2,  0.5,  0.7,  1.3,  1.4,  2.6,  3.8,  4.2,  6.0,   7.3,  12.0,  17.8]
    TD_options     = [19, 49]
    slice_width_um = 4
    tri            = Delaunay(params_lut)

    spatial_json = {}

    for pat_id in Patient:
        spatial_json[f"Patient_{pat_id}"] = {}

        for he in H_E:
            spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"] = {}

            for exp in expansion:
                spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"] = {}

                # Load full-slide cell file (all ROIs combined)
                annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
                annot_path = os.path.join(full_path_dir, annot_file)
                if not os.path.exists(annot_path):
                    continue
                df_annot       = pd.read_csv(annot_path, sep='\t', engine='python')
                df_annot_sorted = df_annot.sort_values(by="Name")
                areas          = df_annot_sorted["Area µm^2"].to_numpy()
                roi_size_um    = np.sqrt(np.min(areas))/2   # window = average ROI size
                print(f"  Patient {pat_id} | H&E {he} | exp {exp} | "
                      f"roi_size_um={roi_size_um:.1f}µm "
                      f"(from min area={np.min(areas):.0f}µm²) | "
                      f"annotations={list(df_annot_sorted['Name'].unique())}")

                cell_file = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
                cell_path = os.path.join(full_path_dir, cell_file)
                if not os.path.exists(cell_path):
                    continue
                df_cells_full = pd.read_csv(cell_path, sep='\t', engine='python')

                # Keep only cells that belong to a named annotation ROI.
                # QuPath exports ALL detected cells including those outside
                # any annotation (Parent = "Root object (Image)"). Counting
                # those in the bootstrap window inflates n_cells and f.
                valid_parents = list(df_annot_sorted["Name"].unique())
                n_before = len(df_cells_full)
                df_cells_full = df_cells_full[df_cells_full["Parent"].isin(valid_parents)]
                print(f"  [filter] {n_before} → {len(df_cells_full)} cells "
                      f"(kept Parent ∈ {valid_parents})")

                # "full_slide" plays the same role as a ROI key in the original format
                spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"] = {}

                for TD in TD_options:
                    spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"][f"TD_{TD}"] = {}

                    means, p_lows, p_highs, stds = [], [], [], []

                    # Only use validated ROI annotations as sampling zones
                    valid_roi_names = list(df_annot_sorted["Name"].unique())

                    # Dex et k_det consensus pondérés pour ce patient et ce TD
                    Dex_best_val  = _dex_consensus(pat_id, TD)
                    k_det_val     = _kdet_consensus(pat_id, TD)
                    print(f"  [consensus] Patient {pat_id} | TD={TD} ms : "
                          f"Dex = {Dex_best_val:.3f} µm²/ms | "
                          f"k_det = {k_det_val:.3f}")

                    for bval in bval_options:
                        res = bootstrap_spatial_roi(
                            df_cells_full  = df_cells_full,
                            df_annot_full  = df_annot_sorted,
                            roi_size_um    = roi_size_um,
                            slice_width_um = slice_width_um,
                            Dex_best       = Dex_best_val,
                            bval           = bval,
                            TD             = TD,
                            tri            = tri,
                            signals_lut    = signals_lut,
                            params_lut     = params_lut,
                            n_bootstrap    = 300,
                            allowed_rois   = valid_roi_names,
                            k_det          = k_det_val,
                        )
                        means.append(res["mean"])
                        stds.append(res["std"])
                        p_lows.append(res["p2.5"])
                        p_highs.append(res["p97.5"])

                    spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"][f"TD_{TD}"][f"Dex_{Dex_best_val:.3f}"] = {
                        "bval":         bval_options,
                        "signals_mean": means,
                        "signals_std":  stds,
                        "p_low":        p_lows,
                        "p_high":       p_highs
                    }

                print(f"  Patient {pat_id} | H&E {he} | exp {exp} | TD={TD}")

    with open(SPATIAL_BOOTSTRAP_FILE, "w") as f:
        json.dump(spatial_json, f, indent=4)
    print(f"\nJSON saved: {SPATIAL_BOOTSTRAP_FILE}")

else:
    with open(SPATIAL_BOOTSTRAP_FILE, "r", encoding="utf-8") as f:
        spatial_json = json.load(f)
    print(f"Spatial bootstrap loaded from {SPATIAL_BOOTSTRAP_FILE}")


# %% [markdown]
# # Shrinkage sensitivity: spatial bootstrap with k ∈ {0.7, 0.8, 0.9}
#
# Physical correction: r_vivo = r_hist / k (cells are larger in vivo).
# Volume fraction f is theoretically conserved under uniform isotropic shrinkage,
# but the Abercrombie denominator changes with k → small residual effect on f.
# k=1.0 corresponds to the reference bootstrap (no shrinkage correction).

# %%
SHRINKAGE_SPATIAL_FILE = "signal_results_spatial_bootstrap_shrinkage.json"

if not os.path.exists(SHRINKAGE_SPATIAL_FILE):

    k_values     = [1.0, 0.9, 0.8, 0.7]
    Dex_options  = [2.4, 2.4]  # patient 1, patient 3
    bval_options = [
        [0.05, 0.35, 0.8, 1.5, 2.4, 3.45, 4.75, 6],
        [0.2, 0.95, 2.3, 4.25, 6.75, 9.85, 13.5, 17.8]
    ]
    TD_options     = [19, 49]
    slice_width_um = 4
    tri            = Delaunay(params_lut)

    shrinkage_spatial_json = {}

    for pat_id in Patient:
        shrinkage_spatial_json[f"Patient_{pat_id}"] = {}

        for he in H_E:
            shrinkage_spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"] = {}

            for exp in expansion:
                shrinkage_spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"] = {}

                annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
                annot_path = os.path.join(full_path_dir, annot_file)
                if not os.path.exists(annot_path):
                    continue
                df_annot        = pd.read_csv(annot_path, sep='\t', engine='python')
                df_annot_sorted = df_annot.sort_values(by="Name")
                areas           = df_annot_sorted["Area µm^2"].to_numpy()
                roi_size_um     = np.sqrt(np.min(areas)) / 2

                cell_file = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
                cell_path = os.path.join(full_path_dir, cell_file)
                if not os.path.exists(cell_path):
                    continue
                df_cells_full = pd.read_csv(cell_path, sep='\t', engine='python')
                valid_parents = list(df_annot_sorted["Name"].unique())
                df_cells_full = df_cells_full[df_cells_full["Parent"].isin(valid_parents)]

                valid_roi_names = list(df_annot_sorted["Name"].unique())
                Dex_best_val    = Dex_options[Patient.index(pat_id)]

                shrinkage_spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"] = {}

                for j, TD in enumerate(TD_options):
                    shrinkage_spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"][f"TD_{TD}"] = {}

                    for k in k_values:
                        means, p_lows, p_highs, stds = [], [], [], []

                        for bval in bval_options[j]:
                            res = bootstrap_spatial_roi(
                                df_cells_full  = df_cells_full,
                                df_annot_full  = df_annot_sorted,
                                roi_size_um    = roi_size_um,
                                slice_width_um = slice_width_um,
                                Dex_best       = Dex_best_val,
                                bval           = bval,
                                TD             = TD,
                                tri            = tri,
                                signals_lut    = signals_lut,
                                params_lut     = params_lut,
                                n_bootstrap    = 300,
                                allowed_rois   = valid_roi_names,
                                k_shrinkage    = k,
                            )
                            means.append(res["mean"])
                            stds.append(res["std"])
                            p_lows.append(res["p2.5"])
                            p_highs.append(res["p97.5"])

                        shrinkage_spatial_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"]["full_slide"][f"TD_{TD}"][f"k_{k}"] = {
                            "bval":         bval_options[j],
                            "signals_mean": means,
                            "signals_std":  stds,
                            "p_low":        p_lows,
                            "p_high":       p_highs,
                        }

                        print(f"  Patient {pat_id} | H&E {he} | exp {exp} | TD={TD} | k={k} done")

    with open(SHRINKAGE_SPATIAL_FILE, "w") as fh:
        json.dump(shrinkage_spatial_json, fh, indent=4)
    print(f"\nJSON saved: {SHRINKAGE_SPATIAL_FILE}")

else:
    with open(SHRINKAGE_SPATIAL_FILE, "r", encoding="utf-8") as fh:
        shrinkage_spatial_json = json.load(fh)
    print(f"Shrinkage spatial bootstrap loaded from {SHRINKAGE_SPATIAL_FILE}")


# %% [markdown]
# # Run bootstrapping for cells variability and save results in JSON

# %%
BOOTSTRAP_FILE = "signal_results_bootstrap.json"

if not os.path.exists(BOOTSTRAP_FILE):

    Din = 1
    Dex_options = [1.5, 2.2, 2.5]
    bval_options = [ #NEEDS TO BE UPDATED FOR THE NEW LUT
        [0.05, 0.35, 0.8, 1.5, 2.4, 3.45, 4.75, 6],
        [0.2, 0.95, 2.3, 4.25, 6.75, 9.85, 13.5, 17.8]
    ]
    TD_options = [19, 49]
    slice_width_um = 4
    tri = Delaunay(params_lut)

    bootstrap_json = {}

    for pat_id in Patient:
        bootstrap_json[f"Patient_{pat_id}"] = {}

        for he in H_E:
            bootstrap_json[f"Patient_{pat_id}"][f"H&E_{he}"] = {}

            for exp in expansion:
                bootstrap_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"] = {}

                # Loading Qupath data
                annot_file = f"Patient_{pat_id}_H&E_{he}_annot.txt"
                annot_path = os.path.join(full_path_dir, annot_file)
                if not os.path.exists(annot_path):
                    continue

                df_annot  = pd.read_csv(annot_path, sep='\t', engine='python')
                df_annot_sorted = df_annot.sort_values(by="Name")
                names  = df_annot_sorted["Name"].to_numpy()
                areas  = df_annot_sorted["Area µm^2"].to_numpy()

                cell_file = f"Patient_{pat_id}_H&E_{he}_exp_{exp}.txt"
                cell_path = os.path.join(full_path_dir, cell_file)
                if not os.path.exists(cell_path):
                    continue

                df_cells = pd.read_csv(cell_path, sep='\t', engine='python')

                for i, roi_name in enumerate(names):
                    if roi_name not in {"ROI1", "ROI2", "ROI3", "ROI4"}:
                        continue

                    df_roi = df_cells[df_cells["Parent"] == roi_name]
                    if len(df_roi) == 0:
                        continue

                    # Raw data for bootstrap
                    cell_areas   = df_roi["Cell: Area"].to_numpy()
                    max_calipers = df_roi["Cell: Max caliper"].to_numpy()
                    min_calipers = df_roi["Cell: Min caliper"].to_numpy()
                    roi_area_um2 = areas[i]

                    bootstrap_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name] = {}

                    for j, TD in enumerate(TD_options):
                        bootstrap_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name][f"TD_{TD}"] = {}

                        for Dex in Dex_options:
                            means, p_lows, p_highs, stds, medians = [], [], [], [], []

                            for bval in bval_options[j]:
                                res = bootstrap_signal_from_cells(
                                    cell_areas, max_calipers, min_calipers,
                                    roi_area_um2, slice_width_um,
                                    Dex_central=Dex,
                                    bval=bval, TD=TD,
                                    tri=tri,
                                    signals_lut=signals_lut,
                                    params_lut=params_lut,
                                    n_bootstrap=300
                                )
                                means.append(res["mean"])
                                medians.append(res["median"])
                                stds.append(res["std"])
                                p_lows.append(res["p2.5"])
                                p_highs.append(res["p97.5"])

                            bootstrap_json[f"Patient_{pat_id}"][f"H&E_{he}"][f"exp_{exp}"][roi_name][f"TD_{TD}"][f"Dex_{Dex}"] = {
                                "bval":          bval_options[j],
                                "signals_mean":  means,
                                "signals_median": medians,
                                "signals_std":   stds,
                                "p_low":         p_lows,   # percentile 2.5%
                                "p_high":        p_highs   # percentile 97.5%
                            }

                        print(f"  ✓ Patient {pat_id} | H&E {he} | exp {exp} | {roi_name} | TD={TD}")

    with open(BOOTSTRAP_FILE, "w") as f:
        json.dump(bootstrap_json, f, indent=4)
    print(f"\nJSON saved : {BOOTSTRAP_FILE}")

else:
    with open(BOOTSTRAP_FILE, "r", encoding="utf-8") as f:
        bootstrap_json = json.load(f)
    print(f"Bootstrap loaded from {BOOTSTRAP_FILE}")


# %% [markdown]
# # Export results for validation.py

# %%
df_summary.to_csv("df_summary.csv", index=False)
print(f"df_summary saved → df_summary.csv  ({len(df_summary)} rows)")

# %%
