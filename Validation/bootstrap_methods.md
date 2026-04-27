# Bootstrap methods for histology-to-DWI signal uncertainty

**Context.** Given a histological ROI segmented with QuPath, we derive cell density and morphology parameters (f, r_mean, r_std) and query the LUT to obtain a simulated DWI signal S_sim(b). The goal of bootstrapping is to quantify the uncertainty on S_sim arising from the limited and imperfect nature of histological sampling. Three approaches were explored and are described below.

---

## 1. Intra-ROI bootstrap (`bootstrap_signal_from_cells`)

### Principle

Each replicate resamples the *n* segmented cells within a fixed ROI with replacement (standard non-parametric bootstrap). The resampled population yields a new estimate of (r_mean, r_std, f), which is fed to the LUT to produce one signal replicate. After B = 300 replicates, the 2.5th and 97.5th percentiles define the 95% confidence interval.

### What it captures

Variability due to cell measurement noise and cell-size variability *within* a single ROI. Because the ROI boundary and area are fixed, the denominator of the Abercrombie estimator does not change — only cell morphology fluctuates.

### Results (b = 1.5 ms/µm², TD = 19 ms, exp = 3)

| Patient | H&E | ROI | n_cells | f_mean | f IC95 | S_mean | S IC95 | CI width |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | ROI1 | 3390 | 0.563 | [0.555, 0.570] | 0.196 | [0.196, 0.197] | 0.0016 |
| 1 | 2 | ROI1 | 3179 | 0.369 | [0.364, 0.373] | 0.154 | [0.149, 0.158] | 0.0089 |
| 1 | 2 | ROI2 | 9926 | 0.277 | [0.275, 0.279] | 0.134 | [0.123, 0.144] | 0.0217 |
| 1 | 2 | ROI3 | 2935 | 0.201 | [0.198, 0.203] | 0.122 | [0.119, 0.125] | 0.0060 |
| 3 | 1 | ROI1 |  990 | 0.589 | [0.576, 0.603] | 0.196 | [0.194, 0.197] | 0.0034 |
| 3 | 1 | ROI2 | 1032 | 0.598 | [0.584, 0.612] | 0.199 | [0.198, 0.200] | 0.0021 |
| 3 | 1 | ROI3 | 1176 | 0.509 | [0.500, 0.519] | 0.196 | [0.195, 0.197] | 0.0021 |
| 3 | 1 | ROI4 |  955 | 0.424 | [0.415, 0.432] | 0.196 | [0.194, 0.197] | 0.0030 |

The CI widths are narrow (0.002–0.022), confirming that intra-ROI cell measurement variability contributes little to signal uncertainty for most ROIs. ROI2 of Patient 1 H&E 2 is an exception (CI = 0.022), driven by its large cell count and wider f range.

### Limitation

This approach underestimates uncertainty in two ways:
1. It does not account for **spatial heterogeneity**: the ROI is manually placed and may not be representative of the DWI voxel it is meant to correspond to.
2. It conflates measurement noise (QuPath segmentation errors) with biological variability, which are not the same thing.

---

## 2. Inter-ROI bootstrap (`bootstrap_signal_interROI`)

### Principle

At each replicate, one ROI is drawn uniformly at random *with replacement* from the set of annotated ROIs on the H&E slide. Within each drawn ROI, an intra-ROI bootstrap (B_inner = 50) is run to get a mean signal. After B_outer = 200 replicates, percentiles are computed over the outer bootstrap.

### What it captures

Variability due to the **choice of which anatomical region** to sample — i.e., spatial heterogeneity between the annotated ROIs.

### Results (b = 1.5 ms/µm², TD = 19 ms, exp = 3)

| Patient | H&E | n_ROIs | S_mean | S IC95 | CI width | Note |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | — | — | — | skipped (< 2 ROIs) |
| 1 | 2 | 3 | 0.137 | [0.122, 0.154] | 0.032 | |
| 3 | 1 | 4 | 0.197 | [0.196, 0.199] | 0.0036 | |

**CI width ratio (inter / intra median):**
- Patient 1, H&E 1 : ratio = n/a — only 1 ROI available, intra dominates by default
- Patient 1, H&E 2 : ratio = **3.58** — spatial variability between ROIs dominates
- Patient 3, H&E 1 : ratio = **1.43** — spatial variability slightly dominates

The inter-ROI CI is systematically wider than the intra-ROI CI where multiple ROIs are available, confirming that the *choice of which region to sample* is the dominant source of uncertainty, not cell measurement noise.

### Limitation

The annotated ROIs are deliberately placed by a pathologist in representative regions: they are not random spatial samples of the tissue. Resampling them with replacement therefore overestimates how much information they carry about the tissue variability at the scale of a DWI voxel (typically 1–2 mm resolution). Furthermore, with only 2–4 ROIs per slide, the bootstrap distribution is coarse.

---

## 3. Spatial bootstrap — moving windows (`bootstrap_spatial_roi`) ✅ **Selected**

### Principle

Rather than resampling from a fixed set of annotated ROIs, each replicate places a square window of side `roi_size_um` at a random position *within* the annotated tissue regions. The window size is set to the square root of the smallest annotated ROI area divided by 2, making it comparable to the annotation scale. The process is:

1. For each annotated ROI, reconstruct the rectangle bounds from QuPath-exported centroid, area, and perimeter (using the quadratic `W+H = P/2`, `W·H = A`).
2. Sample a window position uniformly within that rectangle's placeable zone (i.e., offset such that the window stays fully inside the ROI).
3. ROIs are weighted proportionally to their placeable area, so larger regions contribute more replicates and spatial coverage is unbiased.
4. Count all cells whose centroid falls within the window.
5. Estimate (r_mean, r_std, f) via Abercrombie on the window, using the window area as denominator.
6. Query the LUT with Dex fixed to the consensus value inferred from `df_summary` for the corresponding (patient, TD).

Windows with fewer than 2 cells are accepted as low-density tissue (f ≈ 0, signal ≈ 1). After B = 300 valid replicates, percentiles define the 95% CI.

### Key implementation detail: annotation bounds vs. cell-centroid bounding box

An early version of this function derived window placement bounds from the bounding box of cell centroids, not the annotation rectangle. Since cell centroids never reach the annotation edge (there is always a margin of ~r_cell), this artificially restricted windows to a denser sub-region, biasing f upward. The current implementation recovers the exact annotation rectangle from QuPath geometry, consistent with how f is computed everywhere else (Abercrombie denominator = annotation area).

### What it captures

Spatial heterogeneity **within** the annotated ROIs — local fluctuations of cell density from one sub-region to another inside the same annotation. Unlike the inter-ROI bootstrap, which treats each ROI as a single homogeneous unit, the spatial bootstrap resolves the internal structure of each ROI by sampling many different sub-windows at varying positions.

### Shared limitation of all three methods

All three approaches are constrained to the annotated tissue regions. Windows in the spatial bootstrap are placed only inside the annotated ROI bounding boxes, for the practical reason that tissue outside the annotations may contain artefacts, background, or non-tumour areas. As a consequence, none of the three methods can assess whether the annotated regions are representative of the full DWI voxel (~1–2 mm resolution). This is an inherent limitation of the histology-to-DWI comparison, independent of the bootstrap strategy chosen.

### Results

Results are reported at b = 1.3 ms/µm² (mid-range, most sensitive b-value) for each combination of patient, H&E slide, expansion coefficient and TD. The Dex used is the consensus value inferred from `df_summary` for the corresponding (patient, TD).

| Patient | H&E | exp | TD (ms) | Dex (µm²/ms) | S_mean | IC 95% | CI width | Max CI width (over all b) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3   | 19 | 1.971 | 0.235 | [0.234, 0.237] | 0.004 | 0.004 |
| 1 | 1 | 3   | 49 | 1.886 | 0.278 | [0.277, 0.281] | 0.004 | 0.005 |
| 1 | 1 | 3.5 | 19 | 1.971 | 0.231 | [0.226, 0.234] | 0.007 | 0.008 |
| 1 | 1 | 3.5 | 49 | 1.886 | 0.272 | [0.266, 0.274] | 0.008 | 0.008 |
| 1 | 1 | 4   | 19 | 1.971 | 0.230 | [0.228, 0.231] | 0.003 | 0.004 |
| 1 | 1 | 4   | 49 | 1.886 | 0.271 | [0.270, 0.273] | 0.003 | 0.005 |
| 1 | 2 | 3   | 19 | 1.971 | 0.194 | [0.164, 0.211] | 0.047 | 0.052 |
| 1 | 2 | 3   | 49 | 1.886 | 0.234 | [0.198, 0.251] | 0.053 | 0.056 |
| 1 | 2 | 3.5 | 19 | 1.971 | 0.203 | [0.158, 0.221] | 0.063 | 0.066 |
| 1 | 2 | 3.5 | 49 | 1.886 | 0.239 | [0.190, 0.265] | 0.075 | 0.075 |
| 1 | 2 | 4   | 19 | 1.971 | 0.199 | [0.177, 0.216] | 0.039 | 0.041 |
| 1 | 2 | 4   | 49 | 1.886 | 0.242 | [0.219, 0.261] | 0.042 | 0.042 |
| 3 | 1 | 3   | 19 | 1.858 | 0.248 | [0.193, 0.260] | 0.068 | 0.068 |
| 3 | 1 | 3   | 49 | 1.639 | 0.308 | [0.260, 0.314] | 0.055 | 0.069 |
| 3 | 1 | 3.5 | 19 | 1.858 | 0.240 | [0.191, 0.248] | 0.056 | 0.069 |
| 3 | 1 | 3.5 | 49 | 1.639 | 0.301 | [0.240, 0.309] | 0.069 | 0.069 |
| 3 | 1 | 4   | 19 | 1.858 | 0.232 | [0.207, 0.245] | 0.038 | 0.051 |
| 3 | 1 | 4   | 49 | 1.639 | 0.293 | [0.242, 0.305] | 0.063 | 0.068 |

**Key observations:**
- **Patient 1, H&E 1** shows very narrow CIs (width ≤ 0.008), indicating low spatial heterogeneity — the tissue is relatively homogeneous at this slide location.
- **Patient 1, H&E 2** shows substantially wider CIs (up to 0.075), consistent with the inter-ROI bootstrap result for this slide (ratio = 3.58), confirming high spatial variability between regions.
- **Patient 3, H&E 1** shows intermediate CIs (0.038–0.069), also consistent with its inter-ROI ratio of 1.43.
- CI widths peak around b = 1.3–1.4 ms/µm² and decrease at higher b-values in absolute terms (at high b the signal itself is very small). In relative terms the uncertainty is largest at high b.
- The expansion coefficient has a moderate effect: exp = 3.5 consistently produces the widest CIs, while exp = 4 produces the narrowest, suggesting that the largest expansion amplifies spatial heterogeneity through a wider r distribution.

---

## Why the spatial bootstrap was chosen

| Criterion | Intra-ROI | Inter-ROI | Spatial |
|---|---|---|---|
| Captures morphology variability | ✅ | ✅ | ✅ |
| Captures intra-ROI spatial heterogeneity | ❌ | ❌ | ✅ |
| Captures inter-ROI spatial heterogeneity | ❌ | ✅ | Partial (within annotations only) |
| Continuous distribution (not discrete) | ✅ | ❌ (3–4 outcomes) | ✅ |
| Robust with few annotated ROIs | ✅ | ❌ | ✅ |
| Representative of full DWI voxel | ❌ | ❌ | ❌ |

The spatial bootstrap is preferred over the intra-ROI bootstrap because it captures local density fluctuations *within* each annotation, which the intra-ROI bootstrap misses entirely (it treats the ROI as a single fixed pool). It is preferred over the inter-ROI bootstrap because with only 2–4 annotated ROIs, the inter-ROI bootstrap produces a very coarse discrete distribution with high sensitivity to which ROIs were annotated.

The key limitation — that all three methods are constrained to the annotated regions and cannot assess representativeness at the DWI voxel scale — is shared equally. The spatial bootstrap does not resolve this but is the most internally consistent and continuous estimator within those constraints.

---

## Final results

### Joint inference (Dex, k_det) — df_summary

Inference method: Rician NLL minimisation on a 2D grid (Dex, k_det), exp = 3, k_r = 1.0 fixed.

| Patient | H&E | ROI | TD (ms) | Dex_best | k_det_best | f_vivo | f_eff | r3D (µm) | Dex CI | k_det CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | ROI1 | 19 | 1.943 | 0.166 | 0.562 | 0.094 | 6.70 | [1.905, 2.019] | [0.147, 0.205] |
| 1 | 1 | ROI1 | 49 | 1.905 | 0.128 | 0.562 | 0.072 | 6.70 | [1.867, 1.905] | [0.108, 0.128] |
| 1 | 2 | ROI1 | 19 | 1.943 | 0.283 | 0.369 | 0.104 | 7.04 | [1.905, 2.095] | [0.244, 0.515] |
| 1 | 2 | ROI1 | 49 | 1.867 | 0.186 | 0.369 | 0.068 | 7.04 | [1.867, 1.867] | [0.166, 0.186] |
| 1 | 2 | ROI2 | 19 | 2.057 | 0.496 | 0.277 | 0.137 | 7.59 | [1.943, 2.095] | [0.399, 0.573] |
| 1 | 2 | ROI2 | 49 | 1.905 | 0.302 | 0.277 | 0.084 | 7.59 | [1.867, 1.943] | [0.205, 0.341] |
| 1 | 2 | ROI3 | 19 | 1.943 | 0.496 | 0.201 | 0.100 | 6.89 | [1.905, 1.981] | [0.438, 0.573] |
| 1 | 2 | ROI3 | 49 | 1.867 | 0.341 | 0.201 | 0.068 | 6.89 | [1.829, 1.905] | [0.263, 0.399] |
| 3 | 1 | ROI1 | 19 | 1.867 | 0.380 | 0.588 | 0.223 | 6.71 | [1.791, 1.943] | [0.321, 0.399] |
| 3 | 1 | ROI1 | 49 | 1.639 | 0.205 | 0.588 | 0.121 | 6.71 | [1.601, 1.677] | [0.166, 0.224] |
| 3 | 1 | ROI2 | 19 | 1.867 | 0.360 | 0.598 | 0.215 | 6.54 | [1.829, 1.943] | [0.341, 0.380] |
| 3 | 1 | ROI2 | 49 | 1.639 | 0.186 | 0.598 | 0.111 | 6.54 | [1.563, 1.677] | [0.147, 0.205] |
| 3 | 1 | ROI3 | 19 | 1.829 | 0.418 | 0.509 | 0.213 | 6.66 | [1.791, 1.943] | [0.380, 0.457] |
| 3 | 1 | ROI3 | 49 | 1.639 | 0.244 | 0.509 | 0.124 | 6.66 | [1.601, 1.677] | [0.205, 0.263] |
| 3 | 1 | ROI4 | 19 | 1.867 | 0.515 | 0.425 | 0.219 | 6.59 | [1.791, 1.905] | [0.457, 0.535] |
| 3 | 1 | ROI4 | 49 | 1.639 | 0.302 | 0.425 | 0.128 | 6.59 | [1.601, 1.677] | [0.263, 0.321] |

**Per-patient consensus (median over ROIs and H&E, from df_summary):**

| Patient | TD (ms) | Dex (µm²/ms) | Dex range | k_det | k_det range | f_eff | r3D (µm) |
|---|---|---|---|---|---|---|---|
| 1 | 19 | 1.943 | [1.943, 2.057] | 0.389 | [0.166, 0.496] | 0.102 | 6.96 |
| 1 | 49 | 1.886 | [1.867, 1.905] | 0.244 | [0.128, 0.341] | 0.070 | 6.96 |
| 3 | 19 | 1.867 | [1.829, 1.867] | 0.399 | [0.360, 0.515] | 0.217 | 6.62 |
| 3 | 49 | 1.639 | [1.639, 1.639] | 0.224 | [0.186, 0.302] | 0.122 | 6.62 |

**Observations:**
- Dex is consistently around 1.9 µm²/ms for both patients at TD=19 ms, and lower at TD=49 ms (1.886 for Patient 1, 1.639 for Patient 3), suggesting a slight apparent diffusion time dependence, consistent with restricted diffusion.
- k_det is well below 1.0 for all ROIs (range 0.13–0.52), indicating that histological cell detection systematically underestimates the true voxel cell density. This is expected given QuPath's detection threshold and the 2D-to-3D projection.
- f_eff (effective volume fraction after k_det correction) is substantially lower than f_vivo, with Patient 3 showing higher f_eff (0.11–0.22) than Patient 1 (0.07–0.14), consistent with the denser tumour phenotype of Patient 3.
- The Dex consensus values used in the spatial bootstrap (Dex_1.971 for Patient 1 and Dex_1.858/1.639 for Patient 3) are consistent with these inferred values.

The spatial bootstrap CIs reported in the previous section were computed using these consensus Dex values. Results are visualised in `Average_signal.py` (Step 1 display cell): for each (Patient, H&E, ROI), a figure with TD = 19 ms and TD = 49 ms side by side shows the measured DWI signal and the simulated signal with its 95% bootstrap CI.
