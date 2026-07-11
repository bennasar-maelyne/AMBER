"""
Tumor phenotype catalog for DWI acquisition optimisation.

Each entry defines the biophysical parameter bounds used for Latin Hypercube
sampling in opt_engine.py.  All phenotypes use lookup_table_optim.mat whose
coverage is: f ∈ [0.001, 1.0], Dex ∈ [0.5, 3.0], rmean ∈ [1, 20] µm,
rsd ∈ [0, 12] µm.

Parameter column order (matches LUT params array): [f, Dex, rmean, rsd]

BOUNDS TO VALIDATE WITH JOSH — marked with  # ⚠ VALIDATE
"""

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))

LUT_OPTIM = os.path.join(ROOT, "Validation", "lookup_table_optim.mat")

# ── Catalog ────────────────────────────────────────────────────────────────────
# Each entry:
#   name    : short identifier used in filenames and figure labels
#   label   : display name for figures
#   lut     : path to LUT file
#   bounds  : {param: (min, max)} for the 4 free parameters [f, Dex, rmean, rsd]
#   color   : matplotlib color for plots
#   ref     : literature reference justifying the bounds

CATALOG = [
    {
        "name":  "edema",
        "label": "Edema",
        "lut":   LUT_OPTIM,
        "bounds": {
            "f":     (0.05, 0.15),    # ⚠ VALIDATE — low cell density (oedematous tissue)
            "Dex":   (2.0,  3.0),     # ⚠ VALIDATE — high extracellular diffusivity
            "rmean": (8.0,  15.0),    # ⚠ VALIDATE — µm
            "rsd":   (1.0,  5.0),     # ⚠ VALIDATE — µm
        },
        "color": "#4878d0",
        "t2":   400.0,   # ms at 3T — upper estimate for vasogenic edema; peritumoral zone at 3T ~154 ms (Blystad et al. 2017, DOI: 10.1371/journal.pone.0177135)
        # Ref: Nicholson & Sykova 1998 (Dex ≈ 1.5–3.0 µm²/ms in pathological tissue);
        #      Panagiotaki et al. 2014 PMC5400014 (f, rmean)
        "ref": "Nicholson & Sykova 1998; Panagiotaki et al. 2014",
    },
    {
        "name":  "cyst",
        "label": "Cyst",
        "lut":   LUT_OPTIM,
        "bounds": {
            "f":     (0.005, 0.05),   # ⚠ VALIDATE — very low cell density; LUT min = 0.001
            "Dex":   (2.5,   3.0),   # ⚠ VALIDATE — near free-water diffusivity
            "rmean": (8.0,  15.0),   # ⚠ VALIDATE — µm
            "rsd":   (0.0,   2.0),   # ⚠ VALIDATE — relatively uniform
        },
        "color": "#6acc65",
        "t2":   1000.0,  # ms at 3T — protein-rich tumor cyst (CSF ~2000ms, reduced by protein)
        # Ref: Panagiotaki et al. 2014 PMC5400014
        "ref": "Panagiotaki et al. 2014",
    },
    {
        "name":  "large_cells",
        "label": "Large cells",
        "lut":   LUT_OPTIM,
        "bounds": {
            "f":     (0.35, 0.60),   # ⚠ VALIDATE
            "Dex":   (1.5,  2.5),    # ⚠ VALIDATE — µm²/ms
            "rmean": (12.0, 20.0),   # ⚠ VALIDATE — µm  (defining feature)
            "rsd":   (2.0,  6.0),    # ⚠ VALIDATE — µm
        },
        "color": "#d65f5f",
        "t2":   146.0,   # ms at 3T — IDH-mutant glioma / LGG proxy (Gu et al. 2021, DOI: 10.21037/qims-20-916)
        # Ref: Panagiotaki et al. 2014 PMC5400014
        "ref": "Panagiotaki et al. 2014",
    },
    {
        "name":  "small_cells",
        "label": "Small cells",
        "lut":   LUT_OPTIM,
        "bounds": {
            "f":     (0.50, 0.75),   # ⚠ VALIDATE — high cell packing
            "Dex":   (0.8,  1.5),    # ⚠ VALIDATE — reduced extracellular space
            "rmean": (3.0,  7.0),    # ⚠ VALIDATE — µm  (defining feature)
            "rsd":   (0.5,  2.0),    # ⚠ VALIDATE — µm
        },
        "color": "#ee854a",
        "t2":   124.0,   # ms at 3T — IDH-wildtype glioma / HGG proxy (Gu et al. 2021, DOI: 10.21037/qims-20-916)
        # Ref: Panagiotaki et al. 2014 PMC5400014
        "ref": "Panagiotaki et al. 2014",
    },
    {
        "name":  "fibrosis",
        "label": "Fibrosis",
        "lut":   LUT_OPTIM,
        "bounds": {
            "f":     (0.40, 0.65),   # ⚠ VALIDATE
            "Dex":   (0.5,  1.0),    # ⚠ VALIDATE — restricted extracellular diffusivity (defining feature)
            "rmean": (5.0,  9.0),    # ⚠ VALIDATE — µm
            "rsd":   (1.0,  3.0),    # ⚠ VALIDATE — µm
        },
        "color": "#956cb4",
        "t2":    80.0,   # ms at 3T — lower bound for fibrous meningioma; mean all subtypes 95.6 ± 36.5 ms (Ludovichetti et al. 2022, DOI: 10.1002/brb3.2769)
        # Ref: White et al. 2013 RSI PMC4155409
        "ref": "White et al. 2013",
    },
    # ── Necrosis (two-population model) — DEFERRED ────────────────────────────
    # Blocked: requires lookup_table_two_pop.mat with confirmed kappa values.
    # Parameters: f_alive ∈ [0.05, 0.25], Dex ∈ [1.7, 2.5], rmean ∈ [6, 12], rsd ∈ [1, 4]
    # To add once Josh confirms kappa for the two-population simulation.
]

# ── LUT coverage check ─────────────────────────────────────────────────────────
# Bounds of lookup_table_optim.mat — update if LUT changes
LUT_OPTIM_COVERAGE = {
    "f":     (0.001, 1.0),
    "Dex":   (0.5,   3.0),
    "rmean": (1.0,  20.0),
    "rsd":   (0.0,  12.0),
}


def check_coverage():
    """Print a warning for any catalog bound outside the LUT coverage."""
    ok = True
    for p in CATALOG:
        for param, (lo, hi) in p["bounds"].items():
            lut_lo, lut_hi = LUT_OPTIM_COVERAGE[param]
            if lo < lut_lo or hi > lut_hi:
                print(f"  [WARNING] {p['name']}.{param}: [{lo}, {hi}] partially outside "
                      f"LUT [{lut_lo}, {lut_hi}]")
                ok = False
    if ok:
        print("Coverage check passed — all bounds within LUT range.")
    return ok


def get_phenotype(name):
    """Return catalog entry by name."""
    for p in CATALOG:
        if p["name"] == name:
            return p
    raise KeyError(f"Phenotype '{name}' not found in catalog.")


if __name__ == "__main__":
    print(f"Catalog: {len(CATALOG)} phenotypes\n")
    for p in CATALOG:
        print(f"  {p['label']}")
        for param, (lo, hi) in p["bounds"].items():
            print(f"    {param:6s} : [{lo}, {hi}]")
        print(f"    ref    : {p['ref']}")
        print()
    print("-" * 50)
    check_coverage()
