# AMBER — Contexte du projet (Maëlyne)

## Vue d'ensemble

AMBER est un simulateur agent-based de tumeurs cérébrales sous radiothérapie. Le travail de Maëlyne porte sur deux volets liés à une LUT Monte Carlo (RMS) :
1. **Validation de la LUT** par comparaison signal simulé (histologie → LUT) vs signal DWI acquis in vivo
2. **Optimisation des paramètres d'acquisition DWI** grâce à cette même LUT

L'ensemble est destiné à un **article scientifique** (Paper_v2.docx).

---

## Objectif principal

Montrer qu'il est possible de **reconstruire des IRM de diffusion synthétiques** à partir de données histologiques et d'une LUT Monte Carlo, et que ces IRM synthétiques sont cohérentes avec les IRM DWI acquises in vivo. Puis utiliser la LUT pour **optimiser les paramètres d'acquisition** (b-value, TD) selon le type de tumeur.

---

## Pipeline complet

```
Histologie (QuPath)          LUT Monte Carlo (RMS)
     │                             │
     │  densité cellulaire         │  (f, Dex, rmean, rsd) → signal DWI
     │  rayon cellulaire (2D→3D)   │  interpolation Delaunay
     └──────────────┬──────────────┘
                    │
                    ▼
          Signal simulé S_sim(b)   ←→   S_DWI(b) mesuré in vivo
                    │
                    ▼
          Optimisation paramètres d'acquisition (DWI_opt)
```

---

## Données

- **Patients** : 2 patients avec tumeur cérébrale (Patient 1, Patient 3)
- **IRM DWI** : volumes 4D multi-shell (.mat), masques tumeur/cerveau, cartes de bruit
  - `CON_0101SP_12072016_tumor_all.mat` (Patient 1)
  - `CON_03_V01_tumor_all.mat` (Patient 3)
- **Histologie** : segmentation cellulaire QuPath (aire, calibres) pour 2 lames H&E par patient, coefficients d'expansion (3, 3.5, 4)
- **LUT** : `lookup_table_val.mat` (MATLAB v7.3 HDF5, NPar-normalisé, pas de b=0)
  - Espace paramétrique : `f`, `Dex` [1–3 µm²/ms], `rmean` [1–20 µm], `rsd`
  - B-values : [0.1, 0.2, 0.5, 0.7, 1.3, 1.4, 2.6, 3.8, 4.2, 6.0, 7.3, 12.0, 17.8] ms/µm²
  - TDs : [19, 49] ms

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `Validation/Average_signal.py` | Pipeline principal (version .py de Average_signal.ipynb) |
| `Validation/combined.ipynb` | Comparaison visuelle DWI mesuré vs simulé — **à convertir en .py et mettre à jour** |
| `monte-carlo-simulation-sphere-PGSE-main/` | Code MATLAB de simulation Monte Carlo (RMS) |
| `Validation/signal_results_spatial_bootstrap.json` | Bootstrap spatial : Dex par (patient, TD) depuis df_summary |
| `Validation/signal_results_geom.json` | Signaux pour géométries extrêmes (spécificité) |
| `amber/DWI_opt.ipynb` | Outil d'optimisation des paramètres d'acquisition |
| `Paper/Paper_v2.docx` | Brouillon article scientifique |

---

## Ce qui est implémenté dans Average_signal.py

- **Correction d'Abercrombie** : rayon 2D → 3D, k_r = 1.0 (fixé)
- **Interpolation Delaunay** sur l'espace paramétrique de la LUT
- **Inférence joint chi² (Dex, k_det)** avec IC 95% → `df_summary`
- **Rician NLL** implémenté et validé (Δ négligeable vs chi² → chi² retenu)
- **Bootstrap spatial** (fenêtres aléatoires, Dex consensus pondéré par 1/chi² par patient×TD)
- **Visualisation Step 1** : toutes ROIs × patients, b-values alignées sur le DWI mesuré
- **Consensus Dex** pondéré par 1/chi²_min depuis df_summary

---

## État d'avancement

### Validation LUT — presque terminée
- [x] Pipeline histologie → LUT → signal simulé
- [x] df_summary : inférence Dex + k_det pour tous (Patient, H&E, ROI, TD)
- [x] Bootstrap spatial régénéré avec nouvelle LUT
- [ ] **Document récap bootstrap** pour le tuteur Josh (expliquer les 3 approches et le choix final)
- [ ] **combined.py** : mettre à jour la comparaison DWI mesuré vs simulé

### Spécificité de la LUT
- signal_results_geom.json déjà calculé (géométries extrêmes)
- Approche Josh : threshold sur métrique de comparaison (pas de courbes ROC complètes — overkill)
- À définir : quelle métrique (chi²_min ? RMSE ?) et visualisation

### Optimisation paramètres d'acquisition — prochaine étape majeure
- Outil : `DWI_opt.ipynb` (et potentiellement `DWI_opt_nec.ipynb` pour nécrose)
- Objectif : pour chaque type de tumeur (microstructure donnée), trouver les (b-value, TD) qui maximisent le contraste
- Fonctions de coût : std signal intra-tumeur, CNR, AUC, Cohen d

---

## Paramètres biologiques

| Paramètre | Symbole | Valeurs typiques | Statut |
|---|---|---|---|
| Fraction volumique cellulaire | `f` | 0.1 – 0.8 | Mesuré QuPath |
| Diffusivité extracellulaire | `Dex` | 1.0 – 3.0 µm²/ms | Inféré (chi²) |
| Rayon moyen cellulaire | `rmean` | 3 – 9 µm | Mesuré QuPath |
| Écart-type des rayons | `rsd` | variable | Mesuré QuPath |
| Facteur détection cellulaire | `k_det` | 0.3 – 1.0 | Inféré (chi²) |
| k_r (shrinkage géométrique) | `k_r` | 1.0 | Fixé (décision Josh) |

---

## Notes techniques

- Environnement Python : `.venv` à la racine (activation : `source .venv/Scripts/activate`)
- Branche de travail principale : `dev_mae`
- LUT chargée via `h5py` (MATLAB v7.3) — axes transposés par rapport à l'ancien format
- `Average_signal.py` : ~2982 lignes — lire par chunks avec offset/limit
- `df_summary` est la source de vérité pour les résultats d'inférence (Dex_best, k_det_best, CIs)
