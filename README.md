# Glacier-Elevation-Change

Geodetic glacier elevation-change (dh/dt) and mass-balance pipeline for **Hopper Glacier, western Karakoram**, implementing the methodology of **Hugonnet et al. (2021), *Nature*** ("Accelerated global glacier mass loss in the early twenty-first century").

The workflow chains three DEM epochs — a high-resolution **UAV survey**, **SRTM (2000)**, **TanDEM-X**, and a 2000–2023 **ASTER** time series — into a single, internally consistent elevation record, then derives per-scene elevation-change rates and geodetic mass-balance estimates with full uncertainty propagation.

## Pipeline

| Step | Notebook | Purpose |
|---|---|---|
| 1 | `step1_uav_srtm_bias_correction.ipynb` | Co-register SRTM to the high-accuracy UAV DEM on stable (off-glacier) terrain |
| 2 | `step2_srtm_tandemx_bias_correction.ipynb` | Further tie SRTM's absolute vertical/horizontal position to TanDEM-X, producing `SRTM_final_corrected.tif` — the fixed "epoch ~2000" reference surface |
| 3 | `step3_aster_hugonnet_dh.ipynb` | Co-register each ASTER DEM (2000–2023) to the Step 2 reference, compute dh over the glacier, filter/gap-fill, propagate uncertainty, and export a master results table + dh/dt time series plot |

`dem_utils.py` holds all shared, reusable logic behind the three notebooks.

## Methodology

Each step enforces the same core rules, carried through from `dem_utils.py`:

- **Bias/co-registration is always fit on stable, off-glacier terrain** — never on the glacier itself, since multi-epoch disagreement there is real elevation change, not instrument bias.
- **Robust statistics throughout** — median and NMAD (`1.4826 × median(|x − median(x)|)`) instead of mean/std, which are far less sensitive to outliers.
- **3-D co-registration** (horizontal dE, dN + vertical dZ) via the **Nuth & Kaab (2011)** sinusoidal method for every DEM pair, since horizontal mis-registration can masquerade as vertical bias if skipped. Falls back to a planar de-tilt if Nuth–Kaab doesn't converge.
- **Scene acceptance gate = RMSE, not NMAD** — matching the Hugonnet et al. (2021) text: ASTER scenes are excluded if stable-terrain RMSE after co-registration exceeds 20 m.
- **Per-elevation-band 5×NMAD outlier filtering + hypsometric gap-filling** on the glacier dh field (not a flat whole-glacier iterative sweep), reproducing the paper's actual volume-integration approach.
- **Uncertainty propagation** via Rolstad et al. (2009) spatial-correlation error (documented as a simplification of the paper's full variogram approach) combined with density-conversion uncertainty (850 ± 60 kg/m³) to convert elevation-change rates to geodetic mass balance.
- **No hardcoded regression coefficients** — all bias/co-registration parameters are refit fresh from the current stable-terrain sample on every run.

## Key functions (`dem_utils.py`)

- `nmad`, `rmse`, `robust_rmse`, `print_stats` — robust summary statistics
- `iterative_nmad_filter` — iterative σ×NMAD outlier rejection (5, 4, 3)
- `nuth_kaab_coregister`, `planar_detilt`, `apply_planar_shift` — DEM co-registration
- `vertical_bias_correction` — stable-terrain vertical shift fitting
- `hypsometric_filter_and_fill` — per-elevation-band filtering and gap-filling
- `rolstad_uncertainty`, `density_conversion_uncertainty` — geodetic mass-balance error propagation
- `load_dem`, `reproject_match`, `rasterize_mask`, `save_raster`, `compute_slope_aspect` — I/O and raster utilities

## Outputs

Running Step 3 across all available ASTER granules produces:

- `Hopper_SRTM_ASTER_Hugonnet_results.csv` — one row per ASTER scene, with area, mean/median/hypsometric dh, NMAD, elevation-change rate (m/yr) and geodetic mass balance (m w.e./yr) with uncertainties, co-registration diagnostics, and RMSE acceptance flag
- `dhdt_timeseries.png` — Hopper Glacier's elevation-change rate through time (2000–2023), with error bars
- Per-scene diagnostic figures and co-registered/filtered dh GeoTIFFs

## Data requirements

- UAV-derived DEM of the glacier and surrounding terrain (from Structure-from-Motion photogrammetry)
- SRTM DEM (~Feb 2000)
- TanDEM-X DEM
- ASTER GDEM (AST14DEM) granules spanning 2000–2023
- Glacier outline (RGI or digitized) and a stable/off-glacier terrain mask, as shapefiles

## Dependencies

`numpy` · `xarray` · `rioxarray` · `geopandas` · `rasterio` · `scipy` · `pandas` · `matplotlib`

## Reference

Hugonnet, R., McNabb, R., Rabatel, A. et al. (2021). *Accelerated global glacier mass loss in the early twenty-first century.* **Nature**, 592, 726–731. https://doi.org/10.1038/s41586-021-03436-z

Nuth, C. and Kääb, A. (2011). *Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change.* **The Cryosphere**, 5, 271–290.

Rolstad, C., Haug, T., and Denby, B. (2009). *Spatially integrated geodetic glacier mass balance and its uncertainty based on geostatistical analysis: application to the western Svartisen ice cap, Norway.* **Journal of Glaciology**, 55(192), 666–680.

---

*Part of ongoing glacier surface elevation change and ice flow dynamics research across the western Karakoram.*
