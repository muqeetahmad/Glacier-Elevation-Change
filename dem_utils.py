"""
dem_utils.py
============
Shared, robust utilities for glacier-DEM co-registration, bias correction,
and elevation-change (dh) statistics, following the methodology of
Hugonnet et al. (2021), "Accelerated global glacier mass loss in the
early twenty-first century", Nature.
(https://www.nature.com/articles/s41586-021-03436-z)

Key methodological rules baked into this module — these are the things
that were inconsistent across your notebooks:

1. ALWAYS estimate bias / co-registration parameters on STABLE
   (off-glacier) terrain only, never on the glacier itself. Two DEMs
   from different dates disagree on the glacier because of real
   elevation change, not because of instrument bias — averaging or
   smoothing that difference and calling it "bias" destroys signal.

2. Use robust statistics (median, NMAD = 1.4826 * median(|x - median(x)|))
   everywhere instead of mean/std or scipy.stats.zscore. NMAD is much
   less sensitive to the very outliers you are trying to reject.

3. Co-register EVERY DEM pair in 3-D (horizontal dE, dN + vertical dZ)
   with the Nuth & Kaab (2011) sinusoidal method, not just a vertical
   shift. Horizontal mis-registration masquerades as vertical bias if
   you skip this step.

4. Use the Hugonnet iterative outlier filter (sigma = 5, 4, 3 x NMAD)
   on the FINAL dh product (over the glacier), separately from the
   co-registration filtering (which happens on stable terrain).

5. Never hardcode regression coefficients "from a previous run" (e.g.
   slope/intercept pasted into code) — always fit them fresh, on the
   current stable-terrain sample, every time the script runs.
"""

import os
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from rasterio import features
from rasterio.warp import Resampling
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates

DEFAULT_NODATA = (-9999, -32767, -32768, 65535, -3.4028235e+38, -3.4e38)


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------
def nmad(values):
    """Normalized Median Absolute Deviation - robust std-equivalent."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.median(v)
    return 1.4826 * np.median(np.abs(v - med))


def rmse(values):
    """Root-mean-square error/deviation (NOT robust — matches the paper's
    literal acceptance-gate metric: 'root-mean-square error of the elevation
    difference with TanDEM-X on ice-free terrain', threshold 20 m)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(v ** 2)))


def print_stats(values, label=""):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  {label}: NO DATA")
        return {}
    s = dict(
        mean=float(np.mean(v)), median=float(np.median(v)),
        std=float(np.std(v)), nmad=float(nmad(v)),
        min=float(np.min(v)), max=float(np.max(v)), n=int(v.size),
    )
    print(f"  {label}")
    print(f"    mean={s['mean']:+.3f} m  median={s['median']:+.3f} m  "
          f"NMAD={s['nmad']:.3f} m  std={s['std']:.3f} m  n={s['n']:,}")
    return s


def iterative_nmad_filter(dh, sigmas=(5, 4, 3), verbose=True):
    """
    Hugonnet et al. (2021), SI 2.1: iteratively reject pixels farther than
    sigma * NMAD from the median, tightening sigma each pass.
    Operates on a copy; NaNs mark rejected/invalid pixels.
    """
    dh = dh.copy()
    for sigma in sigmas:
        v = dh[np.isfinite(dh)]
        if v.size == 0:
            break
        med, mad = np.median(v), nmad(v)
        lo, hi = med - sigma * mad, med + sigma * mad
        before = np.sum(np.isfinite(dh))
        dh[np.isfinite(dh) & ((dh < lo) | (dh > hi))] = np.nan
        after = np.sum(np.isfinite(dh))
        if verbose:
            print(f"    sigma={sigma}: removed {before - after:,} px, "
                  f"bounds [{lo:+.2f}, {hi:+.2f}] m "
                  f"({100 * (before - after) / max(before, 1):.1f}%)")
    return dh


def rolstad_uncertainty(nmad_val, area_m2, l_corr_m=500.0):
    """
    Rolstad et al. (2009), eq. 14: variance of the AREA-AVERAGED dh,
    accounting for spatial autocorrelation of DEM errors (pixels are not
    independent observations). Returns sigma_mean (m).

    NOTE ON PROVENANCE (verified against Hugonnet et al. 2021 Methods,
    "Uncertainty analysis of volume changes"): the actual Hugonnet et al.
    (2021) uncertainty is NOT this formula. They sum SEVEN spherical
    variogram terms at correlation lengths of 0.15, 2, 5, 20, 50, 200 and
    500 km (their eq. 4-5), fitted from a global ICESat comparison over a
    full monthly 2000-2019 elevation time series across ~400 million
    pixels. That requires the dense multi-decadal DEM stack their study
    was built on and is not reproducible from a handful of DEM pairs for
    a single glacier. This single-range Rolstad et al. (2009) formula is
    a standard, well-precedented SIMPLIFICATION for exactly this
    two-epoch situation (closer to the "previous studies" the Hugonnet
    paper itself cites as using single-range variograms of 0.2-1 km).
    Report this in your thesis as "Rolstad et al. (2009), applied in the
    spirit of Hugonnet et al. (2021)" - not as the Hugonnet formula itself.
    """
    if area_m2 <= 0 or not np.isfinite(nmad_val):
        return np.nan
    var_mean = (np.pi * l_corr_m ** 2) / (5.0 * area_m2) * nmad_val ** 2
    return float(np.sqrt(var_mean))


def robust_rmse(v):
    """RMSE after no clipping - used for the Hugonnet et al. (2021) DEM
    acceptance test, which is explicitly on RMSE, not NMAD:
    'we excluded all DEMs for which the root-mean-square error of the
    elevation difference with TanDEM-X on ice-free terrain was larger
    than 20 m' (Methods, "Elevation time series")."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(v ** 2)))


def hypsometric_filter_and_fill(dh, elevation, glacier_mask, bin_width=100.0,
                                 nmad_sigma=5.0, verbose=True):
    """
    Reproduces the actual Hugonnet et al. (2021) volume-integration filter
    (Methods, "Integration of elevation into volume changes"):
      - bin the glacier into `bin_width`-metre elevation bands,
      - within each band, reject pixels farther than `nmad_sigma` x NMAD
        from the band's median dh (single pass, per band - NOT the
        iterative whole-glacier [5,4,3]xNMAD some notebooks used),
      - gap-fill any elevation band with no valid pixels left by linear
        interpolation from neighbouring bands (extrapolating from the
        closest bin at the ends).

    This also lets you compute area-averaged dh over the FULL glacier
    hypsometry rather than just the surviving valid-pixel area, which
    matters when voids are elevation-dependent (e.g. ASTER cloud/shadow
    tends to cluster at certain elevations).

    Returns
    -------
    dh_filled : np.ndarray, same shape as dh, glacier-only, with
                bin-median values imputed into empty/void pixels
    band_table : pandas.DataFrame with per-band n, median, nmad
    mean_dh_hypso : float, area-weighted mean dh across the whole
                    glacier hypsometry (bins weighted by glacier area,
                    matching the paper's local hypsometric method)
    """
    import pandas as pd

    dh = np.asarray(dh, dtype=float).copy()
    elevation = np.asarray(elevation, dtype=float)
    glacier_mask = np.asarray(glacier_mask, dtype=bool)

    z_glacier = elevation[glacier_mask]
    z_valid = z_glacier[np.isfinite(z_glacier)]
    if z_valid.size == 0:
        raise RuntimeError("No valid elevation values on the glacier mask.")

    z_min = np.floor(np.nanmin(z_valid) / bin_width) * bin_width
    z_max = np.ceil(np.nanmax(z_valid) / bin_width) * bin_width
    bins = np.arange(z_min, z_max + bin_width, bin_width)

    rows = []
    dh_filled = np.where(glacier_mask, dh, np.nan)

    for b0, b1 in zip(bins[:-1], bins[1:]):
        band_mask = glacier_mask & (elevation >= b0) & (elevation < b1)
        n_band = int(band_mask.sum())
        v = dh_filled[band_mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append(dict(z0=b0, z1=b1, n=n_band, n_valid=0,
                              median=np.nan, nmad=np.nan))
            continue
        med, mad = np.median(v), nmad(v)
        lo, hi = med - nmad_sigma * mad, med + nmad_sigma * mad
        bad = band_mask & np.isfinite(dh_filled) & ((dh_filled < lo) | (dh_filled > hi))
        dh_filled[bad] = np.nan
        v_kept = dh_filled[band_mask]
        v_kept = v_kept[np.isfinite(v_kept)]
        rows.append(dict(z0=b0, z1=b1, n=n_band, n_valid=int(v_kept.size),
                          median=float(np.median(v_kept)) if v_kept.size else np.nan,
                          nmad=float(nmad(v_kept)) if v_kept.size else np.nan))

    band_table = pd.DataFrame(rows)
    band_table["center"] = (band_table["z0"] + band_table["z1"]) / 2.0

    # gap-fill empty bands from neighbours (interpolate, extrapolate at ends)
    medians = band_table["median"].values.copy()
    valid_bins = np.isfinite(medians)
    if valid_bins.sum() == 0:
        raise RuntimeError("No elevation band has any valid dh - cannot gap-fill.")
    if (~valid_bins).any():
        centers = band_table["center"].values
        medians_filled = np.interp(centers, centers[valid_bins], medians[valid_bins])
        n_gapfilled = int((~valid_bins).sum())
        if verbose:
            print(f"    Gap-filled {n_gapfilled}/{len(medians)} empty elevation "
                  f"bands via hypsometric interpolation.")
    else:
        medians_filled = medians
        n_gapfilled = 0
    band_table["median_filled"] = medians_filled

    # impute filled band-median into any glacier pixel left without a value
    for row, med_f in zip(band_table.itertuples(), medians_filled):
        band_mask = glacier_mask & (elevation >= row.z0) & (elevation < row.z1)
        still_missing = band_mask & ~np.isfinite(dh_filled)
        dh_filled[still_missing] = med_f

    # area-weighted (pixel-count weighted) mean across the full hypsometry
    weights = band_table["n"].values.astype(float)
    mean_dh_hypso = float(np.average(medians_filled, weights=weights)) \
        if weights.sum() > 0 else np.nan

    if verbose:
        print(f"    Hypsometric mean dh (full glacier area, gap-filled): "
              f"{mean_dh_hypso:+.3f} m")

    return dh_filled, band_table, mean_dh_hypso


def density_conversion_uncertainty(dh_rate_m_yr, sigma_dh_rate, area_m2, sigma_area_m2,
                                    density=850.0, sigma_density=60.0):
    """
    Convert an elevation-change rate to a mass-balance rate and propagate
    ALL THREE independent error sources the paper accounts for: dh
    uncertainty, area uncertainty, and density-conversion uncertainty
    (850 +/- 60 kg/m3, Methods "Conversion to mass changes").
    Returns dict with mwe_rate, sigma_mwe_rate (m w.e./yr) and
    dV_rate, sigma_dV_rate (m3/yr) using eq. 3 of the paper for the
    volume term, extended with the density term added in quadrature.
    """
    dV_rate = dh_rate_m_yr * area_m2
    sigma_dV_rate = np.sqrt((sigma_dh_rate * area_m2) ** 2 +
                             (area_m2 and (dh_rate_m_yr * sigma_area_m2) ** 2 or 0.0))

    mwe_rate = dh_rate_m_yr * density / 1000.0
    # relative uncertainty from density adds in quadrature with relative dh uncertainty
    rel_dh = (sigma_dh_rate / dh_rate_m_yr) if dh_rate_m_yr not in (0, np.nan) else 0.0
    rel_rho = sigma_density / density
    sigma_mwe_rate = abs(mwe_rate) * np.sqrt(rel_dh ** 2 + rel_rho ** 2) \
        if np.isfinite(rel_dh) else np.nan

    return dict(dV_rate=dV_rate, sigma_dV_rate=sigma_dV_rate,
                mwe_rate=mwe_rate, sigma_mwe_rate=sigma_mwe_rate)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_dem(path, nodata_values=DEFAULT_NODATA):
    """Open a raster with rioxarray, drop band dim, mask known nodata codes."""
    da = rioxarray.open_rasterio(path).squeeze()
    if "band" in da.dims:
        da = da.drop_vars("band")
    da = da.astype("float64")
    for nd in nodata_values:
        da = da.where(np.abs(da - nd) > 1e-3) if abs(nd) > 1e6 else da.where(da != nd)
    return da


def reproject_match(src_da, ref_da, resampling=Resampling.bilinear):
    """Reproject src onto ref's grid (CRS, extent, resolution)."""
    return src_da.rio.reproject_match(ref_da, resampling=resampling)


def rasterize_mask(shp_path_or_gdf, ref_da):
    """Rasterize a polygon shapefile onto ref_da's grid -> boolean mask."""
    gdf = shp_path_or_gdf if isinstance(shp_path_or_gdf, gpd.GeoDataFrame) \
        else gpd.read_file(shp_path_or_gdf)
    gdf = gdf.to_crs(ref_da.rio.crs)
    mask = features.rasterize(
        gdf.geometry, out_shape=ref_da.rio.shape, transform=ref_da.rio.transform(),
        fill=0, default_value=1, all_touched=True, dtype="uint8",
    )
    return mask.astype(bool)


def save_raster(arr, ref_da, crs, path, nodata=np.nan):
    """Save a 2-D numpy array as a GeoTIFF using ref_da's grid/coords."""
    da = xr.DataArray(arr, coords={"y": ref_da.y, "x": ref_da.x}, dims=["y", "x"])
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_nodata(nodata, inplace=True)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    da.rio.to_raster(path, compress="LZW")
    print(f"  Saved: {path}")


def apply_planar_shift(da, dE, dN, dZ):
    """
    Apply a horizontal+vertical shift (as found by nuth_kaab_coregister or
    vertical_bias_correction) to a DataArray on ITS OWN native grid/extent
    -- e.g. the full-extent original SRTM tile, not a small clipped patch.

    This is done by translating the raster's coordinate reference (a pure,
    lossless shift of where the same pixel values sit in space) and
    subtracting the vertical bias, rather than resampling the array onto
    a different (e.g. much smaller) reference grid. Use this to produce a
    corrected output that preserves the SOURCE raster's full spatial
    coverage, which reprojecting onto a small reference grid would not.
    """
    shifted = da - dZ
    shifted = shifted.assign_coords(x=shifted.x + dE, y=shifted.y + dN)
    return shifted


# --------------------------------------------------------------------------
# Terrain derivatives
# --------------------------------------------------------------------------
def compute_slope_aspect(z, res_m):
    """Finite-difference slope (deg) and aspect (deg, 0-360) from an elevation array."""
    dz_dy, dz_dx = np.gradient(z, res_m, res_m)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    aspect = (np.degrees(np.arctan2(-dz_dy, -dz_dx)) + 360) % 360
    return slope, aspect


# --------------------------------------------------------------------------
# Bias correction — VERTICAL ONLY (stable terrain, robust median)
# --------------------------------------------------------------------------
def vertical_bias_correction(src_arr, ref_arr, stable_mask, max_dh=200.0):
    """
    Simplest correction: shift src so its MEDIAN matches ref, using only
    stable-terrain pixels. Returns (corrected_src, bias, dh_stable_mask).
    Use this as a first pass, or when there is too little stable terrain
    / too little aspect variety for Nuth-Kaab to fit reliably.
    """
    dh = src_arr - ref_arr
    m = stable_mask & np.isfinite(dh) & (np.abs(dh) < max_dh)
    if m.sum() < 30:
        raise RuntimeError(f"Only {int(m.sum())} valid stable-terrain pixels; "
                            f"too few for a reliable bias estimate.")
    bias = float(np.median(dh[m]))
    corrected = src_arr - bias
    print(f"  Vertical bias (stable terrain, median): {bias:+.4f} m  (N={int(m.sum()):,})")
    return corrected, bias, m


# --------------------------------------------------------------------------
# Bias correction — FULL 3-D Nuth & Kaab (2011) co-registration
# --------------------------------------------------------------------------
def _downsample_mean(arr, factor):
    """Block-mean downsample by an integer factor (NaN-aware). Used only to
    build a coarser grid for fitting; the final shift is still applied to
    the original, full-resolution array."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    cropped = arr[:h2, :w2]
    reshaped = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=(1, 3))


def nuth_kaab_coregister(src_arr, ref_arr, slope, aspect, res_m, stable_mask,
                          max_dh=200.0, max_iter=15, tol_m=0.05,
                          max_pix_shift=3.0, min_slope_deg=3.0,
                          coreg_grid_res_m=None, verbose=True):
    """
    3-D co-registration of src_arr onto ref_arr, following Nuth & Kaab (2011,
    The Cryosphere): dh / tan(slope) = a*cos(b - aspect) + c, fit iteratively
    on STABLE terrain only. The shift is accumulated analytically and the
    original array is resampled exactly once at the end (avoids compounding
    interpolation error from repeated resampling).

    coreg_grid_res_m : float or None
        If your source DEM is much finer than the reference (e.g. a 0.4 m
        UAV DEM vs 30 m SRTM), the aspect-based fit can be misled by fine
        texture the reference can't resolve, and may fail to converge
        cleanly. Set this to roughly the COARSER DEM's resolution (e.g. 30)
        to fit the shift on a downsampled grid, while still applying the
        final shift to the full-resolution src_arr. Leave as None if the
        two DEMs are already similar resolution.

    Unlike a naive "trust the last iteration" loop, this function tracks
    the BEST iteration by stable-terrain NMAD and returns that one — so
    an unstable/oscillating fit (e.g. bouncing between parameter bounds)
    can never produce a worse result than the starting point.

    Returns
    -------
    corrected_arr : np.ndarray  (src, shifted horizontally + vertically)
    shift : dict with dE, dN, dZ (m), n_iter, converged (bool)
    diag  : dict with pre/post NMAD on stable terrain, for QA plots
    """
    def model(asp_rad, a, b, c):
        return a * np.cos(b - asp_rad) + c

    max_shift_m = max_pix_shift * res_m

    # --- Optionally build a coarser grid purely for fitting -------------
    factor = 1
    if coreg_grid_res_m is not None and coreg_grid_res_m > res_m * 1.5:
        factor = int(round(coreg_grid_res_m / res_m))
        fit_src = _downsample_mean(src_arr, factor)
        fit_ref = _downsample_mean(ref_arr, factor)
        fit_slope = _downsample_mean(slope, factor)
        fit_aspect_sin = _downsample_mean(np.sin(np.radians(aspect)), factor)
        fit_aspect_cos = _downsample_mean(np.cos(np.radians(aspect)), factor)
        fit_aspect = np.degrees(np.arctan2(fit_aspect_sin, fit_aspect_cos)) % 360
        fit_stable_frac = _downsample_mean(stable_mask.astype(float), factor)
        fit_stable_mask = fit_stable_frac > 0.6   # bin counts as stable if mostly stable
        fit_res_m = res_m * factor
        if verbose:
            print(f"  Fitting on a coarsened grid: {res_m:.2f} m -> "
                  f"{fit_res_m:.1f} m (factor {factor}) to avoid fine-texture noise")
    else:
        fit_src, fit_ref, fit_slope, fit_aspect = src_arr, ref_arr, slope, aspect
        fit_stable_mask = stable_mask
        fit_res_m = res_m

    coreg_base = fit_stable_mask & (fit_slope > min_slope_deg) & np.isfinite(fit_ref)

    rows = np.arange(fit_src.shape[0])
    cols = np.arange(fit_src.shape[1])
    cols_2d, rows_2d = np.meshgrid(cols, rows)
    aspect_rad = np.radians(fit_aspect)

    dh0 = fit_src - fit_ref
    m0 = coreg_base & np.isfinite(dh0) & (np.abs(dh0) < max_dh)
    nmad_pre = nmad(dh0[m0]) if m0.sum() else np.nan
    med0 = float(np.median(dh0[m0])) if m0.sum() else 0.0

    # Adaptive bias bound: the vertical term shouldn't need to roam further
    # than the raw offset plus a healthy margin — prevents it pegging at an
    # arbitrary +-200 m wall and oscillating there.
    bias_bound = max(100.0, (abs(med0) + 10 * nmad_pre) * 3) if np.isfinite(nmad_pre) else 300.0

    total_dE = total_dN = total_dZ = 0.0
    prev_nmad = np.inf
    converged = False
    last_dE = last_dN = last_dZ = 0.0

    best_nmad = nmad_pre if np.isfinite(nmad_pre) else np.inf
    best_state = (0.0, 0.0, 0.0)

    if verbose:
        print(f"  {'iter':>4}  {'dE(m)':>7}  {'dN(m)':>7}  {'dZ(m)':>7}  "
              f"{'|shift|':>8}  {'NMAD':>7}")

    it = -1
    for it in range(max_iter):
        dRow = -total_dN / fit_res_m
        dCol = total_dE / fit_res_m
        shifted = map_coordinates(
            fit_src, [rows_2d + dRow, cols_2d + dCol],
            order=1, mode="nearest", prefilter=False,
        ).astype(float)
        shifted[~np.isfinite(fit_src)] = np.nan
        shifted = shifted - total_dZ

        dh = shifted - fit_ref
        m = coreg_base & np.isfinite(dh) & (np.abs(dh) < max_dh)
        if m.sum() < 50:
            if verbose:
                print(f"  {it + 1:>4}  too few stable pixels — stopping")
            break

        v_dh, v_slope, v_aspect = dh[m], fit_slope[m], aspect_rad[m]
        med_v, mad_v = np.median(v_dh), nmad(v_dh)
        keep = np.abs(v_dh - med_v) < 3 * mad_v
        v_dh, v_slope, v_aspect = v_dh[keep], v_slope[keep], v_aspect[keep]
        cur_nmad = nmad(v_dh)

        if np.isfinite(cur_nmad) and cur_nmad < best_nmad:
            best_nmad = cur_nmad
            best_state = (total_dE, total_dN, total_dZ)

        tan_slope = np.clip(np.tan(np.radians(v_slope)), 0.03, None)
        y_nk = v_dh / tan_slope

        # Initial guess for c must be on the same (slope-normalized) scale
        # as y_nk, not raw dh -- on shallow slopes y_nk can be much larger
        # than dh, which previously pushed p0 outside the fit bounds.
        med_y = float(np.median(y_nk)) if y_nk.size else 0.0
        eps = 1e-6
        p0 = [
            float(np.clip(1.0, -max_shift_m + eps, max_shift_m - eps)),
            0.0,
            float(np.clip(med_y, -bias_bound + eps, bias_bound - eps)),
        ]

        try:
            popt, _ = curve_fit(
                model, v_aspect, y_nk, p0=p0,
                bounds=([-max_shift_m, -np.pi, -bias_bound],
                        [max_shift_m, np.pi, bias_bound]),
                maxfev=5000,
            )
        except RuntimeError:
            if verbose:
                print(f"  {it + 1:>4}  curve_fit failed — stopping")
            break

        a_nk, b_nk, c_nk = popt
        dE = a_nk * np.sin(b_nk)
        dN = -a_nk * np.cos(b_nk)
        dZ = c_nk
        shift_mag = np.hypot(dE, dN)

        if verbose:
            flag = ""
            if shift_mag > max_shift_m:
                flag = "  SKIP (too large)"
            elif abs(dZ) > bias_bound * 0.95:
                flag = "  (bias near bound)"
            print(f"  {it + 1:>4}  {dE:+7.3f}  {dN:+7.3f}  {dZ:+7.3f}  "
                  f"{shift_mag:8.4f}  {cur_nmad:7.3f}{flag}")

        if shift_mag > max_shift_m:
            if shift_mag > max_shift_m * 10:
                break
            prev_nmad = cur_nmad
            continue

        last_dE, last_dN, last_dZ = dE, dN, dZ
        total_dE += dE
        total_dN += dN
        total_dZ += dZ
        prev_nmad = cur_nmad

        if shift_mag < tol_m:
            converged = True
            if verbose:
                print(f"  converged at iter {it + 1} (shift={shift_mag:.4f} m)")
            if cur_nmad < best_nmad:
                best_nmad = cur_nmad
                best_state = (total_dE, total_dN, total_dZ)
            break

    # --- Use the BEST iteration seen, not necessarily the last one -------
    total_dE, total_dN, total_dZ = best_state
    if verbose and (total_dE, total_dN, total_dZ) != (last_dE, last_dN, last_dZ):
        print(f"  Using best-NMAD iteration (NMAD={best_nmad:.3f} m), "
              f"not necessarily the final iteration")

    # Apply the final cumulative shift once to the ORIGINAL, full-res array
    rows_full = np.arange(src_arr.shape[0])
    cols_full = np.arange(src_arr.shape[1])
    cols_2d_full, rows_2d_full = np.meshgrid(cols_full, rows_full)

    dRow_f = -total_dN / res_m
    dCol_f = total_dE / res_m
    final = map_coordinates(
        src_arr, [rows_2d_full + dRow_f, cols_2d_full + dCol_f],
        order=1, mode="nearest", prefilter=False,
    ).astype(float)
    final[~np.isfinite(src_arr)] = np.nan

    dh_check = final - ref_arr - total_dZ
    m_final = stable_mask & (slope > min_slope_deg) & np.isfinite(dh_check) & (np.abs(dh_check) < max_dh)
    residual_bias = float(np.median(dh_check[m_final])) if m_final.sum() else 0.0

    corrected = final - total_dZ - residual_bias
    dh_post = (corrected - ref_arr)[m_final]
    nmad_post = nmad(dh_post)

    dh0_full = src_arr - ref_arr
    m0_full = stable_mask & (slope > min_slope_deg) & np.isfinite(dh0_full) & (np.abs(dh0_full) < max_dh)
    nmad_pre_full = nmad(dh0_full[m0_full]) if m0_full.sum() else np.nan

    shift = dict(dE=total_dE, dN=total_dN, dZ=total_dZ + residual_bias,
                 n_iter=it + 1, converged=converged, fit_grid_res_m=fit_res_m)
    diag = dict(nmad_pre=nmad_pre_full, nmad_post=nmad_post,
                dh_post_stable=dh_post, stable_mask_used=m_final)

    if verbose:
        print(f"  Total shift: dE={shift['dE']:+.3f} m  dN={shift['dN']:+.3f} m  "
              f"dZ={shift['dZ']:+.3f} m")
        print(f"  NMAD (stable terrain, full-res): "
              f"{nmad_pre_full:.3f} m -> {nmad_post:.3f} m")

    return corrected, shift, diag


# --------------------------------------------------------------------------
# Fallback: planar de-tilt (use when Nuth-Kaab fails to converge, e.g. too
# little stable terrain, too little aspect variety, or very steep DEM
# artifacts / jitter that a horizontal shift alone cannot capture)
# --------------------------------------------------------------------------
def planar_detilt(src_arr, ref_arr, x_coords, y_coords, stable_mask, max_dh=200.0):
    """
    Fit and remove a planar tilt: dh = a*x + b*y + c, fit on stable terrain,
    then remove the residual median bias. Returns (corrected_arr, coeffs).
    """
    from scipy.linalg import lstsq

    xx, yy = np.meshgrid(x_coords, y_coords)
    dh = src_arr - ref_arr
    m = stable_mask & np.isfinite(dh) & (np.abs(dh) < max_dh)
    if m.sum() < 100:
        raise RuntimeError(f"Only {int(m.sum())} stable pixels; too few to fit a plane.")

    x_fit, y_fit, dh_fit = xx[m], yy[m], dh[m]
    med, mad = np.median(dh_fit), nmad(dh_fit)
    keep = np.abs(dh_fit - med) < 5 * mad
    x_fit, y_fit, dh_fit = x_fit[keep], y_fit[keep], dh_fit[keep]

    A = np.column_stack([x_fit, y_fit, np.ones_like(x_fit)])
    coef, *_ = lstsq(A, dh_fit)
    a, b, c = coef

    plane = a * xx + b * yy + c
    detilted = src_arr - plane

    dh2 = detilted - ref_arr
    m2 = stable_mask & np.isfinite(dh2) & (np.abs(dh2) < max_dh)
    residual_bias = float(np.median(dh2[m2])) if m2.sum() else 0.0
    corrected = detilted - residual_bias

    return corrected, dict(a=a, b=b, c=c, residual_bias=residual_bias)
