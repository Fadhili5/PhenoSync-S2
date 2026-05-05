"""
Shared feature extraction for both training and inference.
Handles multi-region TIFF indexing and efficient batch point sampling.
"""

import os
import re
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen


SENTINEL2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
                   "B07", "B08", "B8A", "B09", "B11", "B12"]

FLAT_STATS = ["mean", "std", "max", "min", "median", "diff_std"]


# ── TIFF INDEXING ──────────────────────────────────────────────────────────────

def parse_tiff_filename(filepath: str):
    """Returns (region, 'YYYY-MM-DD', band) or None."""
    name = os.path.basename(filepath)
    m = re.match(
        r"(region\w+)_(\d{4}-\d{2}-\d{2})-00-00_.+_Sentinel-2_L2A_(B\w+)_\(Raw\)\.tiff?",
        name, re.IGNORECASE
    )
    return (m.group(1), m.group(2), m.group(3).upper()) if m else None


def build_tiff_index(tiff_dirs) -> tuple:
    """
    Args:
      tiff_dirs: str or list[str] — one or more directories containing TIFF files

    Returns:
      tiff_index   : {(region, 'YYYY-MM-DD', band): filepath}
      region_bounds: {region: rasterio.BoundingBox}
    """
    if isinstance(tiff_dirs, str):
        tiff_dirs = [tiff_dirs]

    tiff_index    = {}
    region_bounds = {}

    for d in tiff_dirs:
        files = (glob.glob(os.path.join(d, "*.tiff")) +
                 glob.glob(os.path.join(d, "*.tif")))
        for f in files:
            parsed = parse_tiff_filename(f)
            if not parsed:
                continue
            region, date, band = parsed
            tiff_index[(region, date, band)] = f
            if region not in region_bounds:
                with rasterio.open(f) as src:
                    region_bounds[region] = src.bounds

    regions = sorted(region_bounds)
    dates   = sorted(set(k[1] for k in tiff_index))
    print(f"[index] {len(tiff_index)} TIFFs | {len(regions)} regions | {len(dates)} dates")
    return tiff_index, region_bounds


def find_region_for_point(lon: float, lat: float, region_bounds: dict):
    """Return region name whose bbox contains (lon, lat), else None."""
    for region, b in region_bounds.items():
        if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
            return region
    return None


def csv_date_to_tiff_date(date_str: str) -> str:
    """'YYYY/M/D' or 'YYYY-MM-DD' → 'YYYY-MM-DD'"""
    s = str(date_str).strip()
    if "/" in s:
        return datetime.strptime(s, "%Y/%m/%d").strftime("%Y-%m-%d")
    return s


def nearest_date(target: str, candidates: list) -> str:
    t = datetime.strptime(target, "%Y-%m-%d")
    return min(candidates, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - t).days))


# ── SPECTRAL INDICES ───────────────────────────────────────────────────────────

def compute_indices(bands: dict) -> dict:
    def safe(a, b):
        return (a - b) / (a + b + 1e-9)

    b2, b3, b4 = bands.get("B02", 0), bands.get("B03", 0), bands.get("B04", 0)
    b8, b11    = bands.get("B08", 0), bands.get("B11", 0)

    return {
        "NDVI": safe(b8, b4),
        "LSWI": safe(b8, b11),
        "NDWI": safe(b3, b8),
        "EVI" : 2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1 + 1e-9),
    }


def _feat_vec(bands: dict) -> np.ndarray:
    """16-dim vector: 12 bands + 4 indices in fixed order."""
    idx = compute_indices(bands)
    return np.array(
        [bands.get(b, 0.0) for b in SENTINEL2_BANDS] +
        [idx["NDVI"], idx["LSWI"], idx["NDWI"], idx["EVI"]],
        dtype=np.float32
    )


# ── EFFICIENT BATCH SAMPLING ───────────────────────────────────────────────────

def _batch_sample_tiff(tiff_path: str, coords: list) -> np.ndarray:
    """Open TIFF once, sample all coords. Returns (n_coords,) array."""
    with rasterio.open(tiff_path) as src:
        vals = list(sample_gen(src, coords))
    return np.array([float(v[0]) if v.size > 0 else np.nan for v in vals],
                    dtype=np.float32)


def _extract_sequences_for_points(
    unique_points: list,    # [(lon, lat), ...]
    point_regions: list,    # [region or None, ...]
    tiff_index: dict,
    region_bounds: dict,
) -> list:
    """
    For each unique point returns np.ndarray of shape (n_dates, 16).
    Batches reads per TIFF for efficiency.
    """
    available_dates_by_region = defaultdict(set)
    for (reg, date, _) in tiff_index:
        available_dates_by_region[reg].add(date)
    for r in available_dates_by_region:
        available_dates_by_region[r] = sorted(available_dates_by_region[r])

    # Group points by region so we open each TIFF once per region
    region_to_point_idxs = defaultdict(list)
    for i, (region, (lon, lat)) in enumerate(zip(point_regions, unique_points)):
        if region and region in available_dates_by_region:
            region_to_point_idxs[region].append(i)

    n_pts    = len(unique_points)
    sequences = [None] * n_pts

    for region, idxs in region_to_point_idxs.items():
        dates  = available_dates_by_region[region]
        coords = [unique_points[i] for i in idxs]
        n_d    = len(dates)
        # band_matrix[i, d] = float value at point i on date-index d
        band_data = np.zeros((len(coords), n_d, len(SENTINEL2_BANDS)), dtype=np.float32)

        for d_idx, date in enumerate(dates):
            for b_idx, band in enumerate(SENTINEL2_BANDS):
                key = (region, date, band)
                if key in tiff_index:
                    vals = _batch_sample_tiff(tiff_index[key], coords)
                    band_data[:, d_idx, b_idx] = vals

        # Build (n_dates, 16) sequence per point
        for local_i, global_i in enumerate(idxs):
            seq = np.zeros((n_d, 16), dtype=np.float32)
            for d_idx in range(n_d):
                b = {SENTINEL2_BANDS[j]: float(band_data[local_i, d_idx, j])
                     for j in range(len(SENTINEL2_BANDS))}
                seq[d_idx] = _feat_vec(b)
            sequences[global_i] = seq

    # Out-of-bounds points get zero single-step sequence
    for i in range(n_pts):
        if sequences[i] is None:
            sequences[i] = np.zeros((1, 16), dtype=np.float32)

    return sequences


# ── FLAT FEATURES (XGBoost) ────────────────────────────────────────────────────

def _seq_to_flat(seq: np.ndarray) -> np.ndarray:
    """
    (n_dates, 16) → flat vector of temporal stats.
    Per feature: mean, std, max, min, median, diff_std → 6 stats × 16 = 96 features.
    """
    mean   = seq.mean(axis=0)
    std    = seq.std(axis=0)
    mx     = seq.max(axis=0)
    mn     = seq.min(axis=0)
    med    = np.median(seq, axis=0)
    dstd   = np.diff(seq, axis=0).std(axis=0) if seq.shape[0] > 1 else np.zeros(16)
    return np.concatenate([mean, std, mx, mn, med, dstd])  # 96-dim


# ── MAIN ENTRY POINT ───────────────────────────────────────────────────────────

def extract_all_features(
    df: pd.DataFrame,
    tiff_index: dict,
    region_bounds: dict,
) -> dict:
    """
    Extract features for all rows in df.

    Expects df columns: Longitude, Latitude, phenophase_date

    Returns dict:
      'flat'   : np.ndarray (n_rows, 97)   — temporal stats + DOY (for XGBoost)
      'seqs'   : list of np.ndarray         — (n_dates, 16) per row (for LSTM)
      'lengths': np.ndarray (n_rows,)       — sequence lengths
      'doys'   : np.ndarray (n_rows,)       — obs day-of-year, normalized [0,1]
      'oob'    : np.ndarray bool (n_rows,)  — True if point outside all regions
    """
    unique_lonlat = list(dict.fromkeys(
        zip(df["Longitude"].astype(float), df["Latitude"].astype(float))
    ))
    lonlat_to_idx = {pt: i for i, pt in enumerate(unique_lonlat)}

    unique_regions = [
        find_region_for_point(lon, lat, region_bounds)
        for lon, lat in unique_lonlat
    ]
    print(f"[features] {len(unique_lonlat)} unique points | "
          f"{sum(r is not None for r in unique_regions)} in-bounds")

    sequences_unique = _extract_sequences_for_points(
        unique_lonlat, unique_regions, tiff_index, region_bounds
    )

    n = len(df)
    flat_rows = np.zeros((n, 97), dtype=np.float32)
    seqs      = []
    lengths   = np.zeros(n, dtype=np.int64)
    doys      = np.zeros(n, dtype=np.float32)
    oob       = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(df.iterrows()):
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        pt_idx   = lonlat_to_idx[(lon, lat)]
        region   = unique_regions[pt_idx]
        seq      = sequences_unique[pt_idx]

        obs_date = csv_date_to_tiff_date(str(row["phenophase_date"]).strip())
        doy_raw  = datetime.strptime(obs_date, "%Y-%m-%d").timetuple().tm_yday
        doy_norm = doy_raw / 365.0

        flat = np.append(_seq_to_flat(seq), doy_norm)  # 96 + 1 = 97
        flat_rows[i] = flat
        seqs.append(seq)
        lengths[i] = seq.shape[0]
        doys[i]    = doy_norm
        oob[i]     = region is None

    flat_rows = np.nan_to_num(flat_rows, nan=0.0)
    print(f"[features] flat shape={flat_rows.shape} | "
          f"max_seq_len={max(lengths)} | oob={oob.sum()}")
    return {
        "flat"   : flat_rows,
        "seqs"   : seqs,
        "lengths": lengths,
        "doys"   : doys,
        "oob"    : oob,
    }
