"""
Shared feature extraction for both training and inference.
Handles multi-region TIFF indexing and efficient batch point sampling.
"""

import os
import re
import glob
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen


SENTINEL2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
                   "B07", "B08", "B8A", "B09", "B11", "B12"]

N_BANDS   = len(SENTINEL2_BANDS)   # 12
N_INDICES = 10                      # NDVI, LSWI, NDWI, EVI, RGVI, NDRE, CIre, CWC, NBR, LSWI_EVI
N_DOY_ENC = 2                       # sin(2π·doy/365), cos(2π·doy/365) per timestep
SEQ_DIM   = N_BANDS + N_INDICES + N_DOY_ENC  # 24 per timestep

FLAT_STATS = ["mean", "std", "max", "min", "median", "diff_std"]

# Fixed index positions in the 24-dim sequence vector
IDX_NDVI     = N_BANDS + 0   # 12
IDX_LSWI     = N_BANDS + 1   # 13
IDX_NDWI     = N_BANDS + 2   # 14
IDX_EVI      = N_BANDS + 3   # 15
IDX_RGVI     = N_BANDS + 4   # 16
IDX_NDRE     = N_BANDS + 5   # 17
IDX_CIre     = N_BANDS + 6   # 18
IDX_CWC      = N_BANDS + 7   # 19
IDX_NBR      = N_BANDS + 8   # 20
IDX_LSWI_EVI = N_BANDS + 9   # 21

# Total flat feature dimension (6 stats × 24 + 11 pheno extras + 1 DOY)
FLAT_DIM = SEQ_DIM * 6 + 11 + 1   # 156


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
      tiff_dirs: str or list[str]

    Returns:
      tiff_index   : {(region, 'YYYY-MM-DD', band): filepath}
      region_bounds: {region: rasterio.BoundingBox}
    """
    if isinstance(tiff_dirs, str):
        tiff_dirs = [tiff_dirs]

    tiff_index    = {}
    region_bounds = {}

    for d in tiff_dirs:
        # Scan directory itself + all subdirectories recursively
        files = (glob.glob(os.path.join(d, "*.tiff")) +
                 glob.glob(os.path.join(d, "*.tif"))  +
                 glob.glob(os.path.join(d, "**", "*.tiff"), recursive=True) +
                 glob.glob(os.path.join(d, "**", "*.tif"),  recursive=True))
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
    for region, b in region_bounds.items():
        if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
            return region
    return None


def csv_date_to_tiff_date(date_str: str) -> str:
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

    b2  = bands.get("B02", 0.0)
    b3  = bands.get("B03", 0.0)
    b4  = bands.get("B04", 0.0)
    b5  = bands.get("B05", 0.0)
    b7  = bands.get("B07", 0.0)
    b8  = bands.get("B08", 0.0)
    b8a = bands.get("B8A", 0.0)
    b11 = bands.get("B11", 0.0)
    b12 = bands.get("B12", 0.0)

    ndvi = safe(b8, b4)
    lswi = safe(b8, b11)
    evi  = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + 1e-9)

    return {
        "NDVI"     : ndvi,
        "LSWI"     : lswi,
        "NDWI"     : safe(b3, b8),
        "EVI"      : evi,
        # Red-green VI — rice early growth / corn separability (Paper 2)
        "RGVI"     : safe(b3, b4),
        # Red-edge NDVI — nitrogen-sensitive, corn vs soybean (Paper 5)
        "NDRE"     : safe(b8a, b5),
        # Chlorophyll index red-edge — C4 corn vs soybean ~30% diff at peak (Paper 5)
        "CIre"     : b7 / (b5 + 1e-9) - 1,
        # Canopy water content using SWIR-2 — primary soybean discriminator (Paper 6)
        "CWC"      : safe(b8a, b12),
        # Normalized burn ratio — water stress proxy (Paper 6)
        "NBR"      : safe(b8, b12),
        # Flooding indicator: >0 means water/transplanting dominant (Papers 2, 3)
        "LSWI_EVI" : lswi - evi,
    }


def _feat_vec(bands: dict, doy_rad: float) -> np.ndarray:
    """24-dim vector: 12 bands + 10 indices + 2 per-timestep sinusoidal DOY."""
    idx = compute_indices(bands)
    return np.array(
        [bands.get(b, 0.0) for b in SENTINEL2_BANDS] + [
            idx["NDVI"],     idx["LSWI"],  idx["NDWI"],  idx["EVI"],
            idx["RGVI"],     idx["NDRE"],  idx["CIre"],  idx["CWC"],
            idx["NBR"],      idx["LSWI_EVI"],
            math.sin(doy_rad), math.cos(doy_rad),
        ],
        dtype=np.float32
    )


# ── EFFICIENT BATCH SAMPLING ───────────────────────────────────────────────────

def _batch_sample_tiff(tiff_path: str, coords: list) -> np.ndarray:
    with rasterio.open(tiff_path) as src:
        vals = list(sample_gen(src, coords))
    return np.array([float(v[0]) if v.size > 0 else np.nan for v in vals],
                    dtype=np.float32)


def _extract_sequences_for_points(
    unique_points: list,
    point_regions: list,
    tiff_index: dict,
    region_bounds: dict,
) -> list:
    """
    For each unique point returns np.ndarray of shape (n_dates, SEQ_DIM=24).
    Batches reads per TIFF for efficiency.
    """
    available_dates_by_region = defaultdict(set)
    for (reg, date, _) in tiff_index:
        available_dates_by_region[reg].add(date)
    for r in available_dates_by_region:
        available_dates_by_region[r] = sorted(available_dates_by_region[r])

    region_to_point_idxs = defaultdict(list)
    for i, (region, _) in enumerate(zip(point_regions, unique_points)):
        if region and region in available_dates_by_region:
            region_to_point_idxs[region].append(i)

    n_pts     = len(unique_points)
    sequences = [None] * n_pts

    for region, idxs in region_to_point_idxs.items():
        dates  = available_dates_by_region[region]
        coords = [unique_points[i] for i in idxs]
        n_d    = len(dates)

        band_data = np.zeros((len(coords), n_d, N_BANDS), dtype=np.float32)
        for d_idx, date in enumerate(dates):
            for b_idx, band in enumerate(SENTINEL2_BANDS):
                key = (region, date, band)
                if key in tiff_index:
                    vals = _batch_sample_tiff(tiff_index[key], coords)
                    band_data[:, d_idx, b_idx] = vals

        # Precompute per-timestep DOY radian for sinusoidal encoding
        doy_rads = [
            2 * math.pi * datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday / 365.0
            for d in dates
        ]

        for local_i, global_i in enumerate(idxs):
            seq = np.zeros((n_d, SEQ_DIM), dtype=np.float32)
            for d_idx in range(n_d):
                b = {SENTINEL2_BANDS[j]: float(band_data[local_i, d_idx, j])
                     for j in range(N_BANDS)}
                seq[d_idx] = _feat_vec(b, doy_rads[d_idx])
            sequences[global_i] = seq

    for i in range(n_pts):
        if sequences[i] is None:
            sequences[i] = np.zeros((1, SEQ_DIM), dtype=np.float32)

    return sequences


# ── FLAT FEATURES (XGBoost) ────────────────────────────────────────────────────

def _seq_to_flat(seq: np.ndarray) -> np.ndarray:
    """
    (n_dates, SEQ_DIM=24) → 155-dim flat vector.

    Components:
      6 temporal stats × 24 = 144
      11 phenological/phase features
    = 155 (observation DOY appended separately → FLAT_DIM=156 total)
    """
    T = seq.shape[0]

    # Base temporal statistics (144 features)
    mean = seq.mean(axis=0)
    std  = seq.std(axis=0)
    mx   = seq.max(axis=0)
    mn   = seq.min(axis=0)
    med  = np.median(seq, axis=0)
    dstd = np.diff(seq, axis=0).std(axis=0) if T > 1 else np.zeros(SEQ_DIM, dtype=np.float32)

    # Phenological timing (5 features)
    ndvi_ts = seq[:, IDX_NDVI]
    cwc_ts  = seq[:, IDX_CWC]
    lswi_ts = seq[:, IDX_LSWI]
    evi_ts  = seq[:, IDX_EVI]

    doy_ndvi_peak  = float(np.argmax(ndvi_ts)) / max(T - 1, 1)
    greenup_steps  = np.where(ndvi_ts > 0.2)[0]
    doy_greenup    = float(greenup_steps[0]) / max(T - 1, 1) if len(greenup_steps) else 1.0
    ndvi_amplitude = float(ndvi_ts.max() - ndvi_ts.min())
    cwc_amplitude  = float(cwc_ts.max()  - cwc_ts.min())   # soybean pod-fill indicator

    # Phase masks (Papers 2, 3)
    flood_mask = (lswi_ts > evi_ts) & (ndvi_ts < 0.3)  # transplanting/flooded
    grow_mask  = ndvi_ts > 0.4                           # active canopy

    flooding_fraction = float(flood_mask.sum()) / max(T, 1)

    def _pmean(ts, mask, fallback):
        return float(ts[mask].mean()) if mask.any() else fallback

    flood_mean_ndvi = _pmean(ndvi_ts, flood_mask, float(mean[IDX_NDVI]))
    flood_mean_lswi = _pmean(lswi_ts, flood_mask, float(mean[IDX_LSWI]))
    flood_mean_evi  = _pmean(evi_ts,  flood_mask, float(mean[IDX_EVI]))
    grow_mean_ndvi  = _pmean(ndvi_ts, grow_mask,  float(mean[IDX_NDVI]))
    grow_mean_evi   = _pmean(evi_ts,  grow_mask,  float(mean[IDX_EVI]))
    grow_mean_cwc   = _pmean(cwc_ts,  grow_mask,  float(mean[IDX_CWC]))

    pheno_extras = np.array([
        doy_ndvi_peak,  doy_greenup,     ndvi_amplitude,  cwc_amplitude,
        flooding_fraction,
        flood_mean_ndvi, flood_mean_lswi, flood_mean_evi,
        grow_mean_ndvi,  grow_mean_evi,   grow_mean_cwc,
    ], dtype=np.float32)

    return np.concatenate([mean, std, mx, mn, med, dstd, pheno_extras])  # 155-dim


# ── MAIN ENTRY POINT ───────────────────────────────────────────────────────────

def extract_all_features(
    df: pd.DataFrame,
    tiff_index: dict,
    region_bounds: dict,
) -> dict:
    """
    Extract features for all rows in df.

    Returns dict:
      'flat'   : np.ndarray (n_rows, FLAT_DIM=156)
      'seqs'   : list of np.ndarray  — (n_dates, SEQ_DIM=24) per row
      'lengths': np.ndarray (n_rows,)
      'doys'   : np.ndarray (n_rows,)  — obs DOY normalized [0,1]
      'oob'    : np.ndarray bool (n_rows,)
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

    n         = len(df)
    flat_rows = np.zeros((n, FLAT_DIM), dtype=np.float32)
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

        # OOB: no TIFF coverage — use DOY-encoded single timestep so model
        # gets seasonal position signal instead of a dead-zero sequence.
        if region is None:
            doy_rad = 2 * math.pi * doy_raw / 365.0
            oob_seq = np.zeros((1, SEQ_DIM), dtype=np.float32)
            oob_seq[0, -2] = math.sin(doy_rad)
            oob_seq[0, -1] = math.cos(doy_rad)
            seq = oob_seq

        flat = np.append(_seq_to_flat(seq), doy_norm)  # 155 + 1 = 156
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
