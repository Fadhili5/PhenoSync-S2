# Track 1 — Rice Field Identification & Phenology

## Task
Dual prediction per test point:
1. **Crop type**: `rice` | `corn` | `soybean` | background
2. **Phenological stage**: `Greenup` | `MidGreenup` | `Peak` | `Maturity` | `Senescence` | `MidSenescence` | `Dormancy`

## Input

### points_test.csv columns
| Field | Description |
|-------|-------------|
| `point_id` | Sample point ID |
| `Longitude` | Longitude |
| `Latitude` | Latitude |
| `phenophase_date` | Observation date (YYYY/M/D, no leading zeros) |

### TIFF files
Path: `/input/region_test/*.tiff`
Naming: `regionXX_YYYY-MM-DD-00-00_YYYY-MM-DD-23-59_Sentinel-2_L2A_Bxx_(Raw).tiff`
Bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12

### Training label CSV (points_train_label.csv)
Fields: `point_id`, `Longitude`, `Latitude`, `phenophase_date`, `crop_type`, `phenophase_name`

> NOTE: training CSV uses `crop_type` / `phenophase_name`; test CSV uses `Pre_crop_type` / `Pre_phenophase` — different column names, same meaning.

### MCD12Q2 auxiliary phenology data
- MODIS Land Cover Dynamics product — pixel-level phenological parameters
- Task explicitly requires using this alongside Sentinel-2
- **NOT YET HANDLED in inference.py** — check `/input/` for MCD12Q2 files when full training data arrives
- Provides: Greenup, MidGreenup, Peak, Maturity, Senescence, MidSenescence, Dormancy dates per pixel per year

## Output — result.json
```json
{
  "{Longitude}_{Latitude}_{Date}": ["{CropType}", "{PhenophaseStage}"],
  "124.703696_48.543523_2018/9/1": ["corn", "Senescence"]
}
```
- Key: `lon_lat_date` where date format is `YYYY/M/D` (no leading zeros)
- Value: `[crop_type, phenophase_name]`
- Must include ALL test points or score = 0

## Scoring

```
AlgoScore = (0.4 × MacroF1_Crop + 0.6 × MacroF1_RicePheno) × 100
```

### Crop Classification F1 — 40% weight

```
MacroF1_Crop = (1/3) × Σ F1_Crop_c   for c ∈ {rice, corn, soybean}

F1_Crop_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
Precision_c = TP / (TP + FP)
Recall_c    = TP / (TP + FN)
```

- 3 classes only: rice, corn, soybean — **background excluded**
- Phenology not considered here — only crop label matters
- Edge cases: if TP+FP=0 → Precision=0; if TP+FN=0 → Recall=0; if P+R=0 → F1=0

### Rice Phenology F1 — 60% weight (MAIN METRIC)

```
MacroF1_RicePheno = (1/7) × Σ F1_Rice_i   for i ∈ {1..7 phenostages}

F1_Rice_i = 2 × Precision_Rice_i × Recall_Rice_i / (Precision_Rice_i + Recall_Rice_i)
Precision_Rice_i = TPi / (TPi + FPi)
Recall_Rice_i    = TPi / (TPi + FNi)
```

**Exact / "Double Hit" match required** — both crop AND phenology must be correct simultaneously:

| Element | Condition |
|---------|-----------|
| **TPi** | Actual = (rice, stage i) AND predicted = (rice, stage i) |
| **FPi** | Predicted = (rice, stage i) BUT actual ≠ that. Includes: wrong crop (not rice) OR right crop but wrong stage |
| **FNi** | Actual = (rice, stage i) BUT predicted ≠ that. Includes: missed rice (predicted other crop/background) OR rice but wrong stage |

Phenology errors for corn/soybean do **not** count toward this metric.
Edge cases same as crop: if TP+FP=0 → Precision=0; if TP+FN=0 → Recall=0; if P+R=0 → F1=0.

## Feature Engineering Notes
- Multi-temporal bands → temporal profile per point
- Key indices for rice: NDVI, LSWI (B08-B11)/(B08+B11), EVI
- Rice signature: flooding dip (low NDVI) → rapid greenup → senescence
- MCD12Q2 auxiliary phenology data can augment features

## Useful Indices
| Index | Formula | Rice relevance |
|-------|---------|----------------|
| NDVI | (B08-B04)/(B08+B04) | Vegetation density |
| LSWI | (B08-B11)/(B08+B11) | Water/flooding detection |
| EVI | 2.5*(B08-B04)/(B08+6*B04-7.5*B02+1) | Canopy structure |
| NDWI | (B03-B08)/(B03+B08) | Water surface |

## Band Reference (Sentinel-2)
| Band | Wavelength | Resolution | Use |
|------|-----------|------------|-----|
| B02 | Blue 490nm | 10m | |
| B03 | Green 560nm | 10m | NDWI |
| B04 | Red 665nm | 10m | NDVI |
| B05 | Red Edge 705nm | 20m | |
| B06 | Red Edge 740nm | 20m | |
| B07 | Red Edge 783nm | 20m | |
| B08 | NIR 842nm | 10m | NDVI, LSWI |
| B8A | Narrow NIR 865nm | 20m | |
| B11 | SWIR 1610nm | 20m | LSWI (flooding) |
| B12 | SWIR 2190nm | 20m | |

## Crop Labels
- `rice`
- `corn`
- `soybean`
- background (non-agricultural)

## Phenology Labels (7 stages)
1. `Greenup`
2. `MidGreenup`
3. `Peak`
4. `Maturity`
5. `Senescence`
6. `MidSenescence`
7. `Dormancy`
