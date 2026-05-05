# Model Approach — Track 1: Rice Field Identification & Phenology

## 1. Problem Summary

Dual prediction per (point, date) pair:
1. **Crop type**: `rice` | `corn` | `soybean` | `background`
2. **Phenological stage**: `Greenup` | `MidGreenup` | `Peak` | `Maturity` | `Senescence` | `MidSenescence` | `Dormancy`

Scoring: `AlgoScore = (0.4 × MacroF1_Crop + 0.6 × MacroF1_RicePheno) × 100`
Rice phenology is the dominant metric (60%) and requires **exact double-hit** (crop=rice AND correct stage simultaneously).

---

## 2. Data Understanding

### What we discovered from the sample data
- Test CSV (`points_test.csv`) columns: `point_id`, `Longitude`, `Latitude`, `phenophase_date`, `Pre_crop_type` (empty), `Pre_phenophase` (empty)
- Date format in CSV: `YYYY/M/D` (no leading zeros) — used directly as JSON output key
- TIFF filenames: `regionXX_YYYY-MM-DD-00-00_YYYY-MM-DD-23-59_Sentinel-2_L2A_Bxx_(Raw).tiff`
- Sample test region (`region03`) bounds: lon [124.64, 125.28], lat [48.46, 48.64]
- Only **11 of 171** unique test points fall inside region03 — the rest need their own region TIFFs
- Labeled sample points (6–9) are **outside** region03 — meaning sample labels can't be validated against sample TIFFs
- Training CSV uses `crop_type` / `phenophase_name`; test CSV uses `Pre_crop_type` / `Pre_phenophase`

### Key spatial insight
Each test point belongs to exactly one regionXX. The full test set has multiple regionXX TIFF folders. Our code spatially routes each point to the correct region using TIFF bounding boxes.

---

## 3. Why Multi-Temporal Data?

Single-date spectral signatures overlap heavily between crops. Multi-temporal data enables:
- **Phenological curve shape** discrimination — crops have different NDVI trajectories across the season
- **Flooding detection** — unique to rice, visible only early season
- **Temporal peaks** — corn peaks earlier than soybean in the same region

Research finding: time-series spectral features "reflect crop growth attributes of different phenology information, which can improve crop extraction accuracies" (vs single-date).

---

## 4. Feature Engineering

### Per-date features (16 dimensions per timestep)
| # | Feature | Formula | Crop relevance |
|---|---------|---------|----------------|
| 1–12 | Raw bands | B01–B12 | All spectral info |
| 13 | NDVI | (B08–B04)/(B08+B04) | Vegetation density, temporal curve shape |
| 14 | LSWI | (B08–B11)/(B08+B11) | **Key for rice**: flooding/transplanting detection |
| 15 | NDWI | (B03–B08)/(B03+B08) | Open water surface |
| 16 | EVI | 2.5×(B08–B04)/(B08+6×B04–7.5×B02+1) | Canopy structure, better than NDVI in dense veg |

### Observation date feature
- Day-of-year (DOY) of the observation date, normalized [0,1]
- Tells the model *when in the season* the prediction is for
- Critical for phenology stage prediction

---

## 5. How We Distinguish Each Crop

### Rice (easiest)
Unique LSWI signature across 3 phases:
1. **Pre-transplant flooding**: LSWI > NDVI or EVI — water on surface, very low NDVI
2. **Rapid green-up**: NDVI rises sharply, LSWI drops
3. **Senescence**: both drop

Rule from literature: `LSWI > EVI` during flooding phase → very likely rice. No other crop has this.

### Background (easy)
- Persistently low NDVI across all dates
- No temporal vegetation signal

### Corn vs Soybean (hardest — requires trained model)
Research explicitly states this is "challenging due to similar phenological trajectories and spectral characteristics."

What separates them when using all bands + temporal:

| Signal | Corn | Soybean | Key band |
|--------|------|---------|----------|
| **Phenology timing** | Peaks earlier (~DOY 200) | Peaks later (~DOY 220–240) | NDVI shape |
| **Red-edge response** | Tall vertical canopy → lower B05/B06 | Horizontal broad leaves → higher red-edge | B05, B06, B07 |
| **Canopy water (SWIR)** | Lower water content | Higher → lower B11/B12 reflectance | B11, B12 |
| **Carotenoid at senescence** | Distinct yellowing pattern | Different | B03/B04 ratio |

SWIR bands (B11, B12) improved corn/soybean accuracy by **3.8%** in published research. Red-edge (B05–B07) adds additional separation. Neither rule alone is sufficient — the model learns the joint pattern from labeled training data.

---

## 6. Model Architecture

### Phase 1 — XGBoost (baseline, CPU, fast)

**Input**: flat feature vector (97 dims)
- 16 features (bands + indices) × 6 temporal stats = 96 dims
- Temporal stats per feature: mean, std, max, min, median, diff_std (temporal variation)
- + DOY of observation date (1 dim)

**Two separate classifiers**:
- `xgb_crop.pkl` → predicts crop type (4 classes)
- `xgb_pheno.pkl` → predicts phenological stage (7 classes)

**Why XGBoost first**: no GPU, trains in minutes, interpretable feature importance, strong baseline (~85–90% crop accuracy from literature).

### Phase 2 — BiLSTM with Attention (GPU, higher accuracy)

**Architecture**:
```
Input: (B, T, 16)  ← padded temporal sequence, 16 features per date
    ↓
BiLSTM (2 layers, hidden=128, bidirectional)
    ↓
Attention pooling → context vector (256-dim)
    ↓
Concatenate DOY scalar → (257-dim)
    ↓
[Crop head]          [Phenology head]
Linear(257→64)       Linear(257→64)
ReLU + Dropout       ReLU + Dropout
Linear(64→4)         Linear(64→7)
```

**Key design decisions**:
- **Bidirectional**: sees past and future of the season — valid since we have all dates at inference time
- **Attention pooling**: weights important dates more (e.g., flooding event for rice)
- **Dual head**: shared temporal representation, separate classifiers
- **DOY appended after pooling**: global temporal context without polluting sequence
- **Loss weighting**: 0.4 × crop_loss + 0.6 × pheno_loss — matches competition scoring weights

**Why LSTM over Transformer**: LSTM achieves near-Transformer accuracy (90–93% vs 93–95%) with far less training data required. Training set is limited (4 regions), so LSTM generalizes better given data constraints.

**Why BiLSTM over CNN**: CNN requires spatial image context; our task is point-based prediction. LSTM naturally handles variable-length temporal sequences per point.

---

## 7. Cross-Region Generalization

The scoring evaluates on **hidden test regions** not seen during training. This is the core challenge.

Strategies built into our approach:
1. **Temporal stats features (XGBoost)**: mean/std/max across dates normalize regional differences
2. **Attention mechanism (LSTM)**: learns which phenological signals are universal vs region-specific
3. **DOY normalization**: consistent temporal encoding across regions with different acquisition dates

What we should also consider when training data arrives:
- Train on 3 regions, validate on 4th (cross-region CV) to catch overfitting
- Augment with temporal jitter (shift DOY by ±7 days) to improve robustness

---

## 8. MCD12Q2 Auxiliary Data (Not Yet Implemented)

Task spec requires: *"auxiliary phenology data (MCD12Q2)"*

MCD12Q2 = MODIS Land Cover Dynamics product. Provides pixel-level phenological parameters:
- Greenup date, MidGreenup date, Peak date, Maturity date, Senescence date, MidSenescence date, Dormancy date

**How to use it**: At each test point, sample the MCD12Q2 raster to get the expected phenological calendar → encode as features (e.g., days from Greenup to obs_date, days from Peak to obs_date).

**Status**: Not implemented. Need to identify where MCD12Q2 files live in `/input/` when training data arrives.

---

## 9. Pipeline Flow

```
Training (run once on workstation with GPU):
  points_train_label.csv + region_train_1-4/ TIFFs
          ↓
  features.py → extract_all_features()
          ↓ (flat 97-dim)         ↓ (sequences n_dates×16)
  train.py xgboost           train.py lstm
          ↓                        ↓
  models/xgb_crop.pkl        models/lstm_best.pt
  models/xgb_pheno.pkl       models/lstm_config.pkl
  models/label_encoders.pkl  models/label_encoders.pkl

Inference (runs inside Docker on competition server):
  /input/points_test.csv + /input/region_test/ TIFFs
          ↓
  features.py → extract_all_features()
          ↓
  inference.py → load_models("models/") → run_inference()
          ↓
  /output/result.json
```

---

## 10. Remaining Work

| Task | Status | Notes |
|------|--------|-------|
| Feature extraction | Done | `features.py` |
| XGBoost training | Ready | Runs as soon as training data arrives |
| LSTM training | Ready | Needs GPU workstation |
| MCD12Q2 integration | Not started | Find files in training data first |
| Cross-region validation | Not started | Use StratifiedKFold by region |
| Temporal augmentation | Not started | DOY jitter for LSTM |
| Model size check | Not started | Docker image must be submittable |
| End-to-end Docker test | Not started | Test full pipeline locally |

---

## 11. Key References

- [CTANet — fine crop classification with multi-temporal Sentinel-2](https://www.sciencedirect.com/science/article/pii/S1569843224005284) — 93.9% OA on rice/maize/soybean
- [CNN-RF Hybrid for phenology-based paddy rice mapping](https://www.mdpi.com/2073-431X/14/8/336) — 95% OA using NDVI+EVI+LSWI+RGVI
- [FARM: Sentinel-1 SAR + Sentinel-2 rice mapping](https://www.sciencedirect.com/science/article/abs/pii/S0168169923006506) — LSWI+EVI phenological phases
- [LSTM with temporal random masking for crop type](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003897)
- [Corn and soybean mapping with phenological + biophysical info](https://www.tandfonline.com/doi/full/10.1080/15481603.2025.2609467)
- [Automated soybean mapping using canopy water content](https://www.sciencedirect.com/science/article/pii/S1569843222000036) — SWIR key for soybean
