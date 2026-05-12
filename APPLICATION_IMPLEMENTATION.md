# PhenoSync-S2 — Application Implementation Scheme
## 2026 AI and Space Computing Challenge: Space Intelligence Empowering Zero Hunger
### Track 1 — Crop Type & Phenology Classification

---

## Part I: Overall Task Implementation Architecture

### Full-Process Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA INJECTION                                                      │
│  Multi-temporal Sentinel-2 L2A TIFFs  +  points_train_label.csv    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION  [features.py]                                  │
│  • TIFF index: {(region, date, band) → filepath}                    │
│  • Per-point temporal sequence: (T × 24) array                     │
│    – 12 Sentinel-2 bands (B01–B12)                                  │
│    – 10 spectral indices (NDVI, LSWI, EVI, NDWI, RGVI, NDRE,       │
│       CIre, CWC, NBR, LSWI_EVI)                                     │
│    – 2 per-timestep sinusoidal DOY encodings [sin, cos]             │
│  • Flat 156-dim vector for XGBoost baseline                         │
│    – 6 temporal stats × 24 features = 144                           │
│    – 11 phase-gated phenological features                           │
│    – 1 observation DOY                                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DUAL-STAGE TRAINING  [train.py]                                    │
│                                                                     │
│  Stage 1 — XGBoost Baseline (CPU, ~10–30 min)                      │
│  • Class-weighted gradient boosting                                 │
│  • Validates full pipeline before committing GPU time               │
│  • Saves: xgb_crop.pkl, xgb_pheno.pkl, xgb_remapper.pkl            │
│                                                                     │
│  Stage 2 — BiLSTM + Attention (GPU, ~1–3 hrs, 80 epochs)          │
│  • Stratified 80/20 train/val split                                 │
│  • Class-weighted CrossEntropyLoss + WeightedRandomSampler          │
│  • Temporal Random Masking curriculum (0→20% over 10 epochs)       │
│  • Early stopping patience=15 on val_loss                           │
│  • CosineAnnealingLR                                                │
│  • Saves: lstm_best.pt, lstm_config.pkl, label_encoders.pkl         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INFERENCE  [inference.py]                                          │
│  • Loads trained weights from models/                               │
│  • Extracts features for all test points                            │
│  • LSTM priority; falls back to XGBoost if LSTM not available       │
│  • OOB fallback: DOY-encoded single-step sequence for points        │
│    with no TIFF coverage (preserves temporal signal)                │
│  • Outputs result.json: {Lon_Lat_Date: [CropType, PhenoStage]}     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CONTAINERIZED DEPLOYMENT  [Dockerfile + run.sh]                    │
│  • Docker image bundles code + trained weights                      │
│  • run.sh auto-detects TIFF directory under /input/                 │
│  • Output written to ${OUTPUT_DIR:-/output}/result.json             │
│  • Platform path: /workspace/ (code), /input/ (data), /output/     │
└─────────────────────────────────────────────────────────────────────┘
```

### Modularity and Fault Tolerance

| Module | File | Role |
|--------|------|------|
| Feature extraction | `features.py` | Shared by train and inference — single source of truth |
| Model architecture | `model.py` | Decoupled from training logic |
| Training | `train.py` | XGBoost and LSTM modes independently invokable |
| Inference | `inference.py` | Auto-detects available model; graceful fallback chain |
| Entry point | `run.sh` | TIFF directory auto-detection; path fallback logic |

Fault tolerance mechanisms:
- **OOB point handling**: points outside TIFF spatial coverage receive DOY-encoded sequence rather than failing or returning zeros
- **NaN sanitization**: `np.nan_to_num(flat_rows, nan=0.0)` on all extracted features
- **Model fallback chain**: LSTM → XGBoost → placeholder (prevents silent failure on platform)
- **Auto-detects TIFF subdirectory**: run.sh tries `/input/region_test/`, then `/input/regions/`, then `/input/`

---

## Part II: On-Orbit Feasibility

### Uplink File Size and Compression

| Artifact | Estimated Size | Compression Strategy |
|----------|---------------|---------------------|
| `lstm_best.pt` (hidden_dim=128, 2-layer BiLSTM) | ~8–12 MB | PyTorch fp32 weights; quantizable to fp16 (~4–6 MB) |
| `xgb_crop.pkl` + `xgb_pheno.pkl` | ~2–5 MB | Joblib compression level 3 |
| `label_encoders.pkl` + `xgb_remapper.pkl` | <100 KB | Negligible |
| **Total model payload** | **~15–20 MB** | Compresses to ~10–14 MB with gzip |
| Per-date TIFF (single band, one region) | ~13 MB | Already compressed GeoTIFF (LZW/DEFLATE) |
| Full inference input (~12 bands × N dates) | ~500 MB–2 GB | Streamed band-by-band; never fully loaded into RAM |

For on-orbit deployment, the model weights (~15 MB compressed) represent a realistic uplink payload over S-band at 1 Mbps: ~2 minutes transfer time.

### Resource Consumption Estimates

**Inference on 942 test points (full test set):**

| Resource | Estimate | Basis |
|----------|----------|-------|
| Peak RAM (TIFF loading) | ~400–800 MB | Band-by-band batch sampling with rasterio |
| RAM (model inference) | ~200–300 MB | LSTM batch=256, fp32 tensors |
| GPU VRAM (if available) | ~500 MB | BiLSTM inference batch |
| CPU inference time (no GPU) | ~30–120 sec | Depends on TIFF count and I/O speed |
| GPU inference time | ~5–15 sec | CUDA fp32, batch=256 |
| Storage for result.json | <100 KB | 942 JSON entries |

**Training resource requirements (offline, ground-based):**

| Phase | Time | Hardware |
|-------|------|----------|
| Feature extraction (all 4 regions) | ~30–90 min | CPU, 16 GB RAM |
| XGBoost training | ~10–30 min | CPU only |
| LSTM training (80 epochs) | ~1–3 hrs | NVIDIA GPU recommended |

### Dependent Data Sources

1. **Sentinel-2 L2A surface reflectance TIFFs** — primary input; acquired via Copernicus Ground Segment; bands B01–B12 stored per-date per-region
2. **Auxiliary phenology timing** (MCD12Q2 MODIS) — used indirectly via DOY-based phase-gating thresholds embedded in feature engineering
3. **Training labels** (`points_train_label.csv`) — 5,446 labeled sample points across 4 regions; required only during training

All inference-time dependencies are self-contained within the Docker image (model weights + code). No external API calls or network access required at inference time.

### Single-Execution Time and Worst-Case Latency

| Scenario | Estimated Latency |
|----------|------------------|
| Nominal (GPU, 942 points, pre-indexed TIFFs) | ~2–5 min |
| Nominal (CPU only) | ~5–15 min |
| Worst case (large TIFF directory, slow storage, CPU) | ~30 min |
| On-orbit edge processor (ARM Cortex-A72 class) | ~60–120 min (XGBoost path) |

For on-orbit execution, the XGBoost fallback path is preferable: no GPU dependency, ~2–5 MB model, runs on embedded Linux with 2–4 GB RAM.

---

## Part III: Innovation of Implementation Path

### Key Technical Contributions

#### 1. Dual-Head Architecture — Joint Crop and Phenology Prediction
Single forward pass produces both crop type (4 classes) and phenological stage (7 classes) simultaneously. Shared BiLSTM encoder learns representations useful for both tasks while independent heads specialize.

```
BiLSTM (24→256) → MultiHeadAttnPool → DOY context → ┬→ Crop Head (4)
                                                      └→ Pheno Head (7)
```

#### 2. Multi-Head Temporal Attention Pooling
Standard LSTM pools only the final hidden state, losing temporal dynamics. Our `MultiHeadAttnPool` learns 2 independent attention distributions over the time axis, allowing the model to simultaneously focus on:
- Early-season transplanting phase (rice flooding signal)
- Peak canopy closure (crop discrimination)

```python
scores  = self.scorers(x)                            # (B, T, 2 heads)
weights = torch.softmax(scores, dim=1)               # soft temporal focus
ctx     = torch.einsum('bth,btn->bnh', x, weights)  # head-weighted context
```

#### 3. Sinusoidal DOY Encoding — Two Levels
DOY information injected at two scales:
- **Per-timestep** (in sequence): `[sin(2π·doy/365), cos(2π·doy/365)]` — encodes when each image was acquired relative to the crop calendar
- **Observation-level** (in context): 4-dim `[sin, cos, sin(2x), cos(2x)]` — encodes the target prediction date with two harmonics for fine-grained seasonal positioning

This dual encoding allows the model to distinguish between "what is the crop status at each image date" and "what stage should be predicted at the specific observation date."

#### 4. Phase-Gated Spectral Statistics
Rather than simple temporal statistics, features are conditioned on phenological phase:

```python
flood_mask = (lswi > evi) & (ndvi < 0.3)   # transplanting / flooded
grow_mask  = ndvi > 0.4                     # active canopy

flood_mean_ndvi  # NDVI during flooding phase — rice indicator
flood_mean_lswi  # LSWI during flooding — water presence
grow_mean_cwc    # Canopy water content during growth — soybean key
```

LSWI−EVI > 0 is a published indicator of paddy flooding (transplanting phase). Phase-gated means provide the XGBoost model with crop-discriminative statistics that simple temporal means cannot capture.

#### 5. Cloud-Gap Robustness via Temporal Random Masking
Remote sensing time series frequently contain cloud-contaminated acquisitions. Temporal Random Masking (TRM) randomly zeros timesteps during training with a curriculum:

```python
mask_rate = min(0.20, 0.20 * epoch / min(10, total_epochs))
```

The model is progressively trained to predict correctly even with up to 20% of timesteps missing, directly simulating cloud cover gaps. This is critical for on-orbit inference where cloud-free compositing is not always possible.

#### 6. Targeted Spectral Index Selection
Ten indices selected from literature specifically for rice/corn/soybean discrimination:

| Index | Key Discriminative Role |
|-------|------------------------|
| LSWI−EVI | Paddy flooding detection (>0 = transplanting) |
| CWC (B8A−B12)/(B8A+B12) | Canopy water content — primary soybean pod-fill indicator |
| CIre B07/B05−1 | Chlorophyll red-edge — C4 corn vs soybean ~30% difference at peak |
| NDRE (B8A−B05)/(B8A+B05) | Nitrogen status — corn vs soybean separator |
| RGVI (B03−B04)/(B03+B04) | Rice early-season growth on flooded background |

#### 7. Differences from Ground-Based Solutions

| Aspect | Ground-Based | PhenoSync-S2 On-Orbit |
|--------|-------------|----------------------|
| Data access | Full archive available | Streaming band-by-band, limited RAM |
| Cloud handling | Offline compositing | TRM-trained robustness to missing timesteps |
| Model size | Unconstrained | ~15 MB total weights |
| Inference | Server-class GPU | XGBoost fallback for CPU-only edge hardware |
| Latency | Minutes | Targets <30 min on embedded processor |

---

## Part IV: Value and Application Scenarios

### Primary Value Proposition — Zero Hunger (SDG 2)

Accurate crop type mapping and phenological stage detection directly enables:

1. **Early yield forecasting**: Phenological stage (Greenup → Peak → Maturity) combined with crop type allows regional yield estimation 4–6 weeks before harvest, enabling food security agencies to pre-position aid and adjust import/export policies.

2. **Targeted agricultural interventions**: Identifying which fields are at transplanting vs. peak growth enables irrigation scheduling, fertilizer optimization, and pest management timed to actual crop status rather than calendar averages.

3. **Rice production monitoring at scale**: Rice is the primary caloric crop for 3.5 billion people. Rice-specific phenology detection (the 60%-weighted metric in this challenge) enables national food agencies to monitor planting progress, detect delayed transplanting (drought/flood stress indicator), and estimate paddy area in near-real-time.

### Application Scenarios

#### Scenario 1: National Crop Monitoring Service
Sentinel-2 imagery is acquired every 5 days at 10m resolution globally. Running PhenoSync-S2 on a constellation of edge-compute-enabled satellites enables:
- Daily updated crop type maps without ground station downlink latency
- Phenological stage maps available within the same orbit pass as image acquisition
- Coverage of remote agricultural regions with no ground sensor infrastructure

#### Scenario 2: Smallholder Farm Advisory (Developing World)
In sub-Saharan Africa and Southeast Asia, >70% of food is produced by smallholders (<2 ha). PhenoSync-S2 outputs can drive SMS-based advisory services:
- "Fields in your district are at MidGreenup — apply top-dress nitrogen now"
- No smartphone or internet required; prediction aggregated at district level

#### Scenario 3: Disaster Response — Flood Impact Assessment
During flooding events, the LSWI−EVI > 0 flooding indicator identifies inundated paddy fields. Combined with phenological stage, this distinguishes intentional transplanting flooding from destructive flash floods, enabling rapid damage assessment and insurance payout automation.

#### Scenario 4: Carbon Credit Verification
Rice paddies are significant methane sources. Accurate transplanting date and flooding duration from phenological staging enables satellite-verified carbon accounting for paddy methane mitigation programs (e.g., AWD — Alternate Wetting and Drying verification).

### Satellite-Terrestrial Coordination Model

```
SPACE SEGMENT                    GROUND SEGMENT
─────────────────────────────────────────────────────
Sentinel-2 Satellite             National Food Agency
  │ raw L2A bands                  │ crop maps (daily)
  ▼                                │ yield forecasts
On-orbit inference node  ─────────▶ phenology alerts
  (PhenoSync-S2 LSTM/XGB)          │
  │ compressed result.json          ▼
  │ (~100 KB per 1000 pts)       District Advisory
  ▼                                Services
Ground station downlink           Farmer SMS alerts
```

The result.json format (100 KB for 942 points) is compact enough to downlink over a 1-minute S-band contact window, enabling same-day delivery of crop intelligence.

---

## Part V: Future Roadmap

### Application Prospects

PhenoSync-S2 addresses a gap that will only grow: global demand for food will increase 50–70% by 2050 while arable land is shrinking. Satellite-based crop intelligence at scale is the only mechanism capable of monitoring global food production systems in near-real-time. The technical foundations built here — joint crop/phenology prediction, cloud-robust temporal models, edge-deployable weights — are directly applicable to:

- **Global food early warning systems** (WFP, FAO)
- **Agricultural insurance parametric trigger** systems
- **Carbon market MRV** (Measurement, Reporting, Verification) for paddy methane
- **National agricultural statistics** in countries without ground survey capacity

### Technical Improvement Paths

#### Near-term (0–12 months)

1. **Model quantization**: Convert BiLSTM weights from fp32 to int8 (4× size reduction, ~2× inference speedup, <1% accuracy loss). Enables deployment on ARM Cortex-A class processors without GPU.

2. **Temporal resolution enhancement**: Fuse Sentinel-2A + 2B (combined 5-day revisit) with Landsat-8/9 (15-day revisit) for denser time series, reducing cloud gap impact in monsoon regions.

3. **Expanded crop classes**: Add wheat, cotton, sunflower using the same architecture — only the output head and label set change. BiLSTM encoder is crop-agnostic.

4. **Confidence calibration**: Add temperature scaling to convert logits to calibrated probabilities. Enables uncertainty-aware downstream decisions (e.g., "flag low-confidence predictions for ground verification").

#### Medium-term (1–3 years)

5. **On-orbit fine-tuning**: Freeze BiLSTM encoder, retrain only the classification heads using a small set of downlinked ground truth labels. Adapts to new regions and years without full retraining.

6. **Lightweight transformer variant**: Replace BiLSTM with a 2-layer Temporal Transformer (8 heads, d_model=64). Comparable accuracy, better parallelism on ARM NEON SIMD units found in space-grade processors.

7. **Multi-sensor fusion**: Incorporate SAR (Sentinel-1 C-band backscatter) as a cloud-penetrating channel. Rice flooding detection via SAR is robust during monsoon season when optical imagery has >80% cloud cover.

8. **Distributed constellation inference**: Partition the crop monitoring area across multiple satellites, each running inference on its ground swath, reducing per-satellite compute load and enabling global daily updates.

#### Long-term (3–10 years)

9. **Autonomous on-orbit retraining**: When ground truth labels are available (e.g., from calibration sites with permanent sensors), the satellite autonomously updates model weights during eclipse periods using onboard NVME storage and low-power NPU.

10. **Federated learning across satellites**: Multiple satellites share gradient updates (not raw imagery) during cross-links, collectively training a shared global model while each specializes on its regional data distribution.

### Key Development Milestones for On-Orbit Computing

| Milestone | Target | Metric |
|-----------|--------|--------|
| Int8 quantized inference validated | Year 1 | <1% AlgoScore degradation vs fp32 |
| ARM Cortex-A72 inference benchmark | Year 1 | <30 min for 1000-point batch |
| Sentinel-1 SAR fusion prototype | Year 2 | +5% rice F1 in cloud-heavy regions |
| On-orbit fine-tuning demonstration | Year 3 | Adapt to new region with 200 labels |
| Constellation-distributed inference | Year 5 | Global daily crop map at 20m |

---

## Part VI: Team Introduction

**Team: PhenoSync-S2**

Our team combines expertise in remote sensing, machine learning, and agricultural applications. We have direct experience working with Sentinel-2 multi-temporal data pipelines, training deep learning models for geospatial classification tasks, and deploying inference systems in containerized environments.

**Relevant capabilities demonstrated in this submission:**

- End-to-end implementation: data extraction → feature engineering → model training → containerized inference
- Application of published remote sensing literature to practical model design (spectral index selection, phase-gated features, temporal attention)
- Production-quality code: modular architecture, fault-tolerant inference, Docker deployment
- Understanding of competition-specific constraints: scoring formula optimization (40/60 crop/phenology weighting drove architecture decisions)

**Why this matters to us:** Food security and precision agriculture represent one of the highest-leverage intersections of AI and societal impact. Satellite intelligence that reaches smallholder farmers without requiring ground infrastructure is a genuine step toward SDG 2.

---

## Appendix: Scoring Formula Implementation

```python
# Implemented in inference.py → score_against_labels()

AlgoScore = (0.4 × Macro_F1_Crop + 0.6 × Macro_F1_RicePheno) × 100

# Crop F1: macro average over {rice, corn, soybean}
# (background excluded from crop scoring per competition rules)
f1_crop = f1_score(y_true_crop, y_pred_crop,
                   labels=["rice", "corn", "soybean"],
                   average="macro", zero_division=0)

# Rice Pheno F1: "Double Hit" — both crop AND phenology must be correct
# A prediction of "rice/Peak" for a true "rice/Peak" = TP
# A prediction of "rice/Maturity" for a true "rice/Peak" = FP + FN
# A prediction of "corn/Peak" for a true "rice/Peak" = FN (wrong crop)
y_pred_pheno.append(
    f"rice_{pred_pheno}" if pred_crop == "rice" else "wrong_crop"
)
f1_pheno = f1_score(y_true_pheno, y_pred_pheno, average="macro", zero_division=0)
```

The architecture decision to weight phenology loss at 0.6 and crop loss at 0.4 during training directly mirrors the competition scoring formula, aligning model optimization with evaluation criteria.

---

## Appendix: Model Architecture Summary

```
Input: (B, T, 24)
  ├─ 12 Sentinel-2 bands (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
  ├─ 10 spectral indices (NDVI, LSWI, NDWI, EVI, RGVI, NDRE, CIre, CWC, NBR, LSWI_EVI)
  └─ 2 per-timestep DOY encodings [sin(2π·doy/365), cos(2π·doy/365)]

BiLSTM — 128 hidden × 2 directions → (B, T, 256)
  └─ 2 layers, dropout=0.3 between layers

MultiHeadAttnPool — 2 heads → (B, 256)
  └─ Learned soft attention over T timesteps, per head

4-dim sinusoidal DOY context → concat → (B, 260)
  └─ [sin(2π·doy), cos(2π·doy), sin(4π·doy), cos(4π·doy)]

Crop Head:  Linear(260→128) → LayerNorm → ReLU → Dropout → Linear(128→64) → ReLU → Linear(64→4)
Pheno Head: Linear(260→128) → LayerNorm → ReLU → Dropout → Linear(128→64) → ReLU → Linear(64→7)

Total parameters: ~1.2M
Model size (fp32): ~8–12 MB
```
