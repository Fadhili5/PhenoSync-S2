# PhenoSync-S2 — Track 1: Crop Type & Phenology Classification

Multi-temporal Sentinel-2 crop classification and rice phenology staging.
Predicts **crop type** (rice / corn / soybean / background) and **phenological stage**
(Greenup → Dormancy) for every input point × date combination.

## Scoring Formula

```
AlgoScore    = 0.4 × Crop_MacroF1  +  0.6 × RicePhenology_MacroF1
Final Score  = AlgoScore × 60%  +  Solution Design × 40%
```

Rice phenology F1 is weighted higher (0.6) — rice crop classification accuracy is a prerequisite since phenology is only scored when the crop prediction is correct.

---

## Repository Structure

```
PhenoSync-S2/
├── features.py             # Spectral indices, temporal stats, phase-gated features
├── model.py                # BiLSTM + multi-head attention architecture
├── train.py                # Training script (XGBoost + LSTM modes)
├── inference.py            # Inference + result.json export
├── run.sh                  # Container entry point (platform use only)
├── Dockerfile              # Docker image — bundles code + trained weights
├── requirements.txt        # Python dependencies
├── setup_data.py           # Zip extraction helper (if data arrives zipped)
│
├── DATA/                   # Place training data here before training
│   ├── points_train_label.csv
│   ├── region_train_1/
│   ├── region_train_2/
│   ├── region_train_3/
│   └── region_train_4/
│
├── models/                 # Trained weights saved here (commit before pushing)
│   ├── lstm_best.pt
│   ├── lstm_final.pt
│   ├── lstm_config.pkl
│   ├── xgb_crop.pkl
│   ├── xgb_pheno.pkl
│   ├── xgb_remapper.pkl
│   └── label_encoders.pkl
│
└── test_input_sample/      # Local test data (mirrors /input on platform)
    ├── test_point.csv
    ├── test_data_label_sample.csv
    └── region_test/
```

---

## 1. Environment Setup

Python 3.10+. CUDA GPU strongly recommended for LSTM training.

```powershell
pip install -r requirements.txt
```

Or manually:

```powershell
pip install numpy pandas scikit-learn torch xgboost joblib rasterio
```

> For CUDA support, install PyTorch with the appropriate CUDA version from https://pytorch.org/get-started/locally/

---

## 2. Data Layout

Place extracted training data under `DATA/`:

```
DATA/
├── points_train_label.csv          # 5,446 labeled rows
│                                   # Columns: point_id, Longitude, Latitude,
│                                   #          phenophase_date, crop_type, phenophase_name
│                                   # Crop distribution: rice=2569  corn=1603  soybean=1274
│
├── region_train_1/                 # Flat directory of TIFF files
│   ├── region00_2018-06-07-00-00_..._Sentinel-2_L2A_B01_(Raw).tiff
│   └── ...  (12 bands × all dates × all sub-regions, ~2500 files per dir)
├── region_train_2/
├── region_train_3/
└── region_train_4/
```

**TIFF filename format** (parsed automatically):
```
regionXX_YYYY-MM-DD-00-00_YYYY-MM-DD-23-59_Sentinel-2_L2A_BXX_(Raw).tiff
```

> If data arrived as `.zip` files, run `python setup_data.py` to extract everything and print ready-to-run training commands.

---

## 3. Training

Training is a two-stage pipeline. Run XGBoost first (fast, validates data pipeline), then LSTM (higher accuracy, GPU recommended).

### Stage 1 — XGBoost (CPU, ~10–30 min)

XGBoost operates on 156-dimensional flat feature vectors derived from temporal statistics across all timesteps. Provides a strong baseline and validates the full data loading pipeline quickly.

```powershell
python train.py --mode xgboost `
    --label_csv DATA\points_train_label.csv `
    --tiff_dirs DATA\region_train_1 DATA\region_train_2 DATA\region_train_3 DATA\region_train_4 `
    --output_dir models
```

Expected terminal output:
```
[index]    7698 TIFFs | 3 regions | N dates
[features] 5446 samples | XXXX in-bounds
[features] flat shape=(5446, 156) | max_seq_len=N | oob=X
[xgb]      Train acc — crop=0.9XX  pheno=0.9XX
[log]      Run appended → models/training_log.csv
```

### Stage 2 — BiLSTM (GPU recommended, ~1–3 hours for 80 epochs)

BiLSTM processes the full temporal sequence (T × 24-dim feature vectors per point). Learns phenological trajectories that flat statistics miss — for example, the transplanting flood pulse for rice and the red-edge peak timing that separates corn from soybean.

```powershell
python train.py --mode lstm `
    --label_csv DATA\points_train_label.csv `
    --tiff_dirs DATA\region_train_1 DATA\region_train_2 DATA\region_train_3 DATA\region_train_4 `
    --output_dir models `
    --epochs 80 `
    --batch_size 64 `
    --hidden_dim 128
```

Expected output per epoch:
```
[lstm] Epoch 001/080 | train_loss=1.2341 | val_loss=1.1890 | crop_acc=0.612 | val_crop=0.634 | val_pheno=0.521 | trm=0.02
[lstm] Epoch 010/080 | train_loss=0.8123 | val_loss=0.7654 | crop_acc=0.781 | val_crop=0.803 | val_pheno=0.742 | trm=0.20
...
[lstm] Best val loss=0.4123 | Models saved to models
```

Early stopping triggers after 15 epochs without validation loss improvement. The best checkpoint (`lstm_best.pt`) is saved whenever validation loss improves.

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `xgboost` or `lstm` |
| `--label_csv` | required | Path to labeled training CSV |
| `--tiff_dirs` | required | One or more TIFF directories (space-separated) |
| `--output_dir` | `models` | Where to save model weights |
| `--epochs` | `50` | LSTM training epochs |
| `--batch_size` | `64` | LSTM batch size (reduce to 32/16 if OOM) |
| `--hidden_dim` | `128` | LSTM hidden units per direction (BiLSTM = ×2) |
| `--lr` | `1e-3` | AdamW learning rate |
| `--weight_crop` | `0.4` | Crop loss weight (phenology gets remaining 0.6) |

### Output files saved to `models/`

| File | Description |
|------|-------------|
| `lstm_best.pt` | Best LSTM weights by validation loss — use this for submission |
| `lstm_final.pt` | Final epoch weights |
| `lstm_config.pkl` | Architecture config — required for inference |
| `xgb_crop.pkl` | XGBoost crop classifier |
| `xgb_pheno.pkl` | XGBoost phenology classifier |
| `xgb_remapper.pkl` | Label index remapper |
| `label_encoders.pkl` | Class name encoders — required for inference |
| `training_log.csv` | Accuracy history across all runs |

---

## 4. Local Inference

### Basic inference (no scoring)

```powershell
python inference.py `
    --input_csv  test_input_sample\test_point.csv `
    --tiff_dir   test_input_sample\region_test `
    --output_dir output `
    --model_dir  models
```

### Inference with local scoring

Requires a label CSV with ground truth. Prints Crop MacroF1, Rice Phenology MacroF1, and AlgoScore so you can evaluate quality before committing a submission.

```powershell
python inference.py `
    --input_csv  test_input_sample\test_point.csv `
    --tiff_dir   test_input_sample\region_test `
    --output_dir output `
    --model_dir  models `
    --label_csv  DATA\points_train_label.csv
```

Output:
```
[score] Crop MacroF1=0.XXXX  (n=XXX)
[score] Rice Phenology MacroF1=0.XXXX  (n=XXX)
[score] AlgoScore=XX.XX
```

### Output format — `output/result.json`

```json
{
  "124.703696_48.543523_2018/9/1":    ["rice",    "Maturity"],
  "125.331726_48.768485_2018/6/21":   ["corn",    "Greenup"],
  "125.331726_48.768485_2018/7/31":   ["soybean", "Peak"],
  ...
}
```

**Key format:** `{Longitude}_{Latitude}_{phenophase_date}`

**Critical:** The date string must be preserved **exactly** as it appears in the input CSV (e.g. `2018/9/1` — **not** `2018-09-01`). Every row in the input CSV must have a corresponding key. Missing keys are penalized.

---

## 5. Platform Submission Workflow

The competition platform is **inference-only**. Training must be done locally. Commit trained weights to `models/` and push to `main` — this triggers the CI/CD pipeline, which builds the Docker image and runs inference inside the container.

**Limit: 3 submissions per day.** Each `git push origin main` counts as one submission.

### Full workflow

```
1. Place training data in DATA/
2. python train.py --mode xgboost ...   → trains XGBoost, saves to models/
3. python train.py --mode lstm ...      → trains LSTM, saves to models/
4. python inference.py ... --label_csv  → verify AlgoScore locally
5. git add models/
6. git commit -m "Add trained model weights"
7. git push origin main                 → triggers CI/CD (counts as 1 submission)
```

### Pre-push checklist

- [ ] `models/lstm_best.pt` exists and is non-zero
- [ ] `models/xgb_crop.pkl` and `models/xgb_pheno.pkl` exist
- [ ] `models/label_encoders.pkl` and `models/lstm_config.pkl` exist
- [ ] Local inference completes without errors
- [ ] `result.json` covers **all rows** in the test CSV
- [ ] Date keys use original format (`2018/9/1` not `2018-09-01`)
- [ ] AlgoScore is reasonable (not 0 or suspiciously low)

### Simulate platform locally with Docker

Build and run the container exactly as the platform does, using local test data:

```powershell
docker build -t phenosync-s2 .

docker run --rm `
    -v "${PWD}\test_input_sample:/input" `
    -v "${PWD}\output_docker:/output" `
    phenosync-s2

# Inspect output
cat output_docker\result.json
```

### Platform paths (handled automatically by `run.sh`)

| Container path | Content |
|----------------|---------|
| `/input/test_point.csv` | Test coordinates + dates |
| `/input/region_test/` | Test TIFF files |
| `/workspace/models/` | Trained weights (bundled in image at build time) |
| `/output/result.json` | Required output — must exist after container exits |

`run.sh` auto-detects the TIFF directory under `/input/` and falls back gracefully if the directory name differs from `region_test`.

### Platform access

To push code or pull Docker images, you need a personal access token (not your platform login password):
**GitLab → Avatar → Edit Profile → Access Tokens**

---

## 6. Model Architecture

### Feature vector — 24 dimensions per timestep

**12 Sentinel-2 raw bands:** B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12

**10 spectral indices:**

| Index | Formula | Why it matters |
|-------|---------|----------------|
| NDVI | (B08−B04)/(B08+B04) | Vegetation density, primary growth signal |
| LSWI | (B08−B11)/(B08+B11) | Leaf water content, flooding detection |
| NDWI | (B03−B08)/(B03+B08) | Open water surface |
| EVI | 2.5×(B08−B04)/(B08+6B04−7.5B02+1) | Canopy LAI, less saturated than NDVI in dense canopy |
| RGVI | (B03−B04)/(B03+B04) | Early-season rice growth signal |
| NDRE | (B8A−B05)/(B8A+B05) | Nitrogen status — strong corn/soy separator |
| CIre | B07/B05 − 1 | Chlorophyll red-edge — C4 corn vs. soybean |
| CWC | (B8A−B12)/(B8A+B12) | Canopy water via SWIR-2 — key soybean indicator |
| NBR | (B08−B12)/(B08+B12) | Water stress and SWIR-2 response |
| LSWI−EVI | LSWI − EVI | >0 signals flooded/transplanting — primary rice transplant indicator |

**2 positional encodings per timestep:** `sin(2π·doy/365)`, `cos(2π·doy/365)` — encodes absolute calendar position so the model learns seasonality

### XGBoost flat features (156-dim)

- 6 temporal statistics × 24 features = 144 dims: `mean, std, max, min, median, temporal diff_std`
- Phenological timing: DOY of NDVI peak, DOY of greenup, NDVI amplitude, CWC amplitude, flooding fraction
- Phase-gated means: NDVI/LSWI/EVI during transplanting phase; NDVI/EVI/CWC during growing phase
- 1 observation DOY (query date as a scalar)

### BiLSTM architecture

```
Input (B, T, 24)
    │
BiLSTM — 128 hidden × 2 directions → (B, T, 256)
    │
MultiHeadAttentionPool — 2 heads → (B, 256)
    │
DOY sinusoidal encoding — 4-dim [sin, cos, sin2x, cos2x] → concat → (B, 260)
    │
    ├── Crop Head:  Linear(260→128) → LayerNorm → ReLU → Linear(128→64) → Linear(64→4)
    └── Pheno Head: Linear(260→128) → LayerNorm → ReLU → Linear(128→64) → Linear(64→7)

Outputs: crop logits (4) + phenology logits (7)
```

The multi-head attention pooling mechanism allows the model to selectively weight timesteps — it learns to upweight transplanting-phase observations for rice and NDRE-peak windows for corn/soybean separation.

The DOY sinusoidal encoding injected at the pooled representation conditions both heads on query date, enabling phenological stage prediction relative to the crop's growth calendar.

### Training techniques

| Technique | Detail | Purpose |
|-----------|--------|---------|
| Temporal Random Masking | 0→20% curriculum over 10 epochs | Cloud-gap robustness — simulates missing observations |
| Stratified 80/20 val split | Stratified by crop class | Honest generalization estimate across all classes |
| Early stopping | Patience=15 on val loss | Prevent overfitting |
| Class-weighted CrossEntropyLoss | sklearn `compute_class_weight` | Balances minority class (soybean) learning |
| WeightedRandomSampler | Per-sample weight from class frequency | Balanced mini-batches during training |
| CosineAnnealingLR | T_max = total epochs | Smooth LR decay toward zero |
| AdamW optimizer | Default lr=1e-3, weight_decay | Better generalization than Adam |

---

## 7. Crop Classes & Phenology Stages

**Crop classes (4):** `rice` `corn` `soybean` `background`

**Phenology stages (7):**

| Stage | Description | Typical NDVI range |
|-------|-------------|-------------------|
| Greenup | Vegetation emergence, NDVI rising from baseline | 0.2–0.4 |
| MidGreenup | Rapid canopy development | 0.4–0.6 |
| Peak | Maximum NDVI / canopy closure | 0.7–0.9 |
| Maturity | Canopy stable, grain-filling underway | 0.6–0.8 |
| Senescence | NDVI declining, crop aging | 0.4–0.6 |
| MidSenescence | Continued NDVI decline | 0.2–0.4 |
| Dormancy | Post-harvest / bare soil | <0.2 |

**Important:** Rice phenology is only scored when the crop prediction is also correct. Improving crop F1 — especially rice recall — directly improves the phenology component of AlgoScore.

---

## 8. Performance Tips

- **Low rice recall:** Check LSWI−EVI flooding fraction. Rice transplanting (LSWI−EVI > 0) is the most discriminative rice feature. Sparse in-bounds observations during June–July (transplanting window) degrade rice identification.
- **Low pheno F1:** Verify inference uses `lstm_best.pt` not `lstm_final.pt`. Best checkpoint generalizes better than final epoch.
- **Improving corn/soy separation:** NDRE and CIre are the primary separators — verify B05, B07, B8A are loading correctly from TIFFs.
- **Score plateau after epoch 20:** Try `--lr 5e-4` or `--hidden_dim 256` for more model capacity.
- **Sparse observations:** Points near cloud-heavy regions may have <3 valid dates. DOY-encoded sequences handle this gracefully but very sparse points will have lower accuracy — nothing to do except accept this noise.
- **Ensemble:** The codebase trains both XGBoost and LSTM. Inference uses LSTM by default; consider a soft-vote ensemble if you want to extract more signal from both.

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Missing package | `pip install -r requirements.txt` |
| `0 in-bounds` points | Wrong `--tiff_dir` path | Verify path exists and contains `.tiff` files |
| All predictions = `background` | No trained models found | Run training first, then inference |
| `result.json` missing keys | NaN in Longitude/Latitude | Inspect input CSV for malformed rows |
| Score = 0 on platform | Wrong output path or filename | Must be `/output/result.json` exactly |
| Low pheno F1 | Crop misclassified as non-rice | Phenology only scored when crop = rice; fix crop F1 first |
| CUDA out of memory | Batch too large | Reduce `--batch_size` to 32 or 16 |
| `result.json` date keys wrong format | Date reformatted by pandas | Keys must match input CSV exactly — code preserves raw string |
| Training stalls at same loss | Learning rate too high | Try `--lr 5e-4` |
| Platform CI fails | Models not committed | `git add models/` before push — large `.pt` files need explicit staging |
| Docker build fails | Missing base image | Login to registry first: `docker login gitlab-itu.zero2x.org:5050` |

---

## 10. Platform Notes

- Each `git push origin main` counts as one submission. Limit is **3 per day**.
- The platform runs inference only — no training occurs inside the container.
- Container has no internet access. All dependencies must be bundled in the Docker image via `requirements.txt` and `Dockerfile`.
- Output file must be at `/output/result.json` — any other path or name scores zero.
- Monitor your submission status and logs at the GitLab project web interface after pushing.
