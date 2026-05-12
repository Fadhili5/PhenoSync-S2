We have already created an initial project for you to participate in the ITU competition and written the corresponding `.gitlab-ci.yml` file. When you submit code to the `main` branch or create a new pipeline, a training task will be triggered. You can then go to the web interface to check your project status and runtime logs. Please note that you can only submit **3 times per day**; submissions beyond that will not be accepted.

Our GitLab instance supports logging in with Zero2x platform accounts. However, please note that these accounts cannot be used to push code via Git or pull Docker images directly.

To perform these actions, you must create a personal access token to use as your password. Please navigate to **Avatar > Edit Profile > Access Tokens** to generate your token. For detailed instructions, please refer to the official Jihu GitLab documentation.

Please note the following in `.gitlab-ci.yml`: dataset path, model output address, and launch name can be modified as needed. The paths in your code must be consistent with those in `.gitlab-ci.yml`. Only modify the container mount paths — do **not** change the actual paths to avoid conflicts with other participants.

We also have specific requirements for the naming of your output files. Please follow them strictly, as failure to do so will affect the final scoring of your submitted model.

**Track 1 output:**  
`/your_container_output_path/result.json`

**Track 2 output:**  
`/your_container_output_path/turbidity_result.json`  
`/your_container_output_path/chla_result.json`

**Track 3 output:**  
`/your_container_output_path/result.json`

The base images used for the competition are specified in the `Dockerfile`. If you need to use the base image for local debugging, you can search for the project `itu_docker_images`. The container registry of that project contains the base images used for the competition. You can also pull them with Docker.

```
sudo vim /etc/docker/daemon.json

{
  "insecure-registries": ["gitlab-itu.zero2x.org:5050"]
}

systemctl daemon-reload
systemctl restart docker

docker login http://gitlab-itu.zero2x.org:5050

docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-py310.19
docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images/competition-base:pytorch2.5.1-cuda12.1-cudnn9
```

---

# PhenoSync-S2 — Track 1: Crop Type & Phenology Classification

Multi-temporal Sentinel-2 crop classification and rice phenology staging.  
Predicts **crop type** (rice / corn / soybean / background) and **phenological stage**
(Greenup → Dormancy) for every input point × date combination.

**Scoring formula:**
```
AlgoScore = 0.4 × Crop_MacroF1  +  0.6 × RicePhenology_MacroF1
Final Score = AlgoScore × 60%  +  Solution Design × 40%
```

---

## Repository Structure

```
PhenoSync-S2/
├── features.py            # Spectral indices, temporal stats, phase-gated features
├── model.py               # BiLSTM + multi-head attention architecture
├── train.py               # Training script (XGBoost + LSTM modes)
├── inference.py           # Inference + result.json export
├── run.sh                 # Container entry point (platform use only)
├── Dockerfile             # Docker image — bundles code + trained weights
├── requirements.txt       # Python dependencies
├── setup_data.py          # Zip extraction helper (if data arrives zipped)
│
├── DATA/                  # ← Place training data here before training
│   ├── points_train_label.csv
│   ├── region_train_1/    (from track1_download_link_5)
│   ├── region_train_2/    (from track1_download_link_4, if available)
│   ├── region_train_3/    (from track1_download_link_3)
│   └── region_train_4/    (from track1_download_link_2)
│
├── models/                # ← Trained weights saved here (commit before pushing)
│   ├── lstm_best.pt
│   ├── xgb_crop.pkl
│   ├── xgb_pheno.pkl
│   ├── xgb_remapper.pkl
│   └── label_encoders.pkl
│
└── test_input_sample/     # Local test data (mirrors /input on platform)
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

---

## 2. Data Layout

Place extracted training data under `DATA/`:

```
DATA/
├── points_train_label.csv          # 5,446 labeled rows
│                                   # Columns: point_id, Longitude, Latitude,
│                                   #          phenophase_date, crop_type, phenophase_name
│                                   # Crop dist: rice=2569  corn=1603  soybean=1274
│
├── region_train_1/                 # Flat directory of TIFF files
│   ├── region00_2018-06-07-00-00_..._Sentinel-2_L2A_B01_(Raw).tiff
│   ├── region00_2018-06-07-00-00_..._Sentinel-2_L2A_B02_(Raw).tiff
│   └── ...  (12 bands × all dates × all sub-regions, ~2500 files per dir)
├── region_train_3/
└── region_train_4/
```

**TIFF filename format** (parsed automatically):
```
regionXX_YYYY-MM-DD-00-00_YYYY-MM-DD-23-59_Sentinel-2_L2A_BXX_(Raw).tiff
```

> If data arrived as `.zip` files, run `python setup_data.py` to extract everything and print the training commands.

---

## 3. Training

### Step 1 — XGBoost (CPU, ~10-30 min, validates full pipeline first)

```powershell
python train.py --mode xgboost `
    --label_csv DATA\points_train_label.csv `
    --tiff_dirs DATA\region_train_1 DATA\region_train_3 DATA\region_train_4 `
    --output_dir models
```

Expected output:
```
[index]    7698 TIFFs | 3 regions | N dates
[features] 5446 samples | XXXX in-bounds
[features] flat shape=(5446, 156) | max_seq_len=N | oob=X
[xgb]      Train acc — crop=0.9XX  pheno=0.9XX
[log]      Run appended → models/training_log.csv
```

### Step 2 — BiLSTM (GPU recommended, ~1-3 hours for 80 epochs)

```powershell
python train.py --mode lstm `
    --label_csv DATA\points_train_label.csv `
    --tiff_dirs DATA\region_train_1 DATA\region_train_3 DATA\region_train_4 `
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

Training stops early if validation loss stops improving (patience = 15 epochs).

**If `region_train_2` data arrives later**, add it and retrain — warm-start picks up from existing weights:

```powershell
python train.py --mode lstm `
    --label_csv DATA\points_train_label.csv `
    --tiff_dirs DATA\region_train_1 DATA\region_train_2 DATA\region_train_3 DATA\region_train_4 `
    --output_dir models `
    --epochs 80 --batch_size 64 --hidden_dim 128
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `xgboost` or `lstm` |
| `--label_csv` | required | Path to labeled training CSV |
| `--tiff_dirs` | required | One or more TIFF directories (space-separated) |
| `--output_dir` | `models` | Where to save model weights |
| `--epochs` | `50` | LSTM training epochs |
| `--batch_size` | `64` | LSTM batch size |
| `--hidden_dim` | `128` | LSTM hidden units per direction (BiLSTM = ×2) |
| `--lr` | `1e-3` | AdamW learning rate |
| `--weight_crop` | `0.4` | Crop loss weight (phenology gets remaining 0.6) |

### Output files saved to `models/`

| File | Description |
|------|-------------|
| `lstm_best.pt` | Best LSTM weights (by validation loss) |
| `lstm_final.pt` | Final epoch weights |
| `lstm_config.pkl` | Architecture config — needed by inference |
| `xgb_crop.pkl` | XGBoost crop classifier |
| `xgb_pheno.pkl` | XGBoost phenology classifier |
| `xgb_remapper.pkl` | Label index remapper |
| `label_encoders.pkl` | Class name encoders — needed by inference |
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

```powershell
python inference.py `
    --input_csv  test_input_sample\test_point.csv `
    --tiff_dir   test_input_sample\region_test `
    --output_dir output `
    --model_dir  models `
    --label_csv  DATA\points_train_label.csv
```

Prints:
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

**Key format:** `{Longitude}_{Latitude}_{phenophase_date}` — date string is preserved **exactly** as it appears in the input CSV (e.g. `2018/9/1` NOT `2018-09-01`).  
**Every row** in the input CSV must have a corresponding key. Missing keys = penalty.

---

## 5. Platform Submission Workflow

The platform is **inference-only**. Train locally, commit weights to `models/`, push to `main`.

### Full workflow

```
1. Place training data in DATA/
2. python train.py --mode xgboost ...   ← trains + saves to models/
3. python train.py --mode lstm ...      ← trains + saves to models/
4. python inference.py ... --label_csv  ← verify AlgoScore locally
5. git add models/
6. git commit -m "Add trained model weights"
7. git push origin main                 ← triggers CI/CD (counts as 1 submission)
```

### Pre-push checklist

- [ ] `models/lstm_best.pt` exists
- [ ] `models/xgb_crop.pkl` and `models/xgb_pheno.pkl` exist
- [ ] `models/label_encoders.pkl` exists
- [ ] Local inference produces `result.json` with **all 942 test rows** covered
- [ ] Date keys use original format (`2018/9/1` not `2018-09-01`)
- [ ] Tested `python inference.py` without errors before pushing

### Simulate platform locally with Docker

```powershell
docker build -t phenosync-s2 .

docker run --rm `
    -v "${PWD}\test_input_sample:/input" `
    -v "${PWD}\output_docker:/output" `
    phenosync-s2

# Check result
cat output_docker\result.json
```

### Platform paths (handled automatically by `run.sh`)

| Container path | Content |
|----------------|---------|
| `/input/test_point.csv` | Test coordinates + dates |
| `/input/region_test/` | Test TIFF files |
| `/workspace/models/` | Your trained weights (bundled in image) |
| `/output/result.json` | **Required output — must exist after run** |

> `run.sh` auto-detects the TIFF directory under `/input/` and falls back gracefully if the directory name differs.

---

## 6. Model Architecture

### Feature vector — 24 dims per timestep

**12 Sentinel-2 bands:** B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12

**10 spectral indices:**

| Index | Formula | Why it matters |
|-------|---------|----------------|
| NDVI | (B08−B04)/(B08+B04) | Vegetation density |
| LSWI | (B08−B11)/(B08+B11) | Leaf water, flooding detection |
| NDWI | (B03−B08)/(B03+B08) | Open water |
| EVI | 2.5×(B08−B04)/(B08+6B04−7.5B02+1) | Canopy LAI |
| RGVI | (B03−B04)/(B03+B04) | Rice early-season growth |
| NDRE | (B8A−B05)/(B8A+B05) | Nitrogen status — corn/soy separator |
| CIre | B07/B05 − 1 | Chlorophyll red-edge — C4 corn vs soybean |
| CWC | (B8A−B12)/(B8A+B12) | Canopy water via SWIR-2 — **soybean key** |
| NBR | (B08−B12)/(B08+B12) | Water stress, SWIR-2 based |
| LSWI−EVI | LSWI − EVI | **>0 = flooded/transplanting (rice indicator)** |

**2 positional encodings:** sin(2π·doy/365), cos(2π·doy/365) — per timestep calendar position

**XGBoost flat features (156-dim):**
- 6 stats × 24 features = 144 (mean, std, max, min, median, temporal diff_std)
- Phenological timing: DOY of NDVI peak, DOY of greenup, NDVI amplitude, CWC amplitude, flooding fraction
- Phase-gated means: NDVI/LSWI/EVI during transplanting phase; NDVI/EVI/CWC during growing phase
- 1 observation DOY

### LSTM architecture

```
Input (B, T, 24)
    │
BiLSTM — 128 hidden × 2 directions → (B, T, 256)
    │
MultiHeadAttnPool — 2 heads → (B, 256)
    │
DOY sinusoidal encoding — 4-dim [sin, cos, sin2x, cos2x] → concat → (B, 260)
    │
    ├── Crop Head:  Linear(260→128) → LayerNorm → ReLU → Linear(128→64) → Linear(64→4)
    └── Pheno Head: Linear(260→128) → LayerNorm → ReLU → Linear(128→64) → Linear(64→7)

Outputs: crop logits (4) + phenology logits (7)
```

### Training techniques

| Technique | Detail | Purpose |
|-----------|--------|---------|
| Temporal Random Masking | 0→20% curriculum over 10 epochs | Cloud-gap robustness |
| Stratified 80/20 val split | Stratified by crop class | Honest generalization estimate |
| Early stopping | Patience=15 on val loss | Prevent overfitting |
| Class-weighted CrossEntropyLoss | sklearn `compute_class_weight` | Minority class balance |
| WeightedRandomSampler | Per-sample weight from class freq | Balanced mini-batches |
| CosineAnnealingLR | T_max = epochs | Smooth LR decay |

---

## 7. Crop Classes & Phenology Stages

**Crop classes (4):** `rice` `corn` `soybean` `background`

**Phenology stages (7):**

| Stage | Description |
|-------|-------------|
| Greenup | Vegetation emergence, NDVI rising from baseline |
| MidGreenup | Rapid green-up phase |
| Peak | Maximum NDVI / canopy closure |
| Maturity | Canopy stable, grain fill |
| Senescence | NDVI declining, crop aging |
| MidSenescence | Continued decline |
| Dormancy | Post-harvest / bare soil |

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Missing package | `pip install -r requirements.txt` |
| `0 in-bounds` points | Wrong `--tiff_dir` path | Check TIFF directory path exists and contains `.tiff` files |
| All predictions = `background` | No trained models found | Run training first, then inference |
| `result.json` missing keys | Input CSV has NaN in Longitude/Latitude | Check input CSV for malformed rows |
| Score = 0 on platform | Wrong output filename or path | File must be `/output/result.json` exactly |
| Low pheno F1 | Model predicts wrong crop (not rice) | Rice phenology only scored when crop = rice; improve crop classifier first |
| CUDA out of memory | Batch too large | Reduce `--batch_size` (try 32 or 16) |
