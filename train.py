"""
Training script — XGBoost baseline and BiLSTM model.

Usage:
  # XGBoost (no GPU needed)
  python train.py --mode xgboost `
      --label_csv /path/to/points_train_label.csv `
      --tiff_dirs /path/to/region_train_1 /path/to/region_train_2 `
      --output_dir models/

  # LSTM (GPU recommended)
  python train.py --mode lstm `
      --label_csv /path/to/points_train_label.csv `
      --tiff_dirs /path/to/region_train_1 /path/to/region_train_2 `
      --output_dir models/ `
      --epochs 50 --batch_size 64 --hidden_dim 128
"""

import argparse
import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from features import build_tiff_index, extract_all_features, SEQ_DIM
from model import CropPhenologyLSTM, CROP_CLASSES, PHENO_CLASSES


# ── TRAINING LOG ───────────────────────────────────────────────────────────────

def append_log(output_dir: str, row: dict):
    log_path = os.path.join(output_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[log] Run appended → {log_path}")


# ── LABEL LOADING ──────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Pre_crop_type" in df.columns and "crop_type" not in df.columns:
        df = df.rename(columns={"Pre_crop_type": "crop_type"})
    if "Pre_phenophase" in df.columns and "phenophase_name" not in df.columns:
        df = df.rename(columns={"Pre_phenophase": "phenophase_name"})
    assert "crop_type"       in df.columns, "Missing crop_type column"
    assert "phenophase_name" in df.columns, "Missing phenophase_name column"
    df = df.dropna(subset=["crop_type", "phenophase_name"])
    print(f"[labels] {len(df)} samples | crops={df.crop_type.unique().tolist()}")
    return df


def encode_labels(df: pd.DataFrame):
    le_crop  = LabelEncoder().fit(CROP_CLASSES)
    le_pheno = LabelEncoder().fit(PHENO_CLASSES)
    y_crop   = le_crop.transform(df["crop_type"].fillna("background"))
    y_pheno  = le_pheno.transform(df["phenophase_name"].fillna("Dormancy"))
    return y_crop, y_pheno, le_crop, le_pheno


# ── XGBOOST ────────────────────────────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y_crop: np.ndarray, y_pheno: np.ndarray,
                  output_dir: str):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
        print("[warn] xgboost not installed, using sklearn GBM (slower)")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[xgb] Training crop classifier on {X.shape}...")

    params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        eval_metric="mlogloss", n_jobs=-1, random_state=42,
    )

    from sklearn.preprocessing import LabelEncoder as _LE
    xgb_re_crop  = _LE().fit(y_crop)
    xgb_re_pheno = _LE().fit(y_pheno)
    y_crop_fit   = xgb_re_crop.transform(y_crop)
    y_pheno_fit  = xgb_re_pheno.transform(y_pheno)
    joblib.dump({"crop": xgb_re_crop, "pheno": xgb_re_pheno},
                os.path.join(output_dir, "xgb_remapper.pkl"))

    # Per-sample class weights for balanced training
    def _sample_weights(y_encoded):
        classes   = np.unique(y_encoded)
        cw        = compute_class_weight("balanced", classes=classes, y=y_encoded)
        cw_map    = dict(zip(classes, cw))
        return np.array([cw_map[c] for c in y_encoded], dtype=np.float32)

    sw_crop  = _sample_weights(y_crop_fit)
    sw_pheno = _sample_weights(y_pheno_fit)

    crop_path  = os.path.join(output_dir, "xgb_crop.pkl")
    pheno_path = os.path.join(output_dir, "xgb_pheno.pkl")

    model_crop = XGBClassifier(**params)
    prev_booster_crop = joblib.load(crop_path).get_booster() if os.path.exists(crop_path) else None
    if prev_booster_crop:
        print("[xgb] Warm-starting crop model from previous booster")
    model_crop.fit(X, y_crop_fit, sample_weight=sw_crop, xgb_model=prev_booster_crop)
    joblib.dump(model_crop, crop_path)
    print("[xgb] Crop model saved")

    print("[xgb] Training phenology classifier...")
    model_pheno = XGBClassifier(**params)
    prev_booster_pheno = joblib.load(pheno_path).get_booster() if os.path.exists(pheno_path) else None
    if prev_booster_pheno:
        print("[xgb] Warm-starting pheno model from previous booster")
    model_pheno.fit(X, y_pheno_fit, sample_weight=sw_pheno, xgb_model=prev_booster_pheno)
    joblib.dump(model_pheno, pheno_path)
    print("[xgb] Phenology model saved")

    train_acc_crop  = (model_crop.predict(X)  == y_crop_fit).mean()
    train_acc_pheno = (model_pheno.predict(X) == y_pheno_fit).mean()
    print(f"[xgb] Train acc — crop={train_acc_crop:.3f}, pheno={train_acc_pheno:.3f}")

    append_log(output_dir, {
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode"            : "xgboost",
        "n_samples"       : len(X),
        "train_acc_crop"  : round(float(train_acc_crop),  4),
        "train_acc_pheno" : round(float(train_acc_pheno), 4),
        "val_acc_crop"    : "",
        "val_acc_pheno"   : "",
        "best_loss"       : "",
    })


# ── TEMPORAL RANDOM MASKING ────────────────────────────────────────────────────

def temporal_random_mask(x: torch.Tensor, mask_rate: float) -> torch.Tensor:
    """
    Randomly zero out timesteps during training to simulate cloud gaps (Paper 4).
    x: (B, T, D) — masks whole timestep vectors, not individual features.
    """
    if mask_rate <= 0.0:
        return x
    B, T, D = x.shape
    # Draw per-timestep mask: True = keep, False = zero
    keep = torch.rand(B, T, device=x.device) >= mask_rate
    return x * keep.unsqueeze(-1).float()


# ── LSTM DATASET ───────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    def __init__(self, seqs: list, lengths: np.ndarray, doys: np.ndarray,
                 y_crop: np.ndarray, y_pheno: np.ndarray, max_len: int):
        self.seqs    = seqs
        self.lengths = lengths
        self.doys    = doys
        self.y_crop  = y_crop
        self.y_pheno = y_pheno
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        L   = min(self.lengths[idx], self.max_len)
        padded = np.zeros((self.max_len, SEQ_DIM), dtype=np.float32)
        padded[:L] = seq[:L]
        return (
            torch.tensor(padded),
            torch.tensor(L, dtype=torch.long),
            torch.tensor([[self.doys[idx]]], dtype=torch.float32),
            torch.tensor(self.y_crop[idx],  dtype=torch.long),
            torch.tensor(self.y_pheno[idx], dtype=torch.long),
        )


# ── LSTM TRAINING ──────────────────────────────────────────────────────────────

def train_lstm(features: dict, y_crop: np.ndarray, y_pheno: np.ndarray,
               output_dir: str, epochs: int, batch_size: int,
               hidden_dim: int, lr: float, weight_crop: float):

    os.makedirs(output_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm] Device: {device}")

    n = len(features["seqs"])

    # Stratified train/val split (80/20). Fall back to no split if too few samples.
    if n >= 10:
        idx_all  = np.arange(n)
        idx_tr, idx_val = train_test_split(
            idx_all, test_size=0.2, random_state=42, stratify=y_crop
        )
    else:
        print("[lstm] Too few samples for val split — using all data for both")
        idx_tr  = np.arange(n)
        idx_val = np.arange(n)

    def _subset(arr_or_list, idxs):
        if isinstance(arr_or_list, list):
            return [arr_or_list[i] for i in idxs]
        return arr_or_list[idxs]

    max_len = int(features["lengths"].max())

    tr_seqs = _subset(features["seqs"],    idx_tr)
    tr_lens = _subset(features["lengths"], idx_tr)
    tr_doys = _subset(features["doys"],    idx_tr)
    tr_yc   = y_crop[idx_tr]
    tr_yp   = y_pheno[idx_tr]

    val_seqs = _subset(features["seqs"],    idx_val)
    val_lens = _subset(features["lengths"], idx_val)
    val_doys = _subset(features["doys"],    idx_val)
    val_yc   = y_crop[idx_val]
    val_yp   = y_pheno[idx_val]

    tr_dataset  = CropDataset(tr_seqs,  tr_lens,  tr_doys,  tr_yc, tr_yp,  max_len)
    val_dataset = CropDataset(val_seqs, val_lens, val_doys, val_yc, val_yp, max_len)

    # Class-balanced weighted sampler for training
    crop_classes_present = np.unique(tr_yc)
    cw = compute_class_weight("balanced", classes=crop_classes_present, y=tr_yc)
    cw_map     = dict(zip(crop_classes_present, cw))
    sample_wts = torch.tensor([cw_map[c] for c in tr_yc], dtype=torch.float)
    sampler    = WeightedRandomSampler(sample_wts, num_samples=len(sample_wts), replacement=True)

    tr_loader  = DataLoader(tr_dataset,  batch_size=batch_size, sampler=sampler,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = CropPhenologyLSTM(hidden_dim=hidden_dim).to(device)
    lstm_best_path = os.path.join(output_dir, "lstm_best.pt")
    if os.path.exists(lstm_best_path):
        model.load_state_dict(torch.load(lstm_best_path, map_location=device))
        print("[lstm] Warm-starting from previous best model")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    weight_pheno = 1.0 - weight_crop

    # Class-weighted cross-entropy losses
    def _ce_weights(y_arr, n_classes, label_names):
        present = np.unique(y_arr)
        cw_vals = compute_class_weight("balanced", classes=present, y=y_arr)
        w = torch.ones(n_classes)
        for cls, wv in zip(present, cw_vals):
            w[cls] = wv
        return w.to(device)

    ce_crop  = nn.CrossEntropyLoss(weight=_ce_weights(tr_yc,  len(CROP_CLASSES),  CROP_CLASSES))
    ce_pheno = nn.CrossEntropyLoss(weight=_ce_weights(tr_yp,  len(PHENO_CLASSES), PHENO_CLASSES))

    # TRM curriculum: ramp mask rate from 0 → 0.20 over first 10 epochs
    def _trm_rate(epoch):
        ramp_epochs = min(10, epochs)
        return min(0.20, 0.20 * epoch / ramp_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15  # early stopping

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_c  = correct_p = 0
        n_total    = 0
        mask_rate  = _trm_rate(epoch)

        for x, lengths, doy, yc, yp in tr_loader:
            x, lengths = x.to(device), lengths.to(device)
            doy        = doy.squeeze(1).to(device)
            yc, yp     = yc.to(device), yp.to(device)

            # Temporal Random Masking — simulates cloud gaps (Paper 4)
            x = temporal_random_mask(x, mask_rate)

            opt.zero_grad()
            logits_c, logits_p = model(x, lengths, doy)
            loss = weight_crop  * ce_crop(logits_c, yc) + \
                   weight_pheno * ce_pheno(logits_p, yp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(yc)
            correct_c  += (logits_c.argmax(1) == yc).sum().item()
            correct_p  += (logits_p.argmax(1) == yp).sum().item()
            n_total    += len(yc)

        sched.step()
        avg_train_loss = total_loss / max(n_total, 1)

        # Validation pass
        model.eval()
        val_loss = val_c = val_p = val_n = 0
        with torch.no_grad():
            for x, lengths, doy, yc, yp in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                doy        = doy.squeeze(1).to(device)
                yc, yp     = yc.to(device), yp.to(device)
                lc, lp     = model(x, lengths, doy)
                vl = weight_crop  * ce_crop(lc, yc) + \
                     weight_pheno * ce_pheno(lp, yp)
                val_loss += vl.item() * len(yc)
                val_c    += (lc.argmax(1) == yc).sum().item()
                val_p    += (lp.argmax(1) == yp).sum().item()
                val_n    += len(yc)

        avg_val_loss  = val_loss / max(val_n, 1)
        val_acc_crop  = val_c / max(val_n, 1)
        val_acc_pheno = val_p / max(val_n, 1)

        print(f"[lstm] Epoch {epoch:03d}/{epochs} | "
              f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
              f"crop_acc={correct_c/max(n_total,1):.3f} | "
              f"val_crop={val_acc_crop:.3f} | val_pheno={val_acc_pheno:.3f} | "
              f"trm={mask_rate:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), lstm_best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[lstm] Early stopping at epoch {epoch} (patience={patience})")
                break

    torch.save(model.state_dict(), os.path.join(output_dir, "lstm_final.pt"))
    joblib.dump({"hidden_dim": hidden_dim, "max_len": max_len},
                os.path.join(output_dir, "lstm_config.pkl"))
    print(f"[lstm] Best val loss={best_val_loss:.4f} | Models saved to {output_dir}")

    append_log(output_dir, {
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode"            : "lstm",
        "n_samples"       : n,
        "train_acc_crop"  : "",
        "train_acc_pheno" : "",
        "val_acc_crop"    : round(val_acc_crop,  4),
        "val_acc_pheno"   : round(val_acc_pheno, 4),
        "best_loss"       : round(best_val_loss, 4),
    })


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       required=True, choices=["xgboost", "lstm"])
    parser.add_argument("--label_csv",  required=True)
    parser.add_argument("--tiff_dirs",  required=True, nargs="+")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_crop", type=float, default=0.4)
    args = parser.parse_args()

    df                        = load_labels(args.label_csv)
    tiff_index, region_bounds = build_tiff_index(args.tiff_dirs)
    features                  = extract_all_features(df, tiff_index, region_bounds)
    y_crop, y_pheno, le_crop, le_pheno = encode_labels(df)

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump({"crop": le_crop, "pheno": le_pheno},
                os.path.join(args.output_dir, "label_encoders.pkl"))

    if args.mode == "xgboost":
        train_xgboost(features["flat"], y_crop, y_pheno, args.output_dir)

    elif args.mode == "lstm":
        train_lstm(
            features, y_crop, y_pheno, args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            hidden_dim=args.hidden_dim, lr=args.lr,
            weight_crop=args.weight_crop,
        )


if __name__ == "__main__":
    main()
