"""
Training script — XGBoost baseline and BiLSTM model.

Usage:
  # XGBoost (no GPU needed)
  python train.py --mode xgboost \
      --label_csv /path/to/points_train_label.csv \
      --tiff_dirs /path/to/region_train_1 /path/to/region_train_2 \
                  /path/to/region_train_3 /path/to/region_train_4 \
      --output_dir models/

  # LSTM (GPU recommended)
  python train.py --mode lstm \
      --label_csv /path/to/points_train_label.csv \
      --tiff_dirs /path/to/region_train_1 /path/to/region_train_2 \
                  /path/to/region_train_3 /path/to/region_train_4 \
      --output_dir models/ \
      --epochs 50 --batch_size 64 --hidden_dim 128
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from features import build_tiff_index, extract_all_features
from model import CropPhenologyLSTM, CROP_CLASSES, PHENO_CLASSES


# ── LABEL LOADING ──────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names: accept Pre_crop_type / Pre_phenophase variants
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
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        n_jobs=-1, random_state=42,
    )

    model_crop = XGBClassifier(**params)
    model_crop.fit(X, y_crop)
    joblib.dump(model_crop, os.path.join(output_dir, "xgb_crop.pkl"))
    print(f"[xgb] Crop model saved")

    print(f"[xgb] Training phenology classifier...")
    model_pheno = XGBClassifier(**params)
    model_pheno.fit(X, y_pheno)
    joblib.dump(model_pheno, os.path.join(output_dir, "xgb_pheno.pkl"))
    print(f"[xgb] Phenology model saved")

    # Quick train accuracy
    train_acc_crop  = (model_crop.predict(X)  == y_crop).mean()
    train_acc_pheno = (model_pheno.predict(X) == y_pheno).mean()
    print(f"[xgb] Train acc — crop={train_acc_crop:.3f}, pheno={train_acc_pheno:.3f}")


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
        padded = np.zeros((self.max_len, 16), dtype=np.float32)
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

    max_len = int(features["lengths"].max())
    dataset = CropDataset(
        features["seqs"], features["lengths"], features["doys"],
        y_crop, y_pheno, max_len
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    model = CropPhenologyLSTM(hidden_dim=hidden_dim).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Weight loss: 60% phenology metric weight → upweight phenology head
    weight_pheno = 1.0 - weight_crop
    ce_crop  = nn.CrossEntropyLoss()
    ce_pheno = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_c  = correct_p = 0
        n_total    = 0

        for x, lengths, doy, yc, yp in loader:
            x, lengths = x.to(device), lengths.to(device)
            doy        = doy.squeeze(1).to(device)
            yc, yp     = yc.to(device), yp.to(device)

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
        avg_loss = total_loss / n_total
        print(f"[lstm] Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} | "
              f"crop_acc={correct_c/n_total:.3f} | pheno_acc={correct_p/n_total:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "lstm_best.pt"))

    # Save final + config
    torch.save(model.state_dict(), os.path.join(output_dir, "lstm_final.pt"))
    joblib.dump({"hidden_dim": hidden_dim, "max_len": max_len},
                os.path.join(output_dir, "lstm_config.pkl"))
    print(f"[lstm] Best loss={best_loss:.4f} | Models saved to {output_dir}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       required=True, choices=["xgboost", "lstm"])
    parser.add_argument("--label_csv",  required=True)
    parser.add_argument("--tiff_dirs",  required=True, nargs="+",
                        help="One or more dirs containing training TIFF files")
    parser.add_argument("--output_dir", default="models")
    # LSTM args
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_crop", type=float, default=0.4,
                        help="Loss weight for crop head (0.4 matches scoring split)")
    args = parser.parse_args()

    df                      = load_labels(args.label_csv)
    tiff_index, region_bounds = build_tiff_index(args.tiff_dirs)
    features                = extract_all_features(df, tiff_index, region_bounds)
    y_crop, y_pheno, le_crop, le_pheno = encode_labels(df)

    # Save label encoders (inference needs them)
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
