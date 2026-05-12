import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
import torch

from features import build_tiff_index, extract_all_features, SEQ_DIM
from model import CropPhenologyLSTM, CROP_CLASSES, PHENO_CLASSES


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_test_points(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("Longitude", "Latitude", "phenophase_date"):
        assert col in df.columns, f"Missing column: {col}"
    # Drop rows missing required geo/date fields only
    df = df.dropna(subset=["Longitude", "Latitude", "phenophase_date"])
    print(f"[data] {len(df)} rows from {csv_path}")
    return df


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

def load_models(model_dir: str) -> dict:
    """
    Auto-detects which model(s) are available in model_dir.
    Priority: lstm_best.pt > xgb_crop.pkl
    """
    models = {}
    enc_path = os.path.join(model_dir, "label_encoders.pkl")
    if os.path.exists(enc_path):
        models["encoders"] = joblib.load(enc_path)
        print("[model] Label encoders loaded")

    lstm_path = os.path.join(model_dir, "lstm_best.pt")
    cfg_path  = os.path.join(model_dir, "lstm_config.pkl")
    if os.path.exists(lstm_path) and os.path.exists(cfg_path):
        cfg    = joblib.load(cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net    = CropPhenologyLSTM(hidden_dim=cfg["hidden_dim"]).to(device)
        net.load_state_dict(torch.load(lstm_path, map_location=device))
        net.eval()
        models["lstm"]     = net
        models["lstm_cfg"] = cfg
        models["device"]   = device
        print(f"[model] LSTM loaded ({device}) | max_len={cfg['max_len']}")
        return models

    xgb_crop  = os.path.join(model_dir, "xgb_crop.pkl")
    xgb_pheno = os.path.join(model_dir, "xgb_pheno.pkl")
    if os.path.exists(xgb_crop) and os.path.exists(xgb_pheno):
        models["xgb_crop"]  = joblib.load(xgb_crop)
        models["xgb_pheno"] = joblib.load(xgb_pheno)
        remap_path = os.path.join(model_dir, "xgb_remapper.pkl")
        if os.path.exists(remap_path):
            models["xgb_remapper"] = joblib.load(remap_path)
        print("[model] XGBoost models loaded")
        return models

    print(f"[warn] No trained models found in {model_dir} — using placeholder predictions")
    return models


# ── INFERENCE ──────────────────────────────────────────────────────────────────

def run_inference(features: dict, models: dict) -> tuple:
    n = len(features["flat"])

    le_crop  = models.get("encoders", {}).get("crop")  if models.get("encoders") else None
    le_pheno = models.get("encoders", {}).get("pheno") if models.get("encoders") else None

    # ── LSTM ──
    if "lstm" in models:
        device  = models["device"]
        net     = models["lstm"]
        max_len = models["lstm_cfg"]["max_len"]
        seqs    = features["seqs"]
        lengths = features["lengths"]
        doys    = features["doys"]

        all_c, all_p = [], []
        batch = 256
        for start in range(0, n, batch):
            end    = min(start + batch, n)
            sl     = slice(start, end)
            B      = end - start
            padded = np.zeros((B, max_len, SEQ_DIM), dtype=np.float32)
            lens   = np.zeros(B, dtype=np.int64)
            for i, seq in enumerate(seqs[sl]):
                L = min(len(seq), max_len)
                padded[i, :L] = seq[:L]
                lens[i]       = max(L, 1)

            x_t = torch.tensor(padded).to(device)
            l_t = torch.tensor(lens).to(device)
            d_t = torch.tensor(doys[sl].reshape(-1, 1), dtype=torch.float32).to(device)

            with torch.no_grad():
                lc, lp = net(x_t, l_t, d_t)
            all_c.extend(lc.argmax(1).cpu().tolist())
            all_p.extend(lp.argmax(1).cpu().tolist())

        crop_preds  = [le_crop.inverse_transform([c])[0]  if le_crop  else CROP_CLASSES[c]  for c in all_c]
        pheno_preds = [le_pheno.inverse_transform([p])[0] if le_pheno else PHENO_CLASSES[p] for p in all_p]
        return crop_preds, pheno_preds

    # ── XGBoost ──
    if "xgb_crop" in models:
        X        = features["flat"]
        c_idx    = models["xgb_crop"].predict(X)
        p_idx    = models["xgb_pheno"].predict(X)
        remapper = models.get("xgb_remapper")
        if remapper and le_crop and le_pheno:
            c_orig = remapper["crop"].inverse_transform(c_idx)
            p_orig = remapper["pheno"].inverse_transform(p_idx)
            crop_preds  = le_crop.inverse_transform(c_orig).tolist()
            pheno_preds = le_pheno.inverse_transform(p_orig).tolist()
        else:
            crop_preds  = [le_crop.inverse_transform([c])[0]  if le_crop  else CROP_CLASSES[c]  for c in c_idx]
            pheno_preds = [le_pheno.inverse_transform([p])[0] if le_pheno else PHENO_CLASSES[p] for p in p_idx]
        return crop_preds, pheno_preds

    # ── Placeholder ──
    return ["rice"] * n, ["Greenup"] * n


# ── OUTPUT ─────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, crop_preds: list, pheno_preds: list,
                 oob_mask: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for i, (_, row) in enumerate(df.iterrows()):
        key = f"{row['Longitude']}_{row['Latitude']}_{str(row['phenophase_date']).strip()}"
        if oob_mask[i]:
            results[key] = ["background", "Dormancy"]
        else:
            results[key] = [crop_preds[i], pheno_preds[i]]

    out_path = os.path.join(output_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[output] {len(results)} predictions -> {out_path}")


# ── OPTIONAL SCORING ───────────────────────────────────────────────────────────

def score_against_labels(result_json_path: str, label_csv_path: str):
    from sklearn.metrics import f1_score

    with open(result_json_path) as f:
        preds = json.load(f)

    df_label = pd.read_csv(label_csv_path).dropna(
        subset=["Pre_crop_type", "Pre_phenophase"]
    )

    y_true_crop, y_pred_crop   = [], []
    y_true_pheno, y_pred_pheno = [], []

    for _, row in df_label.iterrows():
        key = f"{row['Longitude']}_{row['Latitude']}_{str(row['phenophase_date']).strip()}"
        if key not in preds:
            continue
        pred_crop, pred_pheno = preds[key]
        true_crop, true_pheno = row["Pre_crop_type"], row["Pre_phenophase"]

        if true_crop in ("rice", "corn", "soybean"):
            y_true_crop.append(true_crop)
            y_pred_crop.append(pred_crop)

        if true_crop == "rice":
            y_true_pheno.append(f"rice_{true_pheno}")
            y_pred_pheno.append(
                f"rice_{pred_pheno}" if pred_crop == "rice" else "wrong_crop"
            )

    f1_crop = f1_pheno = 0.0
    if y_true_crop:
        f1_crop = f1_score(y_true_crop, y_pred_crop,
                           labels=["rice", "corn", "soybean"],
                           average="macro", zero_division=0)
        print(f"[score] Crop MacroF1={f1_crop:.4f}  (n={len(y_true_crop)})")
    if y_true_pheno:
        f1_pheno = f1_score(y_true_pheno, y_pred_pheno,
                            average="macro", zero_division=0)
        print(f"[score] Rice Phenology MacroF1={f1_pheno:.4f}  (n={len(y_true_pheno)})")

    print(f"[score] AlgoScore={(0.4*f1_crop + 0.6*f1_pheno)*100:.2f}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--tiff_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir",  default="models")
    parser.add_argument("--label_csv",  default=None)
    args = parser.parse_args()

    df                        = load_test_points(args.input_csv)
    tiff_index, region_bounds = build_tiff_index(args.tiff_dir)
    features                  = extract_all_features(df, tiff_index, region_bounds)
    models                    = load_models(args.model_dir)
    crop_preds, pheno_preds   = run_inference(features, models)

    save_results(df, crop_preds, pheno_preds, features["oob"], args.output_dir)

    if args.label_csv:
        score_against_labels(
            os.path.join(args.output_dir, "result.json"),
            args.label_csv,
        )


if __name__ == "__main__":
    main()
