"""
Extract training data zips and validate structure.

Run once before training:
    python setup_data.py

Extracts:
    DATA/track1_download_link_2.zip  →  DATA/region_train_4/
    DATA/track1_download_link_3.zip  →  DATA/region_train_3/
    DATA/track1_download_link_5.zip  →  DATA/region_train_1/ + DATA/points_train_label.csv
"""

import os
import sys
import zipfile
import shutil
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "DATA")

ZIPS = {
    "track1_download_link_2.zip": "region_train_4",
    "track1_download_link_3.zip": "region_train_3",
    "track1_download_link_5.zip": "region_train_1",
}

LABEL_CSV_INSIDE = "points_train_label.csv"


def extract_zip(zip_path: str, dest_dir: str):
    print(f"\n[extract] {os.path.basename(zip_path)} -> {dest_dir}")
    z = zipfile.ZipFile(zip_path)
    members = z.namelist()
    total = len(members)
    print(f"[extract] {total} files  ({os.path.getsize(zip_path)/1e9:.1f} GB compressed)")
    t0 = time.time()
    for i, member in enumerate(members, 1):
        z.extract(member, dest_dir)
        if i % 200 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", end="\r")
    print(f"\n[extract] Done in {time.time()-t0:.0f}s")
    z.close()


def main():
    # Disk space check
    _, _, free = shutil.disk_usage(DATA_DIR)
    free_gb = free / 1e9
    needed_gb = 60  # ~20GB each × 3, uncompressed TIFFs are similar size
    print(f"[disk] Free: {free_gb:.0f} GB  |  Estimated needed: ~{needed_gb} GB")
    if free_gb < needed_gb:
        print(f"[WARN] Low disk space. Proceed anyway? (y/n): ", end="")
        if input().strip().lower() != "y":
            sys.exit(1)

    extracted_dirs = []
    label_csv_path = None

    for zip_name, expected_dir in ZIPS.items():
        zip_path   = os.path.join(DATA_DIR, zip_name)
        target_dir = os.path.join(DATA_DIR, expected_dir)

        if not os.path.exists(zip_path):
            print(f"[skip] {zip_name} not found — skipping (download link 4 missing is OK)")
            continue

        if os.path.isdir(target_dir):
            count = len([f for f in os.listdir(target_dir) if f.endswith((".tiff", ".tif"))])
            print(f"[skip] {expected_dir}/ already exists ({count} TIFFs) — skipping")
        else:
            extract_zip(zip_path, DATA_DIR)

        extracted_dirs.append(target_dir)

        # Pull out label CSV if present in this zip
        if label_csv_path is None:
            z = zipfile.ZipFile(zip_path)
            if LABEL_CSV_INSIDE in z.namelist():
                out_csv = os.path.join(DATA_DIR, LABEL_CSV_INSIDE)
                if not os.path.exists(out_csv):
                    print(f"[csv] Extracting {LABEL_CSV_INSIDE} ...")
                    z.extract(LABEL_CSV_INSIDE, DATA_DIR)
                label_csv_path = out_csv
                print(f"[csv] Label CSV at {out_csv}")
            z.close()

    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)

    if label_csv_path:
        import csv
        with open(label_csv_path) as f:
            rows = list(csv.DictReader(f))
        from collections import Counter
        crops = Counter(r["crop_type"] for r in rows)
        print(f"\nLabel CSV   : {label_csv_path}")
        print(f"Total rows  : {len(rows)}")
        print(f"Crop dist   : {dict(crops)}")
    else:
        print("[warn] Label CSV not found — check zip contents")

    print(f"\nExtracted dirs:")
    for d in extracted_dirs:
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith((".tiff", ".tif"))])
            print(f"  {d}  ({n} TIFFs)")

    # Print ready-to-run training commands
    tiff_dirs_arg = " ".join(f'DATA\\{os.path.basename(d)}' for d in extracted_dirs if os.path.isdir(d))
    lbl = f"DATA\\{LABEL_CSV_INSIDE}" if label_csv_path else "<label_csv_path>"

    print("\n" + "="*60)
    print("TRAINING COMMANDS")
    print("="*60)
    print(f"\n# XGBoost (fast, no GPU):")
    print(f"python train.py --mode xgboost `")
    print(f"    --label_csv {lbl} `")
    print(f"    --tiff_dirs {tiff_dirs_arg} `")
    print(f"    --output_dir models")
    print(f"\n# BiLSTM (GPU recommended):")
    print(f"python train.py --mode lstm `")
    print(f"    --label_csv {lbl} `")
    print(f"    --tiff_dirs {tiff_dirs_arg} `")
    print(f"    --output_dir models `")
    print(f"    --epochs 80 --batch_size 64 --hidden_dim 128")


if __name__ == "__main__":
    main()
