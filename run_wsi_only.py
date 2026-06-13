# ============================================================
# WSI-Only Unimodal Baseline
# ============================================================
#
# Purpose:
#   Project WSI embeddings (512-dim) down to 256-dim using PCA
#   fitted only on the training split, then save as fused_train /
#   fused_val / fused_test so run_lazypredict_on_fusion.py can
#   consume this output without any changes.
#
# This is a non-trainable, label-free operation.
# No model checkpoint is saved.
#
# Example:
#   python run_wsi_only.py ^
#     --wsi_dir "D:\embeddings\wsi" ^
#     --split_dir "D:\splits\wsi_clinical_split" ^
#     --output_dir "D:\embeddings\unimodal_wsi"
#
# Arguments:
#   --wsi_dir      Folder containing WSI .npy embeddings. Shape: (512,)
#   --split_dir    Folder containing train/val/test_patients.csv
#   --output_dir   Destination folder.
#   --wsi_dim      Input WSI embedding dimension. Default: 512
#   --out_dim      Output projected dimension. Default: 256
#   --patient_col  Default: patient_id
#   --label_col    Default: label
#   --overwrite    Overwrite existing .npy files.
#
# Output:
#   output_dir/
#       train_patients.csv
#       val_patients.csv
#       test_patients.csv
#       fused_train/
#           patient001.npy   # shape: (256,)
#       fused_val/
#           patient101.npy
#       fused_test/
#           patient201.npy
#       summary.txt
#
# ============================================================

import argparse
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# Helpers
# ============================================================

def load_split(split_csv, emb_dir, wsi_dim, patient_col, label_col):
    df = pd.read_csv(split_csv)
    X, y, ids = [], [], []
    for _, row in df.iterrows():
        pid = str(row[patient_col])
        path = emb_dir / f"{pid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing WSI embedding: {path}")
        emb = np.load(path).astype(np.float32).reshape(-1)
        if emb.shape != (wsi_dim,):
            raise ValueError(f"{pid}: shape {emb.shape}, expected ({wsi_dim},)")
        X.append(emb)
        y.append(int(row[label_col]))
        ids.append(pid)
    return np.stack(X), np.array(y, dtype=np.int64), ids


def save_split(projected, ids, split_name, output_dir, overwrite):
    out = output_dir / f"fused_{split_name}"
    out.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0
    for pid, vec in zip(ids, projected):
        p = out / f"{pid}.npy"
        if p.exists() and not overwrite:
            skipped += 1
            continue
        np.save(p, vec.astype(np.float32))
        saved += 1
    print(f"{split_name}: saved={saved}, skipped={skipped}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir",     required=True, type=str)
    parser.add_argument("--split_dir",   required=True, type=str)
    parser.add_argument("--output_dir",  required=True, type=str)
    parser.add_argument("--wsi_dim",     default=512,   type=int)
    parser.add_argument("--out_dim",     default=256,   type=int)
    parser.add_argument("--patient_col", default="patient_id", type=str)
    parser.add_argument("--label_col",   default="label",      type=str)
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    wsi_dir    = Path(args.wsi_dir)
    split_dir  = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train_patients.csv"
    val_csv   = split_dir / "val_patients.csv"
    test_csv  = split_dir / "test_patients.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    # Copy split CSVs into output dir so LazyPredict can find them.
    shutil.copy2(train_csv, output_dir / "train_patients.csv")
    shutil.copy2(val_csv,   output_dir / "val_patients.csv")
    shutil.copy2(test_csv,  output_dir / "test_patients.csv")

    print("Loading embeddings...")
    X_train, y_train, train_ids = load_split(train_csv, wsi_dir, args.wsi_dim, args.patient_col, args.label_col)
    X_val,   y_val,   val_ids   = load_split(val_csv,   wsi_dir, args.wsi_dim, args.patient_col, args.label_col)
    X_test,  y_test,  test_ids  = load_split(test_csv,  wsi_dir, args.wsi_dim, args.patient_col, args.label_col)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # PCA fitted ONLY on training data.
    # Applying the same projection to val/test is label-safe.
    actual_out_dim = min(args.out_dim, X_train.shape[0], X_train.shape[1])
    if actual_out_dim != args.out_dim:
        print(f"[WARN] Requested out_dim={args.out_dim} capped to {actual_out_dim} (n_train={X_train.shape[0]})")

    print(f"Fitting PCA ({args.wsi_dim} -> {actual_out_dim}) on training data only...")
    pca = PCA(n_components=actual_out_dim, random_state=42)
    pca.fit(X_train)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio: {explained:.4f}")

    X_train_proj = pca.transform(X_train).astype(np.float32)
    X_val_proj   = pca.transform(X_val).astype(np.float32)
    X_test_proj  = pca.transform(X_test).astype(np.float32)

    save_split(X_train_proj, train_ids, "train", output_dir, args.overwrite)
    save_split(X_val_proj,   val_ids,   "val",   output_dir, args.overwrite)
    save_split(X_test_proj,  test_ids,  "test",  output_dir, args.overwrite)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("WSI-Only Unimodal Baseline\n")
        f.write("==========================\n\n")
        f.write(f"WSI dir:    {wsi_dir}\n")
        f.write(f"Split dir:  {split_dir}\n")
        f.write(f"Output dir: {output_dir}\n\n")
        f.write(f"Input dim:  {args.wsi_dim}\n")
        f.write(f"Output dim: {actual_out_dim}\n")
        f.write(f"PCA explained variance: {explained:.4f}\n\n")
        f.write(f"Train: {X_train.shape[0]} patients\n")
        f.write(f"Val:   {X_val.shape[0]} patients\n")
        f.write(f"Test:  {X_test.shape[0]} patients\n")

    print("Finished. Output dim:", actual_out_dim)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
