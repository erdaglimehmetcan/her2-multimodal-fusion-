# ============================================================
# Clinical-Only Unimodal Baseline
# ============================================================
#
# Purpose:
#   Project clinical embeddings (64-dim) up to 256-dim using a
#   fixed random MLP (same approach as run_clinical_embedding.py),
#   then save as fused_train / fused_val / fused_test so
#   run_lazypredict_on_fusion.py can consume this without changes.
#
# This is non-trainable and label-free.
# The MLP uses fixed seed=42 weights — identical to how clinical
# embeddings were originally generated.
# No model checkpoint is saved.
#
# Example:
#   python run_clinical_only.py ^
#     --clinical_dir "D:\embeddings\clinical" ^
#     --split_dir "D:\splits\wsi_clinical_split" ^
#     --output_dir "D:\embeddings\unimodal_clinical"
#
# Arguments:
#   --clinical_dir  Folder containing clinical .npy embeddings. Shape: (64,)
#   --split_dir     Folder containing train/val/test_patients.csv
#   --output_dir    Destination folder.
#   --clinical_dim  Input clinical embedding dimension. Default: 64
#   --out_dim       Output projected dimension. Default: 256
#   --patient_col   Default: patient_id
#   --label_col     Default: label
#   --overwrite     Overwrite existing .npy files.
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
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# Fixed random MLP projection
# ============================================================

class ClinicalProjectionMLP(nn.Module):
    """
    Expands clinical embedding from clinical_dim -> out_dim using
    a fixed random 3-layer MLP (no training).

    Same design principle as ClinicalMLP in run_clinical_embedding.py:
    random weights, seed=42 for reproducibility.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Helpers
# ============================================================

def load_split(split_csv, emb_dir, clinical_dim, patient_col, label_col):
    df = pd.read_csv(split_csv)
    X, y, ids = [], [], []
    for _, row in df.iterrows():
        pid = str(row[patient_col])
        path = emb_dir / f"{pid}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing clinical embedding: {path}")
        emb = np.load(path).astype(np.float32).reshape(-1)
        if emb.shape != (clinical_dim,):
            raise ValueError(f"{pid}: shape {emb.shape}, expected ({clinical_dim},)")
        X.append(emb)
        y.append(int(row[label_col]))
        ids.append(pid)
    return np.stack(X), np.array(y, dtype=np.int64), ids


def project_and_save(model, X, ids, split_name, output_dir, overwrite):
    out = output_dir / f"fused_{split_name}"
    out.mkdir(parents=True, exist_ok=True)

    x_tensor = torch.from_numpy(X)
    with torch.no_grad():
        projected = model(x_tensor).numpy().astype(np.float32)

    saved, skipped = 0, 0
    for pid, vec in zip(ids, projected):
        p = out / f"{pid}.npy"
        if p.exists() and not overwrite:
            skipped += 1
            continue
        np.save(p, vec)
        saved += 1

    print(f"{split_name}: saved={saved}, skipped={skipped}")
    return projected


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_dir", required=True, type=str)
    parser.add_argument("--split_dir",    required=True, type=str)
    parser.add_argument("--output_dir",   required=True, type=str)
    parser.add_argument("--clinical_dim", default=64,    type=int)
    parser.add_argument("--out_dim",      default=256,   type=int)
    parser.add_argument("--patient_col",  default="patient_id", type=str)
    parser.add_argument("--label_col",    default="label",      type=str)
    parser.add_argument("--overwrite",    action="store_true")
    args = parser.parse_args()

    # Fixed seed — identical to run_clinical_embedding.py
    torch.manual_seed(42)
    np.random.seed(42)

    clinical_dir = Path(args.clinical_dir)
    split_dir    = Path(args.split_dir)
    output_dir   = Path(args.output_dir)
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
    X_train, y_train, train_ids = load_split(train_csv, clinical_dir, args.clinical_dim, args.patient_col, args.label_col)
    X_val,   y_val,   val_ids   = load_split(val_csv,   clinical_dir, args.clinical_dim, args.patient_col, args.label_col)
    X_test,  y_test,  test_ids  = load_split(test_csv,  clinical_dir, args.clinical_dim, args.patient_col, args.label_col)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = ClinicalProjectionMLP(input_dim=args.clinical_dim, output_dim=args.out_dim)
    model.eval()

    print(f"Projecting ({args.clinical_dim} -> {args.out_dim}) with fixed random weights...")
    project_and_save(model, X_train, train_ids, "train", output_dir, args.overwrite)
    project_and_save(model, X_val,   val_ids,   "val",   output_dir, args.overwrite)
    project_and_save(model, X_test,  test_ids,  "test",  output_dir, args.overwrite)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Clinical-Only Unimodal Baseline\n")
        f.write("================================\n\n")
        f.write(f"Clinical dir: {clinical_dir}\n")
        f.write(f"Split dir:    {split_dir}\n")
        f.write(f"Output dir:   {output_dir}\n\n")
        f.write(f"Input dim:  {args.clinical_dim}\n")
        f.write(f"Output dim: {args.out_dim}\n")
        f.write(f"Projection: fixed random MLP, seed=42 (non-trainable)\n\n")
        f.write(f"Train: {X_train.shape[0]} patients\n")
        f.write(f"Val:   {X_val.shape[0]} patients\n")
        f.write(f"Test:  {X_test.shape[0]} patients\n")

    print("Finished. Output dim:", args.out_dim)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
