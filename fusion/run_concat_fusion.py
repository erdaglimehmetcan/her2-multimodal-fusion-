# ============================================================
# Concatenation Fusion: WSI + Clinical
# ============================================================
#
# Purpose:
#   Create fused embeddings by directly concatenating WSI and clinical
#   .npy embeddings using a predefined train/val/test split.
#
# Important:
#   Concatenation is non-trainable.
#   Therefore, there is no model training, no epochs, and no checkpoint.
#
# Workflow:
#   1. Load fixed train/val/test CSV files from split_dir
#   2. For each patient:
#        WSI embedding + Clinical embedding
#        -> Concatenated fused embedding
#   3. Save fused .npy embeddings separately for train/val/test
#
# Example:
#   python concat_fusion.py ^
#     --wsi_dir "D:\embeddings\wsi" ^
#     --clinical_dir "D:\embeddings\clinical" ^
#     --split_dir "D:\splits\wsi_clinical_split" ^
#     --output_dir "D:\embeddings\fused_concat"
#
# Arguments:
#   --wsi_dir        Folder containing WSI .npy embeddings.
#   --clinical_dir   Folder containing clinical .npy embeddings.
#   --split_dir      Folder containing train_patients.csv, val_patients.csv, test_patients.csv.
#   --output_dir     Folder where concatenated fused embeddings will be saved.
#   --wsi_dim        WSI embedding dimension. Default: 512
#   --clinical_dim   Clinical embedding dimension. Default: 64
#   --patient_col    Patient ID column name in split CSVs. Default: patient_id
#   --label_col      Label column name in split CSVs. Default: label
#   --dtype          Output dtype. Default: float32
#   --overwrite      If used, overwrite existing fused .npy files.
#   --save_matrices  If used, also save X_train.npy, y_train.npy, etc.
#
# Output:
#   output_dir/
#       train_patients.csv
#       val_patients.csv
#       test_patients.csv
#       fusion_summary.txt
#       failed_cases.txt
#       fused_train/
#           patient001.npy
#       fused_val/
#           patient101.npy
#       fused_test/
#           patient201.npy
#
# If --save_matrices is used:
#       X_train.npy
#       y_train.npy
#       X_val.npy
#       y_val.npy
#       X_test.npy
#       y_test.npy
#
# ============================================================

import argparse
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd


# ============================================================
# Utility functions
# ============================================================

def load_embedding(path, expected_dim, patient_id, modality_name, flatten=True):
    """
    Loads one .npy embedding and checks its dimension.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing {modality_name} embedding for {patient_id}: {path}")

    embedding = np.load(path)

    if flatten:
        embedding = embedding.reshape(-1)

    embedding = embedding.astype(np.float32)

    if embedding.shape != (expected_dim,):
        raise ValueError(
            f"{patient_id}: {modality_name} shape {embedding.shape}, "
            f"expected ({expected_dim},)"
        )

    return embedding


def export_concat_split(
    split_name,
    split_csv,
    wsi_dir,
    clinical_dir,
    output_dir,
    wsi_dim,
    clinical_dim,
    patient_col,
    label_col,
    dtype,
    overwrite,
    save_matrices,
):
    """
    Creates concatenated fused embeddings for one split:
    train, val, or test.
    """

    df = pd.read_csv(split_csv)

    if patient_col not in df.columns:
        raise ValueError(f"{split_csv} does not contain patient column: {patient_col}")

    if save_matrices and label_col not in df.columns:
        raise ValueError(f"{split_csv} does not contain label column: {label_col}")

    fused_dir = output_dir / f"fused_{split_name}"
    fused_dir.mkdir(parents=True, exist_ok=True)

    failed_log = output_dir / "failed_cases.txt"

    X_list = []
    y_list = []
    patient_list = []

    saved_count = 0
    skipped_count = 0
    failed_count = 0

    for _, row in df.iterrows():
        patient_id = str(row[patient_col])

        wsi_path = wsi_dir / f"{patient_id}.npy"
        clinical_path = clinical_dir / f"{patient_id}.npy"
        fused_path = fused_dir / f"{patient_id}.npy"

        try:
            if fused_path.exists() and not overwrite:
                skipped_count += 1
                fused_embedding = np.load(fused_path).reshape(-1)
            else:
                wsi_embedding = load_embedding(
                    path=wsi_path,
                    expected_dim=wsi_dim,
                    patient_id=patient_id,
                    modality_name="WSI",
                )

                clinical_embedding = load_embedding(
                    path=clinical_path,
                    expected_dim=clinical_dim,
                    patient_id=patient_id,
                    modality_name="Clinical",
                )

                # Core concatenation operation:
                # [WSI vector ; Clinical vector]
                fused_embedding = np.concatenate(
                    [wsi_embedding, clinical_embedding],
                    axis=0,
                ).astype(dtype)

                np.save(fused_path, fused_embedding)
                saved_count += 1

            expected_fused_dim = wsi_dim + clinical_dim

            if fused_embedding.shape != (expected_fused_dim,):
                raise ValueError(
                    f"{patient_id}: fused shape {fused_embedding.shape}, "
                    f"expected ({expected_fused_dim},)"
                )

            if save_matrices:
                X_list.append(fused_embedding)
                y_list.append(int(row[label_col]))
                patient_list.append(patient_id)

        except Exception as e:
            failed_count += 1
            print(f"FAILED: {split_name} | {patient_id}")

            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- {split_name} | {patient_id} ---\n")
                f.write(str(e))
                f.write("\n")
                f.write(traceback.format_exc())
                f.write("\n")

    if save_matrices:
        if len(X_list) > 0:
            X = np.stack(X_list).astype(dtype)
            y = np.array(y_list, dtype=np.int64)

            np.save(output_dir / f"X_{split_name}.npy", X)
            np.save(output_dir / f"y_{split_name}.npy", y)

            pd.DataFrame(
                {
                    "patient_id": patient_list,
                    "label": y_list,
                }
            ).to_csv(output_dir / f"{split_name}_matrix_index.csv", index=False)

    print(
        f"{split_name}: "
        f"saved={saved_count}, skipped={skipped_count}, failed={failed_count}"
    )

    return {
        "split": split_name,
        "patients": len(df),
        "saved": saved_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wsi_dir", required=True, type=str)
    parser.add_argument("--clinical_dir", required=True, type=str)
    parser.add_argument("--split_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--wsi_dim", default=512, type=int)
    parser.add_argument("--clinical_dim", default=64, type=int)

    parser.add_argument("--patient_col", default="patient_id", type=str)
    parser.add_argument("--label_col", default="label", type=str)

    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_matrices", action="store_true")

    args = parser.parse_args()

    wsi_dir = Path(args.wsi_dir)
    clinical_dir = Path(args.clinical_dir)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if args.dtype == "float32" else np.float64

    train_csv = split_dir / "train_patients.csv"
    val_csv = split_dir / "val_patients.csv"
    test_csv = split_dir / "test_patients.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    # Save copies of the split files inside this experiment folder.
    shutil.copy2(train_csv, output_dir / "train_patients.csv")
    shutil.copy2(val_csv, output_dir / "val_patients.csv")
    shutil.copy2(test_csv, output_dir / "test_patients.csv")

    fused_dim = args.wsi_dim + args.clinical_dim

    print("Concatenation Fusion")
    print("====================")
    print("WSI dir:", wsi_dir)
    print("Clinical dir:", clinical_dir)
    print("Split dir:", split_dir)
    print("Output dir:", output_dir)
    print("WSI dim:", args.wsi_dim)
    print("Clinical dim:", args.clinical_dim)
    print("Fused dim:", fused_dim)

    summaries = []

    summaries.append(
        export_concat_split(
            split_name="train",
            split_csv=train_csv,
            wsi_dir=wsi_dir,
            clinical_dir=clinical_dir,
            output_dir=output_dir,
            wsi_dim=args.wsi_dim,
            clinical_dim=args.clinical_dim,
            patient_col=args.patient_col,
            label_col=args.label_col,
            dtype=dtype,
            overwrite=args.overwrite,
            save_matrices=args.save_matrices,
        )
    )

    summaries.append(
        export_concat_split(
            split_name="val",
            split_csv=val_csv,
            wsi_dir=wsi_dir,
            clinical_dir=clinical_dir,
            output_dir=output_dir,
            wsi_dim=args.wsi_dim,
            clinical_dim=args.clinical_dim,
            patient_col=args.patient_col,
            label_col=args.label_col,
            dtype=dtype,
            overwrite=args.overwrite,
            save_matrices=args.save_matrices,
        )
    )

    summaries.append(
        export_concat_split(
            split_name="test",
            split_csv=test_csv,
            wsi_dir=wsi_dir,
            clinical_dir=clinical_dir,
            output_dir=output_dir,
            wsi_dim=args.wsi_dim,
            clinical_dim=args.clinical_dim,
            patient_col=args.patient_col,
            label_col=args.label_col,
            dtype=dtype,
            overwrite=args.overwrite,
            save_matrices=args.save_matrices,
        )
    )

    summary_path = output_dir / "fusion_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Concatenation Fusion Summary\n")
        f.write("============================\n\n")

        f.write(f"WSI directory: {wsi_dir}\n")
        f.write(f"Clinical directory: {clinical_dir}\n")
        f.write(f"Split directory: {split_dir}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write(f"WSI dimension: {args.wsi_dim}\n")
        f.write(f"Clinical dimension: {args.clinical_dim}\n")
        f.write(f"Fused dimension: {fused_dim}\n")
        f.write(f"Output dtype: {args.dtype}\n")
        f.write(f"Save matrices: {args.save_matrices}\n\n")

        for item in summaries:
            f.write(
                f"{item['split']}: "
                f"patients={item['patients']}, "
                f"saved={item['saved']}, "
                f"skipped={item['skipped']}, "
                f"failed={item['failed']}\n"
            )

    print("Finished.")
    print("Fused embedding dimension:", fused_dim)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()