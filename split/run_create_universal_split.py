# ============================================================
# Universal Patient Split Script
# ============================================================
#
# Purpose:
#   Create one fixed train/val/test split that can be reused by
#   all attention mechanisms and classifiers.
#
# Why:
#   All models must use the same patient split for fair comparison.
#
# Output:
#   split_dir/
#       train_patients.csv
#       val_patients.csv
#       test_patients.csv
#       matched_patients.csv
#       missing_patients.csv
#       split_summary.txt
#
# Example:
#   python create_universal_split.py ^
#     --labels_csv "D:\embeddings\labels.csv" ^
#     --embedding_dirs "D:\embeddings\wsi" "D:\embeddings\clinical" ^
#     --output_dir "D:\splits\wsi_clinical_split"
#
# Arguments:
#   --labels_csv      CSV file containing patient_id and label columns.
#   --embedding_dirs  One or more embedding folders. Patient must exist in all of them.
#   --output_dir      Folder where train/val/test CSV split files will be saved.
#   --patient_col     Column name for patient IDs. Default: patient_id
#   --label_col       Column name for labels. Default: label
#   --val_size        Validation ratio. Default: 0.15
#   --test_size       Test ratio. Default: 0.15
#   --seed            Random seed for reproducible split. Default: 42
# ============================================================

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels_csv", required=True, type=str)
    parser.add_argument("--embedding_dirs", required=True, nargs="+", type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--patient_col", default="patient_id", type=str)
    parser.add_argument("--label_col", default="label", type=str)

    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--test_size", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    embedding_dirs = [Path(d) for d in args.embedding_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)

    if args.patient_col not in df.columns:
        raise ValueError(f"Missing patient column: {args.patient_col}")

    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    # Keep only required columns.
    df = df[[args.patient_col, args.label_col]].copy()
    df = df.rename(
        columns={
            args.patient_col: "patient_id",
            args.label_col: "label",
        }
    )

    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)

    # Patient IDs must be unique.
    if df["patient_id"].duplicated().any():
        duplicated = df[df["patient_id"].duplicated()]["patient_id"].tolist()
        raise ValueError(f"Duplicated patient IDs found: {duplicated[:10]}")

    matched_rows = []
    missing_rows = []

    # Keep only patients that have embeddings in all required modality folders.
    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        label = row["label"]

        missing_modalities = []

        for emb_dir in embedding_dirs:
            emb_path = emb_dir / f"{patient_id}.npy"

            if not emb_path.exists():
                missing_modalities.append(str(emb_dir))

        if len(missing_modalities) == 0:
            matched_rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                }
            )
        else:
            missing_rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                    "missing_from": "; ".join(missing_modalities),
                }
            )

    matched_df = pd.DataFrame(matched_rows)
    missing_df = pd.DataFrame(missing_rows)

    if len(matched_df) == 0:
        raise RuntimeError("No matched patients found.")

    matched_df.to_csv(output_dir / "matched_patients.csv", index=False)
    missing_df.to_csv(output_dir / "missing_patients.csv", index=False)

    print("Matched patients:", len(matched_df))
    print("Missing patients:", len(missing_df))
    print("Class distribution:")
    print(matched_df["label"].value_counts())

    # --------------------------------------------------------
    # Fixed patient-level split.
    # Stratification preserves HER2 class balance.
    # --------------------------------------------------------

    train_val_df, test_df = train_test_split(
        matched_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=matched_df["label"],
    )

    relative_val_size = args.val_size / (1.0 - args.test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=args.seed,
        stratify=train_val_df["label"],
    )

    train_df.to_csv(output_dir / "train_patients.csv", index=False)
    val_df.to_csv(output_dir / "val_patients.csv", index=False)
    test_df.to_csv(output_dir / "test_patients.csv", index=False)

    summary_path = output_dir / "split_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Universal Patient Split Summary\n")
        f.write("================================\n\n")

        f.write(f"Labels CSV: {labels_csv}\n")
        f.write("Embedding directories:\n")

        for emb_dir in embedding_dirs:
            f.write(f"  - {emb_dir}\n")

        f.write("\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Validation size: {args.val_size}\n")
        f.write(f"Test size: {args.test_size}\n\n")

        f.write(f"Matched patients: {len(matched_df)}\n")
        f.write(f"Missing patients: {len(missing_df)}\n\n")

        f.write(f"Train patients: {len(train_df)}\n")
        f.write(str(train_df["label"].value_counts()))
        f.write("\n\n")

        f.write(f"Validation patients: {len(val_df)}\n")
        f.write(str(val_df["label"].value_counts()))
        f.write("\n\n")

        f.write(f"Test patients: {len(test_df)}\n")
        f.write(str(test_df["label"].value_counts()))
        f.write("\n")

    print("\nSplit created.")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()