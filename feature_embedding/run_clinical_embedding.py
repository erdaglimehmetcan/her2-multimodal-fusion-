# ============================================================
# Clinical Embedding Extraction
#
# Purpose:
#   Reads clinical Excel data, preprocesses variables, and saves
#   one clinical embedding .npy file per patient.
#
# Input:
#   Excel file with one row per patient.
#   Required column: patient_id
#   Do not include HER2 label as a feature.
#
# Output:
#   output_dir/
#       patient001.npy  # shape: (embedding_dim,)
#
# Example:
#   python run_clinical_embedding.py ^
#     --excel_path "D:\data\clinical.xlsx" ^
#     --output_dir "D:\CLINICAL_EMBEDDINGS" ^
#     --patient_id_column "patient_id" ^
#     --embedding_dim 64
#
# Arguments:
#   --excel_path         Clinical Excel file path.
#   --output_dir         Folder for saved .npy embeddings.
#   --patient_id_column  Patient ID column name. Default: patient_id
#   --embedding_dim      Output embedding dimension. Default: 64
#
# ============================================================
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


# ============================================================
# MLP MODEL
# ============================================================

class ClinicalMLP(nn.Module):
    """
    Untrained deep MLP projection network.
    Purpose: project high-dimensional clinical data into a lower-dimensional
    non-linear embedding space.

    Reason for using 5 layers: even with random weights, deeper networks can
    produce richer non-linear representations.

    NOTE: To obtain the same result in every run, torch.manual_seed() and
    np.random.seed() are set inside main().
    """
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            # Final layer: output with embedding_dim dimensions.
            # Values after this activation are used as embeddings.
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# MAIN
# ============================================================

def main(args):

    # Reproducibility: with the same seed, identical embeddings are produced.
    torch.manual_seed(42)
    np.random.seed(42)

    # ========================================================
    # LOAD EXCEL
    # ========================================================

    df = pd.read_excel(args.excel_path)
    print(f"Yuklendi: {df.shape[0]} hasta, {df.shape[1]} sutun")

    # ========================================================
    # GET PATIENT IDS
    # ========================================================

    patient_ids = df[args.patient_id_column].values

    # ========================================================
    # PREPROCESS
    # ========================================================

    for column in df.columns:

        if column == args.patient_id_column:
            continue

        if df[column].dtype == "object":
            # Convert categorical columns to numeric values
            # Example: "Male"/"Female" -> 0/1
            # NOTE: This encoder is not saved. If the same encoding is needed
            # later, the encoder should be saved with pickle.
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))

    # Fill missing values with the median.
    # Using zero filling may be statistically inappropriate for variables
    # such as age or tumor size; median is a more realistic replacement.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fallback for any remaining missing values after categorical conversion.
    df = df.fillna(0)

    # ========================================================
    # FEATURES
    # ========================================================

    # Remove patient ID column from features.
    features = df.drop(columns=[args.patient_id_column]).values

    # ========================================================
    # NORMALIZATION
    # ========================================================

    # StandardScaler scales each column to mean=0 and std=1.
    # This is necessary so the MLP can process all features on a similar scale;
    # otherwise, large-scale variables may dominate the model.
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)

    # ========================================================
    # MODEL
    # ========================================================

    input_dim = features.shape[1]
    print(f"Feature boyutu: {input_dim} -> Embedding boyutu: {args.embedding_dim}")

    model = ClinicalMLP(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim
    )

    model.eval()  # Standard practice, although there is no dropout/batchnorm here.

    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    # torch.from_numpy converts the numpy array to tensor without copying
    # when possible, making it more efficient than torch.tensor().
    x_tensor = torch.from_numpy(features)

    with torch.no_grad():  # No gradients are needed; saves memory.
        embeddings = model(x_tensor)

    embeddings = embeddings.numpy()  # [N_patients, embedding_dim]

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(args.output_dir, exist_ok=True)

    for patient_id, embedding in zip(patient_ids, embeddings):

        save_path = os.path.join(
            args.output_dir,
            f"{patient_id}.npy"
        )

        np.save(save_path, embedding)
        print(f"Kaydedildi: {save_path}")

    print(f"\nTamamlandi. {len(patient_ids)} hasta icin embedding kaydedildi.")


# ============================================================
# ARGPARSE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Clinical data Excel file (.xlsx)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where embedding .npy files will be saved"
    )

    parser.add_argument(
        "--patient_id_column",
        type=str,
        default="patient_id",
        help="Name of the patient ID column in the Excel file"
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Output embedding dimension (default: 64)"
    )

    args = parser.parse_args()

    main(args)

    args = parser.parse_args()

    main(args)
