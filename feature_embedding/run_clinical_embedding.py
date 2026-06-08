# ============================================================
# Clinical Embedding Extraction
# run_clinical_embedding.py
# ============================================================
#
# PURPOSE
# -------
# Reads patient-level clinical data from an Excel file,
# performs preprocessing, and generates a fixed-length
# embedding vector for each patient.
#
# The output of this script is one .npy file per patient:
#
#   patient001.npy
#   patient002.npy
#   patient003.npy
#
# Each .npy file contains a clinical embedding that can later
# be fused with WSI embeddings using:
#
#   - Concatenation Fusion
#   - Gated Attention Fusion
#   - Cross-Attention Fusion
#   - Gated Cross-Attention Fusion
#
#
# PIPELINE POSITION
# -----------------
#
# Clinical Excel
#       ↓
# Clinical Embedding Script
#       ↓
# Clinical .npy Embeddings
#       ↓
# Fusion Methods
#       ↓
# Fused Embeddings
#       ↓
# Classifiers
#       ↓
# HER2 Prediction
#
#
# INPUT REQUIREMENTS
# ------------------
#
# Example Excel:
#
# patient_id | age | er_status | pr_status | tumor_grade
# ------------------------------------------------------
# TCGA-01    | 55  | Positive  | Negative  | 2
# TCGA-02    | 48  | Positive  | Positive  | 3
# TCGA-03    | 67  | Negative  | Negative  | 2
#
# Notes:
#
# - One row must represent one patient.
# - The patient ID column must be unique.
# - Do NOT include HER2 labels in this file.
# - Labels should be stored separately.
#
#
# PREPROCESSING
# -------------
#
# This script automatically:
#
# 1. Encodes categorical variables
#
#       Positive -> 1
#       Negative -> 0
#
# 2. Fills missing numeric values using median imputation
#
# 3. Standardizes features using StandardScaler
#
#       mean = 0
#       std  = 1
#
# 4. Projects clinical features into a lower-dimensional
#    embedding space using an untrained deep MLP.
#
#
# OUTPUT
# ------
#
# Example output directory:
#
# CLINICAL_EMBEDDINGS/
#
# ├── TCGA-01.npy
# ├── TCGA-02.npy
# ├── TCGA-03.npy
# └── ...
#
#
# EXAMPLE OUTPUT SHAPE
# --------------------
#
# If:
#
#   --embedding_dim 64
#
# then:
#
#   TCGA-01.npy
#
# contains:
#
#   shape = (64,)
#
#
# EXAMPLE USAGE
# -------------
#
# Basic:
#
# python run_clinical_embedding.py ^
#   --excel_path "D:\data\clinical.xlsx" ^
#   --output_dir "D:\CLINICAL_EMBEDDINGS"
#
#
# Custom Patient ID Column:
#
# python run_clinical_embedding.py ^
#   --excel_path "D:\data\clinical.xlsx" ^
#   --output_dir "D:\CLINICAL_EMBEDDINGS" ^
#   --patient_id_column "tcga_patient_id"
#
#
# Custom Embedding Dimension:
#
# python run_clinical_embedding.py ^
#   --excel_path "D:\data\clinical.xlsx" ^
#   --output_dir "D:\CLINICAL_EMBEDDINGS" ^
#   --embedding_dim 128
#
#
# ARGUMENTS
# ---------
#
# --excel_path
#     Path to the clinical Excel file (.xlsx).
#
# --output_dir
#     Directory where clinical embeddings will be saved.
#
# --patient_id_column
#     Name of the patient ID column in the Excel file.
#
#     Default:
#         patient_id
#
# --embedding_dim
#     Output embedding dimension.
#
#     Default:
#         64
#
#
# NOTES
# -----
#
# This script does NOT:
#
# - Train a classifier
# - Predict HER2 status
# - Use HER2 labels
#
# It only converts clinical variables into embedding vectors.
#
# Labels should be stored separately and used later during:
#
#   - Dataset splitting
#   - Fusion training
#   - Classifier training
#   - Model evaluation
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
