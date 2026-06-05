# ============================================================
# run_clinical_embedding.py
#
# Excel dosyasindan klinik veriyi okur, on-isleme yapar,
# eğitimsiz derin MLP ile her hasta icin 64-dim embedding
# cikarir ve .npy olarak kaydeder.
#
# Kullanim:
#   python run_clinical_embedding.py \
#     --excel_path "D:/data/clinical.xlsx" \
#     --output_dir "D:/CLINICAL_EMBEDDINGS" \
#     --patient_id_column "patient_id" \
#     --embedding_dim 64
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
    Egitimsiz derin MLP projeksiyon agi.
    Amac: yuksek boyutlu klinik veriyi dusuk boyutlu (embedding_dim)
    bir uzaya non-linear olarak projekte etmek.

    5 katman kullanilmasinin nedeni: rastgele agirliklarla bile
    derin aglar daha zengin non-linear temsil olusturur.

    NOT: Her calistirmada ayni sonucu almak icin main() icinde
    torch.manual_seed() ve np.random.seed() set edilmistir.
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

            # Son katman: embedding_dim boyutunda cikti
            # Bu katmanin aktivasyon sonrasi degerleri embedding olarak kullanilir
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# MAIN
# ============================================================

def main(args):

    # Reproducibility: ayni seed ile her calistirmada ayni embedding'ler cikar
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
            # Kategorik sutunlari sayisala donustur (ornek: "Erkek"/"Kadin" -> 0/1)
            # NOT: Bu encoder kaydedilmiyor. Ayni encoding'i tekrar kullanmak
            # isterseniz encoder'i pickle ile kaydetmeniz gerekir.
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))

    # Eksik degerleri median ile doldur.
    # Sifir ile doldurmak (fillna(0)) yas, tumor boyutu gibi sutunlarda
    # istatistiksel olarak yanlis olabilir; median daha gercekci bir deger koyar.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Kategorik donusum sonrasi kalan bos degerler icin fallback
    df = df.fillna(0)

    # ========================================================
    # FEATURES
    # ========================================================

    # Patient ID sutununu feature'lardan cikar
    features = df.drop(columns=[args.patient_id_column]).values

    # ========================================================
    # NORMALIZATION
    # ========================================================

    # StandardScaler: her sutunu mean=0, std=1 yaparak olcekler.
    # MLP'nin her ozelligi esit agirlikla isliyebilmesi icin gerekli;
    # olceklenmemis veride buyuk degerli sutunlar modele hakim olur.
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

    model.eval()  # dropout/batchnorm katmanlari yoksa fark etmez ama standart pratik

    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    # torch.from_numpy: numpy array'i kopyalamadan tensor'a cevirir (torch.tensor'dan daha verimli)
    x_tensor = torch.from_numpy(features)

    with torch.no_grad():  # gradient hesaplamaya gerek yok, bellek tasarrufu
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
        help="Klinik veri Excel dosyasi (.xlsx)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Embedding .npy dosyalarinin kaydedilecegi klasor"
    )

    parser.add_argument(
        "--patient_id_column",
        type=str,
        default="patient_id",
        help="Excel'deki hasta ID sutununun adi"
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Cikti embedding boyutu (varsayilan: 64)"
    )

    args = parser.parse_args()

    main(args)