
# HER2 Classification via Multimodal Fusion of WSI and Clinical Data
<br clear="right"/>

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/fa5bd5c1-fb0e-4023-b3fa-2318a61b8e5b"
    alt="logo0" 
    width="250"
    align=left
  />
</p>


A modular deep learning pipeline for HER2 binary classification using multimodal fusion of Whole Slide Images (WSI) and clinical tabular data. Four fusion strategies are compared — concatenation, gated attention, cross-attention, and gated cross-attention — followed by a broad classifier sweep using LazyPredict.

---
<br>
<br>
<br>



## Overview

This repository contains the full pipeline for a thesis study on multimodal cancer classification. The pipeline is divided into two stages:

**Stage 1 — Modality-Specific Embedding Extraction**
Each data modality is independently encoded into a fixed-size embedding vector and saved as a `.npy` file per patient.

**Stage 2 — Multimodal Fusion, Attention, and Classification**
WSI and clinical embeddings are fused using one of four attention strategies. The resulting fused embeddings are evaluated across multiple classifiers via LazyPredict.

---

<img width="1536" height="1024" alt="pipeline_abstract_wsi_cli" src="https://github.com/user-attachments/assets/a7435503-df37-4538-abbb-a7a200493559" />

## Pipeline Architecture

```
WSI (.svs)
  └─ TRIDENT (patch extraction + UNI v2 encoder) → patch embeddings (.h5)
       └─ TransMIL (slide-level aggregation)      → WSI embedding (512-dim, .npy)

Clinical Data (.xlsx)
  └─ Preprocessing (encoding + imputation + z-score)
       └─ MLP Encoder (5-layer, untrained)         → Clinical embedding (64-dim, .npy)

MRI (.nii.gz)  [future work]
  └─ MONAI preprocessing
       └─ MedicalNet 3D ResNet18                   → MRI embedding (512-dim, .npy)

WSI embedding + Clinical embedding
  └─ [Concat / Gated Attention / Cross-Attention / Gated Cross-Attention]
       └─ Fused embedding (.npy)
            └─ LazyPredict classifier sweep        → HER2 Positive / Negative
```

---

## Repository Structure

```
├── run_transmil_folder.py          # WSI: patch .h5 → slide embedding .npy via TransMIL
├── run_clinical_embedding.py       # Clinical: Excel → 64-dim embedding .npy via MLP
├── run_mri_embeddings.py           # MRI: NIfTI → 512-dim embedding .npy (future work)
├── run_create_universal_split.py   # Create shared train/val/test split for all models
├── run_concat_fusion.py            # Fusion: simple concatenation (non-trainable baseline)
├── run_gated_attention_fusion.py   # Fusion: gated attention (trainable)
├── run_cross_attention_fusion.py   # Fusion: bidirectional cross-attention (trainable)
├── run_gated_cross_attention.py    # Fusion: gated cross-attention (trainable)
├── run_lazypredict_on_fusion.py    # Classifier sweep on any fused embedding folder
│
├── wsi_embedding_extraction.ipynb       # Notebook: WSI extraction walkthrough
├── clinical_embedding_extraction.ipynb  # Notebook: clinical embedding walkthrough
├── mri_embedding_extraction.ipynb       # Notebook: MRI embedding walkthrough
├── create_universal_split.ipynb         # Notebook: split creation
├── concat_fusion_vector.ipynb           # Notebook: concat fusion
├── gated_attention_fusion.ipynb         # Notebook: gated attention fusion
├── cross_attention_fusion.ipynb         # Notebook: cross-attention fusion
├── gated_cross_attention.ipynb          # Notebook: gated cross-attention fusion
└── lazypredict_on_fusion.ipynb          # Notebook: LazyPredict evaluation
```

---

## Fusion Methods Compared

| Method | Trainable | Output Dim | Description |
|---|---|---|---|
| Concatenation | No | 576 | Direct vector join — non-trainable baseline |
| Gated Attention | Yes | 256 | Sigmoid gate learns per-dim modality contribution |
| Cross-Attention | Yes | 256 | Bidirectional attention; WSI ↔ Clinical |
| Gated Cross-Attention | Yes | 256 | Cross-attention + sigmoid gate over fused context |

All trainable methods use the same fixed train/val/test split for fair comparison.

---

## Requirements

```bash
pip install torch torchvision
pip install monai nibabel SimpleITK
pip install scikit-learn pandas numpy
pip install lazypredict
pip install openpyxl h5py
pip install huggingface_hub
```

For WSI patch extraction, [TRIDENT](https://github.com/mahmoodlab/TRIDENT) is required separately:

```bash
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
pip install -e .
```

---

## Usage

### Step 1 — Extract WSI Embeddings

```bash
# 1a. Extract patch-level features using TRIDENT (run from TRIDENT directory)
python run_batch_of_slides.py \
  --task all \
  --wsi_dir "/path/to/WSI/" \
  --job_dir "/path/to/TRIDENT_OUTPUT" \
  --patch_encoder "uni_v2" \
  --mag 20 \
  --patch_size 256 \
  --batch_size 8 \
  --gpus 0

# 1b. Aggregate patch embeddings into slide-level embeddings via TransMIL
python run_transmil_folder.py \
  --h5_dir "path/to/TRIDENT_OUTPUT/uni_v2/h5_files" \
  --ckpt_path "path/to/transmil_checkpoint.pt" \
  --output_dir "path/to/WSI_EMBEDDINGS" \
  --in_dim 1536 \
  --n_classes 2 \
  --embed_dim 512
```

### Step 2 — Extract Clinical Embeddings

```bash
python run_clinical_embedding.py \
  --excel_path "path/to/clinical.xlsx" \
  --output_dir "path/to/CLINICAL_EMBEDDINGS" \
  --patient_id_column "patient_id" \
  --embedding_dim 64
```

### Step 3 — Create Universal Split

```bash
python run_create_universal_split.py \
  --labels_csv "path/to/labels.csv" \
  --embedding_dirs "path/to/WSI_EMBEDDINGS" "path/to/CLINICAL_EMBEDDINGS" \
  --output_dir "path/to/splits/universal_split"
```

### Step 4 — Run Fusion

Choose one (or all) fusion strategies:

```bash
# Concatenation (baseline)
python run_concat_fusion.py \
  --wsi_dir "path/to/WSI_EMBEDDINGS" \
  --clinical_dir "path/to/CLINICAL_EMBEDDINGS" \
  --split_dir "path/to/splits/universal_split" \
  --output_dir "path/to/fused/concat"

# Gated Attention
python run_gated_attention_fusion.py \
  --wsi_dir "path/to/WSI_EMBEDDINGS" \
  --clinical_dir "path/to/CLINICAL_EMBEDDINGS" \
  --split_dir "path/to/splits/universal_split" \
  --output_dir "path/to/fused/gated_attention"

# Cross-Attention
python run_cross_attention_fusion.py \
  --wsi_dir "path/to/WSI_EMBEDDINGS" \
  --clinical_dir "path/to/CLINICAL_EMBEDDINGS" \
  --split_dir "path/to/splits/universal_split" \
  --output_dir "path/to/fused/cross_attention"

# Gated Cross-Attention
python run_gated_cross_attention.py \
  --wsi_dir "path/to/WSI_EMBEDDINGS" \
  --clinical_dir "path/to/CLINICAL_EMBEDDINGS" \
  --split_dir "path/to/splits/universal_split" \
  --output_dir "path/to/fused/gated_cross_attention"
```

### Step 5 — Classifier Evaluation

```bash
# Run on each fusion output folder
python run_lazypredict_on_fusion.py \
  --fused_dir "path/to/fused/concat" \
  --output_dir "path/to/results/concat"

# Add --evaluate_test only when ready for final evaluation
python run_lazypredict_on_fusion.py \
  --fused_dir "path/to/fused/gated_cross_attention" \
  --output_dir "path/to/results/gated_cross_attention" \
  --evaluate_test
```

---

## Data Format

**labels.csv** — required for split creation and trainable fusion:
```
patient_id,label
patient001,0
patient002,1
...
```
`label`: 0 = HER2 Negative, 1 = HER2 Positive

**Embedding directories** — one `.npy` file per patient, named by patient ID:
```
WSI_EMBEDDINGS/
    patient001.npy    # shape: (512,)
    patient002.npy

CLINICAL_EMBEDDINGS/
    patient001.npy    # shape: (64,)
    patient002.npy
```

---

## Output Structure

Each fusion script produces:
```
fused_output/
├── fused_train/          # fused .npy per patient
├── fused_val/
├── fused_test/
├── train_patients.csv
├── val_patients.csv
├── test_patients.csv
├── best_fusion_model.pt  # (trainable methods only)
└── fusion_summary.txt
```

LazyPredict produces:
```
results/
├── lazy_val_results.csv
├── lazy_val_predictions.csv
├── lazy_test_results.csv      # if --evaluate_test
├── lazy_test_predictions.csv  # if --evaluate_test
└── run_summary.txt
```

---

## Notes

- All embedding extraction is **unsupervised / label-free**. Labels are only used in fusion training and evaluation.
- The **universal split** must be created once and reused across all fusion methods to ensure fair comparison.
- The `--evaluate_test` flag in LazyPredict should only be used **after all architecture decisions are finalized** to avoid test set leakage.
- MRI embedding extraction (`run_mri_embeddings.py`) is implemented but not yet connected to the fusion stage. Planned for future work.
- The clinical MLP uses **fixed random weights** (seed=42) as a projection encoder. Labels are not used during clinical embedding extraction.

---

## Future Work

- Three-way fusion incorporating MRI embeddings
- Supervised clinical embedding encoder
- Hyperparameter search per fusion method
- Explainability / attention weight visualization

---

## Acknowledgements

- [TRIDENT](https://github.com/mahmoodlab/TRIDENT) — patch extraction and encoding
- [MedicalNet](https://github.com/Tencent/MedicalNet) — pretrained 3D ResNet weights
- [TransMIL](https://github.com/szc19990412/TransMIL) — transformer-based MIL aggregation
- [MONAI](https://monai.io/) — medical image preprocessing
- [LazyPredict](https://github.com/shankarpandala/lazypredict) — classifier screening
