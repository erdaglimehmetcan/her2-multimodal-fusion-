# ============================================================
# Gated Attention Fusion: WSI + Clinical
# ============================================================
#
# Purpose:
#   Train a gated attention fusion model using WSI and clinical
#   embeddings, then export new fused .npy embeddings.
#
# Difference from concatenation:
#   Concatenation simply joins vectors:
#       fused = [WSI ; Clinical]
#
#   Gated attention learns modality contribution:
#       fused = gate * WSI_context + (1 - gate) * Clinical_context
#
# Important:
#   Gated attention is trainable.
#   Therefore, this script uses predefined train/val/test splits.
#
# Workflow:
#   1. Load fixed train/val/test CSV files from split_dir
#   2. Train gated attention model on train patients only
#   3. Select best model using validation AUC
#   4. Export fused .npy embeddings for train/val/test
#
# Example:
#   python gated_attention_fusion.py ^
#     --wsi_dir "D:\embeddings\wsi" ^
#     --clinical_dir "D:\embeddings\clinical" ^
#     --split_dir "D:\splits\wsi_clinical_split" ^
#     --output_dir "D:\embeddings\fused_gated_attention"
#
# Arguments:
#   --wsi_dir        Folder containing WSI .npy embeddings.
#   --clinical_dir   Folder containing clinical .npy embeddings.
#   --split_dir      Folder containing train_patients.csv, val_patients.csv, test_patients.csv.
#   --output_dir     Folder where fused embeddings and best model will be saved.
#   --wsi_dim        WSI embedding dimension. Default: 512
#   --clinical_dim   Clinical embedding dimension. Default: 64
#   --fused_dim      Output fused embedding dimension. Default: 256
#   --hidden_dim     Hidden dimension inside gate network. Default: 128
#   --dropout        Dropout rate. Default: 0.1
#   --epochs         Training epochs. Default: 50
#   --batch_size     Batch size. Default: 16
#   --lr             Learning rate. Default: 1e-4
#   --weight_decay   AdamW weight decay. Default: 1e-4
#   --seed           Random seed. Default: 42
#
# Output:
#   output_dir/
#       best_fusion_model.pt
#       train_patients.csv
#       val_patients.csv
#       test_patients.csv
#       fused_train/
#           patient001.npy
#       fused_val/
#           patient101.npy
#       fused_test/
#           patient201.npy
#
# ============================================================

import argparse
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, dataframe, wsi_dir, clinical_dir, wsi_dim, clinical_dim):
        self.df = dataframe.reset_index(drop=True)
        self.wsi_dir = Path(wsi_dir)
        self.clinical_dir = Path(clinical_dir)
        self.wsi_dim = wsi_dim
        self.clinical_dim = clinical_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = str(self.df.loc[idx, "patient_id"])
        label = np.float32(self.df.loc[idx, "label"])

        wsi_path = self.wsi_dir / f"{patient_id}.npy"
        clinical_path = self.clinical_dir / f"{patient_id}.npy"

        if not wsi_path.exists():
            raise FileNotFoundError(f"Missing WSI embedding: {wsi_path}")

        if not clinical_path.exists():
            raise FileNotFoundError(f"Missing clinical embedding: {clinical_path}")

        wsi = np.load(wsi_path).astype(np.float32).reshape(-1)
        clinical = np.load(clinical_path).astype(np.float32).reshape(-1)

        if wsi.shape != (self.wsi_dim,):
            raise ValueError(
                f"{patient_id}: WSI shape {wsi.shape}, expected ({self.wsi_dim},)"
            )

        if clinical.shape != (self.clinical_dim,):
            raise ValueError(
                f"{patient_id}: Clinical shape {clinical.shape}, expected ({self.clinical_dim},)"
            )

        return (
            patient_id,
            torch.from_numpy(wsi),
            torch.from_numpy(clinical),
            torch.tensor(label),
        )


# ============================================================
# Gated Attention Fusion Model
# ============================================================

class GatedAttentionFusion(nn.Module):
    def __init__(
        self,
        wsi_dim=512,
        clinical_dim=64,
        fused_dim=256,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.fused_dim = fused_dim

        # Project both modalities into the same latent space.
        self.wsi_projection = nn.Sequential(
            nn.Linear(wsi_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.clinical_projection = nn.Sequential(
            nn.Linear(clinical_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gate network.
        # Input: concatenated WSI and clinical contexts.
        # Output: gate vector in [0, 1], shape [batch, fused_dim].
        self.gate_network = nn.Sequential(
            nn.Linear(fused_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(fused_dim)

        # Temporary classifier.
        # This is used only to train the fusion encoder.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, wsi, clinical):
        # Convert both modalities to same dimension.
        wsi_context = self.wsi_projection(wsi)
        clinical_context = self.clinical_projection(clinical)

        # Learn modality contribution.
        gate_input = torch.cat([wsi_context, clinical_context], dim=1)
        gate = self.gate_network(gate_input)

        # Gated fusion.
        # gate close to 1 -> more WSI
        # gate close to 0 -> more clinical
        fused = gate * wsi_context + (1.0 - gate) * clinical_context
        fused = self.output_norm(fused)

        return fused

    def forward(self, wsi, clinical):
        fused = self.encode(wsi, clinical)
        logit = self.classifier(fused).squeeze(1)
        return logit


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for _, wsi, clinical, labels in loader:
        wsi = wsi.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(wsi, clinical)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for _, wsi, clinical, labels in loader:
            wsi = wsi.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)

            logits = model(wsi, clinical)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = (all_probs >= 0.5).astype(int)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = np.nan

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    loss = total_loss / len(loader.dataset)

    return loss, auc, acc, f1


# ============================================================
# Export fused embeddings
# ============================================================

def export_fused_embeddings(model, dataset, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()

    with torch.no_grad():
        for patient_ids, wsi, clinical, _ in loader:
            patient_id = str(patient_ids[0])

            wsi = wsi.to(device)
            clinical = clinical.to(device)

            # Only encoder part is used.
            # Classifier is ignored during export.
            fused = model.encode(wsi, clinical)

            fused_np = fused.squeeze(0).cpu().numpy().astype(np.float32)

            save_path = output_dir / f"{patient_id}.npy"
            np.save(save_path, fused_np)

    print(f"Saved fused embeddings to: {output_dir}")


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
    parser.add_argument("--fused_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train_patients.csv"
    val_csv = split_dir / "val_patients.csv"
    test_csv = split_dir / "test_patients.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print("Using predefined split:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    # Save copies of the split files inside this experiment folder.
    shutil.copy2(train_csv, output_dir / "train_patients.csv")
    shutil.copy2(val_csv, output_dir / "val_patients.csv")
    shutil.copy2(test_csv, output_dir / "test_patients.csv")

    train_dataset = EmbeddingDataset(
        train_df,
        args.wsi_dir,
        args.clinical_dir,
        args.wsi_dim,
        args.clinical_dim,
    )

    val_dataset = EmbeddingDataset(
        val_df,
        args.wsi_dir,
        args.clinical_dir,
        args.wsi_dim,
        args.clinical_dim,
    )

    test_dataset = EmbeddingDataset(
        test_df,
        args.wsi_dir,
        args.clinical_dir,
        args.wsi_dim,
        args.clinical_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GatedAttentionFusion(
        wsi_dim=args.wsi_dim,
        clinical_dim=args.clinical_dim,
        fused_dim=args.fused_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Class imbalance handling using only training labels.
    pos_count = train_df["label"].sum()
    neg_count = len(train_df) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_path = output_dir / "best_fusion_model.pt"

    # --------------------------------------------------------
    # Train on train split.
    # Validate on validation split.
    # Test split is not used during training.
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_auc, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val ACC: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    print("Best validation AUC:", best_auc)

    if not best_path.exists():
        raise RuntimeError("No best model was saved. Check validation labels/AUC.")

    # --------------------------------------------------------
    # Export fused embeddings.
    # Same trained fusion encoder is used for all splits.
    # --------------------------------------------------------

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    export_fused_embeddings(
        model,
        train_dataset,
        output_dir / "fused_train",
        device,
    )

    export_fused_embeddings(
        model,
        val_dataset,
        output_dir / "fused_val",
        device,
    )

    export_fused_embeddings(
        model,
        test_dataset,
        output_dir / "fused_test",
        device,
    )

    summary_path = output_dir / "fusion_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Gated Attention Fusion Summary\n")
        f.write("==============================\n\n")
        f.write(f"WSI dim: {args.wsi_dim}\n")
        f.write(f"Clinical dim: {args.clinical_dim}\n")
        f.write(f"Fused dim: {args.fused_dim}\n")
        f.write(f"Hidden dim: {args.hidden_dim}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Best validation AUC: {best_auc}\n")

    print("Finished.")
    print("Fused embedding dimension:", args.fused_dim)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()