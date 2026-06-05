# ============================================================
# Gated Cross-Attention Fusion: WSI + Clinical
# ============================================================
#
# Purpose:
#   Train a gated cross-attention fusion model using WSI and clinical
#   embeddings, then export new fused .npy embeddings.
#
# Important:
#   Attention fusion is trainable.
#   Therefore, patients must be split into train/val/test BEFORE training.
#
# Workflow:
#   1. Read WSI .npy + clinical .npy + labels.csv
#   2. Split patients into train / validation / test
#   3. Train attention fusion model only on train set
#   4. Use validation set only to choose the best model
#   5. Load best model
#   6. Export fused .npy embeddings for train / val / test
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

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
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

        wsi = np.load(wsi_path).astype(np.float32)
        clinical = np.load(clinical_path).astype(np.float32)

        # Safety checks.
        if wsi.shape != (self.wsi_dim,):
            raise ValueError(f"{patient_id}: WSI shape {wsi.shape}, expected ({self.wsi_dim},)")

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
# Gated Cross-Attention Fusion Model
# ============================================================

class GatedCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        wsi_dim=512,
        clinical_dim=64,
        fused_dim=256,
        num_tokens=8,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.fused_dim = fused_dim
        self.num_tokens = num_tokens

        # Converts one WSI embedding into token sequence.
        # [B, 512] -> [B, num_tokens, fused_dim]
        self.wsi_tokenizer = nn.Linear(wsi_dim, num_tokens * fused_dim)

        # Converts one clinical embedding into token sequence.
        # [B, 64] -> [B, num_tokens, fused_dim]
        self.clinical_tokenizer = nn.Linear(clinical_dim, num_tokens * fused_dim)

        self.wsi_norm = nn.LayerNorm(fused_dim)
        self.clinical_norm = nn.LayerNorm(fused_dim)

        # WSI attends to clinical information.
        self.wsi_to_clinical = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Clinical attends to WSI information.
        self.clinical_to_wsi = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gate decides how much WSI context and clinical context contribute.
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(fused_dim)

        # Temporary classifier.
        # This is needed only during training so attention learns meaningful weights.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, wsi, clinical):
        batch_size = wsi.size(0)

        # Tokenize both modalities.
        wsi_tokens = self.wsi_tokenizer(wsi)
        clinical_tokens = self.clinical_tokenizer(clinical)

        wsi_tokens = wsi_tokens.view(batch_size, self.num_tokens, self.fused_dim)
        clinical_tokens = clinical_tokens.view(batch_size, self.num_tokens, self.fused_dim)

        wsi_tokens = self.wsi_norm(wsi_tokens)
        clinical_tokens = self.clinical_norm(clinical_tokens)

        # Cross-attention in both directions.
        wsi_attended, _ = self.wsi_to_clinical(
            query=wsi_tokens,
            key=clinical_tokens,
            value=clinical_tokens,
        )

        clinical_attended, _ = self.clinical_to_wsi(
            query=clinical_tokens,
            key=wsi_tokens,
            value=wsi_tokens,
        )

        # Residual connection + average pooling over tokens.
        wsi_context = (wsi_tokens + wsi_attended).mean(dim=1)
        clinical_context = (clinical_tokens + clinical_attended).mean(dim=1)

        # Gated fusion.
        gate = self.gate(torch.cat([wsi_context, clinical_context], dim=1))

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

    auc = roc_auc_score(all_labels, all_probs)
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
            patient_id = patient_ids[0]

            wsi = wsi.to(device)
            clinical = clinical.to(device)

            # Only the encoder part is used here.
            # The temporary classifier is ignored.
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
    parser.add_argument("--labels_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--wsi_dim", default=512, type=int)
    parser.add_argument("--clinical_dim", default=64, type=int)
    parser.add_argument("--fused_dim", default=256, type=int)

    parser.add_argument("--num_tokens", default=8, type=int)
    parser.add_argument("--num_heads", default=4, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--val_size", default=0.15, type=float)
    parser.add_argument("--test_size", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)

    wsi_dir = Path(args.wsi_dir)
    clinical_dir = Path(args.clinical_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read labels and keep only patients with both embeddings.
    # --------------------------------------------------------

    df = pd.read_csv(args.labels_csv)

    matched_rows = []

    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        label = int(row["label"])

        wsi_path = wsi_dir / f"{patient_id}.npy"
        clinical_path = clinical_dir / f"{patient_id}.npy"

        if wsi_path.exists() and clinical_path.exists():
            matched_rows.append({"patient_id": patient_id, "label": label})

    df = pd.DataFrame(matched_rows)

    print("Matched patients:", len(df))
    print(df["label"].value_counts())

    # --------------------------------------------------------
    # Split FIRST.
    # This prevents data leakage.
    # --------------------------------------------------------

    train_val_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
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

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    # --------------------------------------------------------
    # Create datasets and loaders.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train attention fusion model.
    # Only train set updates the weights.
    # Validation only selects the best epoch.
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GatedCrossAttentionFusion(
        wsi_dim=args.wsi_dim,
        clinical_dim=args.clinical_dim,
        fused_dim=args.fused_dim,
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
    ).to(device)

    pos_count = train_df["label"].sum()
    neg_count = len(train_df) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc = -1.0
    best_path = output_dir / "best_fusion_model.pt"

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

        # Save the model that works best on validation set.
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    print("Best validation AUC:", best_auc)

    # --------------------------------------------------------
    # Load best fusion model.
    # Then export fused embeddings for train / val / test.
    # Labels are not used during export.
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

    print("Finished.")
    print("Fused embedding dimension:", args.fused_dim)


if __name__ == "__main__":
    main()