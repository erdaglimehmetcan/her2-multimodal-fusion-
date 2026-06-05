# ============================================================
# MRI Embedding Extraction Script
# ============================================================
#
# Purpose:
#   This script extracts 512-dimensional feature embeddings from DCE-MRI
#   volumes using a pretrained MedicalNet-style 3D ResNet18 encoder.
#
# Input:
#   A folder containing MRI volumes in .nii or .nii.gz format.
#
# Example input folder:
#   D:\MRI_NIFTI\
#       patient001.nii.gz
#       patient002.nii.gz
#
# Output:
#   A folder containing one .npy embedding file per patient.
#
# Example output folder:
#   D:\MRI_EMBEDDINGS\
#       patient001.npy   # shape: (512,)
#       patient002.npy   # shape: (512,)
#
# Workflow:
#   DCE-MRI volume
#       -> MONAI preprocessing
#       -> MedicalNet 3D ResNet18
#       -> 512-dimensional embedding
#       -> .npy file
#
# Notes:
#   - This script expects MRI files to be already converted to NIfTI format.
#   - DICOM folders should be converted to .nii or .nii.gz before using this script.
#   - Existing output files are skipped unless --overwrite is used.
#   - Failed cases are written into failed_cases.txt.
#
# Example run:
#   python extract_mri_embeddings.py --input_dir "D:\MRI_NIFTI" --output_dir "D:\MRI_EMBEDDINGS" --checkpoint "D:\models\resnet_18_23dataset.pth"
#
# ============================================================




###########################################
####-----------------------------------####
####pip install monai nibabel SimpleITK####
####-----------------------------------####
###########################################


import argparse
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    EnsureTyped,
)


# ============================================================
# MedicalNet-style 3D ResNet18
# ============================================================
#
# This section defines a 3D ResNet18 architecture.
# The input is a 3D MRI volume.
# The output is a 512-dimensional feature vector.
#
# This model is used as an encoder:
#   MRI volume -> ResNet18 -> 512-d embedding
#
# The classification layer is not used.
# We only use the final pooled feature representation.
# ============================================================


def conv3x3x3(in_planes, out_planes, stride=1):
    # Standard 3D convolution block used inside ResNet.
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    # Downsamples the residual connection when feature map size or channel count changes.
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)

    # Adds zero-padding in the channel dimension if needed.
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        device=out.device,
        dtype=out.dtype,
    )

    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    # Basic residual block used in ResNet18.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        # First 3D convolution.
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Second 3D convolution.
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        # Optional residual downsampling.
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        residual = x

        # Main branch.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual branch.
        if self.downsample is not None:
            residual = self.downsample(x)

        # Residual addition.
        out += residual
        out = self.relu(out)

        return out


class MedicalNetResNet18(nn.Module):
    # 3D ResNet18 encoder.
    #
    # Input shape:
    #   [batch, channel, depth, height, width]
    #
    # Output shape:
    #   [batch, 512]
    #
    # For single MRI volume:
    #   [1, 1, 96, 96, 96] -> [1, 512]

    def __init__(self, in_channels=1):
        super().__init__()

        self.in_planes = 64

        # Initial 3D convolution layer.
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet18 has 2 residual blocks per layer.
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Converts final feature map into a single 512-d vector.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        # Builds one ResNet stage.
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = lambda x: downsample_basic_block(
                x,
                planes * block.expansion,
                stride,
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial feature extraction.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual feature extraction.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling.
        x = self.avgpool(x)

        # Flatten from [batch, 512, 1, 1, 1] to [batch, 512].
        x = torch.flatten(x, 1)

        return x


# ============================================================
# Utility functions
# ============================================================


def get_patient_id(path: Path) -> str:
    # Extracts patient ID from filename.
    #
    # Example:
    #   patient001.nii.gz -> patient001
    #   patient001.nii    -> patient001

    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def load_medicalnet_weights(model, checkpoint_path, device):
    # Loads MedicalNet pretrained weights.
    #
    # The checkpoint may contain keys such as:
    #   state_dict
    #   model_state_dict
    #   net
    #   model
    #
    # This function extracts the compatible weights and ignores incompatible layers.
    # Fully connected classification layers are skipped because we only need embeddings.

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    clean_checkpoint = {}

    for key, value in checkpoint.items():
        # Remove DataParallel prefix if present.
        key = key.replace("module.", "")

        # Skip classification layer.
        if key.startswith("fc."):
            continue

        clean_checkpoint[key] = value

    model_state = model.state_dict()

    # Load only layers with matching names and tensor shapes.
    compatible_state = {
        key: value
        for key, value in clean_checkpoint.items()
        if key in model_state and value.shape == model_state[key].shape
    }

    model_state.update(compatible_state)
    model.load_state_dict(model_state)

    print(f"Loaded compatible layers: {len(compatible_state)}")
    print("Skipped layers:", len(clean_checkpoint) - len(compatible_state))

    return model


# ============================================================
# Main extraction script
# ============================================================
#
# This part:
#   1. Reads input/output/checkpoint paths from command line.
#   2. Finds all .nii and .nii.gz MRI files.
#   3. Applies MONAI preprocessing.
#   4. Loads pretrained MedicalNet ResNet18.
#   5. Extracts 512-d embeddings.
#   6. Saves each embedding as .npy.
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    # Folder containing input MRI files.
    parser.add_argument("--input_dir", required=True, type=str)

    # Folder where output .npy embeddings will be saved.
    parser.add_argument("--output_dir", required=True, type=str)

    # Path to MedicalNet ResNet18 checkpoint.
    parser.add_argument("--checkpoint", required=True, type=str)

    # MRI volumes are resized/padded/cropped to this fixed 3D shape.
    parser.add_argument("--spatial_size", nargs=3, type=int, default=[96, 96, 96])

    # If enabled, existing .npy files will be overwritten.
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Failed cases will be written here.
    failed_log = output_dir / "failed_cases.txt"

    # Recursively find all NIfTI files in the input directory.
    mri_files = sorted(
        list(input_dir.rglob("*.nii")) +
        list(input_dir.rglob("*.nii.gz"))
    )

    print(f"Found MRI files: {len(mri_files)}")

    if len(mri_files) == 0:
        raise RuntimeError("No .nii or .nii.gz files found.")

    # Use GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # MONAI preprocessing pipeline.
    #
    # LoadImaged:
    #   Loads the NIfTI MRI file.
    #
    # EnsureChannelFirstd:
    #   Converts image shape to [channel, depth, height, width].
    #
    # ScaleIntensityd:
    #   Normalizes MRI intensity values.
    #
    # ResizeWithPadOrCropd:
    #   Makes every MRI volume the same size.
    #
    # EnsureTyped:
    #   Converts the data into PyTorch-compatible tensor format.
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=tuple(args.spatial_size)),
        EnsureTyped(keys=["image"]),
    ])

    # Build MedicalNet-style ResNet18 encoder.
    model = MedicalNetResNet18(in_channels=1).to(device)

    # Load pretrained MedicalNet weights.
    model = load_medicalnet_weights(model, checkpoint_path, device)

    # Evaluation mode disables dropout/batchnorm training behavior.
    model.eval()

    for mri_path in mri_files:
        patient_id = get_patient_id(mri_path)
        save_path = output_dir / f"{patient_id}.npy"

        # Skip already processed patients.
        # This is important for long extraction jobs.
        if save_path.exists() and not args.overwrite:
            print(f"Skipping {patient_id}, already exists.")
            continue

        try:
            # Load and preprocess one MRI volume.
            sample = transforms({"image": str(mri_path)})

            # Add batch dimension:
            #   [1, D, H, W] -> [1, 1, D, H, W]
            x = sample["image"].unsqueeze(0).float().to(device)

            # Extract embedding without gradient computation.
            with torch.no_grad():
                embedding = model(x)

            # Convert from torch tensor to numpy array.
            embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)

            # Safety check.
            if embedding.shape != (512,):
                raise ValueError(f"Expected embedding shape (512,), got {embedding.shape}")

            # Save patient embedding.
            np.save(save_path, embedding)

            print(f"Saved {patient_id}: {embedding.shape}")

        except Exception as e:
            # If one MRI fails, continue with the next one.
            print(f"FAILED: {patient_id}")

            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- {patient_id} ---\n")
                f.write(str(e))
                f.write("\n")
                f.write(traceback.format_exc())
                f.write("\n")

            # Clear GPU cache after failure.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()