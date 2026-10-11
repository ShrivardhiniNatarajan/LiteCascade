"""Malimg Image Malware Dataset Loader and Synthetic Fixture.

Handles 25-class malware family classification from 64x64 grayscale image representations
of raw binary byte sequences. Provides stratified train/val/test splits and
Leave-One-Family-Out (LOFO) zero-day partitioning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MALIMG_FAMILIES: list[str] = [
    "Adialer.C",
    "Allaple.A",
    "Allaple.L",
    "Alueron.gen!J",
    "Autorun.K",
    "C2LOP.P",
    "C2LOP.gen!g",
    "Dialplatform.B",
    "Dontovo.A",
    "Fakerean",
    "Instantaccess",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "Lolyda.AT",
    "Malex.gen!J",
    "Obfuscator.AD",
    "Rbot!gen",
    "Skintrim.N",
    "Swizzor.gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Wintrim.BX",
    "Yuner.A",
    "Benign",
]


class MalimgDataset(Dataset):
    """PyTorch Dataset for 64x64 grayscale malware images."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # Images: shape (N, 1, 64, 64) float32 in [0, 1]
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def generate_synthetic_malimg(
    num_samples: int = 1000,
    img_size: int = 64,
    num_classes: int = 25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic deterministic 64x64 grayscale malware texture maps."""
    rng = np.random.RandomState(seed)

    # Balanced class assignments
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    labels_list = []
    for c in range(num_classes):
        count = samples_per_class + (1 if c < remainder else 0)
        labels_list.extend([c] * count)
    labels = np.array(labels_list, dtype=np.int64)

    # Class-specific texture patterns simulating binary byte frequency matrices
    images = np.zeros((num_samples, 1, img_size, img_size), dtype=np.float32)
    for i, c in enumerate(labels):
        # Base class frequency pattern
        freq_x = (c % 5 + 1) * 2.0
        freq_y = (c // 5 + 1) * 2.0
        x = np.linspace(0, np.pi * freq_x, img_size)
        y = np.linspace(0, np.pi * freq_y, img_size)
        xx, yy = np.meshgrid(x, y)
        pattern = np.sin(xx) * np.cos(yy)

        # Add Gaussian noise
        noise = rng.randn(img_size, img_size) * 0.2
        img = pattern + noise
        # Normalize to [0.0, 1.0]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images[i, 0] = img_norm.astype(np.float32)

    # Shuffle
    perm = rng.permutation(num_samples)
    return images[perm], labels[perm]


def load_malimg(
    use_synthetic: bool = True,
    num_samples: int = 1000,
    img_size: int = 64,
    seed: int = 42,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Loads Malimg dataset with stratified train/val/test splits."""
    if use_synthetic or root_dir is None or not Path(root_dir).exists():
        logger.info(f"Loading synthetic Malimg fixture (samples={num_samples}, seed={seed})")
        images, labels = generate_synthetic_malimg(
            num_samples=num_samples,
            img_size=img_size,
            num_classes=len(MALIMG_FAMILIES),
            seed=seed,
        )
    else:
        # Load from disk if available
        raise NotImplementedError("Physical Malimg binary folder loader not configured in this environment.")

    # Stratified 70 / 15 / 15 split
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in range(len(MALIMG_FAMILIES)):
        cls_indices = np.where(labels == c)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n))

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train : n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val :])

    return {
        "X_train": images[train_idx],
        "y_train": labels[train_idx],
        "X_val": images[val_idx],
        "y_val": labels[val_idx],
        "X_test": images[test_idx],
        "y_test": labels[test_idx],
        "families": MALIMG_FAMILIES,
        "num_classes": len(MALIMG_FAMILIES),
    }


def create_malimg_lofo(
    family_name: str,
    num_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Creates Leave-One-Family-Out (LOFO) split excluding held-out family from train."""
    if family_name not in MALIMG_FAMILIES:
        raise ValueError(f"Family '{family_name}' not in MALIMG_FAMILIES.")

    held_out_idx = MALIMG_FAMILIES.index(family_name)
    images, labels = generate_synthetic_malimg(num_samples=num_samples, seed=seed)

    train_mask = labels != held_out_idx
    test_mask = labels == held_out_idx

    assert not np.any(labels[train_mask] == held_out_idx), f"Leakage: held-out family in train: {family_name}"

    return {
        "X_train": images[train_mask],
        "y_train": labels[train_mask],
        "X_test": images[test_mask],
        "y_test": labels[test_mask],
        "held_out_family": family_name,
        "held_out_idx": held_out_idx,
    }
