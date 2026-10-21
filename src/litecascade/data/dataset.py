import numpy as np
import torch
from torch.utils.data import Dataset


class CascadeDataset(Dataset):
    """(X, y) windows; optionally also per-window teacher logits aligned by index (for distillation)."""

    def __init__(self, X, y, teacher_logits=None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.t = None if teacher_logits is None else torch.as_tensor(np.asarray(teacher_logits), dtype=torch.float32)
        if self.t is not None and len(self.t) != len(self.X):
            raise ValueError("teacher_logits must be aligned 1:1 with X")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.t is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.t[idx]


def compute_class_weights(y, num_classes: int | None = None) -> torch.Tensor:
    """'Balanced' weights n / (K * count_k); classes absent from y get weight 0."""
    y = np.asarray(y)
    k = num_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(np.float64)
    weights = np.zeros(k)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)
