import torch
from torch.utils.data import Dataset
import numpy as np

class CascadeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_class_weights(y):
    # compute inverse frequency
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)
