import numpy as np
import pandas as pd

def create_group_splits(df, dataset_name, ratios=None, seed=42):
    if ratios is None:
        ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    
    np.random.seed(seed)
    groups = df["group"].unique()
    np.random.shuffle(groups)
    
    n = len(groups)
    train_n = max(1, int(n * ratios.get("train", 0.6)))
    val_n = max(1, int(n * ratios.get("val", 0.2)))
    
    train_groups = groups[:train_n]
    val_groups = groups[train_n:train_n+val_n]
    test_groups = groups[train_n+val_n:]
    if len(test_groups) == 0 and n > 2:
        test_groups = [groups[-1]]
        
    if n == 3:
        train_groups = [groups[0]]
        val_groups = [groups[1]]
        test_groups = [groups[2]]

    # For calibration, we can just use the validation split
    # unless a specific calib split is defined.
    splits = {
        "train": df[df["group"].isin(train_groups)].index.tolist(),
        "val": df[df["group"].isin(val_groups)].index.tolist(),
        "test": df[df["group"].isin(test_groups)].index.tolist(),
        "calib": df[df["group"].isin(val_groups)].index.tolist()
    }
    return splits

def create_lofo_split(df, target_family, dataset_name):
    train_idx = df[df["label_family"] != target_family].index.tolist()
    test_idx = df[df["label_family"] == target_family].index.tolist()
    return {"train": train_idx, "test": test_idx}
