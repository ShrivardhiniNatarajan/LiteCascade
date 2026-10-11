import numpy as np
import pandas as pd

def create_windows(df, feature_cols, T=10, stride=1):
    X = []
    y = []
    for group, group_df in df.groupby("group"):
        group_df = group_df.sort_values("timestamp")
        feats = group_df[feature_cols].values
        labels = group_df["label_binary"].values
        for i in range(0, len(feats) - T + 1, stride):
            X.append(feats[i:i+T])
            y.append(labels[i+T-1])
    return np.array(X), np.array(y)
