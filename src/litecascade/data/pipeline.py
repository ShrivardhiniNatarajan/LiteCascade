"""One leakage-safe data preparation path shared by every training / evaluation script.

load -> clean -> split (group or time, disjoint calib) -> fit scaler + PCA on TRAIN records only ->
window each split separately (windows never cross segments or splits) -> scale.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ciciot2023 import load_ciciot2023
from .clean import feature_columns
from .iot23 import load_iot23
from .nbaiot import load_nbaiot
from .preprocess import Preprocessor
from .splits import DEFAULT_RATIOS, make_splits
from .windows import create_windows

logger = logging.getLogger(__name__)

DATASETS = ["fixture", "fixture-ciciot2023", "fixture-iot23", "nbaiot", "ciciot2023", "iot23"]
SPLITS = ["train", "val", "calib", "test"]


def load_dataset(name: str, root: str | Path | None = None, max_rows_per_file: int | None = None) -> pd.DataFrame:
    """``fixture`` = synthetic data in N-BaIoT layout; ``fixture-<ds>`` = synthetic data in that dataset's layout."""
    if name == "fixture":
        return load_nbaiot(use_synthetic=True)
    if name == "fixture-ciciot2023":
        return load_ciciot2023(use_synthetic=True)
    if name == "fixture-iot23":
        return load_iot23(use_synthetic=True)
    if name == "nbaiot":
        return load_nbaiot(root=root, max_rows_per_file=max_rows_per_file)
    if name == "ciciot2023":
        return load_ciciot2023(root=root, max_rows_per_file=max_rows_per_file)
    if name == "iot23":
        return load_iot23(root=root, max_rows_per_file=max_rows_per_file)
    raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}")


def class_names_for(df: pd.DataFrame, task: str) -> list[str]:
    if task == "binary":
        return ["benign", "malicious"]
    fams = sorted(f for f in df["label_family"].unique() if f != "benign")
    return ["benign"] + fams


@dataclass
class PreparedData:
    dataset: str
    task: str
    T: int
    feature_cols: list[str]
    class_names: list[str]
    preprocessor: Preprocessor
    X: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    is_synthetic: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def summary(self) -> dict:
        out = {}
        for s in SPLITS:
            y = self.y.get(s, np.zeros(0))
            out[s] = {"windows": int(len(y)), "malicious": int((y > 0).sum()), "benign": int((y == 0).sum())}
        return out


def prepare_data(
    dataset: str = "fixture",
    root: str | Path | None = None,
    T: int = 10,
    stride: int = 1,
    split_mode: str = "auto",
    ratios: dict | None = None,
    seed: int = 42,
    task: str = "binary",
    pca_dim: int = 32,
    max_rows_per_file: int | None = None,
    max_train_windows: int | None = None,
    df: pd.DataFrame | None = None,
) -> PreparedData:
    if df is None:
        df = load_dataset(dataset, root=root, max_rows_per_file=max_rows_per_file)
    feats = feature_columns(df)
    class_names = class_names_for(df, task)
    if task == "binary":
        df["label"] = df["label_binary"].astype(int)
    else:
        mapping = {c: i for i, c in enumerate(class_names)}
        df["label"] = df["label_family"].map(mapping).astype(int)

    splits = make_splits(df, mode=split_mode, ratios=ratios or DEFAULT_RATIOS, seed=seed, gap=T)
    train_df = df.loc[splits["train"]]
    pre = Preprocessor(n_components=min(pca_dim, len(feats)), incremental=len(train_df) > 200_000)
    pre.fit(train_df, feats)  # <- train records only

    data = PreparedData(dataset, task, T, feats, class_names, pre, is_synthetic=bool(df["is_synthetic"].iloc[0]))
    rng = np.random.default_rng(seed)
    for s in SPLITS:
        part = df.loc[splits[s]]
        if s == "calib" and splits["calib"] is splits.get("val"):
            data.info["calib_is_val"] = True
        X, y, meta = create_windows(part, feats, T=T, stride=stride, label_col="label", return_meta=True)
        if s == "train" and max_train_windows and len(X) > max_train_windows:
            keep = np.sort(rng.choice(len(X), max_train_windows, replace=False))
            X, y, meta = X[keep], y[keep], meta.iloc[keep].reset_index(drop=True)
        data.X[s] = pre.scale_array(X)
        data.y[s] = y.astype(np.int64)
        data.meta[s] = meta
    data.info.update(
        {
            "split_mode": splits["_mode"],
            "split_groups": splits["_groups"],
            "n_records": int(len(df)),
            "n_features": len(feats),
            "windows": data.summary(),
        }
    )
    for s in SPLITS:
        if len(data.y[s]) == 0:
            logger.warning(f"Split '{s}' has no windows (dataset too small for T={T}?).")
    return data
