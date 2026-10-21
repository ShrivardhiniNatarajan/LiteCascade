"""Shared cleaning + schema helpers used by every dataset loader.

Every loader returns a DataFrame with numeric feature columns plus these metadata columns:

* ``label_binary``  0 = benign, 1 = malicious
* ``label_family``  string family / attack name ("benign" for benign traffic)
* ``group``         split unit (device for N-BaIoT, file for CICIoT2023, capture for IoT-23)
* ``segment``       contiguous record stream; windows never cross segment boundaries
* ``timestamp``     ordering key inside a segment (synthetic monotone time when the source has none)
"""

import numpy as np
import pandas as pd

META_COLS = ["label_binary", "label_family", "group", "segment", "timestamp", "is_synthetic", "row_order"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric, non-metadata columns."""
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def clean_df(df: pd.DataFrame, drop_constant: bool = True) -> pd.DataFrame:
    """Replace +-inf with NaN, drop NaN rows, drop exact duplicates, drop constant feature columns."""
    feats = feature_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats + [c for c in ("label_binary", "group") if c in df.columns])
    df = df.drop_duplicates()
    if drop_constant:
        const = [c for c in feats if df[c].nunique() <= 1]
        df = df.drop(columns=const)
    return df.reset_index(drop=True)


def add_order_columns(df: pd.DataFrame, segment: str, base_time: str = "2000-01-01") -> pd.DataFrame:
    """For sources without timestamps: keep file order as time (one synthetic second per record)."""
    df["segment"] = segment
    df["row_order"] = np.arange(len(df))
    df["timestamp"] = pd.Timestamp(base_time) + pd.to_timedelta(df["row_order"], unit="s")
    return df
