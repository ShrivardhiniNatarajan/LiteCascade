"""Sliding windows of T consecutive records. Windows never cross segment (or group) boundaries.

A window X_t = [x_{t-T+1}, ..., x_t] is labelled with the label of its LAST record, so a window can be
classified as soon as its last record arrives (causal, no look-ahead).
"""

import numpy as np
import pandas as pd


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    T: int = 10,
    stride: int = 1,
    label_col: str = "label_binary",
    segment_col: str | None = None,
    return_meta: bool = False,
    dtype=np.float32,
):
    """Return (X, y) with X of shape (N, T, F); with ``return_meta`` also a DataFrame describing each window.

    ``segment_col`` defaults to "segment" if present, else "group". Records inside a segment are ordered
    by ("timestamp", "row_order") when available.
    """
    if segment_col is None:
        segment_col = "segment" if "segment" in df.columns else "group"
    sort_cols = [c for c in ("timestamp", "row_order") if c in df.columns]

    xs, ys, metas = [], [], []
    for seg, seg_df in df.groupby(segment_col, sort=False):
        if sort_cols:
            seg_df = seg_df.sort_values(sort_cols, kind="stable")
        n = len(seg_df)
        if n < T:
            continue
        feats = seg_df[feature_cols].to_numpy(dtype=dtype)
        starts = np.arange(0, n - T + 1, stride)
        win = np.lib.stride_tricks.sliding_window_view(feats, (T, feats.shape[1]))[:, 0]  # (n-T+1, T, F)
        xs.append(win[starts])
        ends = starts + T - 1
        if label_col in seg_df.columns:
            ys.append(seg_df[label_col].to_numpy()[ends])
        if return_meta:
            meta = pd.DataFrame({"segment": seg, "end_row": seg_df.index.to_numpy()[ends]})
            for c in ("group", "label_family", "timestamp"):
                if c in seg_df.columns:
                    meta[c] = seg_df[c].to_numpy()[ends]
            metas.append(meta)

    n_feat = len(feature_cols)
    X = np.concatenate(xs) if xs else np.zeros((0, T, n_feat), dtype=dtype)
    y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    if return_meta:
        meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame(columns=["segment", "end_row"])
        return X, y, meta
    return X, y
