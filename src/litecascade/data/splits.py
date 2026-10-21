"""Leakage-safe splits.

* ``create_group_splits``: whole groups (devices / files / captures) go to exactly one split.
* ``create_time_splits``: for datasets with too few groups, every segment is cut into contiguous
  chronological blocks (train -> val -> calib -> test) with a purge gap of ``gap`` records between blocks,
  so no sliding window can contain records from two splits.
* ``create_lofo_split``: leave-one-family-out (zero-day) split.

The calibration split is disjoint from train, val and test whenever the ratios include "calib".
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RATIOS = {"train": 0.5, "val": 0.1, "calib": 0.2, "test": 0.2}


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Number of groups per split; every split gets at least one group when n allows it."""
    names = list(ratios)
    total = sum(ratios.values())
    counts = {k: max(1, int(round(n * ratios[k] / total))) for k in names}
    # fix rounding so counts sum to n, taking from / giving to the largest split
    while sum(counts.values()) > n:
        k = max((k for k in names if counts[k] > 1), key=lambda k: counts[k], default=None)
        if k is None:
            break
        counts[k] -= 1
    while sum(counts.values()) < n:
        counts[names[0]] += 1
    return counts


def create_group_splits(df: pd.DataFrame, dataset_name: str = "", ratios: dict | None = None, seed: int = 42, tries: int = 50):
    """Assign whole groups to splits. Returns {split: [row index labels]}.

    Shuffles groups (seeded) and prefers an assignment where every split contains both classes.
    If ``ratios`` has no "calib" key the calibration split falls back to the validation split.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    groups = np.array(sorted(df["group"].unique(), key=str), dtype=object)
    if len(groups) < len(ratios):
        raise ValueError(f"{dataset_name}: {len(groups)} groups < {len(ratios)} splits; use create_time_splits instead")
    counts = _allocate(len(groups), ratios)
    has_label = "label_binary" in df.columns
    group_classes = df.groupby("group")["label_binary"].agg(lambda s: set(s.unique())) if has_label else None

    rng = np.random.default_rng(seed)
    best, best_score = None, -1
    for _ in range(tries):
        perm = rng.permutation(groups)
        assign, start = {}, 0
        for name in ratios:
            assign[name] = list(perm[start : start + counts[name]])
            start += counts[name]
        if not has_label:
            best = assign
            break
        score = sum(len(set().union(*(group_classes[g] for g in assign[name]))) >= 2 for name in ratios)
        if score > best_score:
            best, best_score = assign, score
        if score == len(ratios):
            break

    splits = {name: df.index[df["group"].isin(best[name])].tolist() for name in ratios}
    if "calib" not in splits:
        if "val" in splits:
            logger.warning("No 'calib' ratio given: calibration falls back to the validation split.")
            splits["calib"] = splits["val"]
    splits["_groups"] = {k: [str(g) for g in v] for k, v in best.items()}
    return splits


def create_time_splits(df: pd.DataFrame, ratios: dict | None = None, gap: int = 10, segment_col: str = "segment"):
    """Chronological per-segment blocks with ``gap`` purged records between consecutive blocks."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    names = list(ratios)
    total = sum(ratios.values())
    cum = np.cumsum([ratios[k] / total for k in names])
    seg_col = segment_col if segment_col in df.columns else "group"
    out = {k: [] for k in names}
    for _, seg in df.groupby(seg_col, sort=False):
        order = seg.sort_values("timestamp", kind="stable").index.to_numpy() if "timestamp" in seg else seg.index.to_numpy()
        n = len(order)
        bounds = [0] + [int(round(c * n)) for c in cum]
        for i, name in enumerate(names):
            lo = bounds[i] + (gap if i > 0 else 0)
            hi = bounds[i + 1]
            if hi > lo:
                out[name].extend(order[lo:hi].tolist())
    if "calib" not in out and "val" in out:
        out["calib"] = out["val"]
    out["_groups"] = {"mode": "time", "gap": gap}
    return out


def make_splits(df: pd.DataFrame, mode: str = "auto", ratios: dict | None = None, seed: int = 42, gap: int = 10):
    """``mode``: "group", "time" or "auto" (group split if there are enough groups, else time split)."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    if mode == "auto":
        mode = "group" if df["group"].nunique() >= max(len(ratios), 4) else "time"
    if mode == "group":
        splits = create_group_splits(df, ratios=ratios, seed=seed)
    elif mode == "time":
        splits = create_time_splits(df, ratios=ratios, gap=gap)
    else:
        raise ValueError(f"Unknown split mode {mode}")
    splits["_mode"] = mode
    return splits


def create_lofo_split(df: pd.DataFrame, target_family: str, dataset_name: str = ""):
    """Leave-one-family-out: ``target_family`` never appears in train; test holds that family only.

    Benign records are kept in train (the model must still learn what benign looks like); evaluate the
    held-out family together with a disjoint set of benign records for a meaningful zero-day test.
    """
    train_idx = df[df["label_family"] != target_family].index.tolist()
    test_idx = df[df["label_family"] == target_family].index.tolist()
    return {"train": train_idx, "test": test_idx}
