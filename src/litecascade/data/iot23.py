"""IoT-23 loader (Garcia et al., Stratosphere Lab, 2020) — Zeek ``conn.log.labeled`` files.

Expected layout (the "lighter" IoT-23 release is enough)::

    <root>/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled
    <root>/CTU-Honeypot-Capture-4-1/bro/conn.log.labeled
    ...

Any ``*conn.log.labeled`` file below ``<root>`` is picked up. Each capture is one split group and one
segment; records are ordered by the Zeek ``ts`` field. Numeric Zeek fields are used directly and
``proto`` / ``conn_state`` / ``service`` are one-hot encoded with a fixed vocabulary so every capture
yields the same feature columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .clean import clean_df

DEFAULT_ROOT = Path("data/raw/iot23")

NUMERIC_FIELDS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "id.resp_p",
]  # fmt: skip
PROTO_VOCAB = ["tcp", "udp", "icmp"]
STATE_VOCAB = ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
SERVICE_VOCAB = ["dns", "http", "ssl", "dhcp", "irc", "ssh", "ftp"]


def read_conn_log(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Parse a Zeek conn.log.labeled. The last three fields are space-separated in IoT-23, not tab-separated."""
    fields, rows = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                header = line.split("\t")[1:]
                fields = []
                for h in header:
                    fields.extend(h.split())  # "tunnel_parents   label   detailed-label"
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            tail = parts[-1].split()
            parts = parts[:-1] + tail
            rows.append(parts)
            if max_rows and len(rows) >= max_rows:
                break
    if fields is None:
        raise ValueError(f"{path}: missing '#fields' header; not a Zeek conn log?")
    width = len(fields)
    rows = [r[:width] + ["-"] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=fields)


def conn_log_to_frame(raw: pd.DataFrame, capture: str) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for col in NUMERIC_FIELDS:
        out[col] = pd.to_numeric(raw[col].replace({"-": "0", "(empty)": "0"}), errors="coerce").fillna(0).astype("float32")
    for col, vocab in (("proto", PROTO_VOCAB), ("conn_state", STATE_VOCAB), ("service", SERVICE_VOCAB)):
        vals = raw[col].astype(str)
        for v in vocab:
            out[f"{col}_{v}"] = (vals == v).astype("float32")
    label = raw["label"].astype(str).str.lower()
    detailed = raw.get("detailed-label", pd.Series("-", index=raw.index)).astype(str)
    out["label_binary"] = (label != "benign").astype(int)
    out["label_family"] = np.where(out["label_binary"] == 1, detailed.replace({"-": "malicious"}).str.lower(), "benign")
    out["group"] = capture
    out["segment"] = capture
    ts = pd.to_numeric(raw["ts"], errors="coerce")
    out["timestamp"] = pd.to_datetime(ts, unit="s")
    out["row_order"] = np.arange(len(out))
    return out.sort_values(["timestamp", "row_order"], kind="stable")


def load_iot23(
    use_synthetic: bool = False,
    root: str | Path | None = None,
    max_rows_per_file: int | None = None,
    captures: list[str] | None = None,
) -> pd.DataFrame:
    if use_synthetic:
        from .synthetic import ensure_fixture

        root = ensure_fixture() / "iot23"
    root = Path(root) if root is not None else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"IoT-23 not found at '{root}'. See docs/DATASETS.md.")

    files = sorted(root.rglob("*conn.log.labeled"))
    if not files:
        raise FileNotFoundError(f"No '*conn.log.labeled' files found under '{root}'.")
    frames = []
    for path in files:
        # capture name = first directory below root (e.g. CTU-IoT-Malware-Capture-1-1)
        capture = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else path.stem
        if captures and capture not in captures:
            continue
        frames.append(conn_log_to_frame(read_conn_log(path, max_rows_per_file), capture))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp"])
    df["is_synthetic"] = bool(use_synthetic)
    return clean_df(df)
