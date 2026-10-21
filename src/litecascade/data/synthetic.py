"""Small deterministic SYNTHETIC fixture written in the native on-disk layouts of the three datasets.

Nothing here is real traffic. The fixture exists so the complete pipeline (loaders -> splits -> windows ->
training -> calibration -> compression -> inference -> dashboard) can run on a laptop without the real
datasets. Any metric computed on it says nothing about real-world detection performance.

Signal design (so models have something to learn, including temporal structure):
* traffic is generated from a 6-d latent AR(1) process mixed into the observed features + noise;
* "flood"-type attacks shift the latent mean strongly (easy, the Sentinel should exit on them);
* "combo"-type attacks shift it moderately;
* "scan"-type attacks have almost no mean shift but an alternating (period-2) pattern over time, which
  only a temporal model can pick up reliably;
* attack intensity varies slowly over time, so part of the malicious windows is low-rate and hard;
* benign traffic contains occasional busy episodes (hard negatives); every device has its own offset.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FIXTURE_DIR = Path(os.environ.get("LITECASCADE_FIXTURE_DIR", "data/fixture"))
FIXTURE_VERSION = "4"

# ----------------------------------------------------------------------------------------------
# Real column names of the public datasets
# ----------------------------------------------------------------------------------------------
_LAMBDAS = ["L5", "L3", "L1", "L0.1", "L0.01"]


def nbaiot_columns() -> list[str]:
    cols = []
    for lam in _LAMBDAS:
        cols += [f"MI_dir_{lam}_{s}" for s in ("weight", "mean", "variance")]
    for lam in _LAMBDAS:
        cols += [f"H_{lam}_{s}" for s in ("weight", "mean", "variance")]
    for lam in _LAMBDAS:
        cols += [f"HH_{lam}_{s}" for s in ("weight", "mean", "std", "magnitude", "radius", "covariance", "pcc")]
    for lam in _LAMBDAS:
        cols += [f"HH_jit_{lam}_{s}" for s in ("weight", "mean", "variance")]
    for lam in _LAMBDAS:
        cols += [f"HpHp_{lam}_{s}" for s in ("weight", "mean", "std", "magnitude", "radius", "covariance", "pcc")]
    assert len(cols) == 115
    return cols


CICIOT_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]  # fmt: skip

N_LATENT = 6

# kind -> (latent mean shift, alternating amplitude)
_ATTACK_PROFILES = {
    "benign": (np.zeros(N_LATENT), 0.0),
    "flood": (np.array([3.5, 2.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
    "combo": (np.array([0.0, 0.0, 2.6, 0.0, 1.2, 0.0]), 0.0),
    "scan": (np.array([0.0, 0.0, 0.0, 0.0, 0.35, 0.0]), 1.6),
}


def _latent_stream(kind: str, n: int, rng: np.random.Generator, device_offset: np.ndarray) -> np.ndarray:
    phi = 0.8
    z = np.zeros((n, N_LATENT))
    eps = rng.normal(0, np.sqrt(1 - phi**2), size=(n, N_LATENT))
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]
    z += device_offset
    shift, alt = _ATTACK_PROFILES[kind]
    if kind == "benign":
        # occasional busy episodes that look a bit like attacks (hard negatives)
        busy = np.zeros(n)
        t = 0
        while t < n:
            if rng.random() < 0.02:
                length = int(rng.integers(10, 40))
                busy[t : t + length] = rng.uniform(0.5, 1.0)
                t += length
            t += 1
        z[:, 0] += busy * 1.1
        z[:, 2] += busy * 0.6
        return z
    # slowly varying attack intensity in [0.6, 1.2]
    u = np.cumsum(rng.normal(0, 0.15, size=n))
    intensity = 0.9 + 0.3 * np.tanh(u - u.mean())
    z += intensity[:, None] * shift[None, :]
    if alt:
        sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        z[:, 3] += alt * intensity * sign
    return z


def _observe(z: np.ndarray, mixing: np.ndarray, scale: np.ndarray, offset: np.ndarray, rng) -> np.ndarray:
    x = z @ mixing + rng.normal(0, 0.6, size=(len(z), mixing.shape[1]))
    return (x * scale + offset).astype(np.float32)


def _mixing(n_features: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mixing = rng.normal(0, 1, size=(N_LATENT, n_features)) / np.sqrt(N_LATENT)
    scale = np.exp(rng.normal(0, 1.5, size=n_features))  # heterogeneous magnitudes like real flow stats
    offset = np.abs(rng.normal(0, 3, size=n_features)) * scale
    return mixing, scale, offset


# ----------------------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------------------
NBAIOT_FILES = {
    "benign_traffic.csv": "benign",
    "mirai_attacks/ack.csv": "flood",
    "mirai_attacks/syn.csv": "flood",
    "gafgyt_attacks/combo.csv": "combo",
    "gafgyt_attacks/scan.csv": "scan",
}


def write_nbaiot_fixture(out: Path, n_devices: int = 10, benign_rows: int = 800, attack_rows: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = nbaiot_columns()
    mixing, scale, offset = _mixing(len(cols), rng)
    for d in range(n_devices):
        device_dir = out / f"Synthetic_Device_{d:02d}"
        dev_off = rng.normal(0, 0.3, size=N_LATENT)
        for rel, kind in NBAIOT_FILES.items():
            n = benign_rows if kind == "benign" else attack_rows
            x = _observe(_latent_stream(kind, n, rng, dev_off), mixing, scale, offset, rng)
            path = device_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(x, columns=cols).to_csv(path, index=False, float_format="%.6g")


_CICIOT_LABELS = {
    "BenignTraffic": "benign",
    "DDoS-ICMP_Flood": "flood",
    "DoS-UDP_Flood": "flood",
    "Mirai-greeth_flood": "flood",
    "Recon-PortScan": "scan",
    "SqlInjection": "combo",
}


def write_ciciot_fixture(out: Path, n_files: int = 3, rows: int = 1500, seed: int = 1):
    rng = np.random.default_rng(seed)
    mixing, scale, offset = _mixing(len(CICIOT_COLUMNS), rng)
    labels = list(_CICIOT_LABELS)
    probs = np.array([0.5] + [0.5 / (len(labels) - 1)] * (len(labels) - 1))
    out.mkdir(parents=True, exist_ok=True)
    for f in range(n_files):
        dev_off = rng.normal(0, 0.3, size=N_LATENT)
        blocks, lab = [], []
        total = 0
        while total < rows:
            label = labels[rng.choice(len(labels), p=probs)]
            n = int(rng.integers(20, 80))
            blocks.append(_observe(_latent_stream(_CICIOT_LABELS[label], n, rng, dev_off), mixing, scale, offset, rng))
            lab += [label] * n
            total += n
        df = pd.DataFrame(np.concatenate(blocks)[:rows], columns=CICIOT_COLUMNS)
        df["label"] = lab[:rows]
        df.to_csv(out / f"part-{f:05d}-synthetic-c000.csv", index=False, float_format="%.6g")


_ZEEK_HEADER = (
    "#separator \\x09\n#set_separator\t,\n#empty_field\t(empty)\n#unset_field\t-\n#path\tconn\n"
    "#open\t2019-01-01-00-00-00\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\t"
    "resp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\t"
    "resp_ip_bytes\ttunnel_parents   label   detailed-label\n"
    "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tbool\tcount\t"
    "string\tcount\tcount\tcount\tcount\tset[string]   string   string\n"
)


def _zeek_record(kind: str, ts: float, rng, i: int) -> str:
    if kind == "benign":
        proto, service, state = rng.choice(["udp", "tcp"]), rng.choice(["dns", "http", "ssl", "-"]), "SF"
        dur, ob, rb = rng.exponential(0.5), int(rng.lognormal(5, 1)), int(rng.lognormal(6, 1))
        op, rp, port = int(rng.integers(1, 12)), int(rng.integers(1, 12)), int(rng.choice([53, 80, 443, 123]))
        label, detail = "Benign", "-"
    elif kind == "scan":
        proto, service, state = "tcp", "-", "S0"
        dur, ob, rb, op, rp, port = rng.exponential(2.0), 0, 0, int(rng.integers(1, 4)), 0, int(rng.integers(1, 65535))
        label, detail = "Malicious", "PartOfAHorizontalPortScan"
    elif kind == "cc":
        proto, service, state = "tcp", "irc", rng.choice(["SF", "S3"])
        dur, ob, rb = rng.normal(30, 2), int(rng.normal(300, 20)), int(rng.normal(500, 30))
        op, rp, port = int(rng.integers(5, 9)), int(rng.integers(5, 9)), 6667
        label, detail = "Malicious", "C&C"
    else:  # ddos
        proto, service, state = rng.choice(["udp", "tcp"]), "-", rng.choice(["S0", "OTH"])
        dur, ob, rb = rng.exponential(0.05), int(rng.lognormal(7, 0.5)), 0
        op, rp, port = int(rng.integers(50, 200)), 0, int(rng.choice([80, 23]))
        label, detail = "Malicious", "DDoS"
    fields = [
        f"{ts:.6f}", f"C{i:010d}", "192.168.100.103", str(int(rng.integers(1024, 65535))), "10.0.0.1", str(port),
        proto, service, f"{max(dur, 0):.6f}", str(max(ob, 0)), str(max(rb, 0)), state, "-", "-", "0", "S",
        str(op), str(op * 40 + max(ob, 0)), str(rp), str(rp * 40 + max(rb, 0)),
    ]  # fmt: skip
    return "\t".join(fields) + f"\t(empty)   {label}   {detail}\n"


_IOT23_CAPTURES = {
    "CTU-IoT-Malware-Capture-1-1": ["benign", "scan"],
    "CTU-IoT-Malware-Capture-3-1": ["benign", "cc"],
    "CTU-IoT-Malware-Capture-7-1": ["benign", "ddos"],
    "CTU-Honeypot-Capture-4-1": ["benign"],
}


def write_iot23_fixture(out: Path, rows: int = 800, seed: int = 2):
    rng = np.random.default_rng(seed)
    for capture, kinds in _IOT23_CAPTURES.items():
        path = out / capture / "bro" / "conn.log.labeled"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = 1.5e9
        lines, i = [_ZEEK_HEADER], 0
        while i < rows:
            kind = kinds[0] if (len(kinds) == 1 or rng.random() < 0.5) else kinds[1]
            for _ in range(int(rng.integers(15, 60))):
                ts += rng.exponential(0.5)
                lines.append(_zeek_record(kind, ts, rng, i))
                i += 1
        lines.append("#close\t2019-01-02-00-00-00\n")
        path.write_text("".join(lines), encoding="utf-8")


def write_fixture(out_dir: str | Path = DEFAULT_FIXTURE_DIR) -> Path:
    out = Path(out_dir)
    write_nbaiot_fixture(out / "nbaiot")
    write_ciciot_fixture(out / "ciciot2023")
    write_iot23_fixture(out / "iot23")
    (out / "VERSION").write_text(FIXTURE_VERSION)
    (out / "README.txt").write_text(
        "SYNTHETIC fixture generated by litecascade.data.synthetic. Not real traffic; results on it are not "
        "evidence of real-world detection performance.\n"
    )
    return out


def ensure_fixture(out_dir: str | Path = DEFAULT_FIXTURE_DIR) -> Path:
    out = Path(out_dir)
    version = out / "VERSION"
    if not version.exists() or version.read_text().strip() != FIXTURE_VERSION:
        write_fixture(out)
    return out
