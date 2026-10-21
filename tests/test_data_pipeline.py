import numpy as np
import pandas as pd
import pytest

from litecascade.data.ciciot2023 import label_to_category, load_ciciot2023
from litecascade.data.iot23 import load_iot23, read_conn_log
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.pipeline import prepare_data
from litecascade.data.splits import create_group_splits, create_time_splits
from litecascade.data.synthetic import nbaiot_columns, write_iot23_fixture, write_nbaiot_fixture
from litecascade.data.windows import create_windows


def test_nbaiot_loader_reads_uci_and_kaggle_layouts(tmp_path):
    write_nbaiot_fixture(tmp_path / "uci", n_devices=2, benign_rows=30, attack_rows=20)
    df = load_nbaiot(root=tmp_path / "uci")
    assert df["group"].nunique() == 2 and df["segment"].nunique() == 10
    assert set(df["label_family"]) == {"benign", "mirai_ack", "mirai_syn", "gafgyt_combo", "gafgyt_scan"}
    assert len([c for c in df.columns if c in nbaiot_columns()]) == 115

    flat = tmp_path / "kaggle"
    flat.mkdir()
    cols = nbaiot_columns()
    rng = np.random.default_rng(0)
    for name in ("1.benign.csv", "1.mirai.ack.csv", "2.gafgyt.scan.csv"):
        pd.DataFrame(rng.normal(size=(15, 115)), columns=cols).to_csv(flat / name, index=False)
    df2 = load_nbaiot(root=flat)
    assert set(df2["group"]) == {"device_1", "device_2"}
    assert set(df2["label_family"]) == {"benign", "mirai_ack", "gafgyt_scan"}
    assert df2.groupby("label_family")["label_binary"].first().to_dict() == {"benign": 0, "gafgyt_scan": 1, "mirai_ack": 1}


def test_missing_dataset_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATASETS.md"):
        load_nbaiot(root=tmp_path / "nope")


def test_iot23_parser_handles_space_separated_tail(tmp_path):
    write_iot23_fixture(tmp_path, rows=60)
    raw = read_conn_log(next(tmp_path.rglob("conn.log.labeled")))
    assert {"label", "detailed-label", "tunnel_parents"} <= set(raw.columns)
    assert set(raw["label"]) <= {"Benign", "Malicious"}
    df = load_iot23(root=tmp_path)
    assert df["group"].nunique() == 4
    assert df.groupby("segment")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_ciciot_categories():
    assert label_to_category("BenignTraffic") == "benign"
    assert label_to_category("DDoS-ICMP_Flood") == "ddos"
    assert label_to_category("DoS-UDP_Flood") == "dos"
    assert label_to_category("Recon-PortScan") == "recon"
    assert label_to_category("SqlInjection") == "web"
    assert label_to_category("MITM-ArpSpoofing") == "spoofing"
    df = load_ciciot2023(use_synthetic=True)
    assert df["label_binary"].between(0, 1).all() and "benign" in set(df["label_family"])


def test_group_splits_have_disjoint_calibration():
    df = pd.DataFrame({"group": np.repeat([f"g{i}" for i in range(10)], 5), "label_binary": np.tile([0, 1, 0, 1, 1], 10)})
    s = create_group_splits(df, seed=0)
    names = ["train", "val", "calib", "test"]
    groups = {n: set(df.loc[s[n], "group"]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert groups[a].isdisjoint(groups[b]), (a, b)
    assert sum(len(s[n]) for n in names) == len(df)


def test_time_splits_purge_gap_prevents_window_overlap():
    n, T = 200, 10
    df = pd.DataFrame(
        {
            "segment": "s",
            "group": "g",
            "timestamp": pd.date_range("2026", periods=n, freq="s"),
            "f": np.arange(n),
            "label_binary": 0,
        }
    )
    s = create_time_splits(df, gap=T)
    spans = {k: (min(df.loc[s[k], "f"]), max(df.loc[s[k], "f"])) for k in ("train", "val", "calib", "test")}
    order = ["train", "val", "calib", "test"]
    for a, b in zip(order, order[1:], strict=False):
        assert spans[b][0] - spans[a][1] > T  # a window of T records can never span two splits


def test_windows_respect_segments_and_meta():
    df = pd.DataFrame(
        {"segment": ["a"] * 4 + ["b"] * 3, "f": range(7), "label_binary": [0, 0, 0, 1, 1, 1, 0], "row_order": range(7)}
    )
    X, y, meta = create_windows(df, ["f"], T=3, return_meta=True)
    assert X.shape == (3, 3, 1)
    assert X[:, :, 0].tolist() == [[0, 1, 2], [1, 2, 3], [4, 5, 6]]
    assert y.tolist() == [0, 1, 0]
    assert meta["end_row"].tolist() == [2, 3, 6]


def test_prepare_data_is_leakage_safe():
    data = prepare_data("fixture", T=10)
    g = data.info["split_groups"]
    assert set(g["train"]).isdisjoint(g["test"]) and set(g["calib"]).isdisjoint(g["train"] + g["val"] + g["test"])
    # scaler statistics equal those of the TRAIN records only
    df = load_nbaiot(use_synthetic=True)
    train = df[df["group"].isin(g["train"])]
    med = train[data.feature_cols].median().to_numpy()
    assert np.allclose(data.preprocessor.scaler.center_, med, rtol=1e-4, atol=1e-4)
    for s in ("train", "val", "calib", "test"):
        assert data.X[s].shape[1:] == (10, data.n_features) and len(data.X[s]) == len(data.y[s]) > 0
    # enough malicious calibration windows for eps = 0.5%
    assert data.summary()["calib"]["malicious"] >= 598
