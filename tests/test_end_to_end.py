"""End-to-end: train a tiny cascade on the synthetic fixture, compress it, bundle it, serve it."""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from litecascade.data.nbaiot import load_nbaiot
from litecascade.deploy.bundle import Bundle
from litecascade.deploy.predict import predict_dataframe
from litecascade.train.run_cascade import main as run_cascade


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    report = run_cascade(
        [
            "--dataset", "fixture", "--epochs", "1", "--no-kd", "--prune", "0.3", "--finetune-epochs", "1",
            "--quantize", "qat", "--qat-epochs", "1", "--results", str(tmp / "results"),
            "--artifacts", str(tmp / "artifacts"), "--bundle", "tiny",
        ]
    )  # fmt: skip
    from litecascade.train.run import main as run_baselines

    for arch in ("b7", "b10"):  # two deep-learning baselines as bundles, 1 epoch each
        run_baselines(
            ["--model", arch, "--dataset", "fixture", "--epochs", "1", "--latency-runs", "5", "--results", str(tmp / "results"),
             "--checkpoints", str(tmp / "ck"), "--bundles", str(tmp / "artifacts")]
        )  # fmt: skip
    return tmp, report


def test_report_has_all_stages_and_live_inference_agrees(trained):
    _, report = trained
    names = [s["stage"] for s in report["stages"]]
    assert names == ["joint_kd", "pruned_0.3", "int8_qat_float_gate", "int8_qat_recalibrated"]
    assert report["data_kind"].startswith("SYNTHETIC")
    for st in report["stages"]:
        live = st["test"]["live_inference"]
        # dynamic-INT8 activation scales depend on batch contents, so INT8 stages may differ on a few
        # windows sitting exactly at a threshold; float stages must agree exactly
        tol = 1.0 if st["quantization"] == "none" else 0.99
        assert live["agreement_with_offline"] >= tol and live["exit_agreement"] >= tol
        offline = round((1 - st["test"]["rho_raw"]) * live["windows"])
        assert abs(live["analyst_invocations"] - offline) <= (1 - tol) * live["windows"]
    assert report["stages"][-1]["profile"]["size_kb_serialized"] < report["stages"][0]["profile"]["size_kb_serialized"]


def _sample_csv():
    df = load_nbaiot(use_synthetic=True)
    df = df[df["group"] == df["group"].iloc[0]]
    return df.groupby("segment").head(40)


def test_bundle_predicts_raw_csv(trained):
    tmp, _ = trained
    bundle = Bundle(tmp / "artifacts" / "tiny")
    assert bundle.meta["quantization"] == "qat" and bundle.meta["is_synthetic"]
    raw = _sample_csv()
    upload = raw[["segment", *bundle.feature_cols, "label_family"]].rename(columns={"label_family": "label"})
    res, summary = predict_dataframe(bundle, upload)
    assert summary["windows"] == len(res) == 5 * (40 - bundle.window + 1)
    assert summary["analyst_invocations"] == (res["exit_stage"] == "analyst").sum()
    assert "metrics" in summary and set(res["prediction"]) <= {"benign", "malicious"}
    with pytest.raises(ValueError, match="missing"):
        predict_dataframe(bundle, upload.drop(columns=bundle.feature_cols[:3]))


def test_baseline_bundle_predicts(trained):
    tmp, _ = trained
    b = Bundle(tmp / "artifacts" / "fixture_b10")
    assert b.kind == "baseline" and b.display_name == "B10 TCN" and b.gate is None
    raw = _sample_csv()
    upload = raw[["segment", *b.feature_cols, "label_family"]].rename(columns={"label_family": "label"})
    res, summary = predict_dataframe(b, upload)
    assert summary["windows"] == len(res) and set(res["exit_stage"]) == {"single"} and summary["analyst_invocations"] == 0


def test_web_api(trained, monkeypatch, tmp_path):
    tmp, _ = trained
    import litecascade.web.app as webapp

    monkeypatch.setattr(webapp, "ARTIFACTS", tmp / "artifacts")
    monkeypatch.setattr(webapp, "EXAMPLES", tmp_path)
    monkeypatch.setattr(webapp, "DOC_DIRS", [tmp_path])
    monkeypatch.setattr(webapp, "S", webapp._State())
    (tmp_path / "r.md").write_text("# Report title\n\nbody")
    client = TestClient(webapp.app)
    assert client.get("/").status_code == 200
    models = client.get("/api/models").json()
    ids = {m["id"] for m in models}
    assert ids == {"tiny", "fixture_b7", "fixture_b10"} and all(m["is_synthetic"] for m in models)
    assert next(m for m in models if m["id"] == "tiny")["kind"] == "cascade"

    # live replay: predictions come from the models; labels are only used for scoring
    for _ in range(6):
        out = client.post("/api/live/control", json={"action": "step"}).json()
    ev, summ = out["event"], out["summary"]
    assert set(ev["outputs"]) == ids and "exit" in ev["outputs"]["tiny"] and summ["steps"] == 6
    assert all(summ["models"][m]["n"] == 6 for m in ids)
    fam = summ["families"][0]
    assert client.post("/api/live/inject", json={"family": fam, "windows": 5}).status_code == 200
    nxt = client.post("/api/live/control", json={"action": "step"}).json()["event"]
    assert nxt["family"] == fam and nxt["label"] == 1
    assert client.post("/api/live/inject", json={"family": "nope"}).status_code == 400
    assert client.post("/api/live/models", json={"active": ["fixture_b7"]}).json()["active"] == ["tiny", "fixture_b7"]

    curve = client.get("/api/gate/curve").json()["curve"]
    assert len(curve) > 5 and all(0 <= p["rho"] <= 1 for p in curve)
    pt = client.get("/api/gate/point", params={"epsilon": 0.01}).json()
    assert pt["feasible"] in (True, False) and client.get("/api/gate/point", params={"epsilon": 2}).status_code == 400
    assert client.get("/api/architecture").json()["params"]["total"] > 0

    raw = _sample_csv()
    feats = Bundle(tmp / "artifacts" / "tiny").feature_cols
    csv = raw[["segment", *feats, "label_family"]].rename(columns={"label_family": "label"}).to_csv(index=False)
    r = client.post("/api/analyze", files={"file": ("x.csv", csv, "text/csv")}, data={"models": "tiny,fixture_b10"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["models"] == ["tiny", "fixture_b10"] and all("metrics" in s for s in body["summaries"].values())
    dl = pd.read_csv(io.StringIO(client.get(f"/api/download/{body['job_id']}").text))
    assert {"tiny:prediction", "fixture_b10:prediction", "true_label"} <= set(dl.columns)
    assert len(dl) == body["summaries"]["tiny"]["windows"]

    (tmp_path / "s.csv").write_text(csv)
    assert client.post("/api/analyze_sample", data={"sample": "s.csv", "models": "fixture_b7"}).status_code == 200
    bad = client.post("/api/analyze", files={"file": ("x.csv", "a,b\n1,2\n", "text/csv")}, data={"models": "tiny"})
    assert bad.status_code == 400
    assert client.post("/api/analyze_sample", data={"sample": "../secret.csv", "models": "tiny"}).status_code == 404
    assert client.get("/api/docs").json() == [{"name": "r.md", "title": "Report title"}]
    assert client.get("/api/docs/r.md").text.startswith("# Report")
    assert client.get("/api/docs/secret.md").status_code == 404
    assert client.get("/api/does-not-exist").status_code == 404
