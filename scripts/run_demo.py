"""End-to-end demo on the SYNTHETIC fixture (no real dataset needed, CPU only).

1. writes the synthetic fixture (N-BaIoT / CICIoT2023 / IoT-23 layouts) to data/fixture/
2. trains + profiles the teacher and baselines B1-B8
3. trains LiteCascade-R and -T, calibrates the gate, prunes, runs INT8 QAT, re-calibrates,
   and writes deployable bundles to artifacts/fixture_R_int8 and artifacts/fixture_T_int8
4. (optional) runs the ablation configurations
5. writes reports/fixture.md + reports/fixture.json and sample upload CSVs to examples/

    python scripts/run_demo.py            # ~5-10 min on a laptop CPU
    python scripts/run_demo.py --quick    # ~2 min, fewer epochs (for a smoke test)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.pipeline import prepare_data
from litecascade.data.synthetic import ensure_fixture
from litecascade.eval.report import main as report_main
from litecascade.train.run import main as run_baselines
from litecascade.train.run_cascade import main as run_cascade


def write_examples(out_dir="examples", rows_per_segment=120, seed=42):
    """Raw test-split records (devices never seen in training) as upload samples for the dashboard."""
    data = prepare_data("fixture", seed=seed)
    test_groups = data.info["split_groups"]["test"]
    df = load_nbaiot(use_synthetic=True)
    df = df[df["group"].isin(test_groups)]
    parts = [seg.head(rows_per_segment) for _, seg in df.groupby("segment", sort=False)]
    sample = pd.concat(parts)
    feats = data.feature_cols
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    labelled = sample[["segment", *feats, "label_family"]].rename(columns={"label_family": "label"})
    labelled.to_csv(out / "fixture_sample_labeled.csv", index=False, float_format="%.6g")
    labelled.drop(columns=["label"]).to_csv(out / "fixture_sample_unlabeled.csv", index=False, float_format="%.6g")
    print(f"Wrote {out / 'fixture_sample_labeled.csv'} ({len(sample)} rows, devices {test_groups})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ep, bep, tep = (4, 2, 3) if args.quick else (15, 8, 10)
    ensure_fixture()
    seed = ["--seed", str(args.seed)]
    run_baselines(["--model", "all", "--dataset", "fixture", "--epochs", str(bep), "--latency-runs", "200", *seed])
    common = [
        "--dataset",
        "fixture",
        "--epochs",
        str(ep),
        "--teacher-epochs",
        str(tep),
        "--prune",
        "0.3",
        "--quantize",
        "qat",
        *seed,
    ]
    run_cascade(common + ["--variant", "R", "--bundle", "fixture_R_int8"])
    run_cascade(common + ["--variant", "T", "--bundle", "fixture_T_int8"])
    abl = None
    if args.ablations:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "scripts/run_ablations.py", "--dataset", "fixture", "--epochs", str(ep), *seed], check=True
        )
        abl = "results/ablations_fixture.json"
    report_main(["--dataset", "fixture", "--out", "reports/fixture"] + (["--ablations", abl] if abl else []))
    write_examples(seed=args.seed)
    print("\nDone. Start the dashboard with:  python -m litecascade.web.app  (then open http://127.0.0.1:8000)")


if __name__ == "__main__":
    main()
