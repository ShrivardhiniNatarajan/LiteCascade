"""Run the LiteCascade ablation configurations (paper Table IV, the subset implemented here).

python scripts/run_ablations.py --dataset fixture --epochs 10 --out results/ablations_fixture.json
"""

import argparse
import json
import logging
from pathlib import Path

from litecascade.train.run_cascade import main as run_cascade

CONFIGS = [
    ("Full (PIP init + KD + self-KD + temp. scaling + CP gate)", []),
    ("- PIP initialisation (random PIP)", ["--no-pip-init"]),
    ("- teacher KD", ["--no-kd"]),
    ("- self-KD (Analyst->Sentinel)", ["--no-self-kd"]),
    ("- temperature scaling", ["--no-temp-scaling"]),
    ("naive gate (tau=0.9, no T, no CP bound)", ["--naive-gate"]),
    ("conservative CP gate (all benign-exit misses)", ["--conservative-gate"]),
    ("verify alerts (tau_m = never)", ["--verify-alerts"]),
    ("variant T (no BiGRU)", ["--variant", "T"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fixture")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for name, flags in CONFIGS:
        argv = ["--dataset", args.dataset, "--epochs", str(args.epochs), "--seed", str(args.seed), "--tag", "ablation"]
        argv += ["--results", args.results] + flags
        rep = run_cascade(argv)
        t = rep["stages"][-1]["test"]
        rows.append(
            {
                "name": name,
                "flags": flags,
                "run_id": rep["run_id"],
                "cascade_f1": t["cascade"]["f1_macro"],
                "recall": t["cascade"]["recall_malicious"],
                "rho": t["rho_raw"],
                "exit_miss": t["exit_induced_miss_rate"],
                "analyst_f1": t["analyst_only"]["f1_macro"],
                "sentinel_f1": t["sentinel_only"]["f1_macro"],
            }
        )
    out = Path(args.out or f"{args.results}/ablations_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
