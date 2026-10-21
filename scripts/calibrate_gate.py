"""Re-calibrate the gate of an existing bundle (e.g. with a different epsilon / alpha / target FPR).

python scripts/calibrate_gate.py --bundle artifacts/fixture_R_int8 --epsilon 0.01 --alpha 0.05 [--write]
"""

import argparse
import json

from litecascade.data.pipeline import prepare_data
from litecascade.deploy.bundle import Bundle
from litecascade.train.cascade_trainer import calibrate_model, evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--verify-alerts", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=42, help="must match the seed used for training (same splits)")
    ap.add_argument("--write", action="store_true", help="store the new gate in the bundle")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    data = prepare_data(b.meta["dataset"], root=args.data_root, T=b.window, seed=args.seed, task=b.meta["task"])
    gate = calibrate_model(
        b.model, data, epsilon=args.epsilon, alpha=args.alpha, target_fpr=args.target_fpr, verify_alerts=args.verify_alerts
    )
    rep = evaluate_model(b.model, data, gate)
    print(json.dumps({"gate": gate, "test": {k: rep[k] for k in ("rho_raw", "exit_induced_miss_rate", "cascade")}}, indent=2))
    if args.write:
        b.meta["gate"] = gate
        (b.path / "bundle.json").write_text(json.dumps(b.meta, indent=2, default=float))
        print(f"Updated {b.path / 'bundle.json'}")


if __name__ == "__main__":
    main()
