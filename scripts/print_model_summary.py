"""Parameter / MAC breakdown of LiteCascade for a given input size.

python scripts/print_model_summary.py --variant R --F 115 --K 2
"""

import argparse
import json

from litecascade.eval.profiler import profile_cascade
from litecascade.models.litecascade import LiteCascade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R", choices=["R", "T"])
    ap.add_argument("--F", type=int, default=115)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    args = ap.parse_args()
    model = LiteCascade(F_in=args.F, d=min(args.d, args.F), K=args.K, T_len=args.T, variant=args.variant)
    prof = profile_cascade(model, args.T, args.F, rho=0.8, latency_runs=50)
    print(f"LiteCascade-{args.variant}  F={args.F} T={args.T} d={min(args.d, args.F)} K={args.K}")
    print(json.dumps(prof["params"], indent=2))
    print(
        f"MACs: Sentinel path {prof['macs_sentinel_path']:,} | Analyst incremental {prof['macs_analyst_incremental']:,} "
        f"| full {prof['macs_full']:,}"
    )
    print(f"Analytic speed-up at rho=0.8 (eq. 6): {prof['analytic_speedup_macs']:.2f}x")
    print("Paper design estimates (F=115, T=10): Sentinel exit < 2K params, ~20K total, ~0.06M / ~0.18M MACs.")


if __name__ == "__main__":
    main()
