import argparse
import torch
from litecascade.models.litecascade import LiteCascade

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--F", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()
    
    model = LiteCascade(F_in=args.F, d=16, c=24, hidden=24, K=args.K, variant=args.variant)
    
    sentinel_params = sum(p.numel() for p in model.pip.parameters()) + \
                      sum(p.numel() for p in model.stem.parameters()) + \
                      sum(p.numel() for p in model.sentinel_ds.parameters()) + \
                      sum(p.numel() for p in model.sentinel_fc.parameters()) + 1
                      
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Variant: {args.variant}")
    print(f"Sentinel-only Params: {sentinel_params}")
    print(f"Total Params: {total_params} (vs 20K design estimate)")

if __name__ == "__main__":
    main()
