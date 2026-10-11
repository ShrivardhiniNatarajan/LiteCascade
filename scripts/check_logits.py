import argparse
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    path = f"checkpoints/teacher_logits_{args.dataset}.pt"
    if not Path(path).exists():
        print("Logits file not found!")
        exit(1)
        
    data = torch.load(path, map_location="cpu", weights_only=False)
    print("Train logits shape:", data["train"].shape)
    print("Calib logits shape:", data["calib"].shape)
    
if __name__ == "__main__":
    main()
