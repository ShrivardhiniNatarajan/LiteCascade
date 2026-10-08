import argparse
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from litecascade.data.splits import create_group_splits
from litecascade.data.nbaiot import load_nbaiot
from litecascade.data.preprocess import Preprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fixture")
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    # Always load synthetic for 'fixture'
    df = load_nbaiot(use_synthetic=(args.dataset == "fixture"))
    
    splits = create_group_splits(df, dataset_name=args.dataset, ratios={"train": 0.6, "cal": 0.1, "val": 0.1, "test": 0.2})
    
    print("Checking group overlap across splits...")
    overlap_count = 0
    split_names = list(splits.keys())
    for s1, s2 in itertools.combinations(split_names, 2):
        g1 = set(df.loc[splits[s1], "group"])
        g2 = set(df.loc[splits[s2], "group"])
        overlap = g1.intersection(g2)
        print(f"Overlap {s1}/{s2}: {len(overlap)}")
        overlap_count += len(overlap)
        
    if overlap_count == 0:
        print("All overlap counts are 0.")
        
    print("Fitting IPCA on train split to generate explained-variance plot...")
    train_df = df.loc[splits["train"]]
    feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]
    
    preproc = Preprocessor(n_components=min(32, len(feature_cols)))
    preproc.fit(train_df, feature_cols)
    
    out_dir = Path("results/eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.plot(preproc.pca.explained_variance_ratio_.cumsum(), marker="o")
    plt.title("IPCA Explained Variance (Cumulative)")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plot_path = out_dir / "ipca_variance.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved explained-variance plot to {plot_path}")

if __name__ == "__main__":
    main()
