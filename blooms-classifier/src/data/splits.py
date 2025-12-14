from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import write_parquet
from src.utils.verbs import ALL_BLOOM_LEVELS


def pick_stratify(df: pd.DataFrame) -> np.ndarray:
    labels = df[ALL_BLOOM_LEVELS].values
    # fallback: use argmax; if all zero, assign -1
    max_idx = labels.argmax(axis=1)
    all_zero = labels.sum(axis=1) == 0
    max_idx[all_zero] = -1
    return max_idx


def main():
    parser = argparse.ArgumentParser(description="Create stratified splits")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    strat = pick_stratify(df)

    train_df, temp_df = train_test_split(df, test_size=args.val + args.test, stratify=strat, random_state=args.seed)
    relative_test = args.test / (args.val + args.test)
    strat_temp = pick_stratify(temp_df)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, stratify=strat_temp, random_state=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_df, out_dir / "train.parquet")
    write_parquet(val_df, out_dir / "val.parquet")
    write_parquet(test_df, out_dir / "test.parquet")
    print(f"Splits written to {out_dir}")


if __name__ == "__main__":
    main()
