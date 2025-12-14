from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet
from src.utils.verbs import ALL_BLOOM_LEVELS


def main():
    parser = argparse.ArgumentParser(description="Normalize dataset schema")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = read_parquet(args.input)
    for col in ALL_BLOOM_LEVELS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    if "text" not in df.columns:
        raise SystemExit("input must contain text column")
    df["domain"] = df.get("domain", "unknown")
    df["source"] = df.get("source", "unknown")
    df["synthetic_flag"] = df.get("synthetic_flag", True)
    df["rationale"] = df.get("rationale", "")

    write_parquet(df, args.output)
    print(f"Prepared dataset -> {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
