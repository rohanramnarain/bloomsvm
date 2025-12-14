from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import pandas as pd
from rapidfuzz import fuzz

from src.utils.io import load_jsonl, write_parquet
from src.utils.verbs import ALL_BLOOM_LEVELS

NSFW = {"kill", "suicide", "bomb", "hate", "racist", "nsfw"}


def is_toxic(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in NSFW)


def dedupe(rows: List[dict], max_sim: float) -> List[dict]:
    kept: List[dict] = []
    texts: List[str] = []
    for row in rows:
        text = row["text"].strip()
        if any(fuzz.partial_ratio(text, prev) / 100.0 > max_sim for prev in texts):
            continue
        texts.append(text)
        kept.append(row)
    return kept


def balance(df: pd.DataFrame, max_multiplier: float = 1.5) -> pd.DataFrame:
    label_cols = [col for col in df.columns if col in ALL_BLOOM_LEVELS]
    counts = {col: df[col].sum() for col in label_cols}
    max_count = max(counts.values()) if counts else 0
    frames = [df]
    for col in label_cols:
        deficit = int(min(max_count * max_multiplier, max_count) - counts[col])
        if deficit <= 0:
            continue
        samples = df[df[col] == 1]
        if samples.empty:
            continue
        sampled = samples.sample(n=deficit, replace=True, random_state=42)
        frames.append(sampled)
    return pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Deduplicate and clean dataset")
    parser.add_argument("--in", dest="input_path", type=str, required=True)
    parser.add_argument("--out", dest="output_path", type=str, required=True)
    parser.add_argument("--max-dup-sim", type=float, default=0.92)
    parser.add_argument("--min-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=320)
    args = parser.parse_args()

    rows = []
    for path in Path().glob(args.input_path):
        rows.extend(load_jsonl(path))

    rows = [r for r in rows if args.min_len <= len(r.get("text", "")) <= args.max_len]
    rows = [r for r in rows if not is_toxic(r.get("text", ""))]
    rows = dedupe(rows, args.max_dup_sim)

    if not rows:
        raise SystemExit("No rows after cleaning; adjust thresholds.")

    records = []
    for r in rows:
        labels = r.get("labels") or [0] * len(ALL_BLOOM_LEVELS)
        rec = {
            "id": r.get("id"),
            "text": r.get("text"),
            "domain": r.get("domain", "unknown"),
            "source": r.get("source", "unknown"),
            "synthetic_flag": r.get("synthetic_flag", False),
            "rationale": r.get("rationale", ""),
        }
        for name, val in zip(ALL_BLOOM_LEVELS, labels):
            rec[name] = int(val)
        records.append(rec)

    df = pd.DataFrame(records)
    df = balance(df)
    write_parquet(df, args.output_path)
    print(f"Cleaned {len(df)} rows -> {args.output_path}")


if __name__ == "__main__":
    main()
