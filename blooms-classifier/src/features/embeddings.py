from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.io import ensure_dir


def encode(texts, model, batch_size: int, device: str | None):
    return model.encode(texts, batch_size=batch_size, device=device or model.device, show_progress_bar=True, convert_to_numpy=True)


def main():
    parser = argparse.ArgumentParser(description="Sentence embedding cache")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/cache")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache", type=str, default="data/cache")
    args = parser.parse_args()

    df = pd.read_parquet(args.train)
    ensure_dir(args.cache)
    model = SentenceTransformer(args.model, device=args.device or "cpu")

    embeddings = encode(df["text"].tolist(), model, args.batch, args.device)
    out_dir = Path(args.cache)
    np.save(out_dir / "train_embeddings.npy", embeddings)
    (out_dir / "train_index.txt").write_text("\n".join(df["id"].astype(str)), encoding="utf-8")
    print(f"Saved embeddings to {out_dir}")


if __name__ == "__main__":
    main()
