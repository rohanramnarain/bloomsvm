from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(min_df: int = 2):
    word_v = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    char_v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)
    return word_v, char_v


def main():
    parser = argparse.ArgumentParser(description="TF-IDF feature builder")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, default="data/processed/val.parquet")
    parser.add_argument("--test", type=str, default="data/processed/test.parquet")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val) if Path(args.val).exists() else None
    test_df = pd.read_parquet(args.test) if Path(args.test).exists() else None

    word_v, char_v = build_vectorizer(args.min_df)

    word_v.fit(train_df["text"])
    char_v.fit(train_df["text"])

    train_mat = sparse.hstack([word_v.transform(train_df["text"]), char_v.transform(train_df["text"])]).tocsr()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "train.npz", train_mat)
    joblib.dump((word_v, char_v), out_dir / "vectorizers.joblib")

    if val_df is not None:
        val_mat = sparse.hstack([word_v.transform(val_df["text"]), char_v.transform(val_df["text"])]).tocsr()
        sparse.save_npz(out_dir / "val.npz", val_mat)
    if test_df is not None:
        test_mat = sparse.hstack([word_v.transform(test_df["text"]), char_v.transform(test_df["text"])]).tocsr()
        sparse.save_npz(out_dir / "test.npz", test_mat)
    print(f"TF-IDF features saved to {out_dir}")


if __name__ == "__main__":
    main()
