from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.utils.metrics import LABELS


def load_labels(path: str) -> np.ndarray:
    df = pd.read_parquet(path)
    return df[LABELS].values.astype(int), df


def load_features(kind: str, base_path: str):
    if kind == "tfidf":
        mat = sparse.load_npz(Path(base_path) / "train.npz")
        return mat
    if kind == "embeddings":
        return np.load(Path(base_path) / "train_embeddings.npy")
    raise ValueError("Unknown features kind")


def main():
    parser = argparse.ArgumentParser(description="Train SVM models")
    parser.add_argument("--features", type=str, choices=["tfidf", "embeddings"], required=True)
    parser.add_argument("--kernel", type=str, choices=["linear", "rbf"], default="linear")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--labels", type=str, default="data/processed/train.parquet")
    args = parser.parse_args()

    y, df = load_labels(args.labels)
    X = load_features(args.features, "data/processed/tfidf" if args.features == "tfidf" else "data/cache")
    if args.features == "embeddings":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    if args.kernel == "linear":
        base_model = LinearSVC(class_weight="balanced")
        grid = {"C": [0.5, 1.0, 2.0]}
    else:
        base_model = SVC(kernel="rbf", class_weight="balanced", probability=False)
        grid = {"C": [0.5, 1.0], "gamma": ["scale", 0.1]}

    best_score = -1
    best_model = None
    scorer = make_scorer(lambda yt, yp: f1_score(yt, yp, average="macro"))
    strat = KFold(n_splits=3, shuffle=True, random_state=42)

    for params in (dict(zip(grid.keys(), values)) for values in __import__("itertools").product(*grid.values())):
        model = OneVsRestClassifier(base_model.__class__(**params))
        scores = cross_val_score(model, X, y, cv=strat, scoring=scorer)
        score = scores.mean()
        if score > best_score:
            best_score = score
            best_model = model

    best_model.fit(X, y)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler}, out_dir / "svm.joblib")
    print(f"Best macro-F1 {best_score:.3f}; model saved to {out_dir}")


if __name__ == "__main__":
    main()
