from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.metrics import LABELS, compute_metrics, threshold_probs
from src.utils.plots import plot_confusion


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--truth", type=str, required=True)
    parser.add_argument("--preds", type=str, default=None, help="npy file of probabilities")
    parser.add_argument("--out", type=str, default="outputs/metrics")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_parquet(args.truth)
    y_true = df[LABELS].values.astype(int)

    if args.preds and Path(args.preds).exists():
        y_prob = np.load(args.preds)
    else:
        y_prob = np.zeros_like(y_true, dtype=float)
    y_pred = threshold_probs(y_prob, threshold=args.threshold)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro_f1": metrics.macro_f1,
                "kappa": metrics.kappa,
                "accuracy": metrics.accuracy,
                "per_label_f1": metrics.per_label_f1,
                "per_label_auc": metrics.per_label_auc,
            },
            f,
            indent=2,
        )

    plot_confusion(metrics.confusion, "Confusion", out_dir / "confusion.png")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
