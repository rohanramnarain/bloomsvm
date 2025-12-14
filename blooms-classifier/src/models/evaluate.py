from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.metrics import LABELS, compute_metrics, threshold_probs
from src.utils.plots import plot_confusion


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--truth", type=str, required=True)
    parser.add_argument("--preds", type=str, default=None, help="npy file of probabilities or HF model dir to run inference")
    parser.add_argument("--out", type=str, default="outputs/metrics")
    parser.add_argument("--threshold", type=float, default=0.5, help="single threshold if --thresholds not provided")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None, help="optional list of thresholds to sweep")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-probs", type=str, default=None, help="optional path to save computed probabilities npy")
    args = parser.parse_args()

    df = pd.read_parquet(args.truth)
    y_true = df[LABELS].values.astype(int)

    def infer_probs(model_dir: Path) -> np.ndarray:
        device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
        probs_list: List[np.ndarray] = []
        texts = df["text"].tolist()
        bs = args.batch
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch_texts = texts[i : i + bs]
                enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
        return np.vstack(probs_list)

    y_prob: np.ndarray
    preds_path = Path(args.preds) if args.preds else None
    if preds_path and preds_path.exists():
        if preds_path.is_dir() and (preds_path / "config.json").exists():
            y_prob = infer_probs(preds_path)
        else:
            y_prob = np.load(preds_path)
    else:
        y_prob = np.zeros_like(y_true, dtype=float)

    if args.save_probs:
        Path(args.save_probs).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs, y_prob)

    thresholds: Iterable[float] = args.thresholds if args.thresholds else [args.threshold]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for thr in thresholds:
        y_pred = threshold_probs(y_prob, threshold=thr)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics_path = out_dir / f"metrics_threshold_{thr:.2f}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "macro_f1": metrics.macro_f1,
                    "kappa": metrics.kappa,
                    "accuracy": metrics.accuracy,
                    "per_label_f1": metrics.per_label_f1,
                    "per_label_auc": metrics.per_label_auc,
                    "threshold": thr,
                },
                f,
                indent=2,
            )
        summary[f"{thr:.2f}"] = metrics.macro_f1
        plot_confusion(metrics.confusion, f"Confusion@{thr:.2f}", out_dir / f"confusion_{thr:.2f}.png")

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    out_dir = Path(args.out)
    print(f"Saved metrics to {out_dir}")


if __name__ == "__main__":
    main()
