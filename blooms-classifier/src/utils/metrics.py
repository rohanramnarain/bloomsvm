from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)


@dataclass
class MetricsResult:
    macro_f1: float
    kappa: float
    accuracy: float
    per_label_f1: Dict[str, float]
    per_label_auc: Dict[str, float]
    confusion: np.ndarray


LABELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]


def cohen_kappa_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1:
        return cohen_kappa_score(y_true, y_pred)
    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()
    return cohen_kappa_score(true_flat, pred_flat)


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for i, label in enumerate(LABELS):
        try:
            scores[label] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            scores[label] = float("nan")
    return scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> MetricsResult:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_label_f1 = {label: f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i, label in enumerate(LABELS)}
    kappa = cohen_kappa_multi(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    per_label_auc = macro_auc(y_true, y_prob if y_prob is not None else y_pred)
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1)) if y_true.ndim > 1 else confusion_matrix(y_true, y_pred)
    return MetricsResult(macro_f1, kappa, acc, per_label_f1, per_label_auc, cm)


def threshold_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(int)


def to_numpy(labels: List[List[int]] | np.ndarray) -> np.ndarray:
    arr = np.array(labels)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(int)
