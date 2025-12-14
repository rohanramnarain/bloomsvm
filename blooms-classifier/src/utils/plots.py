from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

from .metrics import LABELS


def plot_confusion(cm: np.ndarray, title: str, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=LABELS, yticklabels=LABELS, ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(labels: List[List[int]], out_path: str | Path) -> None:
    counts = np.array(labels).sum(axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(LABELS, counts)
    ax.set_ylabel("Count")
    ax.set_title("Label distribution")
    plt.xticks(rotation=30)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
