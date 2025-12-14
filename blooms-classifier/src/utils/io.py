import json
import jsonlines
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def utcnow() -> str:
    return datetime.utcnow().isoformat()


def set_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = int(os.getenv("PYTHONHASHSEED", "0")) or 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def pick_device(prefer: str | None = None):
    try:
        import torch

        if prefer == "cpu":
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return "cpu"


def human_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
