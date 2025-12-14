from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import textstat

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV"]


def load_spacy():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def compute_features(df: pd.DataFrame, nlp) -> pd.DataFrame:
    feats: Dict[str, list] = {"len_words": [], "len_chars": [], "ari": []}
    for tag in POS_TAGS:
        feats[f"pos_{tag.lower()}"] = []
    for text in df["text"]:
        feats["len_words"].append(len(text.split()))
        feats["len_chars"].append(len(text))
        feats["ari"].append(textstat.automated_readability_index(text) if text else 0.0)
        if nlp:
            doc = nlp(text)
            total = len(doc) or 1
            counts = {tag: 0 for tag in POS_TAGS}
            for token in doc:
                if token.pos_ in counts:
                    counts[token.pos_] += 1
            for tag in POS_TAGS:
                feats[f"pos_{tag.lower()}"].append(counts[tag] / total)
        else:
            for tag in POS_TAGS:
                feats[f"pos_{tag.lower()}"].append(0.0)
    return pd.DataFrame(feats)


def main():
    parser = argparse.ArgumentParser(description="Engineered feature builder")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.train)
    nlp = load_spacy()
    feats = compute_features(df, nlp)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path / "engineered_train.parquet", index=False)
    print(f"Engineered features saved to {out_path}")


if __name__ == "__main__":
    main()
