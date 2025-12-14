from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz

from src.utils.io import load_jsonl, write_jsonl
from src.utils.verbs import ALL_BLOOM_LEVELS, BLOOM_VERBS


def heuristic_labels(text: str) -> List[int]:
    text_low = text.lower()
    scores: Dict[str, float] = {k: 0.0 for k in ALL_BLOOM_LEVELS}
    for level, verbs in BLOOM_VERBS.items():
        for v in verbs:
            if v in text_low:
                scores[level] += 1.0
        # fuzzy match to catch variants
        for token in re.findall(r"[a-zA-Z]+", text_low):
            scores[level] += fuzz.partial_ratio(token, " ".join(verbs)) / 100.0 * 0.1
    # convert to multi-hot by threshold
    return [1 if scores[lvl] >= 1.0 else 0 for lvl in ALL_BLOOM_LEVELS]


def combine(llm: List[int], heur: List[int]) -> List[int]:
    out = []
    for a, b in zip(llm, heur):
        weight = 0.7 * a + 0.3 * b
        out.append(1 if weight >= 0.5 else 0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Weak supervision merge")
    parser.add_argument("--in", dest="input_glob", type=str, required=True, help="input JSONL glob")
    parser.add_argument("--out", dest="output", type=str, required=True)
    args = parser.parse_args()

    paths = list(Path().glob(args.input_glob))
    rows = []
    for path in paths:
        rows.extend(load_jsonl(path))

    merged = []
    for row in rows:
        heur = heuristic_labels(row["text"])
        llm_labels = row.get("labels") or [0] * len(ALL_BLOOM_LEVELS)
        final = combine(llm_labels, heur)
        row["labels"] = final
        row["weak_supervision"] = {"llm": llm_labels, "heuristic": heur}
        merged.append(row)

    write_jsonl(args.output, merged)
    print(f"Weak-supervised {len(merged)} rows -> {args.output}")


if __name__ == "__main__":
    main()
