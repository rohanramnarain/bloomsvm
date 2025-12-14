from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet
from src.utils.verbs import ALL_BLOOM_LEVELS


def main():
    processed = Path("data/processed/data.parquet")
    if not processed.exists():
        print("No processed data found; run synth pipeline first.")
        return
    df = read_parquet(processed)
    card = Path("outputs/metrics/dataset_card.md")
    card.parent.mkdir(parents=True, exist_ok=True)
    totals = {lbl: int(df[lbl].sum()) for lbl in ALL_BLOOM_LEVELS if lbl in df.columns}
    provenance = df.get("source", pd.Series([])).value_counts().to_dict()
    card.write_text(
        "\n".join(
            [
                "# Dataset card",
                f"Total rows: {len(df)}",
                "## Label counts",
                json.dumps(totals, indent=2),
                "## Provenance",
                json.dumps(provenance, indent=2),
                "Data derived from either synthetic generation or scraping with license filtering.",
            ]
        ),
        encoding="utf-8",
    )
    prov_path = Path("outputs/metrics/provenance.csv")
    prov_path.write_text("source,count\n" + "\n".join([f"{k},{v}" for k, v in provenance.items()]), encoding="utf-8")
    print(f"Wrote dataset card to {card}")


if __name__ == "__main__":
    main()
