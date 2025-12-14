from __future__ import annotations

import argparse
from pathlib import Path

import torch
import weightwatcher as ww
from transformers import AutoModelForSequenceClassification


def run_ww(model_name_or_path: str, out_dir: Path, tag: str, base_model=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    watcher = ww.WeightWatcher(model=model, base_model=base_model)
    # base_model is None for the base run; populated for finetuned to get delta-aware diagnostics
    details = watcher.analyze()
    details_path = out_dir / f"details_{tag}.csv"
    details.to_csv(details_path, index=False)
    summary = watcher.get_summary(details)
    summary_path = out_dir / f"summary_{tag}.md"
    summary_path.write_text("\n".join([f"{k}: {v}" for k, v in summary.items()]), encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run WeightWatcher on models")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--ckpt", type=str, default="outputs/models/bert_finetuned")
    parser.add_argument("--out", type=str, default="outputs/ww")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    base_summary = run_ww(args.model_name, out_dir, "base")
    if Path(args.ckpt).exists():
        finetuned_summary = run_ww(args.ckpt, out_dir, "finetuned", base_model=base_model)
    else:
        finetuned_summary = None

    summary_md = out_dir / "summary.md"
    summary_lines = ["# WeightWatcher summary", f"Base summary: {base_summary}"]
    if finetuned_summary:
        summary_lines.append(f"Finetuned summary: {finetuned_summary}")
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"WeightWatcher results saved to {out_dir}")


if __name__ == "__main__":
    main()
