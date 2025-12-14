# Bloom's Taxonomy Classifier

End-to-end Bloom's taxonomy classifier with three tracks (SVM, BERT, hybrid) and automatic data acquisition. Defaults to a synthetic path that prefers local/open HF models and can be forced to fail rather than fall back to a template.

## Quickstart (synthetic, one command)

```bash
make demo
```
This creates a venv, generates synthetic data, trains a TF-IDF SVM baseline, runs a small BERT multi-label fine-tune, runs WeightWatcher, and renders `outputs/report.html`.

## Latest run snapshot (Dec 2025)

- Synthetic generation with HF chat models (TinyLlama-1.1B-Chat-v0.6, SmolLM2-1.7B-Instruct, Qwen2.5-0.5B-Instruct), `--require-llm` to forbid template fallback.
- 600 raw → weak supervision → dedupe (sim 0.97, len 12–320) → 479 rows, balanced across Bloom labels (≈105–111 each).
- SVM (TF-IDF, linear): test macro-F1 ≈ 0.428 (`outputs/models/svm_latest`).
- BERT fine-tune (6 epochs, batch 8, grad-accum 2): best threshold on val ≈ 0.20, test macro-F1 ≈ 0.375 (`outputs/models/bert_finetuned`).

## Setup

```bash
make venv
source .venv/bin/activate
make setup  # install deps + spacy model
```

## Data acquisition

- **Synthetic (default)**: `python -m src.acquisition.synth --models TinyLlama/TinyLlama-1.1B-Chat-v0.6,HuggingFaceTB/SmolLM2-1.7B-Instruct,Qwen/Qwen2.5-0.5B-Instruct --n 600 --multilabel-p 0.35 --require-llm`
  - Add `--fallback` to force template backend; omit `--require-llm` if you are okay with template fallback.
- **Scrape (respectful, may return zero)**: `python -m src.acquisition.scrape --max-pages 50 --sites https://oercommons.org https://open.umn.edu --allow-licenses cc-by,cc-by-sa --rate-limit 1.0`
- **Weak supervision & cleaning**: `python -m src.acquisition.weak_supervision --in data/raw/*.jsonl --out data/interim/weak.jsonl` then `python -m src.acquisition.dedupe_clean --in data/interim/weak.jsonl --out data/processed/data.parquet`
- **Audit & dataset card**: `python -m src.acquisition.audit`

## Prep & splits

```bash
python -m src.data.prepare --input data/processed/data.parquet --output data/processed/data.parquet
python -m src.data.splits --input data/processed/data.parquet --out data/processed --val 0.1 --test 0.2 --seed 42
```

## Feature tracks

- TF-IDF + engineered: `python -m src.features.tfidf --train data/processed/train.parquet --out data/processed/tfidf`
- Embeddings cache: `python -m src.features.embeddings --train data/processed/train.parquet --model all-MiniLM-L6-v2 --device mps --cache data/cache`
- Engineered stats: `python -m src.features.engineered --train data/processed/train.parquet --out data/processed/eng`

## Model tracks

- SVM (TF-IDF): `python -m src.models.svm_train --features tfidf --kernel linear --out outputs/models/svm`
- BERT multi-label: `python -m src.models.bert_finetune --mode multilabel --device mps --threshold 0.2`
- BERT OVR (logistic heads): `python -m src.models.bert_finetune --mode ovr --device mps`
- Hybrid (embeddings + RBF SVM): `python -m src.models.svm_train --features embeddings --kernel rbf --out outputs/models/svm_hybrid`

## Evaluation & report

- Metrics: `python -m src.models.evaluate --truth data/processed/test.parquet --preds outputs/models/svm/preds.npy --out outputs/metrics`
- WeightWatcher: `python -m src.ww.run_weightwatcher --ckpt outputs/models/bert_finetuned`
- Report: `make report` (executes `notebooks/report.ipynb` -> `outputs/report.html`)

## Make targets

- `make acquire`: synthetic + weak supervision + clean + audit
- `make features`: TF-IDF + embeddings cache
- `make train-svm`, `make train-bert-ovr`, `make train-bert-ml`, `make train-hybrid`
- `make ww`: WeightWatcher diagnostics
- `make demo`: small end-to-end run

## MPS / Apple Silicon notes

- Training auto-selects MPS if available. Reduce `--batch` or increase `--grad-accum` if you see MPS memory pressure.
- Sentence-transformer encoding accepts `--device mps` to stay on GPU.

## Provenance & licensing

- Scraper respects robots.txt and requires allowed licenses. If insufficient labeled data (<800), pipeline falls back to synthetic generation.
- Synthetic generator logs model name, backend, and sample counts in `outputs/metrics/dataset_card.md`.

## Repro checklist (Charles-style)

- Prefetch sentence embeddings (`features/embeddings.py`) then SVM (`models/svm_train.py --features embeddings`).
- Feature engineering: TF-IDF (word+char) and engineered stats (`features/engineered.py`).
- WeightWatcher: `make ww` to compare base vs. fine-tuned BERT (data-free diagnostics).
- Report: `outputs/report.html` captures metrics, confusion matrix, WeightWatcher summary, and dataset card.
