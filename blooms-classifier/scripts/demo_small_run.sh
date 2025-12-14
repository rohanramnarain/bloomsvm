#!/usr/bin/env zsh
set -e
source .venv/bin/activate || { echo "Run make setup first"; exit 1; }
python -m src.acquisition.synth --n 60 --multilabel-p 0.2 --fast
python -m src.acquisition.weak_supervision --in data/raw/synth.jsonl --out data/interim/weak.jsonl
python -m src.acquisition.dedupe_clean --in data/interim/weak.jsonl --out data/processed/data.parquet --max-dup-sim 0.9
python -m src.data.prepare --input data/processed/data.parquet --output data/processed/data.parquet
python -m src.data.splits --input data/processed/data.parquet --out data/processed --val 0.1 --test 0.2
python -m src.features.tfidf --train data/processed/train.parquet --out data/processed/tfidf
python -m src.models.svm_train --features tfidf --kernel linear --out outputs/models/svm_demo
python -m src.models.bert_finetune --mode multilabel --epochs 1 --batch 4 --grad-accum 4 --max-samples 120
python -m src.ww.run_weightwatcher --ckpt outputs/models/bert_finetuned
jupyter nbconvert --to html --execute notebooks/report.ipynb --output outputs/report.html
