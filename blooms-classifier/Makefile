PYTHON ?= python3
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: venv setup scrape synth acquire features train-svm train-bert-ovr train-bert-ml train-hybrid eval ww report demo clean

venv:
	$(PYTHON) -m venv $(VENV)

setup: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && python -m spacy download en_core_web_sm || true

scrape:
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.scrape

synth:
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.synth

acquire:
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.synth --fallback
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.weak_supervision --in data/raw/synth.jsonl --out data/interim/weak.jsonl
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.dedupe_clean --in data/interim/weak.jsonl --out data/processed/data.parquet
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.audit

features:
	$(ACTIVATE) && $(PYTHON) -m src.features.tfidf --train data/processed/train.parquet --out data/processed/tfidf
	$(ACTIVATE) && $(PYTHON) -m src.features.embeddings --train data/processed/train.parquet --model all-MiniLM-L6-v2 --cache data/cache

train-svm:
	$(ACTIVATE) && $(PYTHON) -m src.models.svm_train --features tfidf --kernel linear --out outputs/models/svm

train-bert-ovr:
	$(ACTIVATE) && $(PYTHON) -m src.models.bert_finetune --mode ovr

train-bert-ml:
	$(ACTIVATE) && $(PYTHON) -m src.models.bert_finetune --mode multilabel

train-hybrid:
	$(ACTIVATE) && $(PYTHON) -m src.models.svm_train --features embeddings --kernel rbf --out outputs/models/svm_hybrid

ww:
	$(ACTIVATE) && $(PYTHON) -m src.ww.run_weightwatcher

report:
	$(ACTIVATE) && jupyter nbconvert --to html --execute notebooks/report.ipynb --output report.html --output-dir outputs

demo: setup
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.synth --n 180 --multilabel-p 0.2 --fast
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.weak_supervision --in data/raw/synth.jsonl --out data/interim/weak.jsonl
	$(ACTIVATE) && $(PYTHON) -m src.acquisition.dedupe_clean --in data/interim/weak.jsonl --out data/processed/data.parquet --max-dup-sim 0.9
	$(ACTIVATE) && $(PYTHON) -m src.data.prepare --input data/processed/data.parquet --output data/processed/data.parquet
	$(ACTIVATE) && $(PYTHON) -m src.data.splits --input data/processed/data.parquet --out data/processed --val 0.1 --test 0.2
	$(ACTIVATE) && $(PYTHON) -m src.features.tfidf --train data/processed/train.parquet --out data/processed/tfidf
	$(ACTIVATE) && $(PYTHON) -m src.models.svm_train --features tfidf --kernel linear --out outputs/models/svm_demo
	$(ACTIVATE) && $(PYTHON) -m src.models.bert_finetune --mode multilabel --epochs 1 --batch 4 --grad-accum 4 --max-samples 200
	$(ACTIVATE) && $(PYTHON) -m src.ww.run_weightwatcher --ckpt outputs/models/bert_finetuned
	$(ACTIVATE) && jupyter nbconvert --to html --execute notebooks/report.ipynb --output report.html --output-dir outputs

clean:
	rm -rf $(VENV) data/interim/* data/processed/* data/cache/* outputs/*
