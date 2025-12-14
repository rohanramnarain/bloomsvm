from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from src.utils.metrics import LABELS, threshold_probs


def load_data(train_path: str, val_path: str | None, max_samples: int | None = None):
    train_df = pd.read_parquet(train_path)
    if max_samples and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42)
    val_df = pd.read_parquet(val_path) if val_path and Path(val_path).exists() else None
    return train_df, val_df


def tokenize(tokenizer, texts):
    return tokenizer(texts["text"], truncation=True)


def make_compute_metrics(threshold: float):
    def _compute(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = threshold_probs(probs, threshold)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    return _compute


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT")
    parser.add_argument("--mode", type=str, choices=["ovr", "multilabel"], default="multilabel")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val", type=str, default="data/processed/val.parquet")
    parser.add_argument("--out", type=str, default="outputs/models/bert_finetuned")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.35, help="probability cutoff for F1 metric")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "")

    train_df, val_df = load_data(args.train, args.val, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def df_to_dataset(df: pd.DataFrame) -> Dataset:
        df = df.copy()
        df["labels"] = df[LABELS].astype(float).values.tolist()
        return Dataset.from_pandas(df[["text", "labels"]])

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df) if val_df is not None else None

    tokenized_train = train_ds.map(lambda x: tokenize(tokenizer, x), batched=True)
    tokenized_val = val_ds.map(lambda x: tokenize(tokenizer, x), batched=True) if val_ds else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch" if tokenized_val else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(tokenized_val),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        learning_rate=args.lr,
    )

    compute_fn = make_compute_metrics(args.threshold) if tokenized_val else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)] if tokenized_val else None,
    )

    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved BERT model to {args.out}")


if __name__ == "__main__":
    main()
