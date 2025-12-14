from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import trange

from src.utils.io import ensure_dir, set_seed, utcnow, write_jsonl
from src.utils.verbs import ALL_BLOOM_LEVELS, BLOOM_VERBS

RUBRIC = {
    "remember": "recall basic facts, list, define without interpretation",
    "understand": "summarize or explain ideas in own words",
    "apply": "use a concept in a novel situation or perform calculations",
    "analyze": "break down, compare, find relationships",
    "evaluate": "justify a choice, critique, defend with criteria",
    "create": "design or invent something new using learned ideas",
}

DOMAINS = ["stem", "humanities", "social_sciences"]


@dataclass
class SynthRow:
    id: str
    text: str
    labels: List[int]
    rationale: str
    domain: str
    difficulty: str
    source: str
    synthetic_flag: bool
    timestamp: str


try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


class LocalLLM:
    def __init__(self, model: str, device: str | None = None, allow_template: bool = True):
        self.model = model
        self.device = device
        self.pipeline = None
        self.backend = None
        if requests and self._ollama_alive():
            self.backend = "ollama"
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                tok = AutoTokenizer.from_pretrained(model)
                model_obj = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="auto")
                self.pipeline = pipeline("text-generation", model=model_obj, tokenizer=tok, device_map="auto")
                self.backend = "hf"
            except Exception as exc:
                if not allow_template:
                    raise RuntimeError(f"LLM backend unavailable for {model}: {exc}") from exc
                self.backend = "template"

    def _ollama_alive(self) -> bool:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            resp = requests.get(f"{host}/api/tags", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if self.backend == "ollama" and requests:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            resp = requests.post(
                f"{host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        if self.backend == "hf" and self.pipeline:
            out = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.8)[0]["generated_text"]
            return out[len(prompt) :]
        return self._template_completion(prompt)

    @staticmethod
    def _template_completion(prompt: str) -> str:
        # deterministic fallback: pick verbs and stitch
        verbs = random.choice(list(BLOOM_VERBS.values()))
        sentence = f"Generate {random.choice(verbs)} question about {random.choice(['physics','history','ethics','biology','economics'])}."
        return sentence


def pick_labels(multilabel_p: float) -> List[str]:
    if random.random() < multilabel_p:
        return random.sample(ALL_BLOOM_LEVELS, k=2)
    return [random.choice(ALL_BLOOM_LEVELS)]


def label_to_multi_hot(labels: List[str]) -> List[int]:
    return [1 if level in labels else 0 for level in ALL_BLOOM_LEVELS]


def make_prompt(domain: str, labels: List[str]) -> str:
    targets = ", ".join(labels)
    rubric = "\n".join([f"- {k}: {v}" for k, v in RUBRIC.items()])
    return (
        "You are creating study questions with Bloom's taxonomy labels.\n"
        f"Domain: {domain}. Target Bloom levels: {targets}.\n"
        "Return a single question and a one-sentence rationale."
        "Write concisely."
        f"Rubric:\n{rubric}"
    )


def validate(question: str, labels: List[str]) -> bool:
    low = question.lower()
    for lbl in labels:
        verbs = BLOOM_VERBS.get(lbl, [])
        if any(v in low for v in verbs):
            return True
    return False


def synthesize(n: int, model: str, multilabel_p: float, domains: List[str], fast: bool, allow_template: bool) -> List[SynthRow]:
    llm = LocalLLM(model, allow_template=allow_template)
    rows: List[SynthRow] = []
    difficulties = ["easy", "medium", "hard"]
    for _ in trange(n, desc="synth"):
        domain = random.choice(domains)
        labels = pick_labels(multilabel_p)
        prompt = make_prompt(domain, labels)
        completion = llm.generate(prompt, max_tokens=120 if not fast else 60)
        question = completion.strip().split("?", 1)[0]
        if not question.endswith("?"):
            question = question + "?"
        if not validate(question, labels):
            # second pass validation/regeneration attempt
            prompt = prompt + "\nEnsure the question explicitly uses a verb that signals the Bloom level."
            completion = llm.generate(prompt, max_tokens=60)
            question = completion.strip().split("?", 1)[0] + "?"
        rationale = f"Targets {', '.join(labels)} because it asks to {BLOOM_VERBS[labels[0]][0]}" if labels else ""
        row = SynthRow(
            id=os.urandom(4).hex(),
            text=question.strip(),
            labels=label_to_multi_hot(labels),
            rationale=rationale,
            domain=domain,
            difficulty=random.choice(difficulties),
            source=f"synthetic-{llm.backend}",
            synthetic_flag=True,
            timestamp=utcnow(),
        )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Synthetic Bloom generator")
    parser.add_argument("--model", type=str, default=os.getenv("SYNTH_MODEL", "qwen2.5:3b"), help="legacy single model")
    parser.add_argument(
        "--models",
        type=str,
        default=os.getenv("SYNTH_MODELS", "qwen2.5:3b,llama3.1:8b-instruct,mistral-nemo:12b"),
        help="comma-separated list of open models to sample from",
    )
    parser.add_argument("--n", type=int, default=int(os.getenv("SYNTH_N", "2000")))
    parser.add_argument("--multilabel-p", type=float, default=0.25)
    parser.add_argument("--domains", type=str, default=",".join(DOMAINS))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="shorter generations for demo")
    parser.add_argument("--fallback", action="store_true", help="force template backend if LLM unavailable")
    parser.add_argument("--require-llm", action="store_true", help="raise if no local/HF LLM is available")
    args = parser.parse_args()

    set_seed(42)
    domains = [d for d in args.domains.split(",") if d]
    ensure_dir("data/raw")

    if args.fallback:
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:0"  # force template

    allow_template = not args.require_llm

    models = [m for m in args.models.split(",") if m] if args.models else [args.model]
    total = args.n
    base = total // len(models)
    remainder = total % len(models)
    rows: List[SynthRow] = []
    per_model_meta: List[Dict[str, str | int]] = []
    for idx, model_name in enumerate(models):
        take = base + (1 if idx < remainder else 0)
        if take <= 0:
            continue
        batch = synthesize(take, model_name, args.multilabel_p, domains or DOMAINS, args.fast, allow_template)
        rows.extend(batch)
        backend = batch[0].source if batch else "unknown"
        per_model_meta.append({"model": model_name, "count": len(batch), "backend": backend})

    out_path = Path("data/raw/synth.jsonl")
    write_jsonl(out_path, [r.__dict__ for r in rows])

    meta = {
        "total": len(rows),
        "models": per_model_meta,
        "multilabel_p": args.multilabel_p,
        "domains": domains,
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/dataset_card.md").write_text(
        "\n".join(
            [
                "# Dataset card (synthetic)",
                f"Samples: {meta['total']}",
                f"Models: {[m['model'] for m in per_model_meta]}",
                f"Domains: {domains}",
                f"Backends: {[m['backend'] for m in per_model_meta]}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
