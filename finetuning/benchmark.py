# finetuning/benchmark.py

"""
Benchmark harness — compares every trained adapter on the held-out test set.

For each technique found under finetuning/models/ it reports:
  1. Training time          (from training_stats.json, measured during training)
  2. Peak GPU memory        (from training_stats.json)
  3. Final train/eval loss  (from training_stats.json)
  4. Output quality         (measured HERE: generation on the test set,
                             scored 1-5 by an LLM judge on clarity + accuracy)

LLM-AS-JUDGE DESIGN
  Judge model: whatever Ollama serves (default mistral:7b-instruct — a
  DIFFERENT copy than the tuned one, since the judge runs the base weights).
  Known biases we mitigate:
    - position bias: not applicable (absolute scoring, not pairwise)
    - verbosity bias: rubric explicitly says "short and clear beats long"
    - self-preference: acceptable here because ALL candidates are judged by
      the same judge, so relative ranking is still meaningful
  Each sample is scored on two axes (clarity, accuracy), 1-5 integers only,
  with the SHAP input given as ground truth context.

Also reports the UNTUNED base model as a baseline row, so the table shows
what fine-tuning actually bought.

Output: finetuning/results/benchmark_results.md  (screenshot-ready table)

Usage:
    python -m finetuning.benchmark --max-samples 50
    python -m finetuning.benchmark --techniques qlora dpo --skip-base
"""

import argparse
import json
import os
import re
import statistics
import time
from typing import Dict, List, Optional

import requests

from finetuning.common import (
    BASE_MODEL_UNSLOTH_4BIT, MAX_SEQ_LENGTH, MODELS_DIR, RESULTS_DIR,
    format_prompt, load_sft_splits, require_cuda,
)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

JUDGE_PROMPT = """You are grading an AI-written explanation of a machine-learning prediction.

GROUND TRUTH (the model's actual feature attributions):
{shap_input}

EXPLANATION TO GRADE:
{explanation}

Score two criteria from 1 to 5 (integers only):

CLARITY — could a non-technical manager understand it?
  5 = plain English, short, no jargon, well structured
  3 = understandable but clunky or slightly jargony
  1 = jargon dump or incoherent

ACCURACY — does it match the ground truth attributions?
  5 = names the truly dominant factors with correct directions
  1 = wrong factors or wrong directions

Note: short and clear beats long. Do NOT reward length.

Answer with ONLY this JSON: {{"clarity": <int>, "accuracy": <int>}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Generation with each adapter
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_adapter(adapter_path: Optional[str], test_rows: List[Dict],
                          max_new_tokens: int = 200) -> List[str]:
    """
    Load base (+ adapter if given) with Unsloth in 4-bit inference mode and
    generate an explanation for every test input. Model is freed afterwards
    so multiple adapters can be benchmarked in one process.
    """
    import torch
    from unsloth import FastLanguageModel

    name = adapter_path or BASE_MODEL_UNSLOTH_4BIT
    print(f"  loading {name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # enables Unsloth's fast decoding

    outputs = []
    for i, row in enumerate(test_rows):
        prompt = format_prompt(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        outputs.append(text)
        if (i + 1) % 10 == 0:
            print(f"    generated {i + 1}/{len(test_rows)}")

    # Free VRAM before the next adapter loads
    del model
    torch.cuda.empty_cache()
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge via Ollama
# ─────────────────────────────────────────────────────────────────────────────

def judge_one(shap_input: str, explanation: str) -> Optional[Dict[str, int]]:
    """Score one explanation. Returns {'clarity': int, 'accuracy': int} or None."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": JUDGE_MODEL,
                  "prompt": JUDGE_PROMPT.format(shap_input=shap_input,
                                                explanation=explanation),
                  "format": "json", "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 60}},
            timeout=90,
        )
        r.raise_for_status()
        parsed = json.loads(r.json()["response"])
        clarity = int(parsed["clarity"])
        accuracy = int(parsed["accuracy"])
        if 1 <= clarity <= 5 and 1 <= accuracy <= 5:
            return {"clarity": clarity, "accuracy": accuracy}
    except Exception:
        pass
    return None


def judge_all(test_rows: List[Dict], outputs: List[str]) -> Dict[str, float]:
    """Judge every generation; returns mean clarity/accuracy/overall + n scored."""
    clarities, accuracies = [], []
    for row, output in zip(test_rows, outputs):
        score = judge_one(row["input"], output)
        if score:
            clarities.append(score["clarity"])
            accuracies.append(score["accuracy"])
    if not clarities:
        return {"clarity": float("nan"), "accuracy": float("nan"),
                "overall": float("nan"), "n_scored": 0}
    return {
        "clarity": round(statistics.mean(clarities), 2),
        "accuracy": round(statistics.mean(accuracies), 2),
        "overall": round((statistics.mean(clarities) + statistics.mean(accuracies)) / 2, 2),
        "n_scored": len(clarities),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simple automatic checks (no LLM needed) — sanity metrics beside the judge
# ─────────────────────────────────────────────────────────────────────────────

JARGON_RE = re.compile(r"\b(shap|lime|attribution|logit|coefficient)\b", re.I)


def rule_metrics(outputs: List[str]) -> Dict[str, float]:
    """Deterministic quality signals: jargon leak rate and mean length."""
    n = max(len(outputs), 1)
    return {
        "jargon_leak_pct": round(100 * sum(bool(JARGON_RE.search(o)) for o in outputs) / n, 1),
        "avg_length_chars": round(sum(len(o) for o in outputs) / n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_training_stats(technique: str) -> Dict:
    path = os.path.join(MODELS_DIR, technique, "training_stats.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Benchmark all trained adapters")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Test examples to generate + judge per adapter")
    parser.add_argument("--techniques", nargs="*", default=None,
                        help="Subset to benchmark (default: every dir in finetuning/models)")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the untuned-base baseline row")
    args = parser.parse_args()

    require_cuda("benchmark")

    _, _, test_rows = load_sft_splits()
    test_rows = test_rows[:args.max_samples]
    print(f"Benchmarking on {len(test_rows)} held-out test examples")

    # Discover trained techniques
    if args.techniques:
        techniques = args.techniques
    else:
        techniques = [d for d in sorted(os.listdir(MODELS_DIR))
                      if os.path.isdir(os.path.join(MODELS_DIR, d))] \
            if os.path.exists(MODELS_DIR) else []
    if not techniques and args.skip_base:
        raise SystemExit("No trained adapters found under finetuning/models/. "
                         "Run a finetune_*.py script first.")
    print(f"Techniques found: {techniques or '(none — baseline only)'}")

    candidates = ([] if args.skip_base else [("base (untuned)", None)]) + \
        [(t, os.path.join(MODELS_DIR, t)) for t in techniques]

    results = []
    for name, path in candidates:
        print(f"\n=== {name} ===")
        t0 = time.time()
        outputs = generate_with_adapter(path, test_rows)
        gen_time = time.time() - t0
        print("  judging with LLM-as-judge ...")
        judge = judge_all(test_rows, outputs)
        rules = rule_metrics(outputs)
        stats = load_training_stats(name) if path else {}
        results.append({
            "name": name,
            "train_time": stats.get("train_time_human", "—"),
            "peak_vram": stats.get("peak_gpu_memory_gb", "—"),
            "train_loss": stats.get("final_train_loss", "—"),
            "eval_loss": stats.get("final_eval_loss", "—"),
            **judge, **rules,
            "gen_seconds_per_sample": round(gen_time / max(len(test_rows), 1), 2),
        })
        # Persist raw generations for manual inspection / the interview demo
        os.makedirs(RESULTS_DIR, exist_ok=True)
        gen_path = os.path.join(RESULTS_DIR, f"generations_{name.split(' ')[0]}.jsonl")
        with open(gen_path, "w", encoding="utf-8") as f:
            for row, out in zip(test_rows, outputs):
                f.write(json.dumps({"input": row["input"], "reference": row["output"],
                                    "generated": out}, ensure_ascii=False) + "\n")

    # ── Markdown results table ────────────────────────────────────────────
    results.sort(key=lambda r: (r["overall"] if r["overall"] == r["overall"] else 0),
                 reverse=True)
    lines = [
        "# Fine-Tuning Benchmark Results",
        "",
        f"Test set: {len(test_rows)} held-out examples | "
        f"Judge: {JUDGE_MODEL} via Ollama (clarity + accuracy, 1-5)",
        "",
        "| Technique | Train time | Peak VRAM (GB) | Train loss | Eval loss | "
        "Clarity (1-5) | Accuracy (1-5) | Overall | Jargon leak % | Judged n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['train_time']} | {r['peak_vram']} | "
            f"{r['train_loss']} | {r['eval_loss']} | {r['clarity']} | "
            f"{r['accuracy']} | **{r['overall']}** | {r['jargon_leak_pct']} | "
            f"{r['n_scored']} |")
    lines += [
        "",
        "## How to read this table",
        "- **Train time / Peak VRAM** were recorded during each training run "
        "(training_stats.json) on the same data, seed, and sequence length.",
        "- **Eval loss** is comparable across LoRA/QLoRA/DoRA/GaLore (same SFT "
        "objective). DPO/ORPO optimize a different loss, so compare them via "
        "the judge scores, not the loss column.",
        "- **Overall** = mean of judge clarity and accuracy.",
        "- **Jargon leak %** = generations containing SHAP/LIME/stats terms — "
        "the exact behavior the fine-tune is meant to remove.",
    ]
    md = "\n".join(lines)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n{md}\n\nSaved to {out_path}")


if __name__ == "__main__":
    main()
