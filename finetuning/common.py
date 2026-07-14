# finetuning/common.py

"""
Shared utilities for all fine-tuning experiment scripts.

Everything in this suite targets ONE task:
    input : structured SHAP/LIME attribution output (which features pushed a
            prediction up or down, and by how much)
    output: a 2-4 sentence plain-English explanation a business user can read

Why a shared module?
  - Every technique (LoRA / QLoRA / DoRA / DPO / ORPO / GaLore) must train on
    IDENTICAL data with IDENTICAL prompt formatting, otherwise the benchmark
    comparison in benchmark.py is meaningless.
  - Training time and peak GPU memory are measured the same way everywhere so
    the numbers are directly comparable.
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
FINETUNING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FINETUNING_DIR, "data")
MODELS_DIR = os.path.join(FINETUNING_DIR, "models")
RESULTS_DIR = os.path.join(FINETUNING_DIR, "results")

TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
# Preference pairs (chosen vs rejected) for DPO / ORPO
PREF_TRAIN_FILE = os.path.join(DATA_DIR, "preferences_train.jsonl")
PREF_VAL_FILE = os.path.join(DATA_DIR, "preferences_val.jsonl")

# ── Base models ───────────────────────────────────────────────────────────────
# Full-precision HF id — used by DoRA (PEFT), GaLore, and adapter merging.
BASE_MODEL_HF = "mistralai/Mistral-7B-Instruct-v0.3"
# Unsloth's pre-quantized 4-bit mirror — downloads faster and skips the
# on-the-fly quantization step for QLoRA. Same weights as the HF original.
BASE_MODEL_UNSLOTH_4BIT = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

SEED = 42
MAX_SEQ_LENGTH = 2048  # SHAP inputs + explanation comfortably fit in 2k tokens

# ── Prompt template ───────────────────────────────────────────────────────────
# Mistral-Instruct expects the [INST] ... [/INST] chat format. We keep a fixed
# system-style instruction so the model learns ONE job: SHAP -> plain English.
SYSTEM_INSTRUCTION = (
    "You are the Explainer agent of an AutoML platform. You receive the raw "
    "feature-attribution output of a machine-learning model (SHAP or LIME "
    "values) and must explain the prediction to a non-technical business "
    "user. Write 2-4 short sentences in plain English. Never use the words "
    "'SHAP', 'LIME', or any statistics jargon. Say WHY the prediction was "
    "made, WHICH factors mattered most, and WHAT would change the outcome."
)


def format_prompt(shap_input: str) -> str:
    """Build the Mistral-Instruct prompt WITHOUT the answer (for inference)."""
    return f"[INST] {SYSTEM_INSTRUCTION}\n\n{shap_input} [/INST]"


def format_example(shap_input: str, explanation: str, eos_token: str = "</s>") -> str:
    """Build the full training text WITH the answer (for SFT-style training)."""
    return f"{format_prompt(shap_input)} {explanation}{eos_token}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    """Load a .jsonl file into a list of dicts. Raises a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m finetuning.dataset_prep` first "
            f"to build the training data."
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sft_splits():
    """Return (train, val, test) lists of {'input': ..., 'output': ...} dicts."""
    return load_jsonl(TRAIN_FILE), load_jsonl(VAL_FILE), load_jsonl(TEST_FILE)


def load_preference_splits():
    """Return (train, val) lists of {'prompt', 'chosen', 'rejected'} dicts."""
    return load_jsonl(PREF_TRAIN_FILE), load_jsonl(PREF_VAL_FILE)


# ── GPU memory + time tracking ────────────────────────────────────────────────

class RunTracker:
    """
    Measures wall-clock training time and peak GPU memory for one experiment.

    Usage:
        tracker = RunTracker("qlora")
        tracker.start()
        ... train ...
        tracker.stop(final_loss=..., eval_loss=..., extra={...})
        tracker.save()   # -> finetuning/models/<name>/training_stats.json
    """

    def __init__(self, technique: str, output_dir: Optional[str] = None):
        self.technique = technique
        self.output_dir = output_dir or os.path.join(MODELS_DIR, technique)
        self.stats: Dict = {"technique": technique}
        self._t0 = None

    def start(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.stats["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        self._t0 = time.time()

    def stop(self, final_loss: float = None, eval_loss: float = None,
             extra: Dict = None):
        self.stats["train_time_seconds"] = round(time.time() - self._t0, 1)
        self.stats["train_time_human"] = time.strftime(
            "%Hh %Mm %Ss", time.gmtime(self.stats["train_time_seconds"]))
        try:
            import torch
            if torch.cuda.is_available():
                self.stats["peak_gpu_memory_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        except ImportError:
            pass
        if final_loss is not None:
            self.stats["final_train_loss"] = round(float(final_loss), 4)
        if eval_loss is not None:
            self.stats["final_eval_loss"] = round(float(eval_loss), 4)
        if extra:
            self.stats.update(extra)

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "training_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        print(f"[{self.technique}] Stats saved to {path}")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")


def get_final_losses(trainer) -> tuple:
    """
    Pull the last training loss and last eval loss out of a HF/TRL Trainer's
    log history. Works for SFTTrainer, DPOTrainer, ORPOTrainer and Trainer.
    """
    train_loss, eval_loss = None, None
    for entry in trainer.state.log_history:
        if "loss" in entry:
            train_loss = entry["loss"]
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
    return train_loss, eval_loss


def require_cuda(technique: str):
    """Fail fast with a helpful message when no GPU is available."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            f"[{technique}] No CUDA GPU detected. 7B fine-tuning requires a GPU "
            f"(QLoRA needs ~6-8 GB VRAM, LoRA 16-bit ~18 GB, GaLore ~24 GB). "
            f"On Windows, run inside WSL2 for best library support."
        )
