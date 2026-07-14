# Fine-Tuning Experiment Suite

Teaches **Mistral-7B-Instruct** to do the Explainer agent's job natively:
turn raw SHAP/LIME attribution output into clear plain-English explanations —
no jargon, grounded in the actual features, ending with what would change the
outcome.

Everything runs **locally** (Unsloth/PEFT/TRL for training, Ollama for data
generation, judging, and final serving). No paid APIs, no API keys.

## Setup

```bash
# GPU machine (WSL2 recommended on Windows)
pip install -r finetuning/requirements-finetuning.txt

# Local LLM for synthetic data + judging (optional but recommended)
ollama pull mistral:7b-instruct
```

## Run order

```bash
# 1. Build + optimize the dataset (dedupe, quality filter, class balance)
python -m finetuning.dataset_prep --n-synthetic 800 --use-ollama

# 2. Train — run any subset; all use identical data/seed/prompt format
python -m finetuning.finetune_lora      # LoRA,  16-bit base   (~18 GB VRAM)
python -m finetuning.finetune_qlora     # QLoRA, 4-bit base    (~7 GB VRAM)
python -m finetuning.finetune_dora      # DoRA via PEFT        (~8 GB VRAM)
python -m finetuning.finetune_dpo       # DPO on top of QLoRA  (~8 GB VRAM)
python -m finetuning.finetune_orpo      # ORPO, one-stage      (~8 GB VRAM)
python -m finetuning.finetune_galore    # OPTIONAL full-param  (~24 GB VRAM)

# 3. Compare all trained adapters (+ untuned baseline) on the held-out test set
python -m finetuning.benchmark --max-samples 50

# 4. Export the winner to Ollama (merge -> GGUF -> Modelfile -> ollama create)
python -m finetuning.export_ollama --technique qlora
# then in .env:  OLLAMA_MODEL=datapilot-explainer
```

## What each experiment demonstrates

| Script | Technique | Key idea | Trainable params |
|---|---|---|---|
| `finetune_lora.py` | LoRA | low-rank delta `W + BA` on frozen 16-bit base | ~0.6% |
| `finetune_qlora.py` | QLoRA | same adapters on an NF4 4-bit base → consumer GPU | ~0.6% |
| `finetune_dora.py` | DoRA | decompose weights into magnitude × direction, adapt direction | ~0.65% |
| `finetune_dpo.py` | DPO | prefer *chosen* over *rejected* explanations vs frozen reference | ~0.6% |
| `finetune_orpo.py` | ORPO | SFT + preference in ONE stage, no reference model | ~0.6% |
| `finetune_galore.py` | GaLore | full-parameter training with low-rank **gradient** projection | 100% |

## Outputs

- `data/` — train/val/test.jsonl, preference pairs, `dataset_report.md`
- `models/<technique>/` — adapter + `training_stats.json` (time, peak VRAM, losses)
- `results/benchmark_results.md` — the comparison table (train time, VRAM,
  loss, LLM-judge clarity/accuracy 1-5, jargon-leak rate)
- `results/generations_*.jsonl` — raw generations per technique for manual review
