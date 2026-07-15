# Fine-Tuning Benchmark Results

> **SMOKE TEST - Qwen2.5-0.5B base on RTX A500 4GB, 256/192 train examples, 1 epoch - NOT representative of the Mistral-7B recipes**

Test set: 12 held-out examples | Judge: mistral:7b-instruct via Ollama (clarity + accuracy, 1-5) | Generation backend: transformers

| Technique | Train time | Peak VRAM (GB) | Train loss | Eval loss | Clarity (1-5) | Accuracy (1-5) | Overall | Jargon leak % | Judged n |
|---|---|---|---|---|---|---|---|---|---|
| dpo | 00h 14m 39s | 13.99 | 0.0 | 0.0 | 5 | 4.92 | **4.96** | 0.0 | 12 |
| qlora | 03h 16m 06s | 5.91 | 0.3502 | 0.3641 | 5 | 4.92 | **4.96** | 0.0 | 12 |
| base (untuned) | — | — | — | — | 4.75 | 5 | **4.88** | 100.0 | 12 |

## How to read this table
- **Train time / Peak VRAM** were recorded during each training run (training_stats.json) on the same data, seed, and sequence length.
- **Eval loss** is comparable across LoRA/QLoRA/DoRA/GaLore (same SFT objective). DPO/ORPO optimize a different loss, so compare them via the judge scores, not the loss column.
- **Overall** = mean of judge clarity and accuracy.
- **Jargon leak %** = generations containing SHAP/LIME/stats terms — the exact behavior the fine-tune is meant to remove.