# Running the REAL 7B fine-tuning experiments on a free Colab GPU

The dev machine's NVIDIA RTX A500 has **4 GB VRAM** — enough for real
small-model smoke runs (Qwen2.5-0.5B, `--backend transformers`), but NOT for
the Mistral-7B recipes (QLoRA needs ~7 GB). A free Colab **T4 (16 GB)** runs
every experiment except 16-bit LoRA and GaLore comfortably.

## Exact copy-paste steps

**1. Open** https://colab.research.google.com → New notebook →
Runtime → Change runtime type → **T4 GPU** → Save.

**2. First cell — clone and install** (repo must be pushed to GitHub first):

```python
!git clone https://github.com/Shiva250503ss/DataScienceTeamProject.git
%cd DataScienceTeamProject
!pip install -q unsloth "trl==0.9.6" "peft==0.12.0" "transformers==4.44.2" \
    "accelerate==0.33.0" "datasets==2.21.0" bitsandbytes mlflow rank-bm25 python-dotenv
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**3. Build the dataset** (template mode — Ollama isn't available on Colab;
or commit `finetuning/data/` from your machine and skip this):

```python
!python -m finetuning.dataset_prep --n-synthetic 800
```

**4. Train — QLoRA first (≈40-80 min for 3 epochs on T4):**

```python
!python -m finetuning.finetune_qlora --epochs 3 --rank 16
```

**5. DPO on top of the QLoRA adapter (≈30-60 min):**

```python
!python -m finetuning.finetune_dpo --epochs 1 --beta 0.1
```

**6. Optional extras if time allows:**

```python
!python -m finetuning.finetune_dora --epochs 3     # DoRA comparison
!python -m finetuning.finetune_orpo --epochs 2     # ORPO vs DPO head-to-head
```

**7. Benchmark** (no Ollama on Colab → the LLM-judge scores will show
`n_scored=0`; jargon-leak %, losses, time, and VRAM are still measured.
Run the judge later on any machine with Ollama):

```python
!python -m finetuning.benchmark --max-samples 50
!cat finetuning/results/benchmark_results.md
```

**8. Download the adapters + stats before the session dies:**

```python
!zip -r adapters.zip finetuning/models finetuning/results mlruns
from google.colab import files
files.download("adapters.zip")
```

**9. Back on your machine** — unzip into the repo, then export the winner:

```bash
python -m finetuning.export_ollama --technique qlora
# .env:  OLLAMA_MODEL=datapilot-explainer
```

## VRAM budget cheat-sheet (Mistral-7B, measured ranges from the papers/Unsloth docs)

| Experiment | Min VRAM | Free T4 (16 GB)? |
|---|---|---|
| QLoRA (4-bit) | ~7 GB | ✅ |
| DPO on QLoRA | ~8 GB | ✅ |
| DoRA (4-bit) | ~8 GB | ✅ |
| ORPO (4-bit) | ~8 GB | ✅ |
| LoRA (16-bit) | ~18 GB | ❌ (needs A100/L4 — Colab Pro) |
| GaLore full-param | ~24 GB | ❌ (needs A100) |
