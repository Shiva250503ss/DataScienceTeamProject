# JD Alignment — LLM Fine-Tuning Engineer @ Neon AI

> JD: "R&D on LLM fine-tuning applications, RAG optimizations and improvements,
> training dataset optimization, model benchmark design and analysis, running
> model fine-tuning experiments. Needs proficiency in LLMs, RAG, Git, and
> ideally Transformer training experience."

## Requirement-by-requirement

| JD requirement | Where this repo demonstrates it |
|---|---|
| **LLM fine-tuning applications (R&D)** | `finetuning/` — a real application (SHAP → plain-English Explainer), not a toy: 6 techniques (LoRA, QLoRA, DoRA, DPO, ORPO, GaLore) on one controlled task, with the winner deployed back into the product via `export_ollama.py`. |
| **RAG optimizations and improvements** | `rag/` — the "optimization" story is explicit: baseline dense search → +BM25 hybrid with RRF (identifier matching) → +cross-encoder rerank (precision) → +query rewriting (multi-turn recall) → +knowledge graph (multi-hop). Each module fixes the previous stage's measured weakness, and all of it is wired into two live agents. |
| **Training dataset optimization** | `finetuning/dataset_prep.py` — exact + near-duplicate removal (Jaccard shingles), 4-rule quality filter (incl. groundedness), class balancing by downsampling, before/after statistics report (`dataset_report.md`). |
| **Model benchmark design and analysis** | Two benchmark harnesses: `finetuning/benchmark.py` (train time, peak VRAM, losses, LLM-as-judge clarity/accuracy with documented bias mitigations, jargon-leak rate, untuned baseline row) and the pre-existing `benchmark/` suite (PPO selector regret vs oracle, OpenML-CC18 comparison). |
| **Running fine-tuning experiments** | Six runnable, argparse-driven scripts sharing one seed/data/format so results are comparable; per-run `training_stats.json`. |
| **Proficiency in LLMs** | Local serving (Ollama/Mistral), prompt engineering across 7 agents, JSON-constrained generation, LLM-as-judge design, chat-template handling ([INST]/GGUF/Modelfile). |
| **Proficiency in RAG** | Hybrid retrieval, RRF, rerankers, query transformation, graph retrieval, Qdrant + BM25 dual-store design, graceful degradation. |
| **Git** | Multi-contributor repo, feature branches (`fine-tune`, `rl-model`, `meta-features`, …), PR merges. |
| **Transformer training experience** | PEFT-family training IS transformer training (adapters on attention/MLP projections, gradient checkpointing, 8-bit/paged optimizers); GaLore is genuine full-parameter training with optimizer-state engineering. |

## Honest gaps (know these; each has a talking point)

1. **Benchmark numbers require a GPU run.** The scripts are complete and the
   dataset is built, but training hasn't been executed in this environment.
   *Before the interview: run at least QLoRA + DPO + benchmark on a GPU
   (Colab/Kaggle work) and screenshot `benchmark_results.md`.*
2. **No distributed / large-scale training.** Everything is single-GPU by
   design. Talking point: "the suite isolates technique differences; scaling
   out is FSDP/DeepSpeed config in the same TrainingArguments."
3. **No retrieval-quality metrics (recall@k / nDCG).** RAG is evaluated
   through end-task quality only. Talking point: "next step is a labeled
   query set for retrieval eval — I'd log every retrieval and label hits."
4. **No experiment tracking.** MLflow is in requirements but unused; runs log
   to JSON files. Talking point: "stats are structured JSON — pointing them
   at MLflow/W&B is a one-line `report_to` change."
5. **Commit messages are mostly "update".** If asked about Git hygiene, own
   it: team project, moving fast; you know conventional commits.
6. **Transformer-from-scratch** (pretraining, custom architectures) is not
   shown. The suite demonstrates fine-tuning-level training only — which is
   what the role is about.
