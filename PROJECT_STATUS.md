# PROJECT STATUS — Final Consolidated Pass

Date: 2026-07-15. Every number below was **measured on this machine** during
this pass. Where something was not executed, it says NOT DONE — no estimates,
no "should work".

## 0. Hardware actually used

| Component | Value |
|---|---|
| Machine | Windows 11 Pro, 31.6 GB RAM, 797 GB free disk |
| GPU | NVIDIA RTX A500 Laptop GPU, **4 GB VRAM** (driver 595.95, CUDA 13.2) |
| Python | 3.11.9, torch 2.6.0+cu124, `torch.cuda.is_available() == True` |
| Docker | **Not installed** → real local equivalents used (see §3) |
| LLM serving | Ollama 0.32.0 (portable, `%USERPROFILE%\ollama-portable`), `mistral:7b-instruct` (4.4 GB) pulled and generating |

The 4 GB GPU **cannot** run the Mistral-7B training recipes (QLoRA needs
~7 GB). All training below is REAL GPU training on a smaller base
(Qwen2.5-0.5B-Instruct) — labeled SMOKE TEST throughout. Exact Colab commands
for the 7B runs: [docs/COLAB_FINETUNING.md](docs/COLAB_FINETUNING.md).

## 1. JD requirements — strict REAL vs NOT DONE

| JD requirement | Status | Evidence |
|---|---|---|
| Running fine-tuning experiments | **REAL (smoke scale)** | QLoRA: 3h16m, 5.91 GB peak VRAM, train loss 0.3502, eval 0.3641 (256 ex, 1 epoch). DPO on top of it: 14m39s, train loss → 0.0 (192 pairs). Adapters on disk under `finetuning/models/`. |
| LLM fine-tuning application | **REAL** | Tuned model exported and registered in Ollama as `datapilot-explainer`; answered a real SHAP input (§6). |
| Model benchmark design & analysis | **REAL** | `finetuning/results/benchmark_results.md`: base vs QLoRA vs DPO on 12 held-out examples, judged by local mistral:7b. **Jargon-leak: base 100% → tuned 0%.** Judge overall: 4.96 (both tuned) vs 4.88 (base). |
| Training dataset optimization | **REAL** | dataset_prep executed: 800 synthetic → dedupe/quality-filter/balance → **756 final** (606 train / 75 val / 75 test + preference pairs). Report: `finetuning/data/dataset_report.md`. |
| RAG optimizations | **REAL** | Hybrid+rerank+query-transform+KG wired into live agents; corpus populated by 3 real pipeline runs (16 docs); Qdrant running in embedded local mode. |
| Retrieval evaluation | **REAL** | 12 hand-labeled queries over the real corpus, k=5: bm25 R=.70/nDCG=.62, dense R=.88/nDCG=.72, hybrid R=.82/**MRR=.74 (best)**, hybrid+rerank R=.81. (`rag_store/retrieval_eval_results.md`) |
| Experiment tracking | **REAL** | MLflow: 3 FINISHED runs (qlora + dpo) with losses; UI serving at http://127.0.0.1:5000 (HTTP 200 verified). |
| API/DB/async backend | **REAL** | Full chain verified end-to-end (§3). |
| LoRA (16-bit), DoRA, ORPO, GaLore runs | **NOT DONE** | LoRA-16bit/GaLore exceed 4 GB VRAM by design; DoRA/ORPO would fit but were deprioritized after DPO consumed the debugging budget (3 real crashes fixed, §4). Scripts are ready; Colab commands provided. |
| Mistral-7B (production-scale) training | **NOT DONE on this machine** | 4 GB VRAM. Colab T4 doc has copy-paste steps. |
| Quantized GGUF (q4_k_m) | **PARTIAL** | Export used **f16 GGUF** (~1 GB for 0.5B) because the `llama-quantize` binary isn't built on this machine (needs cmake toolchain). Conversion + Modelfile + `ollama create` all real. |

## 2. Part A audit — resolutions

All 31 findings and fixes are in [docs/AUDIT_LOG.md](docs/AUDIT_LOG.md).
Summary: 3 empty stub packages implemented (api/db/tasks); legacy 32-feature
RL path reconciled to the production 40-feature extractor (`test_agents.py`
now asserts 40 and passes **31/31 — executed**); dead `datapilot/` duplicate
directory deleted; dead SVR block removed; Qdrant given a real embedded-local
fallback; standalone dashboard app switched to Ollama-first.

**Five additional real bugs were found by RUNNING the system** (not by
reading code) and fixed: TargetEncoder crash on string targets (A27), numeric
coercion corrupting multi-number codes → float32 overflow killing training
AND SHAP (A28), pyarrow-24 access violation (A29), trl-0.9.6 Qwen BOS crash
(A30), kombu SQLA broker option crash (A31).

## 3. API → Celery → DB → orchestrator chain (verified with a real example)

Executed by `scripts/e2e_api_test.py` against live services:

```
[1] /health -> db=sqlite:///./datapilot.db  broker=sqla+sqlite:///./datapilot.db
[2] uploaded dataset id=4 (1309 rows x 14 cols)           # Titanic CSV, multipart
[3] queued run id=4 celery_task=aaba398e-...  status=queued
[4] worker picked up the task (status=running) after 12s   # real Celery consume
[5] run finished: status=completed after 571s
    best_model=RandomForestClassifier score=0.776  task_type=classification
[6] /runs/4/results -> 10 result kinds (metrics, cv_scores, shap_importance,
    global_narrative, local_narratives, cleaning_report, profile_report, ...)
    metrics: Accuracy .7861  Precision .7878  Recall .7861  F1 .7868  ROC-AUC .8244
[7] direct SQL row counts: datasets=4 pipeline_runs=4 run_results=21 explanation_docs=8
E2E CHAIN VERIFIED
```

PPO recommendations for that run (real policy output): RandomForest 30.3%,
GaussianNB 29.2%, DecisionTree 22.0%. Two more datasets ran through the same
chain: iris (LogisticRegression, CV 0.96) and abalone (SVR, R² 0.447).

Infra notes (no Docker on this machine): DATABASE_URL probes PostgreSQL and
falls back to SQLite **with the same models and the same applied Alembic
migration** (`86fe5aa40cab`, 4 tables + version stamp verified by direct
sqlite query). Celery probes Redis and falls back to its SQLAlchemy broker —
a real cross-process queue (API enqueues, separate worker process consumes).
`docker-compose.prod.yml` retains the Postgres/Redis path and now includes
`api` and `ollama` services. Runs #1–2 in the DB are artifacts of the debug
iterations (one orphaned by a broker bug, one failed on bug A27) — left in
place as honest history.

## 4. Fine-tuning numbers (Part E) — all measured

> **SMOKE TEST configuration**: base Qwen/Qwen2.5-0.5B-Instruct,
> `--backend transformers`, RTX A500 4 GB — NOT representative of the
> Mistral-7B recipes. The QLoRA wall-clock is inflated by heavy CPU/GPU
> contention (the Titanic pipeline + Ollama ran concurrently).

| Metric | QLoRA (SFT) | DPO (on QLoRA) |
|---|---|---|
| Train time | 3h 16m 06s (contended) | 14m 39s (exclusive) |
| Peak GPU memory | 5.91 GB* | 13.99 GB* |
| Final train loss | 0.3502 | 0.0 |
| Final eval loss | 0.3641 | ~0.0 |
| Data | 256 SFT examples, 1 epoch | 192 preference pairs, 1 epoch |

\* torch `max_memory_allocated` — exceeds the 4 GB card because CUDA on
Windows (WDDM) spills into shared system RAM.

DPO's zero loss is itself a real finding: the template-written "rejected"
examples are trivially separable; harder pairs (Ollama-written) are the
obvious next iteration.

**Benchmark** (12 held-out examples, judge = mistral:7b-instruct via Ollama,
absolute 1–5 rubric): full table in `finetuning/results/benchmark_results.md`.

| Model | Clarity | Accuracy | Overall | **Jargon leak %** |
|---|---|---|---|---|
| DPO | 5.00 | 4.92 | 4.96 | **0.0** |
| QLoRA | 5.00 | 4.92 | 4.96 | **0.0** |
| base (untuned) | 4.75 | 5.00 | 4.88 | **100.0** |

The deterministic jargon-leak metric is the meaningful result at this scale
(judge scores are near ceiling): the base model mentioned SHAP/statistics in
**every** answer; both tuned models in **none**.

## 5. Retrieval eval + MLflow (Part F) — measured

Corpus: 16 real explanation documents (from the 3 pipeline runs). Labels: 12
hand-written queries with binary relevance (`rag/eval_queries.json`). k=5.

| Config | recall@5 | nDCG@5 | MRR |
|---|---|---|---|
| bm25 | 0.7014 | 0.6156 | 0.6764 |
| dense (Qdrant local) | **0.8819** | **0.7227** | 0.6667 |
| hybrid (RRF) | 0.8194 | 0.7198 | **0.7361** |
| hybrid+rerank | 0.8056 | 0.7021 | 0.7153 |

Honest read: at 16 documents the cross-encoder adds nothing (nothing hard to
re-rank); dense wins recall, hybrid wins MRR. Rerun this as the corpus grows —
the harness takes one command.

MLflow: experiment `datapilot-finetuning` contains 3 FINISHED runs
(`qlora-Qwen2.5-0.5B-Instruct` with loss 0.3502/0.3641 and the DPO runs).
Verified via `mlflow.search_runs` and the UI is serving.

## 6. Working model + before/after (Parts G/H)

`datapilot-explainer` registered in Ollama (merge → f16 GGUF → Modelfile →
`ollama create`, all executed). Same real SHAP input from Titanic run 4
(persisted at `docs/_before_after_example.json`):

**Input (real SHAP context from run 4):**
```
Prediction: 1 Confidence: 90.9%
Top factors behind this prediction (SHAP values):
  - boat = 1.12 -> increases prediction (SHAP: +0.2195)
  - sex = -1.00 -> increases prediction (SHAP: +0.2017)
  - age = -1.93 -> increases prediction (SHAP: +0.0584)
  - pclass_3.0 = -1.00 -> increases prediction (SHAP: +0.0538)
  - body = -0.66 -> decreases prediction (SHAP: -0.0264)
```

**BEFORE — mistral:7b-instruct (prompted, production LLM):**
> "…Specifically, having a boat added 0.2195 to the prediction, the person's
> sex added 0.2017, and their passenger class increased it by 0.0538…"
> *(leaks raw SHAP numbers; rambles about features not in the input)*

**AFTER — datapilot-explainer (fine-tuned, 14× smaller):**
> "The model predicted '1' mainly because sex is -1.00, which strongly pushed
> the decision in this direction. Age (-1.93) added further weight to the
> same conclusion. On the other hand, body (-0.66) worked against this
> outcome, but not strongly enough. Changing sex would have the biggest
> effect on this prediction."
> *(clean structure, zero SHAP jargon, ends with the counterfactual — the
> trained format. Known smoke-scale limitation: it ranked sex above boat,
> whose SHAP is marginally higher, and echoes scaled feature values — both
> inherited from template-written targets.)*

To put it in production: `.env` → `OLLAMA_MODEL=datapilot-explainer`.

## 7. Open the running system

```powershell
# everything at once (4 windows):
powershell scripts\start_services.ps1

# individually:
python -m streamlit run ui/app.py --server.port 8501   # UI    -> http://localhost:8501
python -m uvicorn api.main:app --port 8000             # API   -> http://localhost:8000/docs
python -m mlflow ui --port 5000                        # MLflow-> http://127.0.0.1:5000
python -m celery -A tasks.worker worker --pool=solo --loglevel=info
& "$env:USERPROFILE\ollama-portable\ollama.exe" serve  # LLM   -> http://localhost:11434
```

All four HTTP services returned 200 during this pass. **One item needs your
hands:** clicking through the Streamlit UI in a browser (upload → run →
tabs). The identical orchestrator entry point was executed 3× via the API
chain, and the UI is reachable, but I cannot operate a browser — do one
upload of `datasets/titanic.csv` (target: `target`) to see it live.

## 8. Decisions I made without asking (review these)

1. **Legacy 32-feature RL path was UPDATED, not removed** — `rl_selector/train.py`
   depends on `environment.py`/`data_collection.py`, so I reconciled them to
   the shared 40-feature extractor + production model lists instead of
   deleting the retraining capability. Old 32-feature training JSONs (if any
   exist elsewhere) are now rejected with a clear error.
2. **Local fallbacks for missing Docker** (SQLite / SQLA-broker / embedded
   Qdrant) — always announced at startup, never silent; prod compose path
   unchanged.
3. **Smoke base model = Qwen2.5-0.5B-Instruct** — largest instruct model that
   trains in 4 GB; scripts keep Mistral-7B as the default.
4. **Exported DPO (not QLoRA)** — benchmark tie; DPO represents the full
   SFT→preference recipe.
5. **requirements.txt modernized** — the old langchain pins were mutually
   unresolvable by pip; replaced with the resolvable set actually installed;
   pyarrow pinned to 16.1.0 (documented crash with 24.x).
6. **Kept runs #1–2** (debug artifacts) in the database as honest history.
