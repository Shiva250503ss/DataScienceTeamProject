# Cleanup Plan — Part A Audit

Full-repo scan performed before any deletion. Methodology: filesystem walk
for cache/junk patterns, `git ls-files` to check tracked-vs-untracked status
of every candidate, `grep` across all `.py` files to confirm nothing
references a candidate path by name, and a direct query against `mlflow.db`
to verify its contents rather than guess.

---

## 1. SAFE TO AUTO-REMOVE (executed in Part B, no confirmation needed)

| Item | Count | Notes |
|---|---|---|
| `__pycache__/` directories | 17 dirs, 75 `.pyc` files | One per package (`agents/`, `api/`, `db/`, `finetuning/`, `rag/`, `rl_selector/`, `tasks/`, `ui/`, `utils/`, `orchestrator/`, `benchmark/`, `alembic/`, `alembic/versions/`, `scripts/`, `RL_MODEL_PPO_CORRECT/`, `ai_dashboard_generator/services/`, and repo root). Pure bytecode cache, regenerated automatically on next import. |
| `.ipynb_checkpoints/` | 0 found | None exist. |
| Editor/OS junk (`.DS_Store`, `Thumbs.db`, `*.swp`, `*~`) | 0 found | None exist. |
| Empty leftover folders | 0 found | None exist — every directory in the tree has at least one file. |
| Build artifacts (`dist/`, `build/`, `*.egg-info`) | 0 found | None exist (outside the vendored `llama.cpp/` clone, which is handled separately below). |
| Previously-identified dead code not yet removed | 0 found | Everything flagged as dead in a past audit (`datapilot/` duplicate directory, the dead SVR block in `agents/modeler.py`, the legacy 32-feature extractor) was already removed in a prior session — verified against `docs/AUDIT_LOG.md`'s resolution notes. Nothing outstanding. |

**Total Part B action: delete the 17 `__pycache__` directories. Nothing else qualifies for auto-removal.**

---

## 2. LIKELY REMOVABLE — ASK FIRST (not deleted; awaiting your decision per item)

| # | Item | Size | Tracked in git? | Why it's a candidate | My recommendation |
|---|---|---|---|---|---|
| 1 | `llama.cpp/` (repo root) | 189.3 MB | No (untracked) | Full clone of the external llama.cpp project, used only as build tooling for `export_ollama.py`'s GGUF quantization step. `cmake` was never installed locally, so `llama-quantize` was never actually built here — this checkout has not yet served its purpose on this machine. The new Colab notebook (`finetuning/DataPilot_Mistral7B_Colab.ipynb`) clones and builds its own copy independently. | Delete, unless you plan to install `cmake` locally and quantize on this machine too. |
| 2 | `finetuning/models/export/dpo_merged/` | 953.3 MB | No (untracked) | The **intermediate** merged fp16 HuggingFace model — a stepping stone `export_ollama.py` produces on the way to the GGUF, not the final artifact. The final artifact (`export/dpo_f16.gguf`, protected — see category 3) already exists downstream of it. `export_ollama.py` only keeps this around to skip a slow re-merge if you re-run the export with a different quant level. | Delete — regenerable from `finetuning/models/dpo/` (the adapter, protected) + the base model in ~10-15 min if ever needed again. |
| 3 | `output/data_analysis/`, `output/explanations/`, `output/visualizations/` (the loose top-level files, **not** `output/run_2` through `run_6`) | ~2.5 MB | **Yes, tracked/committed** | Sample pipeline outputs from before the API/Celery refactor introduced the `output/run_<id>/` numbering scheme (`tasks/worker.py` writes to `output/run_{run.id}/`; these loose files predate that and came from an earlier Streamlit-only run using the old default `output_dir="./output"`). No code references them by this literal path. | Delete — but flagging that these are git-tracked, so removal needs a `git rm` to take effect in the next commit, unlike everything else in this list. |
| 4 | `output_test/` | 24 files, ~0.9 MB | **Yes, tracked/committed** | Same vintage/shape as item 3 — a duplicate set of generic sample outputs (`shap_beeswarm.html`, `confusion_matrix.html`, etc.) with no unique identifying content. Already flagged as a cleanup candidate in a past audit (`docs/AUDIT_LOG.md` A18: "leftover run artifacts... noted for optional cleanup") — this pass is that follow-up. | Delete. |
| 5 | `output_test2/` | 24 files, ~0.9 MB | **Yes, tracked/committed** | Identical situation to item 4 — a second duplicate set. | Delete. |
| 6 | `uploads/` (6 CSVs: 4× `titanic.csv`, 1× `abalone.csv`, 1× `iris.csv`, each with a random hash prefix) | ~0.6 MB | No (untracked) | These are the actual files the FastAPI upload endpoint saved, and `datapilot.db`'s `datasets.upload_path` column (in the **protected** database) points at them by exact path. Deleting a file here would orphan its `Dataset` row — the completed `pipeline_runs`/`run_results` tied to it would be unaffected (their results are already persisted separately), but re-fetching or re-running against that dataset by ID would break. | **Keep**, or let me cross-check which of the 6 are still referenced by a `Dataset` row before removing any — I did not do that DB cross-check in this audit pass since it wasn't requested yet. |
| 7 | `logs/` (11 files: `api.log`, `celery.log`, `celery.err.log`, `dpo_run.log`, `benchmark_run.log`, `export_run.log`, `mlflow.log`, `mlflow.err.log`, `streamlit.log`, `streamlit.err.log`, `api.err.log`) | ~250 KB | No (untracked) | Session logs from running the services locally. Two of them (`celery.err.log`, `export_run.log`) contain the actual tracebacks that `docs/AUDIT_LOG.md` entries A27–A31 cite as evidence of real bugs found during live verification — deleting them doesn't invalidate the audit log's written conclusions, but removes the raw evidence backing them. Small total size. | Keep (low cost to keep; regenerate-on-next-run anyway, but no urgency to delete). |
| 8 | `iris.csv` (repo root) | 3.4 KB | **Yes, tracked/committed** | Small sample dataset. Referenced by name in `FULL_PROJECT_EXPLAINED.md` (a **protected** doc) as the example used to explain SHAP output for a classification walkthrough. | Keep — it's cited by name in a protected document. |
| 9 | `main.log` (repo root) | 1.6 KB | **Yes, tracked/committed** | Stray log file at repo root, no references found anywhere in code or docs. | Delete. |

**Nothing in this category has been deleted.** Reply with which numbers to delete (e.g. "delete 1, 2, 4, 5, 9; keep 3, 6, 7, 8") and I'll execute exactly that in Part C, or hold off entirely.

---

## 3. NEVER TOUCH (confirmed protected — not modified, not even considered for the ask-first list)

Everything you explicitly named:
- `docs/AUDIT_LOG.md`, `docs/JD_ALIGNMENT.md`, `docs/COLAB_FINETUNING.md`, `FULL_PROJECT_EXPLAINED.md`, `PROJECT_STATUS.md`, `README.md`
- `finetuning/results/` (`benchmark_results.md`, `generations_*.jsonl`) and `rag_store/` (`retrieval_eval_results.md`, `documents.json`, `knowledge_graph.json`, `qdrant_local/`) — all real measured numbers
- `datapilot.db` and `alembic/` (migration history)

Plus items I'm treating as protected **by extension of the same principle**, flagging this interpretation explicitly rather than silently deciding it:

| Item | Why I'm protecting it even though not named explicitly |
|---|---|
| `mlflow.db` (repo root, 933.9 KB) | I queried it directly rather than assumed: it contains exactly the 3 `FINISHED` MLflow runs (`dpo-qlora` ×2, `qlora-Qwen2.5-0.5B-Instruct`) that `PROJECT_STATUS.md` cites as evidence ("MLflow: experiment `datapilot-finetuning` contains 3 FINISHED runs with real losses"). Deleting it would invalidate a specific, verifiable claim in a protected document. |
| `finetuning/models/qlora/`, `finetuning/models/dpo/` (44.6 MB each) | The actual trained adapters (not intermediate checkpoints) — their `training_stats.json` files are the real numbers cited in `PROJECT_STATUS.md` and `finetuning/results/benchmark_results.md`. |
| `finetuning/models/export/dpo_f16.gguf` (948.1 MB) + `finetuning/models/export/Modelfile` | This is literally "the final exported adapter/GGUF" — `datapilot-explainer` in Ollama was registered from this exact file, and it's the subject of the before/after inference demo in `PROJECT_STATUS.md`. |
| `output/run_2` through `output/run_6/` | Untracked, recent, contain `local_narratives.json` files that feed the RAG corpus and are the underlying evidence for the real pipeline runs described in `PROJECT_STATUS.md` (e.g. run 4 = Titanic, RandomForest, 0.786 accuracy). Not touched, not even listed in category 2. |
| `RL_MODEL_PPO_CORRECT/` (entire directory) | Already resolved in a past audit (A19) as intentionally kept for provenance — it's the actual script that trained the shipped `rl_model_selector_*.pkl` files. Not re-litigated here. |
| `datasets/` (187.3 MB) | Every CSV has a matching `_meta.json` companion, no orphans or duplicates found — all actively used by `rl_selector` training/data collection and `benchmark/`. Not a cleanup target. |
| `benchmark/results_amlb_comparison/`, `results_quickcheck/`, `results_test/`, `results_validation/` | Reviewed in a past audit (A13) as legitimate historical benchmark outputs, not dead weight. Not re-litigated here. |
| `.claude/` (`settings.json`, `settings.local.json`) | Claude Code's own tooling config, outside the scope of a codebase cleanup. |

---

## Verification performed for this audit (so you can trust the categorization)

- Grepped all `.py` files for `output_test`, `dpo_merged`, `llama.cpp`, `iris.csv`, `main.log` — confirmed zero code paths reference the ask-first candidates by name (all `llama.cpp` mentions are either inside the vendored clone's own source or in `export_ollama.py`'s legitimate references to it as a build dependency).
- Ran `git ls-files` against every candidate directory/file individually to get exact tracked/untracked status (reported per-item above) rather than assuming.
- Directly queried `mlflow.db` with `mlflow.search_runs()` to confirm its actual contents instead of guessing from the filename.
- Confirmed no `.ipynb_checkpoints`, OS junk, empty directories, or build artifacts exist anywhere in the tree (outside the vendored `llama.cpp/`).
