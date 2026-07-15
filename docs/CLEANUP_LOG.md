# Cleanup Log

Factual record of the cleanup pass. Full reasoning/audit trail is in
`docs/CLEANUP_PLAN.md`.

## Auto-removed (Part B, category 1 — no confirmation needed)

- 17 `__pycache__/` directories, 75 `.pyc` files (regenerated automatically
  by Python; deleted twice more over the course of this pass as
  verification imports regenerated them — 0 remain as of the last check).

## Approved / rejected from the ask-first list (Part B, category 2)

User approved: **1, 2, 3, 4, 5, 9**. Executed exactly that — nothing else.

| # | Item | Decision | Result |
|---|---|---|---|
| 1 | `llama.cpp/` (189.3 MB) | Approved — deleted | Removed (untracked, plain delete) |
| 2 | `finetuning/models/export/dpo_merged/` (953.3 MB) | Approved — deleted | Removed (untracked, plain delete) |
| 3 | `output/{data_analysis,explanations,visualizations}/` loose files | Approved — deleted | Removed via `git rm -r` (was tracked); empty parent shells cleaned up after (git doesn't track/remove empty dirs) |
| 4 | `output_test/` | Approved — deleted | Removed via `git rm -r` (was tracked); same empty-shell cleanup |
| 5 | `output_test2/` | Approved — deleted | Removed via `git rm -r` (was tracked); same empty-shell cleanup |
| 9 | `main.log` | Approved — deleted | Removed via `git rm` (was tracked) |
| 6 | `uploads/` | Not approved — kept | No action |
| 7 | `logs/` | Not approved — kept | No action |
| 8 | `iris.csv` | Not approved — kept | No action |

**Freed: ~1.25 GB.** Repo working-tree size (excluding `.git`) went from
~2,485 MB to 1,239.5 MB.

Items 3, 4, 5, 9 were git-tracked — their removal is **staged** (`git rm`),
not committed. Nothing was pushed or committed during this pass.

## Reorganization (Part C)

**C.1 — move stray root scripts into proper folders**: nothing moved.
Checked every candidate against actual import paths and protected-doc
references before deciding, per your instruction not to guess:
- `meta_features.py` — imported via bare `from meta_features import ...`
  from 6 files across 4 subpackages (`agents/profiler.py`,
  `rl_selector/environment.py`, `rl_selector/data_collection.py`,
  `benchmark/amlb_comparison.py`, `benchmark/ppo_benchmark.py`,
  `test_agents.py`). Moving it means updating all 6 — left in place rather
  than risk the pipeline for a cosmetic move.
- `test_agents.py` — referenced by name in 3 protected docs
  (`PROJECT_STATUS.md`, `FULL_PROJECT_EXPLAINED.md`, `docs/AUDIT_LOG.md`).
  Left in place per your instruction to confirm first; **flagging for your
  decision** — the import itself (`sys.path.insert(0, ".")`, CWD-relative)
  would survive a move into a `tests/` folder as long as it's still invoked
  from the repo root, but the protected docs' references to the bare
  filename would become stale.
- `rl_model_selector_classification.pkl` / `rl_model_selector_regression.pkl`
  — referenced by name in `FULL_PROJECT_EXPLAINED.md` (protected), loaded
  via a root-relative path in `rl_selector/inference.py`, and also
  referenced by the historical, provenance-protected scripts under
  `RL_MODEL_PPO_CORRECT/`. Left in place; **flagging for your decision** —
  moving is mechanically a 1-line fix in `inference.py`, but touches a
  protected-doc reference and the historical scripts.
- Nothing else at repo root is a stray script — the rest
  (`.dockerignore`, `.env.example`, `alembic.ini`, `docker-compose*.yml`,
  `Dockerfile`, `requirements.txt`, `DST_Paper.pdf`, `main.tex`) are
  config/document files that conventionally live at the repo root.

**C.2 — standardize naming**: nothing to do. Re-checked for `_v2`,
`_final`, `_old`, `_new`, `_copy`, `_backup`, `.bak` patterns across the
whole tree (excluding `.git`) both in the Part A audit and again after the
deletions — zero matches both times.

**C.3 — `.gitignore`**: updated. Added, verified with `git check-ignore`:
`__pycache__/`, `*.py[cod]`, `.ipynb_checkpoints/`, `datapilot.db`,
`mlflow.db`, `mlruns/`, `finetuning/models/`, `llama.cpp/`,
`rag_store/qdrant_local/` (only the embedded vector-index cache — verified
`rag_store/documents.json`, `knowledge_graph.json`, and
`retrieval_eval_results.md` remain trackable), `uploads/`, `logs/`,
`output/`.

## Verification (Part D)

- `python test_agents.py`: **31/31 checks passed (100%)** — matches the
  pre-cleanup baseline exactly (same count as recorded in
  `PROJECT_STATUS.md`).
- Broader smoke test (not part of the formal suite, run for extra
  confidence given the scope of deletions): every core subsystem still
  imports cleanly (`agents`, `orchestrator`, `rag`, `db`, `api`, `tasks`,
  `finetuning`); `rl_selector.inference` still loads both root-level
  `.pkl` models (`use_rl=True`, device=cuda); the RAG corpus still loads
  its 16 real documents; the knowledge graph still loads its 33 nodes /
  81 edges.
- Nothing broke. No fixes were needed.

## Protected content — final confirmation

Re-verified present and untouched after all deletions: `output/run_2`
through `run_6`, `finetuning/models/{qlora,dpo}/`,
`finetuning/models/export/{dpo_f16.gguf,Modelfile}`, `mlflow.db`,
`datapilot.db`, `iris.csv`, `uploads/`, `logs/`, and all six named
protected docs plus `rag_store/`'s real-results files.
