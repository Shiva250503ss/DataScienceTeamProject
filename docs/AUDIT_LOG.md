# Codebase Audit Log — Stubs, Hardcoded Values, Dummy Data, Dead Code

Full-repo scan performed BEFORE fixing anything (Part A of the final pass).
Machine facts at audit time: Windows 11, Python 3.11.9, NVIDIA RTX A500 (4 GB
VRAM, CUDA 13.2 driver), 31.6 GB RAM, Docker NOT installed, Ollama NOT installed.

Legend: 🔴 stub/dummy that must be implemented · 🟠 conflict/dead code ·
🟡 reviewed, acceptable as-is (documented reason) · ✅ fixed in an earlier session

## 🔴 Stubs (empty files posing as modules)

| # | Location | Finding | Resolution |
|---|----------|---------|------------|
| A1 | `api/main.py`, `api/dependencies.py`, `api/__init__.py` | 0 bytes. README advertised "FastAPI backend (extensible)" — there is no API at all. | Implement real FastAPI app (Part B). |
| A2 | `db/__init__.py` | 0 bytes. No models, no session, no migrations, despite Postgres in docker-compose. | Implement SQLAlchemy models + Alembic migrations (Part B). |
| A3 | `tasks/__init__.py` | 0 bytes. `docker-compose.prod.yml` worker service runs `celery -A tasks.worker worker` — **that module does not exist**; the worker container would crash on boot. | Implement real Celery app + task (Part B). |

## 🟠 Conflicting / dead code paths

| # | Location | Finding | Resolution |
|---|----------|---------|------------|
| A4 | `rl_selector/environment.py` | Legacy **32**-feature observation space (`Box(shape=(32,))`); returns `np.random.rand(32)` as dummy observations; model list includes XGB/LGBM/CatBoost which the production policy was never trained on. Conflicts with the production 40-feature system. | Rewrite to 40 features via shared `meta_features.py`, production model lists, zero-vector (not random) terminal observations (Part C). |
| A5 | `rl_selector/data_collection.py` | Contains an embedded **duplicate 32-feature extractor** (~160 lines) that silently diverges from `meta_features.py`. Trains 10-11 models incl. boosting libs the production policy can't recommend. | Replace embedded extractor with the shared 40-feature one; align model lists with production (Part C). |
| A6 | `rl_selector/train.py` | Docstrings say 32 features / "10-11 candidate models"; both wrong for production. Depends on A4+A5 (this is the only dependent of the legacy path). | Update to match reconciled 40-feature path (Part C). |
| A7 | `test_agents.py:167` | Asserts "Exactly 32 meta-features extracted" — the profiler produces 40. Test was failing-by-design. | Assert `N_META_FEATURES` (40) imported from the shared module (Part C). |
| A8 | `datapilot/` (dir) | Stale duplicate copies of `agents/modeler.py` and `meta_features.py`. **Imported by nothing** (verified by grep). | Delete (Part C). |
| A9 | `agents/modeler.py` ~lines 638-642 | Duplicate dead `if model_name == 'SVR':` block — unreachable (identical condition returned 6 lines earlier). | Remove dead block. |
| A10 | `ai_dashboard_generator/app/streamlit_app.py` | Standalone entry of the sub-project still auto-connects Groq-from-secrets first (the integrated path via `ui/insights_tab.py` is already Ollama-first). | Default to OllamaClient, same as the integrated path. |

## 🔴 Hardcoded / environment-dependent values

| # | Location | Finding | Resolution |
|---|----------|---------|------------|
| A11 | `rag/indexer.py` | Assumed a Qdrant **server** at `QDRANT_URL`; with no Docker on this machine the dense index was permanently disabled — "deployed but unused" all over again. | Add embedded local-mode fallback (`qdrant_client` local storage) so the dense index is real even without a server; server still preferred when reachable. |
| A12 | `tasks` broker assumption | Celery was implied Redis-only; Redis requires Docker here. | Broker URL comes from config; local integration testing uses Celery's SQLAlchemy transport against the same DB, prod compose keeps Redis. Documented, not hardcoded. |
| A13 | `benchmark/quick_check.py` | Fine — thin wrapper calling the real benchmark with tiny settings; no fake data. | 🟡 No change. |

## 🟡 Reviewed and acceptable (not stubs — documented why)

| # | Location | Finding |
|---|----------|---------|
| A14 | ~30 bare `pass` statements across agents/, ai_dashboard_generator/, benchmark/ | All are `except:`-handler fallbacks (defensive degradation), not unimplemented functions. `agents/base.py:100` is an `@abstractmethod` body — correct Python. |
| A15 | `test_agents.py` synthetic dataset, `finetuning/dataset_prep.py` scenario generator | Random data is the *point* (controlled test fixtures / synthetic training scenarios with documented generation logic), not dummy stand-ins for real data. |
| A16 | Cleaning/feature thresholds (5%/30% missing, IQR×1.5, VIF 10, cardinality 10/20…) | Documented heuristics in docstrings; standard practice values. Config-lifting them adds indirection without benefit. |
| A17 | `rl_selector/data_collection.py` failure score 0.5/0.0 | Documented penalty score when a candidate model errors during collection — intentional. |
| A18 | Root-level `iris.csv`, `main.log`, `output_test*/` dirs | Leftover run artifacts; harmless, kept (git history). Noted for optional cleanup. |
| A19 | `RL_MODEL_PPO_CORRECT/train_rl_model_selector.py` said "32 meta-features" in 3 places (module docstring line 17, section header line 518, policy comment ~line 980) | The *file's own code* always built 40 features (the shipped pkls match 40 — verified against `meta_features.py` which mirrors it); these 3 spots were stale prose self-contradicting the adjacent correct docstrings/code. **Updated in final verification pass (2026-07-15)**: all 3 now say 40. The `"32f"` pseudocode variable-name shorthand (8 occurrences, e.g. `32f = env.reset()`) was left as-is — informal notation, not a factual claim, and renaming it is cosmetic. |

## 🔴 Found and fixed DURING live verification (final pass — real runs, not code reading)

| # | Location | Finding | Resolution |
|---|----------|---------|------------|
| A27 | `agents/feature.py` | `TargetEncoder.fit_transform(X[col], y)` crashes with "'numpy.ndarray' object has no attribute 'groupby'" whenever the target is a string Series (every classification run with a text target — category_encoders 2.7 internally converts non-numeric y to ndarray). Killed the whole Titanic pipeline. | Label-encode y once before target encoding (numeric y is what target encoding mathematically needs anyway). Verified: Titanic feature stage passes, 25 features. |
| A28 | `agents/profiler.py` + `cleaner.py` | Disguised-numeric coercion CONCATENATES multi-number codes: Titanic `cabin` "B57 B59 B63" → 575963… → 3.1e42, which is finite in float64 but overflows float32 → sklearn "value too large for dtype" killed RF/DT training AND SHAP. | (a) Coercion now rejects columns where >20% of values contain two whitespace-separated digit groups; (b) final X guards in feature.py + modeler.py clip to ±float32 max. Verified: run 4 completed with all 10 result kinds incl. shap_importance. |
| A29 | `finetuning/finetune_dpo.py` env | Hard 0xC0000005 access violation on Windows: pyarrow **24.0** + torch 2.6 + bitsandbytes in one process (crash location nondeterministic). | Downgraded to pyarrow **16.1.0** (datasets 2.21-compatible); also build arrow Datasets BEFORE loading the CUDA model. Verified: DPO trains. |
| A30 | `finetuning/finetune_dpo.py` | trl 0.9.6 `DPOTrainer` prepends `tokenizer.bos_token_id` unconditionally; Qwen2 tokenizers have **no BOS** (None) → `torch.tensor([None, …])` TypeError at step 0. | Set `bos_token = eos_token` when missing. Verified: DPO training progresses. |
| A31 | `tasks/worker.py` | `broker_transport_options={"polling_interval": …}` is forwarded verbatim to `create_engine()` by kombu's SQLAlchemy transport → crash. | Removed; option only valid for Redis/SQS transports. Verified: worker connects and consumes. |

## ✅ Already fixed in the previous session (for completeness)

| # | Location | Finding |
|---|----------|---------|
| A20 | `agents/base.py` + `utils/config.py` + `.env.example` | README claimed Ollama/Llama-3.1; code actually called Gemini + Groq cloud APIs with keys. Now Ollama/Mistral-first with optional cloud fallback. |
| A21 | `agents/explainer.py` | `NameError`: `background` used before assignment in the TreeExplainer→KernelExplainer fallback. |
| A22 | `rag/hybrid_search.py` | BM25Okapi IDF = ln(1.5/1.5) = 0 on small corpora → silent empty retrieval. Switched to BM25Plus + token-overlap filter. |
| A23 | `orchestrator/state.py` | Comment claimed 32 meta-features (actual: 40). |
| A24 | `docker-compose*.yml` | README referenced a `datapilot-ollama` container that existed in no compose file. Added. |
| A25 | `ui/app.py` footer | Said "Powered by … Groq" while README said Ollama. Aligned. |
| A26 | `finetuning/dataset_prep.py` | Template writer produced self-contradicting explanations (opposing factor logic). Fixed relative to top driver. |
