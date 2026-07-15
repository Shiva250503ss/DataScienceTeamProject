# DataPilot AI Pro — Full Project Walkthrough (Interview Prep)

This document explains **every significant file** in pipeline order — the order data flows through the system. Read it top to bottom and it works as one rehearsal script.

For each file you get:
- **What it does** — based on the actual code, not the README
- **Why it was built this way** — the design reasoning
- **Say this out loud** — a short script for screen-sharing
- **Likely questions** — with short model answers

---

## 0. The 60-Second Elevator Pitch

Say this when they ask "walk me through your project":

> "DataPilot is an AutoML platform. You upload a CSV. Seven agents process it in sequence: profile, clean, engineer features, train models, visualize, and explain. A LangGraph orchestrator routes the data between agents. A PPO reinforcement-learning agent picks which ML algorithms to train, based on 40 meta-features of the dataset. The LLM parts run locally on Mistral-7B through Ollama — no API keys.
>
> On top of that base, I built two things on my own. First, a fine-tuning suite: I trained Mistral to turn raw SHAP output into plain-English explanations, and I compared six techniques — LoRA, QLoRA, DoRA, DPO, ORPO, and GaLore — on training time, GPU memory, and output quality with an LLM judge. Second, a RAG pipeline: hybrid retrieval over Qdrant plus BM25 with reciprocal rank fusion, cross-encoder reranking, query rewriting for the chat mode, and a small knowledge graph for multi-hop questions. Both are wired into the live agents, not standalone demos."

---

## 1. Honest Facts — Know These Before the Interview

These are true statements about the repo. If an interviewer digs, you must not be surprised:

1. **`api/`, `db/`, `tasks/` are empty stubs.** There is no FastAPI backend, no database models, no Celery tasks. The docker-compose worker service references `tasks.worker`, which does not exist yet. Say: *"Those are reserved for a planned FastAPI/Celery layer. The platform currently runs fully through Streamlit."*
2. **There used to be two meta-feature systems; now there's one.** The production path uses **40** meta-features (`meta_features.py`, shared with the real PPO training in `RL_MODEL_PPO_CORRECT/`). `rl_selector/environment.py` + `data_collection.py` originally embedded a diverging 32-feature extractor from an earlier iteration; both were reconciled to import the shared 40-feature `meta_features.py` and the production sklearn model lists. `test_agents.py` asserts `N_META_FEATURES` (40) and passes.
3. **The PPO models in production** (`rl_model_selector_classification.pkl`, `rl_model_selector_regression.pkl`) were trained by `RL_MODEL_PPO_CORRECT/train_rl_model_selector.py` on OpenML datasets, choosing among **sklearn-only models** (8 classifiers, 9 regressors). XGBoost/LightGBM/CatBoost appear in the UI list but the RL agent never recommends them — a user can still select them manually.
4. **The LLM was previously Gemini + Groq (cloud).** It is now Mistral-7B via Ollama (local) — that swap was part of my independent work. The optional cloud fallback still exists if keys are set in `.env`.
5. **The Insights tab embeds a second sub-project** (`ai_dashboard_generator/`) with its own services and LLM clients. It was developed separately and mounted into the main UI via `ui/insights_tab.py`.
6. **Qdrant was deployed but unused** until the RAG work. Now `rag/indexer.py` writes to it.

---

## 2. Entry Point — The UI

### `ui/app.py` (~950 lines)

**What it does.** The Streamlit app. Two tabs. Tab 1 "Data Insights" renders the AI dashboard generator. Tab 2 "Data Pipeline" is the AutoML flow: upload CSV → pick target column (radio list) → pick algorithm or "Auto-Select (PPO)" → run pipeline → results in eight sub-tabs (Models, Profile, Cleaning, Features, Visualizations, Explanations, Error Analysis, Predict).

**Why this way.** Heavy ML imports are lazy-loaded (`_get_pipeline_functions`) so the app starts fast. The target picker is mandatory (`index=None`) so users cannot run with a wrong auto-detected target. The Predict tab replays the exact same encoders and scalers from training on user-typed values — it reindexes the input row to the final feature list (`df_pred.reindex(columns=feature_names, fill_value=0)`), which automatically handles one-hot expansion and VIF-dropped columns.

**Say this out loud.** "The UI is thin. It collects the CSV, target, and model choice, then calls one function — `run_ml_pipeline` from the orchestrator. Everything else is displaying the state dict that comes back. The Predict tab is the trickiest part: it re-applies the stored label encoders, one-hot mappings, target encoders, and the scaler, in the same order as training, so predictions are consistent."

**Likely questions.**
- *Q: How do you avoid training/serving skew in the Predict tab?* — "The Feature agent stores every fitted encoder and scaler in the pipeline state. Predict replays them and then aligns columns with `reindex` to the exact training feature list. Same transforms, same order."
- *Q: Why Streamlit and not React?* — "Speed of iteration for a data product. The UI is not the point; the pipeline is."

### `ui/insights_tab.py` + `ai_dashboard_generator/`

**What it does.** Mounts the separate AI Dashboard Generator project inside Tab 1. It handles a nasty import problem: both projects have a `utils/` package, so it surgically swaps `sys.modules` entries so each project resolves its own `utils`. The sub-project profiles any CSV/Excel, auto-derives metrics, builds a rule-based dashboard, and optionally enhances it with the LLM. It has a chat mode ("chat with your data").

**Why this way.** The dashboard generator existed as its own repo. Mounting it was cheaper than rewriting it. The LLM clients (`ai_dashboard_generator/services/llm_clients.py`) now have an `OllamaClient` as primary with the same `generate`/`chat_completion` interface as the old Gemini/Groq clients — a drop-in swap.

**Say this out loud.** "Tab 1 is a separately-built dashboard generator we integrated. The interesting engineering is the module-name collision fix — both projects define `utils`, so the integration code evicts cached modules that came from the wrong root before importing."

---

## 3. Configuration and the LLM Layer

### `utils/config.py`

**What it does.** One dataclass, reads `.env`. Ollama URL and model first (`mistral:7b-instruct`), then optional cloud keys, ML settings, paths, and RAG model names.

**Say this out loud.** "Single config object. The default is fully local — Ollama plus two small sentence-transformers models. If someone sets a Gemini or Groq key, those become automatic fallbacks, but nothing requires them."

### `agents/base.py`

**What it does.** `BaseAgent` — the abstract parent of all seven agents. Gives each agent `self.llm` (built once by `_build_llm()`), `ask_llm(prompt)`, and `log()`. `_build_llm()` creates a `ChatOllama` pointed at local Mistral; if cloud keys exist it wraps it with LangChain's `with_fallbacks()`.

**Why this way.** One LLM entry point for the whole platform. Swapping Llama → Gemini → Mistral over the project's life only ever touched this one function.

**Likely questions.**
- *Q: Why Mistral-7B-Instruct?* — "Best quality-per-GB in the 7B class for instruction following, Apache-2.0 licensed, first-class GGUF/Ollama support, and Unsloth ships a pre-quantized 4-bit build — so the same base model serves inference AND fine-tuning."
- *Q: What happens if Ollama is down?* — "`with_fallbacks` retries the cloud models if keys exist. Otherwise `ask_llm` returns a clear setup message instead of crashing — every agent treats LLM output as optional enrichment."

---

## 4. The Orchestrator

### `orchestrator/state.py`

**What it does.** `PipelineState` — a TypedDict documenting every key each agent reads and writes: `raw_data`, `profile_report`, `current_data`, `X`/`y`, `trained_models`, `explanations`, etc.

**Say this out loud.** "The state is the contract between agents. Each agent reads what it needs, adds its outputs, and passes the dict on. This file documents who writes what."

### `orchestrator/graph.py`

**What it does.** Builds the LangGraph `StateGraph`. Conditional entry: `run_mode='ml_pipeline'` or `'both'` → Context Analyzer (LLM reads columns/samples and guesses domain, target, cleaning hints) → Profiler → Cleaner → Feature → Modeler → Visualizer → Explainer; `'data_analysis'` → straight to Data Analyzer. After Explainer, a conditional edge runs Data Analyzer too if mode is `'both'`. Every node wraps its agent in try/except with timing — an agent failure appends to `state['errors']` and the pipeline continues where possible.

**Why this way.** LangGraph gives conditional routing and a clean DAG for free. Agents are instantiated once at module level and reused — model loading (like the PPO pkl) happens once, not per run.

**Likely questions.**
- *Q: Why LangGraph instead of just calling functions in order?* — "Conditional routing between three modes, uniform error handling per node, and it makes the pipeline extensible — adding a node is two lines. It also matches how we think about the system: a graph of agents."
- *Q: What if the Modeler crashes?* — "The node catches it, logs into `state['errors']`, and downstream agents check for missing keys and skip gracefully. The UI shows the errors in an expander."

---

## 5. The Agents (in execution order)

### Stage 0 — `agents/data_analyzer.py::execute_context_phase`

**What it does.** Before any processing, sends column names, sample rows, and describe() stats to the LLM. Gets back JSON: domain, suggested target, task type, cleaning hints per column, feature ideas. Stored as `state['data_context']`; can set the target if the user did not.

**Why.** A human data scientist looks at the data before touching it. This gives every downstream agent "domain awareness" — e.g. the Cleaner logs LLM hints like "zero means missing in blood_pressure".

### Stage 1 — `agents/profiler.py` (~620 lines)

**What it does, step by step.**
0. Coerces "disguised numerics" — `"$1,234.56"`, `"50%"`, `"(500)"`, `"10 kg"` → floats (regex strip; convert if ≥80% of values parse).
1. Detects column types: numeric, categorical, binary, datetime, text, id (heuristics on dtype, cardinality, name patterns).
2. Auto-detects the target (name conventions → last column → lowest-cardinality categorical) if not given.
3. Decides classification vs regression (categorical target, or numeric with ≤20 unique → classification). The LLM's context suggestion can override.
4. Computes per-column stats (mean, std, skewness, missing %, top values).
5. Extracts **40 meta-features** via the shared `meta_features.py` — this exact vector goes to the PPO selector.
6. Quality score 0-100 (deductions for missing, duplicates, constants, high-cardinality).
7. Human-readable warnings.
8. describe()-based anomaly detection (max >> Q3+3·IQR, mean far from median, CV > 2, dominant values).
9. Category uniformity check ('Male'/'male'/'MALE' variants) — the Cleaner fixes what this finds.

**Say this out loud.** "The Profiler is the eyes of the pipeline. The most interesting part is the disguised-numeric coercion — real business CSVs are full of currency strings — and the 40 meta-features, because that vector is the observation the RL agent uses to pick models."

**Likely questions.**
- *Q: How do you detect an ID column?* — "Two paths: integer columns need BOTH an id-like name and >90% unique values; string columns just need >95% unique. That avoids dropping a real feature that happens to be unique."
- *Q: Why 20 unique values as the classification cutoff?* — "Heuristic. Integer targets like ratings 1-10 are classification; house prices with thousands of values are regression. The LLM context phase can override edge cases."

### `meta_features.py` (repo root — the single source of truth)

**What it does.** Computes the 40 normalized [0,1] meta-features: 6 basic (shape/type ratios), 3 missing, 10 statistical (skew/kurtosis/correlations/outlier rate), 3 categorical cardinality, 3 target properties, 3 PCA intrinsic-dimension, 4 landmark scores (3-fold CV of tiny models: decision tree, naive bayes, logistic regression, kNN), 8 signal features (feature-target correlations, nonlinearity gap = tree landmark minus linear landmark, class count, sparsity).

**Why.** Landmarking is the clever bit: quick cheap models act as probes. If a depth-3 tree already scores high but logistic regression doesn't, the data is nonlinear — the PPO agent learns that pattern favors tree ensembles. The file must produce identical output to the training script, or the PPO policy's inputs would be distribution-shifted — that is why it is one shared module.

- *Q: Why normalize everything to [0,1]?* — "The PPO observation space is `Box(0,1)`. Neural policies train poorly on unbounded features."

### Stage 2 — `agents/cleaner.py` (~450 lines)

**What it does.** Duplicate removal; missing-value strategy per column (numeric: <5% → mean/median by skewness, 5-30% → KNN imputer, >30% → median + missing-indicator column; categorical: <10% → mode, else a 'Missing' category; datetime: ffill/bfill); IQR outlier detection with winsorization only when outliers are <5% of rows; dtype fixing; category standardization using the Profiler's uniformity findings (most frequent variant wins).

**Say this out loud.** "Every cleaning decision is recorded in `cleaning_report`, so the UI can show exactly what was done to the data — auditability was a design goal. The strategy selection is adaptive: skewed columns get median, correlated columns get KNN, high-missing columns get an indicator so the model can learn from missingness itself."

- *Q: Why winsorize instead of dropping outliers?* — "Dropping rows loses signal in other columns. Clipping to the IQR fence keeps the row but caps its leverage. And if more than 5% of rows are 'outliers', that's probably real distribution shape, so we keep them untouched."

### Stage 3 — `agents/feature.py` (~470 lines)

**What it does.** Drops id/datetime/text columns; snapshots original column names for the Predict UI; encodes categoricals (2 values → label, ≤10 → one-hot with drop_first, >10 → target encoding with smoothing); feature selection only if >50 features (correlation filter at 0.95 then mutual-information top-50); **VIF multicollinearity removal** (iterative, threshold 10, skipped for <10 features; drops the candidate least correlated with the target, not blindly the highest-VIF one); scaling AFTER VIF (RobustScaler if mean |skew| > 1 else StandardScaler); label-encodes the target; final float64 enforcement.

**Say this out loud.** "Two details I like: VIF removal keeps the collinear feature that is more predictive of the target — petal length and petal width are collinear, but you keep whichever correlates more with the species. And scaling runs after VIF so the scaler is fitted on the final feature set — otherwise predict-time feature names mismatch."

- *Q: Why target encoding for high cardinality?* — "One-hot on 500 categories creates 500 columns. Target encoding replaces each category with a smoothed mean of the target — one column, keeps signal, smoothing prevents overfitting rare categories."
- *Q: VIF vs correlation filter — why both?* — "Pairwise correlation misses multicollinearity spread across 3+ features. VIF regresses each feature on all others, catching that."

### Stage 4 — `agents/modeler.py` (~1070 lines)

**What it does.** Asks the RL selector for top-3 models (or uses the user's explicit choice); trains each with Optuna hyperparameter search (25 trials / 90s cap, fewer for SVMs and big data) and 5-fold CV; falls back through simpler models if all fail; builds a soft-voting ensemble (hard voting if predict_proba unavailable); evaluates the ensemble with CV; picks the best of {3 singles, ensemble} as the final model; overfitting detection (score > 0.98 suspicion, train-vs-CV gap > 0.05, cross-model variance); error analysis on out-of-fold predictions (per-class error rates, confusion matrix; for regression the 10 worst predictions and error-by-value-range); comprehensive metrics via `cross_val_predict` (accuracy/precision/recall/F1/ROC-AUC or R²/MAE/MSE/RMSE).

**Say this out loud.** "Three things worth showing: Optuna tuning with model-specific search spaces and time budgets; the honesty checks — the pipeline flags its own suspicious scores instead of celebrating 99% accuracy; and everything is evaluated with out-of-fold predictions, never on training data."

- *Q: Why can a single model beat the ensemble?* — "Voting averages probabilities. If one model is clearly better and the others add noise, the average is worse. So we compare all four CV scores and ship the winner."
- *Q: How do you avoid leakage in the metrics?* — "`cross_val_predict` — every prediction comes from a fold that did not train on that row."

### The RL Model Selector

**Files:** `rl_selector/environment.py`, `train.py`, `data_collection.py` (all now share the 40-feature `meta_features.py` extractor), `inference.py` (production), `RL_MODEL_PPO_CORRECT/train_rl_model_selector.py` (the script that trained the shipped pkls, 40 features), `benchmark/` (evaluation harness incl. OpenML-CC18 comparison vs AutoML baselines).

**What it does.** Model selection framed as a one-step RL episode. Observation: the 40 meta-features. Action: pick one model from a fixed list (8 sklearn classifiers or 9 regressors). Reward: that model's real cross-validated score on that dataset, +0.1 bonus if it picked within 0.01 of the best. Training data: hundreds of OpenML datasets where ALL candidate models were trained and scored, so the environment can reward any action instantly (a form of offline/bandit RL). PPO with a 256→128→64 MLP policy. At inference (`inference.py`), the policy network's action probabilities are read directly and the top-3 become recommendations with confidences; sensible defaults if the pkl is missing.

**Say this out loud.** "It's a contextual bandit solved with PPO. One step per episode: see the dataset's meta-features, pick a model, get the model's real CV score as reward. Because we pre-computed every model's score on every training dataset, no model training happens inside the RL loop — that's what makes training feasible. At inference we don't just take the argmax; we read the full action distribution and return the top-3 with probabilities, and the Modeler trains all three plus an ensemble."

- *Q: Why RL and not a supervised classifier predicting the best model?* — "Supervised 'predict the winner' treats a model that loses by 0.001 the same as one that loses by 0.3. The RL reward is the actual score, so the policy learns to minimize regret, not just match labels. It also naturally handles ties and near-ties."
- *Q: Why is the action order hard-coded in inference.py?* — "The policy outputs an index. Index N must map to the same model name as during training, so the list order is frozen and documented."
- *Q: What did the benchmark show?* — "The `benchmark/` harness compares PPO picks against always-RandomForest and against each dataset's oracle best on OpenML datasets — the metric is regret. Results CSVs are in `benchmark/results_*`."

### Stage 5 — `agents/visualizer.py` (~1090 lines)

**What it does.** ~19 Plotly charts in 4 groups: profiling (types, missing, distributions, correlation heatmap, quality gauge), cleaning (before/after missing, outlier boxplots), features (importance, engineering summary), models (comparison, ranking table, confusion matrix or residuals, RL recommendations, ensemble-vs-best). Saves each as HTML; builds a combined dashboard.

**Say this out loud.** "Pure presentation layer — it reads the reports the other agents produced and turns them into charts. No LLM, no state mutation beyond adding figures."

### Stage 6 — `agents/explainer.py` (~1200 lines) — **the fine-tuning target**

**What it does.** Picks the best single model (SHAP on ensembles is awkward); TreeExplainer for tree models, KernelExplainer otherwise (with sample caps for slow models like SVR); handles multiclass SHAP shapes across SHAP versions; builds importance bar / beeswarm / waterfall / dependence-interaction charts; LIME explanations for ~5 diverse samples; **LLM narratives**: a global 6-section business report built from the whole pipeline's reports, and local per-prediction explanations from SHAP values.

**RAG integration (my work).** Before writing the global narrative it retrieves similar explanations from past runs (hybrid + rerank) and appends them as style references. After the run it persists `local_narratives.json` (narrative + the SHAP context that produced it — these become real fine-tuning examples), indexes all narratives into Qdrant/BM25, and records structured facts (feature→target SHAP strengths, strong feature-feature correlations) into the knowledge graph.

**Say this out loud.** "This agent is why I chose the fine-tuning task. At runtime, it formats SHAP values into a text block and asks the LLM to explain the prediction in plain English. That's exactly a (input, output) pair — so the fine-tune teaches Mistral to do this job natively, better and faster than prompting. And every real run now saves its pairs, so the training set grows with usage."

- *Q: TreeExplainer vs KernelExplainer?* — "TreeExplainer is exact and fast for tree models — polynomial algorithm over tree paths. KernelExplainer is model-agnostic but samples coalitions, so it's slow; we cap background and explained samples, more aggressively for O(n²) models like SVR."
- *Q: Why do you explain the best single model instead of the ensemble?* — "SHAP for a VotingClassifier means explaining every member and averaging — slow and muddy. The best single model is usually within a point of the ensemble and gives crisp attributions."

### The Data Analyzer — `agents/data_analyzer.py` (~2300 lines)

**What it does.** The "manager mode" agent. Auto mode: quick profile → LLM proposes 6-8 insights as JSON chart specs → each spec is validated by a significance gate (drops near-uniform comparisons and correlations below 0.15) → Plotly charts + LLM narratives + KPIs + data-quality warnings → combined dashboard. Prompt mode (`analyze_with_prompt`): user asks in natural language, LLM returns a chart spec JSON, chart gets built.

**RAG integration (my work).** In prompt mode: the query is first rewritten by `rag/query_transform.py` (resolves "it", "that" from chat history, maps vague words to real column names), then related past insights (hybrid + rerank) and knowledge-graph facts are appended to the LLM context. After auto mode, discovered insights are indexed into the RAG store.

**Say this out loud.** "The design principle is: the LLM decides WHAT to show, deterministic code decides HOW. The LLM only ever returns a JSON spec — column names, chart type, aggregation. Python validates the spec against real columns and builds the chart. The LLM never touches the data itself, which kills hallucinated numbers."

- *Q: What if the LLM returns an invalid spec?* — "JSON parsing is defensive (regex extraction, fallbacks), specs referencing missing columns are dropped, and if everything fails there's a rule-based fallback that builds standard overview charts."

---

## 6. RAG Optimization Pipeline (`rag/`) — Independent Work

Build order matters: each module improves on the previous one's weakness.

### `rag/embeddings.py` + `rag/indexer.py` (foundation)

**What.** MiniLM-L6-v2 local embedder (384-dim, lazy singleton). The indexer is the single write path: every document goes to Qdrant (dense vectors) AND a JSON mirror (the BM25 corpus). Content-hash IDs make re-indexing idempotent. Three doc types: explanations, insights, dataset profiles.

**Why the JSON mirror.** BM25 needs the whole corpus in memory to compute IDF. Scrolling it out of Qdrant per query would be slow; a local file loads in milliseconds. Both stores stay in sync because all writes go through one class.

- *Q: Why MiniLM and not a bigger embedder?* — "384 dims, 22M params, embeds thousands of docs per second on CPU. Retrieval quality is dominated by the reranker anyway — the embedder just needs decent recall."

### `rag/hybrid_search.py` (module 1)

**What.** Dense search (Qdrant) + sparse search (BM25Plus) fused with Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank)`.

**Why hybrid.** Dense understands paraphrase but fails on exact rare tokens — our corpus is full of snake_case column names like `debt_to_income_ratio`. BM25 nails identifiers but has zero semantics. They fail in opposite directions, so the union covers both. The tokenizer splits snake_case AND keeps the whole identifier, so both query styles match.

**Why RRF.** Cosine scores (~0-1) and BM25 scores (unbounded) are not comparable. RRF fuses on ranks only — no score normalization needed, one constant (k=60 from the original paper).

**A real bug I fixed (great interview story).** With BM25Okapi, a term appearing in 1 of 2 documents has IDF = ln(1.5/1.5) = **exactly zero** — on a young knowledge base every match scored 0 and sparse retrieval silently returned nothing. I switched to BM25Plus (which lower-bounds term contributions) and replaced the score>0 filter with an explicit token-overlap check.

### `rag/reranker.py` (module 2)

**What.** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescoring of the hybrid candidates: retrieve ~20 wide, rerank, keep top 3-5.

**Why.** Bi-encoders encode query and document separately — they cannot see interactions like negation or direction ("income REDUCES risk" vs "income INCREASES risk" embed nearly the same). A cross-encoder reads the pair jointly and fixes precision. It is too slow to run on the whole corpus, hence the two-stage retrieve-then-rerank design — the standard production RAG pattern.

### `rag/query_transform.py` (module 3)

**What.** Before retrieval, chat questions like "why is it dropping?" are rewritten by local Mistral into self-contained queries using the chat history and real column names; plus 2 paraphrase expansions to boost BM25 recall. A short-circuit skips the LLM call for long, specific first-turn questions. Falls back to the raw query on any failure.

**Why.** Retrieval quality is bounded by query quality. Multi-turn chat produces queries with pronouns and ellipsis that neither BM25 nor embeddings can resolve — only the conversation can. Rewrite once, use it for both retrieval and the chart-spec generation.

### `rag/graph_retrieval.py` (module 4)

**What.** A small JSON-backed knowledge graph. Nodes: datasets, columns, targets. Edges with plain-English "fact" sentences: `belongs_to`, `predicts` (weight = mean |SHAP|), `correlates_with` (weight = |r|). Built automatically after each Explainer run. Querying: fuzzy entity matching in the question → BFS up to 2 hops → return the traversed facts, nearest and strongest first.

**Why.** Vector search finds documents similar to the question. It cannot join evidence across runs — "how does monthly_charges relate to what drove churn?" needs two facts from different pipeline stages. A 2-hop graph walk answers it directly. Deliberately no graph database — adjacency dicts over JSON are enough at this scale and keep the demo dependency-free.

- *Q: Why not GraphRAG with an LLM building the graph?* — "Our facts are already structured — SHAP importances and correlation matrices come out of the pipeline as numbers. Extracting entities with an LLM would add cost and hallucination risk for zero gain. LLM-based graph extraction makes sense for unstructured text corpora; ours is structured."

### Where RAG is wired in (not standalone)

1. **Explainer** retrieves 3 reference explanations from past runs before writing the global narrative (style consistency, recurring patterns), and indexes its outputs after.
2. **Data Analyzer chat** rewrites the query, retrieves related insights + graph facts into the LLM context, and indexes discovered insights after auto-analysis.
3. Everything degrades gracefully: Qdrant down → BM25-only; models missing → RRF order; all failing → agents run exactly as before RAG existed.

---

## 7. Fine-Tuning Suite (`finetuning/`) — Independent Work

**The task.** Input: SHAP attribution text (prediction, features, values, signed SHAP scores). Output: 2-4 plain-English sentences. No jargon, grounded in the named features, ends with what would change the outcome. This is literally the Explainer agent's runtime job.

### `finetuning/common.py`

Shared constants so all six experiments are comparable: same base model (Mistral-7B-Instruct-v0.3), same `[INST]` prompt template, same data splits, same seed, and a `RunTracker` that measures wall-clock time and peak GPU memory identically everywhere. **If the data or format differed between experiments, the benchmark would be meaningless — that is the whole reason this file exists.**

### `finetuning/dataset_prep.py` — dataset optimization (a JD keyword!)

**What.** Generates synthetic SHAP scenarios across 8 business domains (churn, credit, housing, medical, HR, sales, fraud, marketing) with logically consistent signed attributions; writes good explanations (via local Ollama, or deterministic templates as fallback so the script always runs) and deliberately bad ones (three failure modes: jargon dump, vague filler, wrong driver); harvests REAL pairs from past Explainer runs (`local_narratives.json`). Then the optimization pass:
1. **Dedupe** — exact (md5 of normalized text) + near-duplicate (5-gram Jaccard > 0.7)
2. **Quality filter** — length bounds, jargon-leak check, degenerate-repetition check, and a groundedness check (the explanation must mention at least one input feature)
3. **Class balance** — downsampling to 1.5× the median (domain × prediction-direction) bucket; downsampling, never duplication
4. **Report** — before/after counts, filter reasons, domain coverage → `dataset_report.md`

Outputs both SFT data (input/output) and preference pairs (prompt/chosen/rejected) from the same scenarios.

- *Q: Why synthetic data — isn't that cheating?* — "The format is what matters: the SHAP text block is generated by our own pipeline code, so the synthetic inputs are distributionally identical to production inputs. And real pairs from actual runs are mixed in and grow over time. Synthetic bootstraps; real data takes over."
- *Q: Why downsample instead of oversample?* — "Duplicated examples cause memorization, and an LLM judge would reward the memorized phrasing. Downsampling keeps every example unique."

### The six experiments

| Script | One-sentence pitch |
|---|---|
| `finetune_lora.py` | Freeze the 7B, train low-rank matrices beside each linear layer (`W + BA`, r=16 ≈ 0.6% of params) — Unsloth kernels make it 2× faster. |
| `finetune_qlora.py` | Same adapters on a 4-bit NF4 base — the QLoRA paper's three tricks (NF4, double quantization, paged optimizers) drop VRAM to ~7 GB, single consumer GPU. Only variable vs LoRA is base precision, so the benchmark isolates the quantization cost. |
| `finetune_dora.py` | Decompose each weight into magnitude × direction, adapt the direction low-rank and the magnitude separately — closer to how full fine-tuning moves weights; one flag in PEFT (`use_dora=True`). |
| `finetune_dpo.py` | Train on chosen-vs-rejected pairs against a frozen reference: RLHF's objective in closed form, no reward model, no PPO rollouts. Runs on top of the QLoRA adapter (SFT-then-DPO, standard practice). LR is 40× lower than SFT; 1 epoch. |
| `finetune_orpo.py` | SFT + preference in ONE stage via an odds-ratio penalty — no reference model, no separate SFT run, about half the total compute of the DPO path. Head-to-head with DPO on identical pairs. |
| `finetune_galore.py` | (Experimental) True full-parameter training in ~24 GB by keeping Adam's states in a low-rank projection of the GRADIENT — the frontier beyond adapters. |

**Say this out loud (the comparison story).** "LoRA vs QLoRA isolates quantization cost. LoRA vs DoRA isolates the decomposition trick at the same rank. DPO vs ORPO is two-stage-with-reference versus one-stage-without. GaLore asks whether full-parameter training is worth it at all here. One task, one dataset, one seed — six points on the quality/memory/time frontier."

- *Q: Explain DPO's loss in one breath.* — "Maximize the margin by which the policy prefers the chosen answer over the rejected one, measured relative to a frozen reference model so it can't drift into gibberish; beta controls that leash."
- *Q: Why is 'a good explanation' a preference problem and not SFT?* — "There is no single gold explanation — many are fine. What we reliably know is that grounded plain English beats jargon. Pairs encode exactly that knowledge; a single target string doesn't."
- *Q: LoRA rank — why 16?* — "Common sweet spot for a narrow task on 7B. Higher ranks add capacity we don't need and slow training; the benchmark would show if r=16 underfit — eval loss would plateau high."

### `finetuning/benchmark.py`

**What.** For every trained adapter (plus the untuned base as a baseline): generate on the held-out test set, then score each output 1-5 on **clarity** and **accuracy** with an LLM judge (local Mistral via Ollama, JSON-constrained, temperature 0), plus deterministic checks (jargon-leak %, length). Combines with the training-time stats into one markdown table.

- *Q: Judge biases?* — "Absolute scoring kills position bias; the rubric explicitly says short-beats-long to counter verbosity bias; self-preference exists but is constant across candidates, so the RANKING stays meaningful. The jargon-leak regex is a judge-free sanity metric alongside."
- *Q: Why is eval loss not comparable for DPO/ORPO?* — "They optimize preference losses, not next-token cross-entropy. Compare them on the judge scores."

### `finetuning/export_ollama.py`

**What.** Merge the winning adapter into fp16 weights → convert to GGUF via llama.cpp → quantize q4_k_m (~4.4 GB) → write a Modelfile (Mistral chat template + the Explainer system prompt baked in) → `ollama create datapilot-explainer`. Then one `.env` line (`OLLAMA_MODEL=datapilot-explainer`) puts the fine-tuned model into production. No code changes, no API key, no token limits.

**Say this out loud.** "This closes the loop: the platform generates training data, the suite trains and benchmarks, the winner gets merged, quantized, and served by the same Ollama the agents already talk to."

---

## 8. Infrastructure

- **`docker-compose.yml`** — dev: Postgres, Redis, Qdrant, Ollama. **`docker-compose.prod.yml`** — adds the Streamlit app container (with `OLLAMA_BASE_URL=http://ollama:11434`) and an optional Celery worker profile (worker code not yet implemented). **`Dockerfile`** — two-stage build, non-root user, healthcheck.
- **`requirements.txt`** — platform deps incl. RAG (sentence-transformers, rank-bm25, qdrant-client). **`finetuning/requirements-finetuning.txt`** — training deps kept separate because they are heavy and CUDA-specific.
- **`test_agents.py`** — synthetic-data test of Profiler + Cleaner (types, target detection, imputation, outliers, standardization). Its meta-feature count check asserts `N_META_FEATURES` (40) and passes.
- **`benchmark/`** — PPO selector evaluation: regret vs oracle, comparison on OpenML-CC18 against AutoML baselines; results CSVs committed.
- **`DST_Paper.pdf` / `main.tex`** — the academic write-up of the base platform.

---

## 9. Cross-Cutting Questions You Should Expect

**Q: What was YOUR contribution vs the team's?**
"The base platform — agents, orchestrator, UI, RL selector — was collaborative. My independent work: the switch to fully local LLM serving via Ollama/Mistral, the entire `finetuning/` experiment suite, the entire `rag/` pipeline, and wiring both into the Explainer and Data Analyzer agents."

**Q: Why fine-tune at all — why not just prompt harder?**
"Three reasons. Consistency: prompting still leaks jargon a few percent of the time; the benchmark's jargon-leak metric quantifies the fix. Latency and cost: a tuned 7B needs no long few-shot prompt. And capability: the model internalizes the format, so it stays robust when the SHAP input gets messy."

**Q: How would you scale this?**
"Retrieval: Qdrant already scales horizontally; move BM25 to the same store or Elasticsearch when the corpus outgrows memory. Training: the suite is single-GPU by design; multi-GPU means FSDP/DeepSpeed in the TrainingArguments. Serving: Ollama for single-node; vLLM with the merged weights for throughput."

**Q: What would you do next?**
"Three things. Evaluate the fine-tuned model against the judge continuously as real narratives accumulate — the dataset grows with usage. Add retrieval evaluation (recall@k on a labeled query set) instead of only end-task quality. And implement the reserved FastAPI/Celery layer so long pipelines run async."

**Q: What was the hardest bug?**
Pick one of these true stories:
1. "BM25Okapi scoring exactly zero on small corpora — IDF is ln(1.5/1.5)=0 for a term in half the docs. Silent empty results. Fixed with BM25Plus plus an explicit token-overlap filter."
2. "Two projects with colliding `utils` packages in one Streamlit process — fixed by evicting wrongly-cached modules from `sys.modules` before import."
3. "SHAP returns different shapes per version and model type — list of arrays, 3D arrays — normalized in one place with explicit dimension handling."
