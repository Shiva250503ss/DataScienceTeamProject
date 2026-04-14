# DataPilot AI Pro

**Enterprise Autonomous Data Science Platform** — Upload a CSV, get ML models, explainability, and business insights automatically.

---

## Features

| Feature | Description |
|---------|-------------|
| **AutoML Pipeline** | Profiling → Cleaning → Feature Engineering → RL-powered Model Selection → Ensemble Training → SHAP/LIME Explanations |
| **Data Analyzer** | LLM-powered business intelligence — auto-discovers insights for managers with interactive Plotly dashboards |
| **Explainability** | SHAP global/local importance, LIME explanations, AI-generated narratives |
| **RL Model Selector** | PPO agent trained on real OpenML datasets to recommend the best algorithm |
| **LangGraph Orchestrator** | Full pipeline coordination with conditional routing between ML and analysis modes |

---

## Quick Start

### Option A — Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start infrastructure (Postgres, Redis, Qdrant, Ollama)
docker-compose up -d

# 3. Pull the LLM model
docker exec datapilot-ollama ollama pull llama3.1:8b

# 4. (Optional) Train RL model selector on real data
python -m rl_selector.train --collect --n_datasets 30 --task classification --timesteps 50000
python -m rl_selector.train --collect --n_datasets 20 --task regression --timesteps 50000

# 5. Run the UI
streamlit run ui/app.py
```

### Option B — Docker (Production)

```bash
# Build and run everything
docker compose -f docker-compose.prod.yml up --build -d

# With Celery worker for async jobs:
docker compose -f docker-compose.prod.yml --profile worker up --build -d
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
├── agents/
│   ├── base.py            # BaseAgent ABC (Ollama LLM integration)
│   ├── profiler.py        # Data profiling — types, stats, quality score
│   ├── cleaner.py         # Missing values, outliers, duplicates
│   ├── feature.py         # Encoding, scaling, feature selection
│   ├── modeler.py         # RL-guided model training + ensemble
│   ├── visualizer.py      # 19 chart types across 4 groups
│   ├── explainer.py       # SHAP + LIME + LLM narratives
│   └── data_analyzer.py   # LLM-powered business insights
├── orchestrator/
│   ├── state.py           # PipelineState TypedDict
│   └── graph.py           # LangGraph StateGraph (7 nodes)
├── rl_selector/
│   ├── environment.py     # Gymnasium env (32 meta-features → model selection)
│   ├── train.py           # PPO training with real OpenML data
│   ├── inference.py       # RLModelSelector for production
│   └── data_collection.py # OpenML dataset collection & evaluation
├── ui/
│   └── app.py             # Streamlit web interface (3 modes)
├── utils/
│   └── config.py          # Config dataclass
├── api/                   # FastAPI backend (extensible)
├── db/                    # Database models (extensible)
├── tasks/                 # Celery async tasks (extensible)
├── tests/                 # Unit & integration tests
├── Dockerfile             # Multi-stage production image
├── docker-compose.yml     # Dev infrastructure
├── docker-compose.prod.yml# Production compose (all services)
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variables
```

---

## Architecture

```
                    ┌──────────────┐
                    │  Streamlit   │   ← User uploads CSV
                    │   UI (app)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LangGraph   │   ← Routes: ML / Analysis / Both
                    │ Orchestrator │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌─────────┐  ┌──────────┐  ┌─────────────┐
       │Profiler │  │  Data     │  │  Explainer  │
       │Cleaner  │  │ Analyzer  │  │ (SHAP/LIME) │
       │Feature  │  │  (LLM)   │  └─────────────┘
       │Modeler  │  └──────────┘
       │Visualizer│
       └─────────┘
              │
       ┌──────▼───────┐
       │ RL Selector  │   ← PPO recommends best model
       │ (PPO Agent)  │
       └──────────────┘
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://datapilot:datapilot@localhost:5432/datapilot` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama LLM server |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `PPO_MODEL_PATH` | `rl_selector/models/ppo_model_selector` | RL model path |

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **RL**: stable-baselines3 (PPO), Gymnasium, PyTorch
- **LLM**: LangChain, LangGraph, Ollama (llama3.1:8b)
- **Explainability**: SHAP, LIME
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data**: OpenML (real datasets), category_encoders
- **Infrastructure**: PostgreSQL, Redis, Qdrant, Docker
- **Frontend**: Streamlit
