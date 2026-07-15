# api/main.py

"""
DataPilot AI Pro — FastAPI backend.

Endpoints (see /docs for the interactive OpenAPI UI):

  GET  /health                     liveness + which DB/broker are in use
  POST /datasets                   upload a CSV (multipart) -> Dataset row
  GET  /datasets                   list uploaded datasets
  POST /runs                       queue a pipeline run (Celery task)
  GET  /runs                       list runs with status
  GET  /runs/{run_id}              status of one run
  GET  /runs/{run_id}/results      all persisted results (metrics, SHAP,
                                   narratives, reports) once completed
  GET  /explanations               RAG-indexed explanation documents (SQL
                                   mirror of the vector store)

Architecture: the API never runs the pipeline in-request. POST /runs writes
a PipelineRun row (status=queued) and dispatches tasks.run_pipeline to the
Celery worker, which executes the real LangGraph orchestrator and persists
results. Clients poll GET /runs/{id} until status=completed.

Run locally:
    uvicorn api.main:app --port 8000
    celery -A tasks.worker worker --pool=solo --loglevel=info   (second shell)
"""

import os
import shutil
import uuid
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.dependencies import get_db
from db.models import Dataset, ExplanationDoc, PipelineRun, RunResult
from utils.config import config

app = FastAPI(
    title="DataPilot AI Pro API",
    description="AutoML pipeline as a service — upload data, queue runs, "
                "fetch models/explanations. Async execution via Celery.",
    version="1.0.0",
)

UPLOAD_DIR = config.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DatasetOut(BaseModel):
    id: int
    name: str
    n_rows: int
    n_cols: int
    columns: list

    class Config:
        from_attributes = True


class RunRequest(BaseModel):
    dataset_id: int
    target_column: Optional[str] = None   # auto-detected if omitted
    run_mode: str = "ml_pipeline"         # ml_pipeline | data_analysis | both
    user_selected_model: Optional[str] = None  # None => PPO auto-select


class RunOut(BaseModel):
    id: int
    dataset_id: int
    status: str
    run_mode: str
    target_column: Optional[str]
    task_type: Optional[str]
    best_model_name: Optional[str]
    best_score: Optional[float]
    error: Optional[str]
    celery_task_id: Optional[str]

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness + which concrete backends this process resolved to."""
    from db.session import resolve_database_url
    from tasks.worker import celery_app
    return {
        "status": "ok",
        "database": resolve_database_url().split("@")[-1],  # no credentials
        "celery_broker": celery_app.conf.broker_url.split("@")[-1],
        "ollama": config.OLLAMA_BASE_URL,
    }


@app.post("/datasets", response_model=DatasetOut, status_code=201)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV. Stored on disk; shape metadata goes to the database."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted")

    dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{file.filename}")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        df = pd.read_csv(dest)
    except Exception as e:
        os.remove(dest)
        raise HTTPException(400, f"File is not parseable CSV: {e}")

    ds = Dataset(
        name=os.path.splitext(file.filename)[0],
        filename=file.filename,
        upload_path=dest,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=df.columns.tolist(),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


@app.get("/datasets", response_model=List[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.id.desc()).all()


@app.post("/runs", response_model=RunOut, status_code=202)
def create_run(req: RunRequest, db: Session = Depends(get_db)):
    """
    Queue a pipeline run. Returns immediately with status=queued; the Celery
    worker picks it up, executes the LangGraph orchestrator, and persists
    results. Poll GET /runs/{id} for status.
    """
    ds = db.get(Dataset, req.dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")
    if req.run_mode not in ("ml_pipeline", "data_analysis", "both"):
        raise HTTPException(400, "run_mode must be ml_pipeline | data_analysis | both")
    if req.target_column and req.target_column not in ds.columns:
        raise HTTPException(400, f"target_column '{req.target_column}' not in "
                                 f"dataset columns {ds.columns}")

    run = PipelineRun(
        dataset_id=ds.id,
        status="queued",
        run_mode=req.run_mode,
        target_column=req.target_column,
        user_selected_model=req.user_selected_model,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Dispatch to the real Celery worker (import here so the API can still
    # serve reads if the broker is down — queueing would fail loudly instead).
    from tasks.worker import run_pipeline_task
    async_result = run_pipeline_task.delay(run.id)
    run.celery_task_id = async_result.id
    db.commit()
    db.refresh(run)
    return run


@app.get("/runs", response_model=List[RunOut])
def list_runs(db: Session = Depends(get_db)):
    return db.query(PipelineRun).order_by(PipelineRun.id.desc()).all()


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


@app.get("/runs/{run_id}/results")
def get_run_results(run_id: int, kind: Optional[str] = None,
                    db: Session = Depends(get_db)):
    """
    All persisted results for a run, keyed by kind (metrics, cv_scores,
    shap_importance, global_narrative, local_narratives, profile_report,
    cleaning_report, error_analysis, model_recommendations, ...).
    Filter with ?kind=metrics for a single payload.
    """
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")

    query = db.query(RunResult).filter(RunResult.run_id == run_id)
    if kind:
        query = query.filter(RunResult.kind == kind)
    results = query.all()
    if run.status != "completed" and not results:
        return {"run_id": run_id, "status": run.status,
                "detail": "No results yet — run has not completed."}
    return {
        "run_id": run_id,
        "status": run.status,
        "best_model": run.best_model_name,
        "best_score": run.best_score,
        "output_dir": run.output_dir,
        "results": {r.kind: r.payload for r in results},
    }


@app.get("/explanations")
def list_explanations(dataset_name: Optional[str] = None, limit: int = 50,
                      db: Session = Depends(get_db)):
    """RAG-indexed explanation documents (SQL mirror of the vector store)."""
    query = db.query(ExplanationDoc).order_by(ExplanationDoc.id.desc())
    if dataset_name:
        query = query.filter(ExplanationDoc.dataset_name == dataset_name)
    docs = query.limit(limit).all()
    return [{"doc_id": d.doc_id, "doc_type": d.doc_type,
             "dataset_name": d.dataset_name, "text": d.text,
             "run_id": d.run_id} for d in docs]
