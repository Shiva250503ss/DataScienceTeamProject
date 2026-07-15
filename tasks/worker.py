# tasks/worker.py

"""
Celery application + the pipeline task.

Broker resolution (a real broker either way, no eager/mock mode):
  1. CELERY_BROKER_URL env var, else REDIS_URL from config — Redis in
     production (docker-compose provides it; docker-compose.prod.yml's
     worker service runs exactly this module: `celery -A tasks.worker worker`).
  2. If Redis is unreachable (dev machine without Docker), fall back to
     Celery's SQLAlchemy transport on the same local database — a real
     broker: the worker genuinely polls a queue table and the API process
     genuinely enqueues into it, across separate OS processes.

Windows note: run the worker with `--pool=solo` (prefork is POSIX-only):
    celery -A tasks.worker worker --pool=solo --loglevel=info
"""

import datetime
import os

from celery import Celery

from utils.config import config


def _resolve_broker() -> tuple:
    """Return (broker_url, result_backend_url) — probe Redis, else SQLA."""
    broker = os.getenv("CELERY_BROKER_URL", config.REDIS_URL)
    try:
        import redis as _redis
        _redis.Redis.from_url(broker, socket_connect_timeout=2).ping()
        return broker, os.getenv("CELERY_RESULT_BACKEND", broker)
    except Exception:
        # Fall back to the SQLAlchemy transport on the resolved database.
        from db.session import resolve_database_url
        db_url = resolve_database_url()
        sqla_broker = f"sqla+{db_url}"
        sqla_backend = f"db+{db_url}"
        print(f"[tasks] Redis unreachable — using SQLAlchemy broker on "
              f"{db_url} (start docker-compose for Redis in production)")
        return sqla_broker, sqla_backend


_broker_url, _backend_url = _resolve_broker()

celery_app = Celery("datapilot", broker=_broker_url, backend=_backend_url)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
# NOTE: broker_transport_options must stay empty for the SQLAlchemy
# transport — kombu forwards them verbatim to create_engine(), so options
# like polling_interval (valid for Redis/SQS) crash the SQLA broker.


@celery_app.task(name="tasks.run_pipeline", bind=True)
def run_pipeline_task(self, run_id: int) -> dict:
    """
    Execute one pipeline run end-to-end and persist everything.

    Steps:
      1. Load the PipelineRun + Dataset rows, mark the run 'running'
      2. Read the uploaded CSV and invoke the REAL LangGraph orchestrator
         (run_ml_pipeline / run_data_analysis — the same code path the
         Streamlit UI uses)
      3. Persist structured outputs as RunResult rows (metrics, SHAP
         importance, narratives, reports) and RAG-indexed explanation
         documents as ExplanationDoc rows
      4. Mark the run 'completed' (or 'failed' with the error)

    Runs inside the Celery worker process, so the API request that queued it
    returns immediately.
    """
    import pandas as pd

    from db.models import Dataset, ExplanationDoc, PipelineRun, RunResult
    from db.session import create_all_tables, get_session_factory

    create_all_tables()
    session = get_session_factory()()

    run = session.get(PipelineRun, run_id)
    if run is None:
        session.close()
        raise ValueError(f"PipelineRun {run_id} not found")

    dataset = session.get(Dataset, run.dataset_id)
    run.status = "running"
    run.started_at = datetime.datetime.now(datetime.timezone.utc)
    run.celery_task_id = self.request.id
    session.commit()

    try:
        df = pd.read_csv(dataset.upload_path)
        output_dir = os.path.join("output", f"run_{run.id}")
        os.makedirs(output_dir, exist_ok=True)
        run.output_dir = output_dir
        session.commit()

        # ── The real orchestrator — same entry points as the UI ──────────
        from orchestrator.graph import run_data_analysis, run_ml_pipeline
        if run.run_mode == "data_analysis":
            result = run_data_analysis(df, dataset_name=dataset.name,
                                       output_dir=output_dir)
        else:
            result = run_ml_pipeline(
                df,
                target_column=run.target_column,
                dataset_name=dataset.name,
                output_dir=output_dir,
                user_selected_model=run.user_selected_model,
            )

        # ── Persist structured outputs ─────────────────────────────────
        def add_result(kind: str, payload):
            if payload is None:
                return
            session.add(RunResult(run_id=run.id, kind=kind, payload=payload))

        add_result("metrics", result.get("comprehensive_metrics"))
        add_result("cv_scores", result.get("cv_scores"))
        add_result("model_recommendations", [
            {"model": m, "confidence": c}
            for m, c in (result.get("model_recommendations") or [])
        ] or None)
        add_result("profile_report", _json_safe(result.get("profile_report")))
        add_result("cleaning_report", _json_safe(result.get("cleaning_report")))
        add_result("error_analysis", _json_safe(result.get("error_analysis")))
        add_result("overfitting_analysis",
                   _json_safe(result.get("overfitting_analysis")))

        explanations = result.get("explanations") or {}
        add_result("global_narrative", explanations.get("global_narrative"))
        add_result("local_narratives",
                   _json_safe(explanations.get("local_narratives")))
        shap_imp = explanations.get("shap_importance")
        if shap_imp is not None and hasattr(shap_imp, "to_dict"):
            add_result("shap_importance", shap_imp.to_dict("records"))

        analysis = result.get("data_analysis") or {}
        if analysis:
            add_result("insights", _json_safe(analysis.get("insights")))
            add_result("analysis_summary", analysis.get("summary"))

        # ── Mirror RAG-indexed explanation docs into SQL ────────────────
        try:
            from rag.indexer import RagIndexer
            for doc in RagIndexer.load_corpus():
                if doc["dataset_name"] == dataset.name:
                    exists = session.query(ExplanationDoc).filter_by(
                        doc_id=doc["id"]).first()
                    if not exists:
                        session.add(ExplanationDoc(
                            run_id=run.id, doc_id=doc["id"],
                            doc_type=doc["doc_type"],
                            dataset_name=doc["dataset_name"],
                            text=doc["text"],
                            doc_metadata=doc.get("metadata", {}),
                        ))
        except Exception as rag_err:
            print(f"[tasks] RAG doc mirroring skipped: {rag_err}")

        run.status = "completed"
        run.task_type = result.get("task_type")
        run.target_column = result.get("target_column", run.target_column)
        run.best_model_name = result.get("best_model_name")
        score = result.get("ensemble_score")
        run.best_score = float(score) if score is not None else None
        errors = result.get("errors") or []
        if errors:
            run.error = " | ".join(str(e) for e in errors)[:4000]
        run.finished_at = datetime.datetime.now(datetime.timezone.utc)
        session.commit()

        return {"run_id": run.id, "status": run.status,
                "best_model": run.best_model_name, "score": run.best_score}

    except Exception as e:
        session.rollback()
        run.status = "failed"
        run.error = str(e)[:4000]
        run.finished_at = datetime.datetime.now(datetime.timezone.utc)
        session.commit()
        raise
    finally:
        session.close()


def _json_safe(obj):
    """Recursively convert numpy/pandas scalars so JSON columns accept them."""
    import numpy as np
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
