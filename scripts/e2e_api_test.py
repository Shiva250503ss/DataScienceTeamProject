# scripts/e2e_api_test.py

"""
End-to-end integration test of the API -> Celery -> DB -> orchestrator chain.

Prerequisites (three shells, or the -Background helpers below):
    uvicorn api.main:app --port 8000
    celery -A tasks.worker worker --pool=solo --loglevel=info
    (optional) Ollama running for LLM narratives

What it does — every step hits REAL infrastructure:
  1. GET  /health                       — API is up, shows resolved DB/broker
  2. POST /datasets (multipart upload)  — dataset row lands in the database
  3. POST /runs                         — run row created, Celery task queued
  4. poll GET /runs/{id}                — worker picks it up (status: running)
  5. wait for completed                 — orchestrator ran all agents
  6. GET  /runs/{id}/results            — metrics/SHAP/narratives from the DB
  7. direct SQL check                   — counts rows in every table

Usage:
    python scripts/e2e_api_test.py [--csv datasets/titanic.csv] [--target target]
                                   [--model RandomForestClassifier] [--timeout 3600]
"""

import argparse
import json
import os
import sys
import time

import httpx

# Allow running from anywhere: put the repo root on sys.path for step 7's
# direct database check.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/titanic.csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--model", default=None,
                        help="user_selected_model (None => PPO auto-select)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds to wait for run completion")
    args = parser.parse_args()

    client = httpx.Client(timeout=60)

    # 1. Health
    health = client.get(f"{API}/health").json()
    print(f"[1] /health -> {health}")

    # 2. Upload dataset
    with open(args.csv, "rb") as f:
        resp = client.post(f"{API}/datasets",
                           files={"file": (args.csv.split("/")[-1].split("\\")[-1],
                                           f, "text/csv")})
    resp.raise_for_status()
    ds = resp.json()
    print(f"[2] uploaded dataset id={ds['id']} "
          f"({ds['n_rows']} rows x {ds['n_cols']} cols)")

    # 3. Queue run
    payload = {"dataset_id": ds["id"], "target_column": args.target,
               "run_mode": "ml_pipeline"}
    if args.model:
        payload["user_selected_model"] = args.model
    resp = client.post(f"{API}/runs", json=payload)
    resp.raise_for_status()
    run = resp.json()
    print(f"[3] queued run id={run['id']} celery_task={run['celery_task_id']} "
          f"status={run['status']}")

    # 4/5. Poll until the worker finishes
    t0 = time.time()
    seen_running = False
    while True:
        time.sleep(10)
        run = client.get(f"{API}/runs/{run['id']}").json()
        elapsed = int(time.time() - t0)
        if run["status"] == "running" and not seen_running:
            seen_running = True
            print(f"[4] worker picked up the task (status=running) after {elapsed}s")
        if run["status"] in ("completed", "failed"):
            print(f"[5] run finished: status={run['status']} after {elapsed}s")
            break
        if elapsed > args.timeout:
            print(f"TIMEOUT after {elapsed}s (status={run['status']})")
            sys.exit(2)

    if run["status"] == "failed":
        print(f"RUN FAILED: {run['error']}")
        sys.exit(1)

    print(f"    best_model={run['best_model_name']} score={run['best_score']} "
          f"task_type={run['task_type']}")

    # 6. Results from the DB via API
    results = client.get(f"{API}/runs/{run['id']}/results").json()
    kinds = sorted(results["results"].keys())
    print(f"[6] /runs/{run['id']}/results -> {len(kinds)} result kinds: {kinds}")
    metrics = results["results"].get("metrics", {})
    print(f"    metrics: {json.dumps(metrics)}")
    narrative = results["results"].get("global_narrative")
    if narrative:
        print(f"    narrative (first 200 chars): {narrative[:200]}...")

    # 7. Direct database check (no API in the loop)
    from db.models import Dataset, ExplanationDoc, PipelineRun, RunResult
    from db.session import get_session_factory
    s = get_session_factory()()
    counts = {t.__tablename__: s.query(t).count()
              for t in (Dataset, PipelineRun, RunResult, ExplanationDoc)}
    s.close()
    print(f"[7] direct SQL row counts: {counts}")

    print("\nE2E CHAIN VERIFIED: API -> DB -> Celery -> orchestrator -> DB -> API")


if __name__ == "__main__":
    main()
