# db/models.py

"""
SQLAlchemy models — persistence layer for the platform.

Four tables:
  datasets          — uploaded CSV files (path + shape metadata)
  pipeline_runs     — one row per pipeline execution (status machine:
                      queued -> running -> completed | failed)
  run_results       — the structured outputs of a run, one row per result
                      kind (metrics, shap_importance, narrative, reports...)
                      stored as JSON so the API can serve them directly
  explanation_docs  — mirror of RAG-indexed explanation documents, so the
                      knowledge base is queryable with SQL as well as vector
                      search (doc_id matches the RAG store's content hash)

Design notes:
  - SQLAlchemy 2.0 declarative style with typed Mapped[] columns.
  - JSON columns use the portable sqlalchemy.JSON type: JSONB on PostgreSQL
    (production, via docker-compose) and TEXT-serialized JSON on SQLite
    (local development without Docker). Same code, both backends.
  - Timestamps are UTC, set server-side where possible.
"""

import datetime
from typing import Any, Optional

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String,
                        Text, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    upload_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[Any] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["PipelineRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued",
                                        nullable=False)  # queued/running/completed/failed
    run_mode: Mapped[str] = mapped_column(String(32), default="ml_pipeline")
    target_column: Mapped[Optional[str]] = mapped_column(String(255))
    user_selected_model: Mapped[Optional[str]] = mapped_column(String(128))
    task_type: Mapped[Optional[str]] = mapped_column(String(32))
    best_model_name: Mapped[Optional[str]] = mapped_column(String(128))
    best_score: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(Text)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64))
    output_dir: Mapped[Optional[str]] = mapped_column(String(1024))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True))

    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    results: Mapped[list["RunResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan")


class RunResult(Base):
    __tablename__ = "run_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_runs.id"), nullable=False)
    # kind: metrics | cv_scores | shap_importance | global_narrative |
    #       local_narratives | cleaning_report | feature_report |
    #       profile_report | model_recommendations | error_analysis
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Any] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())

    run: Mapped["PipelineRun"] = relationship(back_populates="results")


class ExplanationDoc(Base):
    __tablename__ = "explanation_docs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pipeline_runs.id"))
    doc_id: Mapped[str] = mapped_column(String(32), unique=True,
                                        nullable=False)  # RAG content hash
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
