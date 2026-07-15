# db/session.py

"""
Database engine + session factory.

Backend resolution (real database either way, no mocks):
  1. DATABASE_URL from .env / environment — PostgreSQL in production
     (docker-compose provides it at postgres:5432).
  2. If that server is unreachable (e.g. dev machine without Docker), fall
     back to SQLite at ./datapilot.db — a real file-backed database using
     the SAME models and the SAME Alembic migrations. A warning is printed
     so the fallback is never silent.

The resolved URL is cached so the probe only happens once per process.
"""

import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from utils.config import config

SQLITE_FALLBACK_URL = "sqlite:///./datapilot.db"

_engine = None
_SessionLocal = None
_resolved_url: Optional[str] = None


def resolve_database_url() -> str:
    """Probe the configured DB; fall back to SQLite with a visible warning."""
    global _resolved_url
    if _resolved_url:
        return _resolved_url

    url = os.getenv("DATABASE_URL", config.DATABASE_URL)
    if url.startswith("sqlite"):
        _resolved_url = url
        return url
    try:
        probe = create_engine(url, connect_args={"connect_timeout": 3})
        with probe.connect() as conn:
            conn.execute(text("SELECT 1"))
        probe.dispose()
        _resolved_url = url
    except Exception as e:
        print(f"[db] PostgreSQL unreachable ({type(e).__name__}) — "
              f"falling back to {SQLITE_FALLBACK_URL}. "
              f"Start docker-compose for the production database.")
        _resolved_url = SQLITE_FALLBACK_URL
    return _resolved_url


def get_engine():
    global _engine
    if _engine is None:
        url = resolve_database_url()
        kwargs = {}
        if url.startswith("sqlite"):
            # Needed because FastAPI + Celery access sessions from
            # different threads within one process during local dev.
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def create_all_tables():
    """
    Create tables directly from the models (used by tests and first-run
    convenience). Production schema changes go through Alembic migrations:
        alembic upgrade head
    """
    from db.models import Base
    Base.metadata.create_all(get_engine())
