# api/dependencies.py

"""FastAPI dependencies — one DB session per request, always closed."""

from typing import Generator

from sqlalchemy.orm import Session

from db.session import create_all_tables, get_session_factory

_tables_ready = False


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; ensure the schema exists on first use."""
    global _tables_ready
    if not _tables_ready:
        create_all_tables()
        _tables_ready = True
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
