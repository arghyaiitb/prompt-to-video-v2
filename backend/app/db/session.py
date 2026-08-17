"""SQLite engine + FastAPI session dependency.

Two writers touch this file: request handlers on the event loop's threadpool and the
pipeline worker. So:
  * ``check_same_thread=False`` — the connection pool hands connections across threads.
  * WAL journal — readers (status polling, every ~1s) never block on the writer.
  * ``busy_timeout`` — a poll that lands mid-write waits instead of raising "database is
    locked".

The engine is created lazily so tests can swap in a throwaway file before first use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

_engine: Engine | None = None


def _apply_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def make_engine(db_path: Path) -> Engine:
    """Build an engine for `db_path` with the SQLite settings this app needs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _apply_pragmas)
    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine(get_settings().db_path)
    return _engine


def set_engine(engine: Engine | None) -> None:
    """Test seam: point the whole app at a different database (or reset with None)."""
    global _engine
    _engine = engine


def init_db() -> None:
    """Create tables, then add any columns an older DB is missing. Idempotent.

    ``create_all`` creates missing *tables* only — it never alters one that already
    exists. A dev database from a previous release therefore needs the additive column
    migration in ``app.db.models``, or every INSERT fails on the new columns.
    """
    # Import for side effect: registers Job on SQLModel.metadata before create_all.
    from app.db import models

    SQLModel.metadata.create_all(get_engine())
    models.migrate(get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(get_engine()) as session:
        yield session


def session_scope() -> Session:
    """Non-dependency session for the worker thread. Caller closes it (use `with`)."""
    return Session(get_engine())
