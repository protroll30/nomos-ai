"""LanceDB connection helpers (local-first, under `database/lancedb_data/`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import lancedb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = REPO_ROOT / "database" / "lancedb_data"
LEGAL_CHUNKS_TABLE = "legal_chunks"
VECTOR_DIM = 384  # matches common `sentence-transformers` encoders (e.g. MiniLM)


@lru_cache
def get_db_path() -> Path:
    return DEFAULT_DB_DIR


def connect():
    path = get_db_path()
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def list_table_names(db) -> list[str]:
    """Normalize LanceDB `list_tables()` response to table name strings."""
    return list(db.list_tables().tables)


def open_legal_table():
    db = connect()
    return db.open_table(LEGAL_CHUNKS_TABLE)
