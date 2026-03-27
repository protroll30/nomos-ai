"""Create local LanceDB instance and the `legal_chunks` table if missing."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow as pa

from database.client import LEGAL_CHUNKS_TABLE, VECTOR_DIM, connect, list_table_names


def _legal_chunks_schema() -> pa.schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("citation", pa.string()),
            pa.field("instrument", pa.string()),
            pa.field("section_ref", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def init_legal_corpus_table() -> None:
    db = connect()
    if LEGAL_CHUNKS_TABLE in list_table_names(db):
        return
    empty = pa.Table.from_pylist([], schema=_legal_chunks_schema())
    db.create_table(LEGAL_CHUNKS_TABLE, data=empty)


def main() -> int:
    init_legal_corpus_table()
    db = connect()
    names = list_table_names(db)
    row_count = db.open_table(LEGAL_CHUNKS_TABLE).count_rows() if LEGAL_CHUNKS_TABLE in names else 0
    print(f"LanceDB ready at {db.uri}. Tables: {names}. legal_chunks rows: {row_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
