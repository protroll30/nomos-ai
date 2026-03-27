"""
Nomos legal corpus MCP server — exposes local LanceDB to Cursor via MCP (stdio).

Run from repository root:
    python -m mcp_servers.nomos_legal_db.mcp_server
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from database.client import LEGAL_CHUNKS_TABLE, connect, list_table_names

mcp = FastMCP("Nomos Legal DB")


@mcp.tool()
def nomos_legal_db_stats() -> dict:
    """Summarize local LanceDB tables and row counts for the legal corpus."""
    db = connect()
    names = list(db.list_tables())
    out: dict[str, int | str] = {"uri": db.uri, "tables": len(names)}
    for name in names:
        out[f"rows:{name}"] = db.open_table(name).count_rows()
    return out


@mcp.tool()
def nomos_legal_text_search(query: str, limit: int = 20) -> list[dict]:
    """
    Case-insensitive substring search over `text` (full scan; for small corpora / diagnostics).
    """
    if not query.strip():
        return []
    q = query.lower()
    db = connect()
    if LEGAL_CHUNKS_TABLE not in list_table_names(db):
        return []
    tbl = db.open_table(LEGAL_CHUNKS_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return []
    mask = df["text"].astype(str).str.lower().str.contains(q, na=False)
    hits = df.loc[mask].head(max(1, min(limit, 100)))
    return hits[["id", "citation", "instrument", "section_ref", "text"]].to_dict(orient="records")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
