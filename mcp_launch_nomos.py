from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SERVER = _ROOT / "mcp_servers" / "nomos_legal_db" / "mcp_server.py"
runpy.run_path(str(_SERVER), run_name="__main__")
