"""Convert synthetic_all_rows.json to chat JSONL for Llama-3 / OpenAI-style SFT."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "synthetic_all_rows.json"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "eval.jsonl"

SYSTEM_TEXT = (
    "You are Nomos, an expert AI compliance auditor enforcing the EU AI Act "
    "(Regulation 2024/1689). Your task is to review FastAPI architectures and output a "
    "strict JSON response. Assume the system is classified as high-risk (Annex III)."
)


def _strip_inline_hash_outside_strings(line: str) -> str:
    """Remove ``# ...`` at end of line when ``#`` is not inside ' or \" string literals."""
    in_single = False
    in_double = False
    escape = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and (in_single or in_double):
            escape = True
            i += 1
            continue
        if not in_double and c == "'" and not in_single:
            in_single = True
        elif in_single and c == "'":
            in_single = False
        elif not in_single and c == '"' and not in_double:
            in_double = True
        elif in_double and c == '"':
            in_double = False
        elif not in_single and not in_double and c == "#":
            return line[:i].rstrip()
        i += 1
    return line


def scrub_python_comments(code: str) -> str:
    """
    Strip single-line Python comments: full lines starting with # (regex) and inline #
    outside string literals (quote-aware tail strip).
    """
    # Whole-line comments only (regex).
    code = re.sub(r"^\s*#.*$", "", code, flags=re.MULTILINE)
    out_lines: list[str] = []
    for raw_line in code.splitlines():
        if not raw_line.strip():
            out_lines.append(raw_line)
            continue
        out_lines.append(_strip_inline_hash_outside_strings(raw_line))
    return "\n".join(out_lines)


def row_to_messages(row: dict) -> dict:
    bad = scrub_python_comments(str(row.get("non_compliant_code", "")))
    good = scrub_python_comments(str(row.get("compliant_fix", "")))
    user_content = (
        "Audit the following code for EU AI Act compliance:\n\n"
        f"```python\n{bad}\n```"
    )
    assistant_obj = {
        "legal_anchor": row["legal_anchor"],
        "clause": row["clause"],
        "violation_subtlety": row["violation_subtlety"],
        "uncertainty_factor": row["uncertainty_factor"],
        "compliance_justification": row["compliance_justification"],
        "compliant_fix": good,
    }
    assistant_content = json.dumps(assistant_obj, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> int:
    raw = INPUT_PATH.read_text(encoding="utf8")
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON array")

    random.seed(42)
    random.shuffle(rows)

    formatted: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        r = {k: v for k, v in row.items() if not str(k).startswith("_")}
        formatted.append(row_to_messages(r))

    n = len(formatted)
    n_train = (n * 9) // 10
    train = formatted[:n_train]
    eval_ = formatted[n_train:]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w", encoding="utf8") as ft:
        for obj in train:
            ft.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with EVAL_PATH.open("w", encoding="utf8") as fe:
        for obj in eval_:
            fe.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        f"OK: wrote {len(train)} rows to {TRAIN_PATH} and {len(eval_)} rows to {EVAL_PATH} "
        f"(from {n} shuffled examples)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
