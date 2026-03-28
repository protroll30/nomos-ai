"""Generate SFT rows: default path is DeepSeek-only (golden few-shot + synthetic rows); optional Claude legacy phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from database.init_lancedb import init_legal_corpus_table
from database.client import LEGAL_CHUNKS_TABLE, VECTOR_DIM, connect, list_table_names

SYSTEM_PROMPT = """Role: You are a Senior Regulatory ML Engineer generating supervised fine-tuning examples for a compliance model. Outputs must be high-fidelity and statistically diverse.

LEGAL GROUNDING (NON-NEGOTIABLE)

No Hallucinations: Use only Regulation (EU) 2024/1689. Do not invent articles.

Real Citations: Reference real Articles (e.g., Art. 8–15) or Annexes.

Clause Extraction: Use a faithful paraphrase or a verbatim excerpt from the user-provided golden set.

Legal Anchor: Every JSON object must include a legal_anchor string (e.g., "Art. 14(4), Reg. 2024/1689").

TASK: GENERATE SYNTHETIC TRAINING ROWS
Each row must include:

"clause": The specific legal obligation.

"violation_subtlety": One of overt (obvious violation) | borderline (debatable/nuanced) | omission (missing a required component).

"uncertainty_factor": A float from 0.1 to 1.0.

0.1-0.3: Extremely subtle; likely to pass a basic linter but fail a deep audit.

0.8-1.0: Blatant violation of safety/transparency.

"non_compliant_code": Realistic Python/FastAPI code showing the violation.

"chain_of_thought": Step-by-step reasoning. Crucial: For low uncertainty_factor items, explain the legal ambiguity.

"compliant_fix": The corrected code snippet.

"complexity": simple_route | middleware | background_task | dependency_injection.

BAYESIAN DIVERSITY REQUIREMENT

30% of examples should be borderline. These are the most valuable for training the Bayesian Posterior Density curves.

Vary the coding styles (e.g., some use Pydantic v2, some use raw dictionaries, some use custom decorators).

DOMAIN / USE-CASE DIVERSITY (FICTIONAL CONTEXT ONLY)

Across batches, vary the implied business context in non_compliant_code and compliant_fix: route names, docstrings, and variable semantics should suggest different high-risk deployer settings (e.g. clinical decision support, HR screening, credit or fraud scoring, remote proctoring, public-sector eligibility). Use plausible fictional product or company names. This is for scenario diversity only—legal_anchor and clause must still follow only real obligations from the provided regulation excerpts and golden set; do not invent Articles or map industries to fake rules.

OUTPUT FORMAT

Valid JSON array of objects.

No markdown fences inside strings."""

# Prepended to every DeepSeek system message (API text; ** reads as emphasis in markdown UIs).
DEEPSEEK_JSON_DISCIPLINE = """**OUTPUT RAW, VALID JSON ONLY. DO NOT WRAP IN MARKDOWN. DO NOT ADD CONVERSATIONAL TEXT. YOUR ENTIRE RESPONSE MUST START WITH '[' AND END WITH ']'.**"""

REQUIRED_KEYS = (
    "legal_anchor",
    "clause",
    "violation_subtlety",
    "uncertainty_factor",
    "non_compliant_code",
    "chain_of_thought",
    "compliant_fix",
    "complexity",
)

SUBTLETY = frozenset({"overt", "borderline", "omission"})
COMPLEXITY = frozenset({"simple_route", "middleware", "background_task", "dependency_injection"})

DEFAULT_CLAUDE_MODEL = "claude-opus-4-6-20260219"
# DeepSeek-V3.2-Speciale: reasoning-first; API does not use tool-calling for this variant (HF + DeepSeek docs).
# Hugging Face local deployment recommends temperature=1.0, top_p=0.95 for V3.2 / Speciale.
DEFAULT_DEEPSEEK_MODEL = "deepseek-v3.2-speciale"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_TEMPERATURE = 1.0
DEFAULT_DEEPSEEK_TOP_P = 0.95
PHASE1_PATH = ROOT / "data" / "synthetic_phase1_claude.json"
COMBINED_PATH = ROOT / "data" / "synthetic_all_rows.json"

DEFAULT_TOTAL_ROWS = 300
DEFAULT_DEEPSEEK_ONLY_BATCH = 50


def load_golden(path: Path) -> list:
    raw = path.read_text(encoding="utf8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("golden_set.json must be a JSON array")
    return data


def _extract_json_array_payload(text: str) -> str:
    """Take first '[' ... matching ']' as balanced segments, respecting JSON string boundaries (handles ] inside code strings)."""
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    start = text.find("[")
    if start == -1:
        raise ValueError("no JSON array start '[' in response")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unclosed or malformed JSON array in response")


def extract_json_array(text: str) -> list:
    """Strip markdown fences / chatter; parse JSON array via balanced '[' … ']' extraction then json.loads."""
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    payload = _extract_json_array_payload(text)
    return json.loads(payload)


def validate_row(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    d = obj
    for k in REQUIRED_KEYS:
        if k not in d:
            return False
    if not isinstance(d["legal_anchor"], str) or not d["legal_anchor"].strip():
        return False
    va = d["violation_subtlety"]
    if va not in SUBTLETY:
        return False
    try:
        u = float(d["uncertainty_factor"])
    except (TypeError, ValueError):
        return False
    if not (0.1 <= u <= 1.0):
        return False
    for k in ("non_compliant_code", "chain_of_thought", "compliant_fix", "clause"):
        if not isinstance(d[k], str) or not d[k].strip():
            return False
    if d["complexity"] not in COMPLEXITY:
        return False
    if "2024/1689" not in d["legal_anchor"]:
        return False
    return True


def row_fingerprint(d: dict) -> str:
    h = hashlib.sha256(
        (d["clause"][:400] + "|" + d["non_compliant_code"][:600]).encode("utf8")
    ).hexdigest()
    return h


def strip_provenance(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "_provenance"} for r in rows]


def normalize_golden_row(raw: dict) -> dict:
    """Map golden_set.json shape (e.g. reasoning, text) to REQUIRED_KEYS."""
    clause = (raw.get("clause") or raw.get("text") or "").strip()
    cot = (raw.get("chain_of_thought") or raw.get("reasoning") or "").strip()
    la = str(raw.get("legal_anchor", "")).strip()
    out = {
        "legal_anchor": la,
        "clause": clause,
        "violation_subtlety": raw["violation_subtlety"],
        "uncertainty_factor": float(raw["uncertainty_factor"]),
        "non_compliant_code": str(raw.get("non_compliant_code", "")).strip(),
        "chain_of_thought": cot,
        "compliant_fix": str(raw.get("compliant_fix", "")).strip(),
        "complexity": raw["complexity"],
    }
    if not validate_row(out):
        rid = raw.get("id", "?")
        raise ValueError(f"golden row failed validation after normalize: {rid}")
    out["_provenance"] = "golden"
    return out


def normalize_golden_rows(golden: list) -> list[dict]:
    return [normalize_golden_row(r) for r in golden]


def load_regulation_context(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf8").strip()


def build_system_blocks_cached(
    golden: list,
    regulation_excerpt: str,
    cache_ttl: str,
) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": SYSTEM_PROMPT}]
    if regulation_excerpt:
        blocks.append(
            {
                "type": "text",
                "text": "=== Regulation (EU) 2024/1689 — static excerpts (do not fabricate articles outside this regime) ===\n"
                + regulation_excerpt,
            }
        )
    golden_block = (
        "=== FEW-SHOT GROUND TRUTH (golden set JSON) ===\n"
        "Match citation style and fidelity; synthesise training rows using only real Articles/Annexes from Reg. (EU) 2024/1689.\n"
        + json.dumps(golden, indent=2, ensure_ascii=False)
    )
    blocks.append(
        {
            "type": "text",
            "text": golden_block,
            "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
        }
    )
    return blocks


def build_user_message_dynamic(
    batch_idx: int,
    batch_total: int,
    batch_size: int,
    borderline_needed: int,
    *,
    golden_in_system_prompt: bool,
) -> str:
    borderline_line = ""
    if borderline_needed > 0:
        borderline_line = (
            f"In this batch, include at least {borderline_needed} rows with "
            f'"violation_subtlety" exactly equal to "borderline".\n'
        )
    anchor = (
        "The system prompt already contains the EU AI Act excerpts and the golden set JSON — do not ask for them.\n\n"
        if golden_in_system_prompt
        else ""
    )
    return f"""{anchor}Batch {batch_idx} of {batch_total}. Generate exactly {batch_size} unique training rows.

Project-wide target (across all batches): approximately 30% of all rows must have violation_subtlety \"borderline\".

{borderline_line}
Return ONLY a JSON array. First non-whitespace character must be [. No markdown outside the JSON."""


def build_user_message_with_golden_inline(
    golden: list,
    batch_idx: int,
    batch_total: int,
    batch_size: int,
    borderline_needed: int,
) -> str:
    return (
        build_user_message_dynamic(
            batch_idx,
            batch_total,
            batch_size,
            borderline_needed,
            golden_in_system_prompt=False,
        )
        + "\n\nFEW-SHOT GROUND TRUTH (golden set):\n"
        + json.dumps(golden, indent=2, ensure_ascii=False)
    )


def _log_prompt_cache_usage(msg: object, batch_idx: int) -> None:
    u = getattr(msg, "usage", None)
    if u is None:
        return
    inp = getattr(u, "input_tokens", None)
    cr = getattr(u, "cache_read_input_tokens", None)
    cc = getattr(u, "cache_creation_input_tokens", None)
    if cr is None and cc is None:
        return
    print(
        f"  prompt_cache batch {batch_idx}: input_tokens={inp} "
        f"cache_read_input_tokens={cr or 0} cache_creation_input_tokens={cc or 0}"
    )


def call_anthropic(
    *,
    model: str,
    user_text: str,
    golden: list,
    regulation_excerpt: str,
    cache_ttl: str,
    use_prompt_cache: bool,
    batch_idx: int,
    max_retries: int = 5,
) -> list:
    import anthropic

    client = anthropic.Anthropic()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if use_prompt_cache:
                system = build_system_blocks_cached(golden, regulation_excerpt, cache_ttl)
                msg = client.messages.create(
                    model=model,
                    max_tokens=16384,
                    system=system,
                    messages=[{"role": "user", "content": user_text}],
                )
            else:
                msg = client.messages.create(
                    model=model,
                    max_tokens=16384,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_text}],
                )
            _log_prompt_cache_usage(msg, batch_idx)
            parts: list[str] = []
            for b in msg.content:
                if b.type == "text":
                    parts.append(b.text)
            return extract_json_array("".join(parts))
        except Exception as e:
            last_err = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Anthropic failed after retries: {last_err}") from last_err


def build_deepseek_system(
    regulation_excerpt: str,
    claude_few_shot: list[dict],
) -> str:
    parts = [SYSTEM_PROMPT]
    if regulation_excerpt:
        parts.append(
            "=== Regulation (EU) 2024/1689 — static excerpts ===\n" + regulation_excerpt
        )
    parts.append(
        "=== FEW-SHOT EXAMPLES (Claude Opus — replicate this schema, tone, and JSON shape exactly; generate NEW rows, do not copy) ===\n"
        + json.dumps(strip_provenance(claude_few_shot), indent=2, ensure_ascii=False)
    )
    return "\n\n".join(parts)


def build_deepseek_system_golden(
    regulation_excerpt: str,
    golden_few_shot: list[dict],
) -> str:
    parts = [SYSTEM_PROMPT]
    if regulation_excerpt:
        parts.append(
            "=== Regulation (EU) 2024/1689 — static excerpts ===\n" + regulation_excerpt
        )
    parts.append(
        "=== FEW-SHOT GROUND TRUTH (verified golden set — replicate schema, tone, and JSON shape exactly; generate NEW rows, do not copy or paraphrase these rows) ===\n"
        + json.dumps(strip_provenance(golden_few_shot), indent=2, ensure_ascii=False)
    )
    return "\n\n".join(parts)


def call_deepseek(
    *,
    model: str,
    base_url: str,
    system_text: str,
    user_text: str,
    batch_idx: int,
    temperature: float,
    top_p: float,
    max_retries: int = 5,
) -> list:
    from openai import OpenAI

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=key, base_url=base_url)
    full_system = DEEPSEEK_JSON_DISCIPLINE + "\n\n" + system_text
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=16384,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_text},
                ],
            )
            choice = resp.choices[0]
            content = choice.message.content
            if not content:
                raise ValueError("empty DeepSeek content")
            u = getattr(resp, "usage", None)
            if u:
                print(
                    f"  deepseek batch {batch_idx}: prompt_tokens={getattr(u, 'prompt_tokens', None)} "
                    f"completion_tokens={getattr(u, 'completion_tokens', None)}"
                )
            return extract_json_array(content)
        except Exception as e:
            last_err = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"DeepSeek failed after retries: {last_err}") from last_err


def run_claude_phase(
    *,
    target: int,
    batch_size: int,
    golden: list,
    regulation: str,
    claude_model: str,
    cache_ttl: str,
    use_prompt_cache: bool,
) -> list[dict]:
    seen: set[str] = set()
    acc: list[dict] = []
    batch_total = max(1, (target + batch_size - 1) // batch_size)
    batch_idx = 0
    stale = 0
    while len(acc) < target:
        prior = len(acc)
        batch_idx += 1
        if batch_idx > 250:
            raise RuntimeError("claude batch limit exceeded")
        need = min(batch_size, target - len(acc))
        borderline_count = sum(1 for r in acc if r.get("violation_subtlety") == "borderline")
        target_borderline = int(round(0.3 * target))
        borderline_needed = max(0, min(need, target_borderline - borderline_count))
        if use_prompt_cache:
            user_text = build_user_message_dynamic(
                batch_idx,
                batch_total,
                need,
                borderline_needed,
                golden_in_system_prompt=True,
            )
            raw_list = call_anthropic(
                model=claude_model,
                user_text=user_text,
                golden=golden,
                regulation_excerpt=regulation,
                cache_ttl=cache_ttl,
                use_prompt_cache=True,
                batch_idx=batch_idx,
            )
        else:
            user_text = build_user_message_with_golden_inline(
                golden, batch_idx, batch_total, need, borderline_needed
            )
            raw_list = call_anthropic(
                model=claude_model,
                user_text=user_text,
                golden=golden,
                regulation_excerpt="",
                cache_ttl=cache_ttl,
                use_prompt_cache=False,
                batch_idx=batch_idx,
            )
        if not isinstance(raw_list, list):
            raise ValueError("Claude did not return a JSON array")
        for obj in raw_list:
            if len(acc) >= target:
                break
            if not validate_row(obj):
                continue
            fp = row_fingerprint(obj)
            if fp in seen:
                continue
            seen.add(fp)
            obj["_provenance"] = "claude"
            acc.append(obj)
        if len(acc) == prior:
            stale += 1
            if stale >= 12:
                raise RuntimeError("Claude: no new valid rows in 12 consecutive batches")
        else:
            stale = 0
        print(f"  [Claude] batch {batch_idx}: valid {len(acc)}/{target}")
    return acc


def run_deepseek_phase(
    *,
    target: int,
    batch_size: int,
    claude_rows: list[dict],
    regulation: str,
    deepseek_model: str,
    deepseek_base_url: str,
    deepseek_temperature: float,
    deepseek_top_p: float,
) -> list[dict]:
    if not claude_rows:
        raise ValueError("DeepSeek phase requires non-empty Claude few-shot rows")
    seen: set[str] = set()
    for r in claude_rows:
        seen.add(row_fingerprint(r))
    acc: list[dict] = []
    system_text = build_deepseek_system(regulation, claude_rows)
    batch_total = max(1, (target + batch_size - 1) // batch_size)
    batch_idx = 0
    stale = 0
    while len(acc) < target:
        prior = len(acc)
        batch_idx += 1
        if batch_idx > 250:
            raise RuntimeError("deepseek batch limit exceeded")
        need = min(batch_size, target - len(acc))
        borderline_count = sum(1 for r in acc if r.get("violation_subtlety") == "borderline")
        target_borderline = int(round(0.3 * target))
        borderline_needed = max(0, min(need, target_borderline - borderline_count))
        user_text = build_user_message_dynamic(
            batch_idx,
            batch_total,
            need,
            borderline_needed,
            golden_in_system_prompt=True,
        )
        user_text = (
            "The system message contains Regulation excerpts and 100 Claude-generated few-shot examples. "
            "Produce NEW synthetic rows that match their JSON schema and style exactly.\n\n"
        ) + user_text
        raw_list = call_deepseek(
            model=deepseek_model,
            base_url=deepseek_base_url,
            system_text=system_text,
            user_text=user_text,
            batch_idx=batch_idx,
            temperature=deepseek_temperature,
            top_p=deepseek_top_p,
        )
        if not isinstance(raw_list, list):
            raise ValueError("DeepSeek did not return a JSON array")
        for obj in raw_list:
            if len(acc) >= target:
                break
            if not validate_row(obj):
                continue
            fp = row_fingerprint(obj)
            if fp in seen:
                continue
            seen.add(fp)
            obj["_provenance"] = "deepseek"
            acc.append(obj)
        if len(acc) == prior:
            stale += 1
            if stale >= 12:
                raise RuntimeError("DeepSeek: no new valid rows in 12 consecutive batches")
        else:
            stale = 0
        print(f"  [DeepSeek] batch {batch_idx}: valid {len(acc)}/{target}")
    return acc


def run_deepseek_golden_only_phase(
    *,
    target: int,
    batch_size: int,
    golden_rows_normalized: list[dict],
    regulation: str,
    deepseek_model: str,
    deepseek_base_url: str,
    deepseek_temperature: float,
    deepseek_top_p: float,
) -> list[dict]:
    """Generate `target` new rows using DeepSeek; golden set is few-shot only (fingerprints seeded to avoid duplicates)."""
    if target < 1:
        return []
    seen: set[str] = set()
    for r in golden_rows_normalized:
        clean = {k: v for k, v in r.items() if k != "_provenance"}
        seen.add(row_fingerprint(clean))
    acc: list[dict] = []
    system_text = build_deepseek_system_golden(regulation, golden_rows_normalized)
    batch_total = max(1, (target + batch_size - 1) // batch_size)
    batch_idx = 0
    stale = 0
    while len(acc) < target:
        prior = len(acc)
        batch_idx += 1
        if batch_idx > 250:
            raise RuntimeError("deepseek batch limit exceeded")
        need = min(batch_size, target - len(acc))
        borderline_count = sum(1 for r in acc if r.get("violation_subtlety") == "borderline")
        target_borderline = int(round(0.3 * target))
        borderline_needed = max(0, min(need, target_borderline - borderline_count))
        user_text = (
            "The system message contains Regulation excerpts and the verified golden-set few-shot examples. "
            "Produce NEW synthetic rows that match their JSON schema and style exactly.\n\n"
        ) + build_user_message_dynamic(
            batch_idx,
            batch_total,
            need,
            borderline_needed,
            golden_in_system_prompt=True,
        )
        raw_list = call_deepseek(
            model=deepseek_model,
            base_url=deepseek_base_url,
            system_text=system_text,
            user_text=user_text,
            batch_idx=batch_idx,
            temperature=deepseek_temperature,
            top_p=deepseek_top_p,
        )
        if not isinstance(raw_list, list):
            raise ValueError("DeepSeek did not return a JSON array")
        for obj in raw_list:
            if len(acc) >= target:
                break
            if not validate_row(obj):
                continue
            fp = row_fingerprint(obj)
            if fp in seen:
                continue
            seen.add(fp)
            obj["_provenance"] = "deepseek"
            acc.append(obj)
        if len(acc) == prior:
            stale += 1
            if stale >= 12:
                raise RuntimeError("DeepSeek: no new valid rows in 12 consecutive batches")
        else:
            stale = 0
        print(f"  [DeepSeek golden-only] batch {batch_idx}: valid {len(acc)}/{target}")
    return acc


def embed_batch(
    texts: list[str], model_name: str, encode_batch: int = 32
) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    out: list[list[float]] = []
    for i in range(0, len(texts), encode_batch):
        chunk = texts[i : i + encode_batch]
        vecs = m.encode(chunk, convert_to_numpy=True, normalize_embeddings=False)
        for row in vecs:
            v = row.astype("float32")[:VECTOR_DIM]
            if v.shape[0] != VECTOR_DIM:
                raise RuntimeError("embedding dim mismatch")
            out.append(v.tolist())
    return out


def rows_to_lance_payloads(rows: list[dict], embedder_model: str) -> list[dict]:
    rows_clean = []
    for r in rows:
        c = {k: v for k, v in r.items() if not str(k).startswith("_")}
        rows_clean.append(c)
    texts = [
        "\n\n".join(
            [
                r["clause"],
                r["non_compliant_code"],
                r["chain_of_thought"],
                r["compliant_fix"],
            ]
        )
        for r in rows_clean
    ]
    vectors = embed_batch(texts, embedder_model)
    payloads: list[dict] = []
    for r, vec in zip(rows_clean, vectors):
        blob = json.dumps(r, ensure_ascii=False)
        payloads.append(
            {
                "id": f"syn-{uuid.uuid4().hex[:16]}",
                "citation": r["legal_anchor"][:512],
                "instrument": "Reg. (EU) 2024/1689",
                "section_ref": "",
                "text": blob,
                "vector": vec,
            }
        )
    return payloads


def ingest_lance(payloads: list[dict]) -> int:
    init_legal_corpus_table()
    db = connect()
    if LEGAL_CHUNKS_TABLE not in list_table_names(db):
        raise RuntimeError("legal_chunks table missing")
    tbl = db.open_table(LEGAL_CHUNKS_TABLE)
    if payloads:
        tbl.add(payloads)
    return tbl.count_rows()


def write_health(total: int) -> None:
    out = ROOT / "data" / "dataset_health.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"legal_chunks_total": total}, indent=2) + "\n",
        encoding="utf8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Default: DeepSeek-only 300-row dataset (15 golden few-shot + 285 generated in batches). "
            "Legacy: Claude then DeepSeek with Claude few-shot."
        ),
    )
    ap.add_argument("--claude-count", type=int, default=100)
    ap.add_argument("--deepseek-count", type=int, default=200)
    ap.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL_ROWS,
        help="deepseek-only: total rows in combined output (golden + generated). Default 300.",
    )
    ap.add_argument(
        "--phase",
        choices=("deepseek-only", "all", "claude", "deepseek", "ingest-only"),
        default="deepseek-only",
        help=(
            "deepseek-only=golden few-shot + generate remainder with DeepSeek (default); "
            "all=Claude then DeepSeek (--pause-after-claude supported); "
            "claude=phase1 only; deepseek=legacy phase2 (needs synthetic_phase1_claude.json); "
            "ingest-only=embed existing combined JSON"
        ),
    )
    ap.add_argument(
        "--pause-after-claude",
        action="store_true",
        help=(
            "With --phase all: run Claude only, write phase1 file, then exit. "
            "Review the file, then run again with --phase deepseek (same paths)."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_DEEPSEEK_ONLY_BATCH,
        help="DeepSeek request batch size (default 50; max 50).",
    )
    ap.add_argument(
        "--claude-model",
        default=os.environ.get("CLAUDE_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or DEFAULT_CLAUDE_MODEL,
        help="Claude model for first N examples (e.g. Opus 4.6)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="deprecated alias for --claude-model",
    )
    ap.add_argument(
        "--deepseek-model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        help=(
            "Default: deepseek-v3.2-speciale (DeepSeek-V3.2-Speciale). "
            "If the official API rejects this id, set DEEPSEEK_MODEL=deepseek-chat and DEEPSEEK_BASE_URL "
            "to the Speciale route from https://api-docs.deepseek.com/updates/"
        ),
    )
    ap.add_argument(
        "--deepseek-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        help="OpenAI-compatible base URL (default https://api.deepseek.com). Speciale may use a path under the same host per DeepSeek docs.",
    )
    ap.add_argument(
        "--deepseek-temperature",
        type=float,
        default=float(os.environ.get("DEEPSEEK_TEMPERATURE", DEFAULT_DEEPSEEK_TEMPERATURE)),
    )
    ap.add_argument(
        "--deepseek-top-p",
        type=float,
        default=float(os.environ.get("DEEPSEEK_TOP_P", DEFAULT_DEEPSEEK_TOP_P)),
    )
    ap.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument("--golden", type=Path, default=ROOT / "data" / "golden_set.json")
    ap.add_argument(
        "--regulation-context",
        type=Path,
        default=ROOT / "data" / "eu_ai_act_cache_context.txt",
    )
    ap.add_argument(
        "--cache-ttl",
        choices=("5m", "1h"),
        default="1h",
    )
    ap.add_argument("--no-prompt-cache", action="store_true")
    ap.add_argument(
        "--phase1-out",
        type=Path,
        default=PHASE1_PATH,
        help="where to save Claude rows (JSON array)",
    )
    ap.add_argument(
        "--combined-out",
        type=Path,
        default=COMBINED_PATH,
        help="all rows before ingest",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = ap.parse_args()

    claude_model = args.model if args.model else args.claude_model

    regulation = load_regulation_context(args.regulation_context)
    golden_preview: list = []
    if args.golden.is_file():
        try:
            golden_preview = load_golden(args.golden)
        except (json.JSONDecodeError, OSError) as e:
            if args.dry_run:
                golden_preview = []
                print(f"warning: could not load golden for preview: {e}", file=sys.stderr)
            else:
                raise

    if args.dry_run:
        print(f"phase: {args.phase}")
        print(f"total={args.total}")
        print(f"pause_after_claude={args.pause_after_claude}")
        print(f"claude_count={args.claude_count} deepseek_count={args.deepseek_count}")
        print(f"batch_size={args.batch_size}")
        print(f"claude_model={claude_model}")
        print(f"deepseek_model={args.deepseek_model}")
        print(f"deepseek_base_url={args.deepseek_base_url}")
        print(
            f"deepseek sampling: temperature={args.deepseek_temperature} top_p={args.deepseek_top_p}"
        )
        print(f"regulation chars: {len(regulation)}")
        print(f"phase1_out={args.phase1_out}")
        print(f"combined_out={args.combined_out}")
        return 0

    if args.phase in ("all", "claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY for Claude phase", file=sys.stderr)
        return 1
    need_deepseek = args.phase in ("deepseek-only", "deepseek") or (
        args.phase == "all" and not args.pause_after_claude
    )
    if need_deepseek and not os.environ.get("DEEPSEEK_API_KEY"):
        print("Set DEEPSEEK_API_KEY for DeepSeek phase", file=sys.stderr)
        return 1

    use_cache = not args.no_prompt_cache
    if use_cache and len(regulation) < 4000 and args.phase in ("all", "claude"):
        print(
            "warning: short regulation excerpt may reduce Anthropic cache hit rate.",
            file=sys.stderr,
        )

    if args.phase == "ingest-only":
        if not args.combined_out.is_file():
            print(f"missing {args.combined_out}", file=sys.stderr)
            return 1
        combined = json.loads(args.combined_out.read_text(encoding="utf8"))
        if not isinstance(combined, list):
            raise ValueError("combined file must be a JSON array")
        for r in combined:
            if not validate_row(r):
                raise ValueError("invalid row in combined file")
        payloads = rows_to_lance_payloads(combined, args.embedder)
        count = ingest_lance(payloads)
        write_health(count)
        print(f"LanceDB legal_chunks total rows: {count}")
        return 0

    golden = load_golden(args.golden)
    batch_size = max(1, min(args.batch_size, 50))

    if args.phase == "deepseek-only":
        golden_norm = normalize_golden_rows(golden)
        n_golden = len(golden_norm)
        need = args.total - n_golden
        if need < 0:
            print(
                f"--total {args.total} is smaller than normalized golden set size ({n_golden})",
                file=sys.stderr,
            )
            return 1
        deep_only: list[dict] = []
        if need > 0:
            deep_only = run_deepseek_golden_only_phase(
                target=need,
                batch_size=batch_size,
                golden_rows_normalized=golden_norm,
                regulation=regulation,
                deepseek_model=args.deepseek_model,
                deepseek_base_url=args.deepseek_base_url,
                deepseek_temperature=args.deepseek_temperature,
                deepseek_top_p=args.deepseek_top_p,
            )
        combined = golden_norm + deep_only
        args.combined_out.parent.mkdir(parents=True, exist_ok=True)
        args.combined_out.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
            encoding="utf8",
        )
        print(
            f"wrote combined: {args.combined_out} ({len(combined)} rows; "
            f"{n_golden} golden + {len(deep_only)} DeepSeek)"
        )
        payloads = rows_to_lance_payloads(combined, args.embedder)
        count = ingest_lance(payloads)
        write_health(count)
        print(f"LanceDB legal_chunks total rows: {count}")
        print(f"wrote {ROOT / 'data' / 'dataset_health.json'}")
        return 0

    claude_rows: list[dict] = []
    if args.phase in ("all", "claude"):
        claude_rows = run_claude_phase(
            target=args.claude_count,
            batch_size=batch_size,
            golden=golden,
            regulation=regulation,
            claude_model=claude_model,
            cache_ttl=args.cache_ttl,
            use_prompt_cache=use_cache,
        )
        args.phase1_out.parent.mkdir(parents=True, exist_ok=True)
        args.phase1_out.write_text(
            json.dumps(claude_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf8",
        )
        print(f"wrote Claude phase: {args.phase1_out} ({len(claude_rows)} rows)")
        if args.phase == "claude":
            print(
                "Review the file above, then continue with DeepSeek + ingest, for example:\n"
                f"  python scripts/generate_synthetic_data.py --phase deepseek "
                f"--phase1-out {args.phase1_out} --combined-out {args.combined_out}"
            )
            return 0
        if args.pause_after_claude:
            print(
                "Paused after Claude (--pause-after-claude). Review the phase1 file, then run:\n"
                f"  python scripts/generate_synthetic_data.py --phase deepseek "
                f"--phase1-out {args.phase1_out} --combined-out {args.combined_out}"
            )
            return 0
    elif args.phase == "deepseek":
        if not args.phase1_out.is_file():
            print(f"missing {args.phase1_out}; run --phase claude first", file=sys.stderr)
            return 1
        claude_rows = json.loads(args.phase1_out.read_text(encoding="utf8"))
        if not isinstance(claude_rows, list) or len(claude_rows) < 1:
            raise ValueError("phase1 file must be a non-empty JSON array")

    deep_rows: list[dict] = []
    if args.phase in ("all", "deepseek"):
        if not claude_rows:
            claude_rows = json.loads(args.phase1_out.read_text(encoding="utf8"))
        deep_rows = run_deepseek_phase(
            target=args.deepseek_count,
            batch_size=batch_size,
            claude_rows=claude_rows,
            regulation=regulation,
            deepseek_model=args.deepseek_model,
            deepseek_base_url=args.deepseek_base_url,
            deepseek_temperature=args.deepseek_temperature,
            deepseek_top_p=args.deepseek_top_p,
        )
        print(f"DeepSeek phase: {len(deep_rows)} rows")

    combined = claude_rows + deep_rows
    args.combined_out.parent.mkdir(parents=True, exist_ok=True)
    args.combined_out.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    print(f"wrote combined: {args.combined_out} ({len(combined)} rows)")

    payloads = rows_to_lance_payloads(combined, args.embedder)
    count = ingest_lance(payloads)
    write_health(count)
    print(f"LanceDB legal_chunks total rows: {count}")
    print(f"wrote {ROOT / 'data' / 'dataset_health.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
