"""LLM-as-judge on prediction JSONL from eval_unsloth_lora.py (--dump-preds)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

JUDGE_SYSTEM = """You evaluate two EU AI Act compliance audit outputs for the same coding scenario.
The REFERENCE is high-quality supervised data (treat as the target style and substance).
The CANDIDATE is a model-generated audit.
Score the CANDIDATE against the REFERENCE and task requirements, not lexical overlap.
Respond with one JSON object only, no markdown fences, using exactly these keys:
legal_substance_alignment (integer 1-5): Article fit and legal reasoning vs reference.
argument_coherence (integer 1-5): Internal consistency and clarity.
remediation_quality (integer 1-5): Whether compliant_fix is concrete and plausible.
overall (integer 1-5): Holistic usefulness for a compliance reviewer.
verdict (string): one of strong, adequate, weak, invalid (use invalid if candidate is not a usable JSON audit).
rationale (string): brief justification (2-4 sentences)."""


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_judge_output(text: str) -> dict | None:
    try:
        obj = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    need = (
        "legal_substance_alignment",
        "argument_coherence",
        "remediation_quality",
        "overall",
        "verdict",
        "rationale",
    )
    if not all(k in obj for k in need):
        return None
    for k in need[:-2]:
        try:
            v = int(obj[k])
        except (TypeError, ValueError):
            return None
        if not 1 <= v <= 5:
            return None
    if obj["verdict"] not in ("strong", "adequate", "weak", "invalid"):
        return None
    if not isinstance(obj["rationale"], str):
        return None
    return obj


def _call_openai_judge(*, model: str, user_text: str) -> dict | None:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return None
    client = OpenAI()
    req: dict = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_text},
        ],
    }
    raw_temp = os.environ.get("OPENAI_JUDGE_TEMPERATURE")
    if raw_temp is not None:
        req["temperature"] = float(raw_temp)
    elif not model.lower().startswith("gpt-5"):
        req["temperature"] = 0
    resp = client.chat.completions.create(**req)
    content = resp.choices[0].message.content
    if not content:
        return None
    return _parse_judge_output(content)


def _call_anthropic_judge(*, model: str, user_text: str) -> dict | None:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return None
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM + " Output raw JSON only, no other text.",
        messages=[{"role": "user", "content": user_text}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not parts:
        return None
    return _parse_judge_output("".join(parts))


def _build_user_block(row: dict) -> str | None:
    msgs = row.get("messages")
    pred = row.get("pred_assistant")
    if not isinstance(msgs, list) or not msgs or not isinstance(pred, str):
        return None
    if msgs[-1].get("role") != "assistant":
        return None
    gold = msgs[-1].get("content", "")
    ctx_parts = []
    for m in msgs[:-1]:
        role = m.get("role", "")
        content = m.get("content", "")
        ctx_parts.append(f"[{role}]\n{content}")
    ctx = "\n\n".join(ctx_parts)
    return (
        f"=== Audit context ===\n{ctx}\n\n"
        f"=== Reference (gold) ===\n{gold}\n\n"
        f"=== Candidate (model) ===\n{pred}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge on dumped LoRA predictions.")
    ap.add_argument(
        "--preds",
        type=Path,
        required=True,
        help="JSONL from eval_unsloth_lora.py --dump-preds",
    )
    ap.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    ap.add_argument(
        "--model",
        default="",
        help="Override model id (defaults: OPENAI_JUDGE_MODEL / ANTHROPIC_JUDGE_MODEL or built-ins).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max rows (0 = all).")
    ap.add_argument("--sleep", type=float, default=0.25, help="Seconds between API calls.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSONL: input row + judge subobject.",
    )
    args = ap.parse_args()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.preds.is_file():
        print(f"missing {args.preds}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with args.preds.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("no rows in preds file", file=sys.stderr)
        return 1

    if args.provider == "openai":
        model = args.model or os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-4o"
        judge_fn = lambda t: _call_openai_judge(model=model, user_text=t)
    else:
        model = args.model or os.environ.get("ANTHROPIC_JUDGE_MODEL") or "claude-3-5-haiku-20241022"
        judge_fn = lambda t: _call_anthropic_judge(model=model, user_text=t)

    sums = {k: 0 for k in ("legal_substance_alignment", "argument_coherence", "remediation_quality", "overall")}
    verdicts: dict[str, int] = {}
    ok = 0
    out_f = args.out.open("w", encoding="utf-8") if args.out else None

    for i, row in enumerate(rows):
        block = _build_user_block(row)
        if block is None:
            print(f"row {i}: skip (bad shape)", file=sys.stderr)
            continue
        result = None
        for attempt in range(2):
            result = judge_fn(block)
            if result is not None:
                break
            time.sleep(1.0 + attempt)
        if result is None:
            print(f"row {i}: judge parse/API failed", file=sys.stderr)
            if out_f:
                rec = {**row, "judge": None, "judge_error": True}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        ok += 1
        for k in sums:
            sums[k] += int(result[k])
        verdicts[result["verdict"]] = verdicts.get(result["verdict"], 0) + 1
        print(f"row {i + 1}/{len(rows)} verdict={result['verdict']} overall={result['overall']}")
        if out_f:
            rec = {**row, "judge": result}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if args.sleep > 0 and i + 1 < len(rows):
            time.sleep(args.sleep)

    if out_f:
        out_f.close()
        print(f"wrote {args.out}")

    print()
    print(f"judge_ok={ok}/{len(rows)} provider={args.provider} model={model}")
    if ok:
        for k, s in sums.items():
            print(f"mean_{k}={s / ok:.3f}")
        print(f"verdict_counts={json.dumps(verdicts)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
