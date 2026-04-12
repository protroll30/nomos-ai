"""Generate on eval JSONL with a trained Unsloth LoRA adapter; report JSON + field overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_obj(s: str) -> dict | None:
    s = s.strip()
    if not s:
        return None
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate LoRA SFT adapter on messages JSONL.")
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "outputs" / "nomos-lora",
        help="Directory with adapter weights + tokenizer (trainer.save_model output).",
    )
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval.jsonl")
    ap.add_argument(
        "--model-name",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    )
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Max rows (0 = all).")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        print("CUDA required.", file=__import__("sys").stderr)
        return 1
    if not args.eval.is_file():
        print(f"missing {args.eval}", file=__import__("sys").stderr)
        return 1
    if not (args.adapter_dir / "adapter_config.json").is_file():
        print(f"missing adapter in {args.adapter_dir}", file=__import__("sys").stderr)
        return 1

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    FastLanguageModel.for_inference(model)

    rows: list[dict] = []
    with args.eval.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.limit > 0:
        rows = rows[: args.limit]

    n = len(rows)
    json_ok = 0
    key_match = 0
    f1_num = f1_den_p = f1_den_r = 0

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for i, row in enumerate(rows):
        msgs = row["messages"]
        if not msgs or msgs[-1].get("role") != "assistant":
            print(f"row {i}: expected last message role assistant", file=__import__("sys").stderr)
            return 1
        gold_text = msgs[-1].get("content") or ""
        prompt_msgs = msgs[:-1]
        input_ids = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen_ids = out[0, input_ids.shape[1] :]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        gold_obj = _parse_obj(gold_text)
        pred_obj = _parse_obj(pred_text)
        if pred_obj is not None:
            json_ok += 1
        if gold_obj is not None and pred_obj is not None:
            gk = set(gold_obj.keys())
            pk = set(pred_obj.keys())
            if gk == pk:
                key_match += 1
            inter = gk & pk
            f1_num += len(inter)
            f1_den_p += len(pk)
            f1_den_r += len(gk)

        print(f"--- row {i + 1}/{n} ---")
        print(pred_text[:500] + ("…" if len(pred_text) > 500 else ""))

    prec = f1_num / f1_den_p if f1_den_p else 0.0
    rec = f1_num / f1_den_r if f1_den_r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print()
    print(f"rows={n}")
    print(f"json_valid_pred={json_ok}/{n} ({json_ok / n:.1%})" if n else "rows=0")
    print(
        f"exact_keyset_match={key_match}/{n} ({key_match / n:.1%})" if n else "",
    )
    print(f"key_micro_f1={f1:.3f} (pred vs gold keys, rows where both parse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
