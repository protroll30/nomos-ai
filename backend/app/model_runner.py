from __future__ import annotations

import json
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_TEXT = (
    "You are Nomos, an expert AI compliance auditor enforcing the EU AI Act "
    "(Regulation 2024/1689). Your task is to review FastAPI architectures and output a "
    "strict JSON response. Assume the system is classified as high-risk (Annex III)."
)

_model = None
_tokenizer = None
_load_error: str | None = None
_lock = threading.Lock()


def _adapter_dir() -> Path:
    raw = os.environ.get("NOMOS_ADAPTER_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (REPO_ROOT / "outputs" / "nomos-lora").resolve()


def _model_name() -> str:
    return os.environ.get(
        "NOMOS_MODEL_NAME",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    )


def _load() -> None:
    global _model, _tokenizer, _load_error
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = _adapter_dir()
    if not (adapter / "adapter_config.json").is_file():
        _load_error = f"No LoRA adapter at {adapter} (expected adapter_config.json)."
        return

    if not torch.cuda.is_available():
        _load_error = "CUDA is not available; inference requires a GPU."
        return

    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            _model_name(),
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(_model_name(), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        _model = model
        _tokenizer = tokenizer
        _load_error = None
    except Exception as e:
        _load_error = str(e)
        _model = None
        _tokenizer = None


def ensure_loaded() -> str | None:
    global _load_error
    with _lock:
        if _model is not None and _tokenizer is not None:
            return None
        if _load_error and _model is None:
            _load_error = None
        _load()
        return _load_error


def inference_ready() -> bool:
    return _model is not None and _tokenizer is not None


def last_load_error() -> str | None:
    return _load_error


def status_snapshot() -> dict:
    return {
        "adapter_dir": str(_adapter_dir()),
        "model_name": _model_name(),
        "inference_ready": inference_ready(),
        "load_error": last_load_error(),
    }


def build_messages(code: str) -> list[dict]:
    user = (
        "Audit the following code for EU AI Act compliance:\n\n"
        f"```python\n{code.strip()}\n```"
    )
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user},
    ]


def generate_audit(code: str, *, max_new_tokens: int = 1024) -> tuple[str, dict | None, str | None]:
    err = ensure_loaded()
    if err:
        raise RuntimeError(err)
    assert _model is not None and _tokenizer is not None

    import torch

    msgs = build_messages(code)
    enc = _tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(_model.parameters()).device
    pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id

    if isinstance(enc, torch.Tensor):
        input_ids = enc.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        input_ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        attention_mask = (
            am.to(device)
            if am is not None
            else torch.ones_like(input_ids, dtype=torch.long, device=device)
        )

    with torch.inference_mode():
        out = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
            repetition_penalty=1.12,
        )
    gen_ids = out[0, input_ids.shape[1] :]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    parse_err: str | None = None
    parsed: dict | None = None
    try:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        obj = json.loads(candidate)
        parsed = obj if isinstance(obj, dict) else None
        if parsed is None:
            parse_err = "Top-level JSON is not an object."
    except json.JSONDecodeError as e:
        parse_err = str(e)

    return text, parsed, parse_err
