# Nomos AI

Proof-of-concept pipeline for **instruction-tuning a small auditor-style model** on synthetic **EU AI Act–themed** FastAPI snippets. Outputs are **structured JSON** (legal anchor, clause excerpt, justification, remediation sketch). This is **not** legal advice, **not** a certified compliance product, and **not** a substitute for human or professional review.

## Contents

- **`data/train.jsonl`**, **`data/eval.jsonl`**: OpenAI-style chat JSONL (`system` / `user` / `assistant`) for SFT and eval.
- **`scripts/train_unsloth_sft.py`**: LoRA SFT with [Unsloth](https://github.com/unslothai/unsloth) + TRL on a CUDA GPU (default base: Llama 3.1 8B Instruct 4-bit).
- **`scripts/eval_unsloth_lora.py`**: Run the trained adapter on `eval.jsonl` (defaults to a **Transformers + PEFT** backend to avoid Unsloth fast-generate issues on some `transformers` 5.x stacks).
- **`scripts/eval_llm_judge.py`**: Optional **LLM-as-judge** (OpenAI or Anthropic) on dumped predictions; prints aggregate scores and verdict counts.
- **`scripts/prepare_sft_data.py`**: Builds `train.jsonl` / `eval.jsonl` from `data/synthetic_all_rows.json` when you generate that file locally (large synthetic outputs are gitignored by default).
- **`scripts/verify_unsloth_env.py`**: Quick import/CUDA check before training.
- **`backend/`**: FastAPI service with **`POST /v1/audit`** (GPU on the API host) plus **`POST /v1/codebase/ast`** for **CPython `ast`** scans of a multi-file map (routes/imports/defs heuristics).
- **`dashboard/`**: Next.js UI with a **Live audit** panel that calls that API.

## Requirements

- **Python 3.10+** (3.11 used in CI-style setups).
- **NVIDIA GPU with CUDA** for Unsloth training and for local eval generation.
- **Hugging Face**: account/token if you pull gated models or want higher Hub rate limits (`export HF_TOKEN=...` or `huggingface-cli login`).
- **OpenAI** and/or **Anthropic** API keys only if you run **`eval_llm_judge.py`**.

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Used for |
|----------|-----------|
| `HF_TOKEN` | Hugging Face Hub (optional; rate limits / gated models) |
| `OPENAI_API_KEY` | `eval_llm_judge.py` (`--provider openai`); **`NOMOS_AUDIT_BACKEND=openai`** for `/v1/audit` |
| `OPENAI_JUDGE_MODEL` | Judge model id (default: `gpt-4o`) |
| `OPENAI_JUDGE_TEMPERATURE` | Optional override (e.g. `gpt-4o` usually `0`) |
| `ANTHROPIC_API_KEY` | `eval_llm_judge.py` with `--provider anthropic` |
| `ANTHROPIC_JUDGE_MODEL` | Judge model id when using Anthropic |
| `NOMOS_AUDIT_BACKEND` | `hf` (default): local GPU model. `openai`: Chat Completions via `OPENAI_API_KEY` (no GPU on API host). |
| `NOMOS_OPENAI_MODEL` | OpenAI model id when `NOMOS_AUDIT_BACKEND=openai` (default: `gpt-4o-mini`) |
| `NOMOS_ADAPTER_DIR` | Absolute path to LoRA adapter dir (default: `<repo>/outputs/nomos-lora`) |
| `NOMOS_USE_LORA` | Default **`0`**: **base model only** (no adapter; typical PoC). Set **`1`** after training to merge the LoRA adapter from `NOMOS_ADAPTER_DIR`. |
| `NOMOS_MODEL_NAME` | Base HF model id (default: Unsloth Llama 3.1 8B 4-bit) |
| `NOMOS_PRELOAD_MODEL` | If `1`, load weights at API startup (otherwise first request pays load cost) |
| `NOMOS_MAX_NEW_TOKENS` | Default generation cap for `/v1/audit` |
| `NOMOS_CORS_ORIGINS` | Comma-separated origins for the dashboard (default includes `localhost:3000`) |

Copy secrets into a **local** `.env` if you use `python-dotenv` in the judge script, or `export` them in the shell (e.g. on a remote GPU pod). **Do not commit `.env`.**

## Training

From the repo root, with CUDA visible:

```bash
python scripts/verify_unsloth_env.py
python scripts/train_unsloth_sft.py \
  --epochs 2 \
  --lr 2e-4 \
  --output-dir outputs/nomos-lora-e2
```

Artifacts go under **`outputs/<run>/`** (adapter + tokenizer; ignored by git). Adjust hyperparameters as needed.

**AST during SFT:** Before `apply_chat_template`, each **user** turn that contains a Markdown-style Python code fence is augmented with the same **ast summary** text as the API (`backend/app/code_intel.py`: routes/imports/defs heuristics). This aligns training with inference when you use `include_ast_summary` on `/v1/audit`. Disable with **`--no-ast-augment`** on the trainer (and use the same flag on eval for adapters trained that way).

## Evaluating the adapter

**1. Generate predictions** (default backend: `hf`). On Linux/macOS use **forward slashes** in paths:

```bash
python scripts/eval_unsloth_lora.py \
  --adapter-dir outputs/nomos-lora-e2 \
  --dump-preds outputs/preds_eval.jsonl
```

By default, eval **prepends the same ast context** as training. If the adapter was trained with **`--no-ast-augment`**, pass **`--no-ast-augment`** here too.

Optional: `--backend unsloth` uses Unsloth’s loader (can hit `transformers` 5.x + fast-generate bugs on some setups).

**2. Optional LLM judge** (after `OPENAI_API_KEY` is set):

```bash
python scripts/eval_llm_judge.py \
  --preds outputs/preds_eval.jsonl \
  --out outputs/preds_eval_judged.jsonl
```

Judge output includes per-row scores and a final summary (`mean_*`, `verdict_counts`). The judge scores **alignment with your reference and rubric**, not legal truth.

## Regenerating train/eval JSONL

If you have `data/synthetic_all_rows.json` (from your own data generation pipeline):

```bash
python scripts/prepare_sft_data.py
```

Some large synthetic artifacts are listed in `.gitignore`; the committed **`train.jsonl` / `eval.jsonl`** are enough to reproduce training without them.

## Live audit API + dashboard

The **model runs on the API host** (needs **CUDA** + adapter weights). The **browser only calls HTTP**.

**1. API (from repo root, same venv as training):**

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Optional: `export NOMOS_ADAPTER_DIR=/path/to/outputs/nomos-lora-e2` if not using the default `outputs/nomos-lora`.

**2. Dashboard:**

```bash
cd dashboard
cp .env.example .env.local
# edit .env.local → NEXT_PUBLIC_NOMOS_API_URL=http://127.0.0.1:8000 (or your pod URL)
npm install
npm run dev
```

Open the app, use **Live audit (LoRA)** — paste Python, **Run audit**. Responses are **PoC quality**; the endpoint is **unauthenticated** (do not expose publicly without auth/TLS).

### API summary

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/audit/status` | Adapter path, load errors, whether weights are in memory |
| `POST` | `/v1/audit` | Model audit: `{"code": "..."}` **or** `{"files": {"src/app.py": "..."}}` (mutually exclusive). Response adds optional `ast_summary` when AST context was prepended to the prompt. |
| `POST` | `/v1/codebase/ast` | No model: `{"files": { "path": "source" }}` → parse trees summarized per file, merged routes/imports (FastAPI-style decorators are detected heuristically). |

**AST behavior:** For `POST /v1/audit`, if you send **`files`**, an `ast` summary is **prepended to the user message by default** (set `include_ast_summary: false` to disable). For a single **`code`** string only, AST is **off** by default unless you set `include_ast_summary: true`. Limits: 64 files, 250k characters total, paths normalized with `/`, `..` rejected.

## Before pushing to GitHub

- Confirm **no** `.env`, API keys, or **adapter weights** under `outputs/` are committed.
- `.gitignore` already excludes `.env`, `outputs/`, checkpoints, and common caches.

## License / disclaimer

Outputs may be **incorrect or incomplete** relative to the EU AI Act or any jurisdiction. Use at your own risk for research and prototyping only.
