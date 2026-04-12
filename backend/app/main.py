"""Nomos FastAPI entrypoint: health + LoRA audit inference for the dashboard."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import model_runner

_origins = os.environ.get(
    "NOMOS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app = FastAPI(title="Nomos", description="Regulation-to-code intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "nomos"}


@app.get("/v1/audit/status")
def audit_status() -> dict:
    if os.environ.get("NOMOS_PRELOAD_MODEL", "").lower() in ("1", "true", "yes"):
        if not model_runner.inference_ready():
            model_runner.ensure_loaded()
    return model_runner.status_snapshot()


class AuditBody(BaseModel):
    code: str = Field(..., min_length=1, max_length=120_000)
    max_new_tokens: int | None = Field(default=None, ge=64, le=4096)


class AuditResponse(BaseModel):
    raw_text: str
    parsed: dict | None
    parse_error: str | None


@app.post("/v1/audit", response_model=AuditResponse)
def audit(body: AuditBody) -> AuditResponse:
    max_tok = body.max_new_tokens or int(os.environ.get("NOMOS_MAX_NEW_TOKENS", "1024"))
    try:
        raw, parsed, parse_err = model_runner.generate_audit(
            body.code,
            max_new_tokens=max_tok,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return AuditResponse(raw_text=raw, parsed=parsed, parse_error=parse_err)


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("NOMOS_PRELOAD_MODEL", "").lower() in ("1", "true", "yes"):
        model_runner.ensure_loaded()
