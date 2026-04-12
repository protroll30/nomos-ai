"""Nomos FastAPI entrypoint: health + LoRA audit inference for the dashboard."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from app import code_intel
from app import model_runner

_origins = os.environ.get(
    "NOMOS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

_MAX_AUDIT_FILES = 64
_MAX_AUDIT_TOTAL_CHARS = 250_000
_MAX_FILE_PATH_LEN = 512

app = FastAPI(title="Nomos", description="Regulation-to-code intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_files(raw: dict[str, str]) -> dict[str, str]:
    if len(raw) > _MAX_AUDIT_FILES:
        raise ValueError(f"Too many files (max {_MAX_AUDIT_FILES})")
    out: dict[str, str] = {}
    for k, v in raw.items():
        kp = k.strip().replace("\\", "/")
        if not kp or len(kp) > _MAX_FILE_PATH_LEN:
            raise ValueError(f"Invalid file path: {k!r}")
        if ".." in kp.split("/"):
            raise ValueError(f"Path must not contain '..': {k!r}")
        out[kp] = v if isinstance(v, str) else str(v)
    total = sum(len(k) + len(val) for k, val in out.items())
    if total > _MAX_AUDIT_TOTAL_CHARS:
        raise ValueError(f"Total source exceeds {_MAX_AUDIT_TOTAL_CHARS} characters")
    if not any(s.strip() for s in out.values()):
        raise ValueError("All file contents are empty")
    return out


def _resolved_audit_input(
    code: str | None,
    files: dict[str, str] | None,
    include_ast_summary: bool,
) -> tuple[str, str | None]:
    if files is not None:
        merged = code_intel.merge_files_for_prompt(files)
        summary: str | None = None
        if include_ast_summary:
            summary = code_intel.format_scan_for_prompt(code_intel.scan_codebase(files))
        return merged, summary
    assert code is not None
    if include_ast_summary:
        scan = code_intel.scan_module(code, "snippet.py")
        return code, code_intel.format_scan_for_prompt(scan)
    return code, None


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
    code: str | None = Field(None, max_length=120_000)
    files: dict[str, str] | None = None
    include_ast_summary: bool | None = Field(
        default=None,
        description="If null, AST summary is included only when `files` is set.",
    )
    max_new_tokens: int | None = Field(default=None, ge=64, le=4096)

    @model_validator(mode="after")
    def _validate_source(self) -> AuditBody:
        has_code = bool(self.code and self.code.strip())
        has_files = bool(
            self.files and any((v or "").strip() for v in self.files.values())
        )
        if has_code and has_files:
            raise ValueError("Provide either code or files, not both")
        if not has_code and not has_files:
            raise ValueError("Provide code or a non-empty files map")
        if has_files and self.files is not None:
            self.files = _normalize_files(dict(self.files))
        return self


class AuditResponse(BaseModel):
    raw_text: str
    parsed: dict | None
    parse_error: str | None
    ast_summary: str | None = None


@app.post("/v1/audit", response_model=AuditResponse)
def audit(body: AuditBody) -> AuditResponse:
    max_tok = body.max_new_tokens or int(os.environ.get("NOMOS_MAX_NEW_TOKENS", "1024"))
    use_ast = (
        body.include_ast_summary
        if body.include_ast_summary is not None
        else (body.files is not None)
    )
    code, ast_summary = _resolved_audit_input(
        body.code,
        body.files,
        use_ast,
    )
    try:
        raw, parsed, parse_err = model_runner.generate_audit(
            code,
            max_new_tokens=max_tok,
            ast_summary=ast_summary,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return AuditResponse(
        raw_text=raw,
        parsed=parsed,
        parse_error=parse_err,
        ast_summary=ast_summary,
    )


class CodebaseAstBody(BaseModel):
    files: dict[str, str]

    @model_validator(mode="after")
    def _norm(self) -> CodebaseAstBody:
        self.files = _normalize_files(dict(self.files))
        return self


@app.post("/v1/codebase/ast")
def codebase_ast(body: CodebaseAstBody) -> dict:
    return code_intel.scan_codebase(body.files)


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("NOMOS_PRELOAD_MODEL", "").lower() in ("1", "true", "yes"):
        model_runner.ensure_loaded()
