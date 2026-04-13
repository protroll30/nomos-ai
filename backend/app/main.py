"""Nomos FastAPI entrypoint: health + audit inference (local HF or OpenAI) for the dashboard."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
_DOTENV_LOADED = load_dotenv(_ENV_FILE)

from app import code_intel
from app import model_runner

_LOGGER = logging.getLogger("nomos.audit")

_origins = os.environ.get(
    "NOMOS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

_CORS_ALLOW_HEADERS = [
    "Accept",
    "Accept-Language",
    "Content-Type",
    "Authorization",
    "X-Nomos-Audit-Backend",
]

_MAX_AUDIT_FILES = 64
_MAX_AUDIT_TOTAL_CHARS = 250_000
_MAX_FILE_PATH_LEN = 512
_AUDIT_BACKEND_HEADER = "X-Nomos-Audit-Backend"


def _client_audit_backend(request: Request) -> str | None:
    raw = request.headers.get(_AUDIT_BACKEND_HEADER)
    if raw is None:
        return None
    s = raw.strip()
    return s or None


def _resolve_client_backend(
    request: Request,
    body_audit_backend: str | None = None,
) -> str | None:
    q = request.query_params.get("audit_backend")
    for candidate in (q, _client_audit_backend(request), body_audit_backend):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def _debug_audit_enabled() -> bool:
    return os.environ.get("NOMOS_DEBUG_AUDIT", "").strip().lower() in ("1", "true", "yes")


def _log_audit_trace(request: Request, client: str | None, label: str) -> None:
    if not _debug_audit_enabled():
        return
    eff = model_runner.effective_audit_backend(client)
    _LOGGER.info(
        "%s path=%s query=%r header=%r resolved=%r effective=%r",
        label,
        request.url.path,
        request.query_params.get("audit_backend"),
        request.headers.get(_AUDIT_BACKEND_HEADER),
        client,
        eff,
    )


app = FastAPI(title="Nomos", description="Regulation-to-code intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=_CORS_ALLOW_HEADERS,
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
def audit_status(request: Request) -> dict:
    client = _resolve_client_backend(request)
    _log_audit_trace(request, client, "status")
    if os.environ.get("NOMOS_PRELOAD_MODEL", "").lower() in ("1", "true", "yes"):
        if model_runner.effective_audit_backend(client) == "hf":
            if not model_runner.inference_ready(client):
                model_runner.ensure_loaded(client)
    return model_runner.status_snapshot(client)


@app.get("/v1/audit/debug")
def audit_debug(request: Request) -> dict:
    if not _debug_audit_enabled():
        raise HTTPException(status_code=404, detail="NOMOS_DEBUG_AUDIT is not enabled")
    client = _resolve_client_backend(request)
    _log_audit_trace(request, client, "debug")
    return {
        "http": {
            "query_audit_backend": request.query_params.get("audit_backend"),
            "header_X_Nomos_Audit_Backend": _client_audit_backend(request),
            "resolved_client_override": client,
        },
        "paths": {
            "repo_root": str(_REPO_ROOT.resolve()),
            "env_file": str(_ENV_FILE.resolve()),
            "env_file_exists": _ENV_FILE.is_file(),
            "dotenv_loaded_vars": _DOTENV_LOADED,
        },
        "runner": model_runner.debug_audit_bundle(client),
        "status_snapshot": model_runner.status_snapshot(client),
    }


class AuditBody(BaseModel):
    code: str | None = Field(None, max_length=120_000)
    files: dict[str, str] | None = None
    include_ast_summary: bool | None = Field(
        default=None,
        description="If null, AST summary is included only when `files` is set.",
    )
    max_new_tokens: int | None = Field(default=None, ge=64, le=4096)
    audit_backend: str | None = Field(
        default=None,
        description="Per-request backend: openai or hf (same as query audit_backend / X-Nomos-Audit-Backend).",
    )

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
def audit(request: Request, body: AuditBody) -> AuditResponse:
    client = _resolve_client_backend(request, body.audit_backend)
    _log_audit_trace(request, client, "audit")
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
            client_backend=client,
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
        if model_runner.effective_audit_backend(None) == "hf":
            model_runner.ensure_loaded()
