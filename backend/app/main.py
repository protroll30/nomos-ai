"""Nomos FastAPI entrypoint (Phase 1 skeleton)."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Nomos", description="Regulation-to-code intelligence API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "nomos"}
