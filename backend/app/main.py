"""ASGI entrypoint. `uv run uvicorn app.main:app --reload --port 8000`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import jobs, voices
from app.core.config import get_settings
from app.db.session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Vite dev server, both spellings — the browser sends whichever the user typed.
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.video_output_dir.mkdir(parents=True, exist_ok=True)
    settings.video_cache_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title="prompt-to-video", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(voices.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
