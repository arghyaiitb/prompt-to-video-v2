"""Job endpoints. POST returns in milliseconds; the render takes minutes.

The handoff is `asyncio.create_task` on a tracked set — fire-and-forget without letting
the GC collect a live task. The pipeline itself pushes every blocking call into a thread,
so a running render does not slow down the status polling that drives the UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, desc, select

from app.api.themes import (
    PALETTE_FIELDS,
    ThemeCustom,
    default_theme_name,
    known_theme_ids,
    review_palette,
    suggest_palette_fix,
)
from app.api.voices import engine_for_voice
from app.core.config import get_settings
from app.core.models import JobStatus
from app.db.models import CUSTOM_THEME_NAME, DEFAULT_BULLETS_PER_SLIDE, FALLBACK_TTS_ENGINE, Job
from app.db.session import get_session
from app.worker import factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["jobs"])

RECENT_LIMIT = 20

Tone = Literal["new_hires", "all_staff", "technical", "executives"]
"""Audience register. A closed set: the script prompt has to say something concrete for
each one, so an arbitrary string would silently do nothing."""

# Strong refs to in-flight renders; asyncio only holds weak ones.
_tasks: set[asyncio.Task[None]] = set()

SessionDep = Annotated[Session, Depends(get_session)]


class JobCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    slide_count: int = Field(default=5, ge=2, le=10)
    voice: str = Field(default="", max_length=100)
    music: bool = False

    tts_engine: str | None = Field(default=None, max_length=40)
    """Engine id from GET /api/engines. Unknown ids fall back to the default engine.

    Cross-checked against `voice`: an engine paired with another engine's voice is a 422,
    not a guess — see `_resolve_voice`."""

    theme: str | None = Field(default=None, max_length=60)
    """Preset id from GET /api/themes. Unknown ids fall back to the default palette."""

    theme_custom: ThemeCustom | None = None
    """Own-colours override. Gated on contrast — see `_check_custom_theme`."""

    bullets_per_slide: int = Field(default=DEFAULT_BULLETS_PER_SLIDE, ge=3, le=5)
    """Fewer than three looks empty in the left panel; more than five cannot be read
    before the scene ends."""

    tone: Tone | None = None


class JobCreated(BaseModel):
    job_id: str
    theme_warnings: list[str] = Field(default_factory=list)
    """Palette advice that cleared WCAG AA but missed our AAA recommendation.

    Additive and optional: a client that ignores it behaves exactly as before.
    """


class JobStatusOut(BaseModel):
    job_id: str
    topic: str
    status: str
    progress: int
    current_stage: str | None = None
    error: str | None = None
    video_url: str | None = None
    created_at: datetime
    """Exposed so job-history cards can show relative time."""

    theme: str
    """What the job is actually rendered with, not what was requested — an unknown id
    is normalised to the default at creation time."""

    theme_custom: dict[str, Any] | None = None
    bullets_per_slide: int
    tone: str | None = None

    tts_engine: str
    """The engine that actually narrated this job, resolved at creation time."""

    voice: str
    """The resolved voice — an omitted one is filled in from the engine's default."""


def _as_utc(value: datetime) -> datetime:
    """Re-attach UTC before serialising.

    ``_now()`` stores ``datetime.now(UTC)``, but SQLite has no tz type and drops the
    tzinfo on the way back out. A naive ISO string is read as *local* time by
    ECMAScript, so clients in non-UTC zones misreport age by their whole offset.
    """
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _video_url(job: Job) -> str | None:
    if job.status == JobStatus.DONE.value and job.video_path:
        return f"/api/jobs/{job.id}/video"
    return None


def _to_out(job: Job) -> JobStatusOut:
    return JobStatusOut(
        job_id=job.id,
        topic=job.topic,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error=job.error,
        video_url=_video_url(job),
        created_at=_as_utc(job.created_at),
        # `or` rather than a bare read: rows written before these columns existed are
        # backfilled by the migration, but a hand-edited DB can still hold NULL.
        theme=job.theme or default_theme_name(),
        theme_custom=job.custom_palette(),
        bullets_per_slide=job.bullets_per_slide or DEFAULT_BULLETS_PER_SLIDE,
        tone=job.tone,
        tts_engine=job.tts_engine or FALLBACK_TTS_ENGINE,
        voice=job.voice or "",
    )


def _resolve_theme_name(requested: str | None) -> str:
    """Normalise the preset id to one we can actually render.

    An unknown id is not an error — `get_theme` falls back — but storing it would make
    GET /api/jobs/{id} claim the video used a palette that does not exist.
    """
    default = default_theme_name()
    if not requested:
        return default
    known = known_theme_ids()
    if known and requested not in known:
        logger.info("unknown theme %r requested; falling back to %r", requested, default)
        return default
    return requested


def _resolve_tts_engine(requested: str | None) -> str:
    """Normalise the engine id, exactly as `_resolve_theme_name` does for palettes.

    An unknown id is not fatal — the render succeeds on the default engine — but storing
    the raw request would make GET /api/jobs/{id} claim narration came from an engine that
    does not exist, and the engine determines whether SSML was used at all.
    """
    resolved = factory.resolve_speech_engine(requested)
    if requested and resolved != requested.strip().lower():
        logger.info("unknown tts_engine %r requested; falling back to %r", requested, resolved)
    return resolved


def _resolve_voice(requested: str, engine: str) -> str:
    """The voice this job narrates with, or a 422 naming the mismatch.

    Two different rules, on purpose:

    * No voice at all is normalised to the engine's default. "Switch engine, keep the rest
      of the form" has to keep working, and there is exactly one sensible answer.
    * A voice that demonstrably belongs to a *different* engine is REJECTED rather than
      normalised. It is always a client bug — the voice list is served per engine by this
      same API — and the voice is the most audible property of a six-minute render, so
      silently substituting one ships a video in a voice nobody chose. Failing loudly here
      costs a round trip; normalising costs a re-render.
    """
    voice = (requested or "").strip()
    if not voice:
        return factory.default_voice(engine)

    owner = engine_for_voice(voice)
    if owner is not None and owner != engine:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "voice_engine_mismatch",
                "message": (
                    f"voice {voice!r} belongs to the {owner!r} engine but tts_engine is "
                    f"{engine!r}; pick a voice from GET /api/voices?engine={engine} "
                    f"or omit voice to use {factory.default_voice(engine)!r}"
                ),
                "voice": voice,
                "voice_engine": owner,
                "tts_engine": engine,
                "engine_default_voice": factory.default_voice(engine),
            },
        )
    return voice


def _check_custom_theme(custom: ThemeCustom) -> list[str]:
    """Reject a custom palette that fails contrast, naming every failure.

    Text is burned into the video's pixels, so an unreadable palette cannot be corrected
    after the render — it has to be caught here. The 422 body carries a fixed palette so
    the UI can offer a one-click correction instead of a dead end.
    """
    theme = custom.to_theme()
    failures, warnings = review_palette(theme)
    if not failures:
        return warnings

    fixed = suggest_palette_fix(theme)
    raise HTTPException(
        # Literal 422 rather than the starlette constant, which was renamed
        # (UNPROCESSABLE_ENTITY -> UNPROCESSABLE_CONTENT) and now warns on the old name.
        status_code=422,
        detail={
            "error": "theme_contrast_failed",
            "message": (
                "the custom palette is not readable on screen; "
                "narration text would be burned into the video unreadably"
            ),
            "failures": failures,
            "contrast": theme.contrast_report(),
            "suggested_fix": {field: getattr(fixed, field) for field in PALETTE_FIELDS},
            "suggested_contrast": fixed.contrast_report(),
        },
    )


def _spawn(job_id: str) -> None:
    """Kick off the render. Imported lazily so the API boots without the providers."""
    from app.worker import pipeline

    async def _runner() -> None:
        await pipeline.run_job(job_id)

    try:
        task = asyncio.get_running_loop().create_task(_runner(), name=f"job:{job_id}")
    except RuntimeError:  # no loop (sync context) — run it on a private loop in a thread
        import threading

        threading.Thread(
            target=lambda: asyncio.run(_runner()), name=f"job:{job_id}", daemon=True
        ).start()
        return
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _get_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@router.post("/jobs", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_job(payload: JobCreate, session: SessionDep) -> JobCreated:
    warnings: list[str] = []
    if payload.theme_custom is not None:
        warnings = _check_custom_theme(payload.theme_custom)

    tts_engine = _resolve_tts_engine(payload.tts_engine)
    job = Job(
        topic=payload.topic.strip(),
        slide_count=payload.slide_count,
        voice=_resolve_voice(payload.voice, tts_engine),
        tts_engine=tts_engine,
        music=payload.music,
        theme=(
            CUSTOM_THEME_NAME
            if payload.theme_custom is not None
            else _resolve_theme_name(payload.theme)
        ),
        theme_custom=(
            json.dumps(payload.theme_custom.model_dump())
            if payload.theme_custom is not None
            else None
        ),
        bullets_per_slide=payload.bullets_per_slide,
        tone=payload.tone,
        status=JobStatus.QUEUED.value,
        progress=0,
        current_stage=JobStatus.QUEUED.value,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    _spawn(job.id)
    return JobCreated(job_id=job.id, theme_warnings=warnings)


@router.get("/jobs", response_model=list[JobStatusOut])
def list_jobs(session: SessionDep) -> list[JobStatusOut]:
    rows = session.exec(
        select(Job).order_by(desc(Job.created_at), desc(Job.id)).limit(RECENT_LIMIT)
    ).all()
    return [_to_out(job) for job in rows]


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
def get_job(job_id: str, session: SessionDep) -> JobStatusOut:
    return _to_out(_get_or_404(session, job_id))


@router.get("/jobs/{job_id}/timeline")
def get_job_timeline(job_id: str, session: SessionDep) -> dict[str, Any]:
    """Debug view of the persisted Timeline — real word timings, plans, paths."""
    job = _get_or_404(session, job_id)
    from app.worker.pipeline import timeline_from_job

    timeline = timeline_from_job(job)
    if timeline is None:
        raise HTTPException(status_code=404, detail="no timeline recorded yet")
    return timeline.model_dump(mode="json")


@router.get("/jobs/{job_id}/video")
def get_job_video(job_id: str, session: SessionDep) -> FileResponse:
    job = _get_or_404(session, job_id)
    if not job.video_path:
        raise HTTPException(status_code=404, detail="video not rendered yet")
    path = Path(job.video_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="video file is missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, session: SessionDep) -> None:
    job = _get_or_404(session, job_id)
    job_dir = get_settings().video_output_dir / job_id
    session.delete(job)
    session.commit()
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)
