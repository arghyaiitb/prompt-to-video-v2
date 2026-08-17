"""Job endpoints. POST returns in milliseconds; the render takes minutes.

The handoff is `asyncio.create_task` on a tracked set — fire-and-forget without letting
the GC collect a live task. The pipeline itself pushes every blocking call into a thread,
so a running render does not slow down the status polling that drives the UI.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, desc, select

from app.core.config import get_settings
from app.core.models import JobStatus
from app.db.models import Job
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["jobs"])

RECENT_LIMIT = 20

# Strong refs to in-flight renders; asyncio only holds weak ones.
_tasks: set[asyncio.Task[None]] = set()

SessionDep = Annotated[Session, Depends(get_session)]


class JobCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    slide_count: int = Field(default=5, ge=2, le=10)
    voice: str = Field(default="", max_length=100)
    music: bool = False


class JobCreated(BaseModel):
    job_id: str


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
        created_at=job.created_at,
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
    settings = get_settings()
    job = Job(
        topic=payload.topic.strip(),
        slide_count=payload.slide_count,
        voice=payload.voice or settings.video_default_tts_voice,
        music=payload.music,
        status=JobStatus.QUEUED.value,
        progress=0,
        current_stage=JobStatus.QUEUED.value,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    _spawn(job.id)
    return JobCreated(job_id=job.id)


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
