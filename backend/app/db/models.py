"""The Job row: everything the frontend polls for, plus the Timeline snapshot.

`timeline_json` is written after every pipeline stage. It is the debug trail and the
resume point — a job that died during rendering still has its narration timings on disk.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel

from app.core.models import JobStatus


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Job(SQLModel, table=True):
    """One render request. Status/progress are advanced by app.worker.pipeline."""

    id: str = Field(default_factory=_new_id, primary_key=True)

    topic: str
    slide_count: int
    voice: str
    music: bool = False

    status: str = Field(default=JobStatus.QUEUED.value, index=True)
    progress: int = 0
    current_stage: str | None = None
    error: str | None = None

    timeline_json: str | None = None
    video_path: str | None = None

    created_at: datetime = Field(default_factory=_now, index=True)
    updated_at: datetime = Field(default_factory=_now)
