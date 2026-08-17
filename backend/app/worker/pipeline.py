"""The orchestrator. One job, eight stages, one rule: audio is the clock.

Word counts are a lie — a voice model's real pace varies with punctuation, numbers and
mood. So narration is synthesized first, aligned second, and only then do scene
boundaries exist. Everything downstream (visual plans, clip lengths, xfade offsets,
music duration) is derived from those measured durations.

Stage order and progress values are contractual: the frontend stepper renders them.

    scripting 10 -> imaging 30 -> narrating 50 -> aligning 60
              -> scoring 70 -> rendering 90 -> assembling 95 -> done 100

Provider calls are synchronous and blocking (HTTP via requests-style clients, ffmpeg via
subprocess), so every one of them goes through ``asyncio.to_thread``. Nothing in here may
stall the event loop — the API is answering status polls the whole time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.models import (
    JobStatus,
    RenderProfile,
    Scene,
    Timeline,
    Word,
)
from app.db.models import Job
from app.db.session import session_scope
from app.worker import factory

logger = logging.getLogger(__name__)

#: stage name -> progress percentage reported when the stage *starts*.
STAGE_PROGRESS: dict[JobStatus, int] = {
    JobStatus.SCRIPTING: 10,
    JobStatus.IMAGING: 30,
    JobStatus.NARRATING: 50,
    JobStatus.ALIGNING: 60,
    JobStatus.SCORING: 70,
    JobStatus.RENDERING: 90,
    JobStatus.ASSEMBLING: 95,
    JobStatus.DONE: 100,
}

#: Fallback when neither ffprobe nor the aligner can tell us how long a clip is.
_MIN_SCENE_DURATION = 1.0

#: Breath at the end of each scene so a transition never clips the last consonant.
_SCENE_TAIL_PAD = 0.25


# --------------------------------------------------------------------------- DB helpers


def _load_job(job_id: str) -> Job:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise LookupError(f"job {job_id} not found")
        session.expunge(job)
        return job


def _patch_job(job_id: str, **fields: Any) -> None:
    """Single-statement update. Keeps the write window tiny under WAL."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(UTC)
        session.add(job)
        session.commit()


async def _set_stage(job_id: str, status: JobStatus, timeline: Timeline | None = None) -> None:
    fields: dict[str, Any] = {
        "status": status.value,
        "progress": STAGE_PROGRESS[status],
        "current_stage": status.value,
    }
    if timeline is not None:
        fields["timeline_json"] = timeline.model_dump_json()
    await asyncio.to_thread(_patch_job, job_id, **fields)


async def _save_timeline(job_id: str, timeline: Timeline) -> None:
    await asyncio.to_thread(_patch_job, job_id, timeline_json=timeline.model_dump_json())


# ------------------------------------------------------------------------ audio measure


def _probe_duration(path: Path) -> float | None:
    """Real container duration via ffprobe. None if ffprobe is missing or unhappy."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _measured_duration(audio_path: Path, words: list[Word]) -> float:
    """Duration of one scene's narration, most trustworthy source first."""
    probed = _probe_duration(audio_path)
    if probed and probed > 0:
        return probed + _SCENE_TAIL_PAD
    if words:
        return max(w.end for w in words) + _SCENE_TAIL_PAD
    return _MIN_SCENE_DURATION


# ----------------------------------------------------------------------------- stages


async def _stage_script(job: Job, job_dir: Path) -> Timeline:
    provider = factory.script_provider()
    script = await asyncio.to_thread(provider.generate, job.topic, job.slide_count)
    scenes = [
        Scene(
            id=s.id,
            narration=s.narration,
            heading=s.heading,
            image_prompt=s.image_prompt,
        )
        for s in script.scenes
    ]
    return Timeline(
        job_id=job.id,
        topic=job.topic,
        title=script.title,
        scenes=scenes,
        voice=job.voice,
        profile=RenderProfile(),
    )


async def _stage_images(timeline: Timeline, job_dir: Path) -> None:
    provider = factory.image_provider()
    profile = timeline.profile
    for scene in timeline.scenes:
        out = job_dir / f"scene_{scene.id:02d}.png"
        path = await asyncio.to_thread(
            provider.generate,
            scene.image_prompt,
            out,
            profile.width * profile.upscale_factor,
            profile.height * profile.upscale_factor,
        )
        scene.image_path = str(path)


async def _stage_narrate(timeline: Timeline, job_dir: Path) -> None:
    synth = factory.speech_synthesizer()
    for scene in timeline.scenes:
        out = job_dir / f"scene_{scene.id:02d}.mp3"
        path = await asyncio.to_thread(synth.synthesize, scene.narration, timeline.voice, out)
        scene.audio_path = str(path)


async def _stage_align(timeline: Timeline) -> None:
    """Align, then lay scenes end-to-end on the real audio clock.

    Word timings come back relative to each scene's own file; they are rebased onto the
    global timeline so `Timeline.scenes[*].words` and `start/end` share one origin.
    """
    aligner = factory.aligner()
    cursor = 0.0
    for scene in timeline.scenes:
        if not scene.audio_path:
            raise RuntimeError(f"scene {scene.id} has no audio to align")
        audio_path = Path(scene.audio_path)
        words = await asyncio.to_thread(aligner.align, audio_path, scene.narration)
        duration = await asyncio.to_thread(_measured_duration, audio_path, words)

        scene.start = cursor
        scene.end = cursor + duration
        scene.words = [
            Word(
                word=w.word,
                start=w.start + cursor,
                end=w.end + cursor,
                confidence=w.confidence,
                punctuated_word=w.punctuated_word,
            )
            for w in words
        ]
        cursor = scene.end


async def _stage_music(timeline: Timeline, job_dir: Path) -> None:
    provider = factory.music_provider()
    out = job_dir / "music.mp3"
    path = await asyncio.to_thread(
        provider.generate,
        f"instrumental underscore for a short explainer about {timeline.topic}",
        timeline.narration_duration,
        out,
    )
    timeline.music_path = str(path)


async def _stage_plan(timeline: Timeline) -> Timeline:
    """Creative decisions. Runs after alignment because plans depend on real durations."""
    planner = factory.visual_planner()
    return await asyncio.to_thread(planner.plan, timeline)


async def _stage_render(timeline: Timeline, job_dir: Path) -> Timeline:
    """One clip per scene, from the planned timeline.

    Delegated to the backend's batch entry point so scenes render concurrently and
    frame counts are rounded cumulatively — looping ``render_scene`` here rounded
    each scene independently, leaking up to half a frame per scene.
    """
    backend = factory.video_backend()
    for scene in timeline.scenes:
        if scene.plan is None:
            raise RuntimeError(f"planner returned no VisualPlan for scene {scene.id}")
        if not scene.image_path:
            raise RuntimeError(f"scene {scene.id} has no image to render")

    rendered = await asyncio.to_thread(backend.render_all, timeline, job_dir)

    missing = [s.id for s in rendered.scenes if not s.clip_path]
    if missing:
        raise RuntimeError(f"backend returned no clip for scenes {missing}")
    return rendered


async def _stage_assemble(timeline: Timeline, job_dir: Path) -> Path:
    backend = factory.video_backend()
    out = job_dir / "video.mp4"
    return await asyncio.to_thread(backend.assemble, timeline, out)


# ------------------------------------------------------------------------------ driver


async def run_job(job_id: str) -> None:
    """Drive one job to completion. Never raises — failure lands in the DB row."""
    settings = get_settings()
    timeline: Timeline | None = None
    try:
        job = await asyncio.to_thread(_load_job, job_id)
        job_dir = settings.job_dir(job_id)

        await _set_stage(job_id, JobStatus.SCRIPTING)
        timeline = await _stage_script(job, job_dir)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.IMAGING, timeline)
        await _stage_images(timeline, job_dir)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.NARRATING, timeline)
        await _stage_narrate(timeline, job_dir)
        await _save_timeline(job_id, timeline)

        # Audio is the clock: scene start/end only become real here.
        await _set_stage(job_id, JobStatus.ALIGNING, timeline)
        await _stage_align(timeline)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.SCORING, timeline)
        if job.music:
            await _stage_music(timeline, job_dir)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.RENDERING, timeline)
        # Persist the plans before the expensive part, so a render crash is diagnosable.
        timeline = await _stage_plan(timeline)
        await _save_timeline(job_id, timeline)
        timeline = await _stage_render(timeline, job_dir)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.ASSEMBLING, timeline)
        video_path = await _stage_assemble(timeline, job_dir)

        await asyncio.to_thread(
            _patch_job,
            job_id,
            status=JobStatus.DONE.value,
            progress=STAGE_PROGRESS[JobStatus.DONE],
            current_stage=JobStatus.DONE.value,
            error=None,
            video_path=str(video_path),
            timeline_json=timeline.model_dump_json(),
        )
    except asyncio.CancelledError:
        # Server shutting down mid-render: leave a truthful row, don't swallow the cancel.
        await asyncio.to_thread(
            _patch_job,
            job_id,
            status=JobStatus.FAILED.value,
            error="cancelled: server shut down before the render finished",
        )
        raise
    except Exception as exc:  # noqa: BLE001 - a stuck "running" row is worse than any bug
        logger.exception("job %s failed", job_id)
        message = f"{type(exc).__name__}: {exc}"
        fields: dict[str, Any] = {"status": JobStatus.FAILED.value, "error": message[:2000]}
        if timeline is not None:
            fields["timeline_json"] = timeline.model_dump_json()
        try:
            await asyncio.to_thread(_patch_job, job_id, **fields)
        except Exception:
            logger.exception("could not record failure for job %s", job_id)


def timeline_from_job(job: Job) -> Timeline | None:
    """Rehydrate the persisted Timeline, tolerating a half-written row."""
    if not job.timeline_json:
        return None
    try:
        return Timeline.model_validate(json.loads(job.timeline_json))
    except Exception:  # noqa: BLE001 - debug aid must not break a status response
        return None
