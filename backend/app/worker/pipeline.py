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
import functools
import inspect
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.models import (
    BulletPoint,
    JobStatus,
    RenderProfile,
    Scene,
    SceneRole,
    Theme,
    Timeline,
    Word,
)
from app.db.models import Job
from app.db.session import session_scope
from app.providers.bullet_timing import time_bullets
from app.providers.gemini_script import scene_clip_prompt, scene_role
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

#: Crossfade budget added on top of the configured pause.
#:
#: ``xfade`` consumes its overlap, so a scene's tail is shortened by the transition that
#: follows it. Padding by only the desired pause would leave roughly half of it audible.
#: Matches ``VisualPlan.transition_duration``'s default; scenes whose transition is
#: clamped shorter simply get a slightly longer breath, which is harmless.
_XFADE_BUDGET = 0.5


def _scene_tail_pad() -> float:
    """Breath after each scene: the pause the user hears, plus the crossfade it loses."""
    return max(0.0, get_settings().video_scene_pause_s) + _XFADE_BUDGET


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
        return probed + _scene_tail_pad()
    if words:
        return max(w.end for w in words) + _scene_tail_pad()
    return _MIN_SCENE_DURATION


# ----------------------------------------------------------------------------- stages


def _script_kwargs(provider: Any, job: Job) -> dict[str, Any]:
    """The generation knobs `provider.generate` actually accepts.

    `ScriptProvider.generate` grew `bullets_per_slide`/`tone` as keyword-only arguments,
    but the concrete providers are developed separately and may still be on the two-arg
    signature. Inspecting is better than catching TypeError, which would also swallow a
    genuine TypeError raised *inside* generate() and silently rerun the whole call.
    """
    requested = {"bullets_per_slide": job.bullets_per_slide, "tone": job.tone}
    try:
        params = inspect.signature(provider.generate).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return requested
    accepted = {name: value for name, value in requested.items() if name in params}
    ignored = sorted(set(requested) - set(accepted))
    if ignored:
        logger.warning(
            "%s.generate() does not accept %s; those choices are being dropped",
            type(provider).__name__,
            ", ".join(ignored),
        )
    return accepted


async def _stage_script(job: Job, job_dir: Path, theme: Theme | None = None) -> Timeline:
    provider = factory.script_provider()
    script = await asyncio.to_thread(
        functools.partial(
            provider.generate, job.topic, job.slide_count, **_script_kwargs(provider, job)
        )
    )
    scenes = [
        Scene(
            id=s.id,
            # The script decides the shape of the video; everything downstream reads it
            # from here. A provider that does not plan roles reports CONTENT, which is
            # exactly what an unstructured script is.
            role=scene_role(s),
            narration=s.narration,
            heading=s.heading,
            image_prompt=s.image_prompt,
            clip_prompt=scene_clip_prompt(s),
            # Carried untimed: `appear_at` needs the word timings, which do not exist
            # until narration has been synthesised and aligned. See _stage_bullets.
            bullets=[
                BulletPoint(text=text)
                for text in _within_bullet_budget(scene_role(s), [b for b in s.bullets])
            ],
        )
        for s in script.scenes
    ]
    timeline = Timeline(
        job_id=job.id,
        topic=job.topic,
        title=script.title,
        scenes=scenes,
        voice=job.voice,
        profile=RenderProfile(),
    )
    if theme is not None:
        # Stamped on the Timeline so the palette is persisted with the job's debug trail
        # and visible to the planner. The render backend still takes it by constructor.
        timeline.theme = theme
    return timeline


def _api_concurrency() -> int:
    """In-flight calls per provider. Caps burst so a 12-scene job can't trip rate limits."""
    return max(1, get_settings().video_api_concurrency)


async def _gather_scenes(label: str, coros: list[Any]) -> list[Any]:
    """Run per-scene work concurrently, reporting every failure rather than the first.

    ``return_exceptions=True`` so one bad scene doesn't cancel siblings mid-flight and
    leave half-written files behind; we surface the whole picture instead.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    errors = [f"scene {i + 1}: {r}" for i, r in enumerate(results) if isinstance(r, BaseException)]
    if errors:
        raise RuntimeError(f"{label} failed for {len(errors)} scene(s): " + "; ".join(errors))
    return results


async def _stage_images(timeline: Timeline, job_dir: Path) -> None:
    """All images at once — they depend on nothing but their own prompt."""
    provider = factory.image_provider()
    profile = timeline.profile
    limit = asyncio.Semaphore(_api_concurrency())

    async def one(scene: Scene) -> None:
        out = job_dir / f"scene_{scene.id:02d}.png"
        async with limit:
            path = await asyncio.to_thread(
                provider.generate,
                scene.image_prompt,
                out,
                profile.width * profile.upscale_factor,
                profile.height * profile.upscale_factor,
            )
        scene.image_path = str(path)

    await _gather_scenes("image generation", [one(s) for s in timeline.scenes])


async def _stage_narrate(timeline: Timeline, job_dir: Path) -> None:
    """All narration at once — each take is independent of the others."""
    synth = factory.speech_synthesizer()
    limit = asyncio.Semaphore(_api_concurrency())

    async def one(scene: Scene) -> None:
        out = job_dir / f"scene_{scene.id:02d}.mp3"
        async with limit:
            path = await asyncio.to_thread(synth.synthesize, scene.narration, timeline.voice, out)
        scene.audio_path = str(path)

    await _gather_scenes("narration", [one(s) for s in timeline.scenes])


async def _stage_align(timeline: Timeline) -> None:
    """Align every scene concurrently, then lay them end-to-end on the audio clock.

    The alignment calls are independent, but the rebasing is NOT: each scene's offset is
    the sum of all previous durations. So fan out the network work, then walk the scenes
    in order to assign the cursor. Doing both in the same loop would serialise the slow
    part for no reason.
    """
    aligner = factory.aligner()
    limit = asyncio.Semaphore(_api_concurrency())

    for scene in timeline.scenes:
        if not scene.audio_path:
            raise RuntimeError(f"scene {scene.id} has no audio to align")

    async def one(scene: Scene) -> tuple[list[Word], float]:
        audio_path = Path(scene.audio_path or "")
        async with limit:
            words = await asyncio.to_thread(aligner.align, audio_path, scene.narration)
        duration = await asyncio.to_thread(_measured_duration, audio_path, words)
        return words, duration

    aligned = await _gather_scenes("alignment", [one(s) for s in timeline.scenes])

    cursor = 0.0
    for scene, (words, duration) in zip(timeline.scenes, aligned, strict=True):
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
    """Generate the music bed. Best-effort by design.

    The bed is a nice-to-have; the narration carries the video. A generative music
    model can return an empty candidate (observed: ``finishReason='OTHER'``), and
    losing a completed script, four images and four narration takes to that would be
    absurd. On failure we log, leave ``music_path`` unset, and let ``assemble``
    produce a narration-only mix — which it already supports.
    """
    provider = factory.music_provider()
    out = job_dir / "music.mp3"
    try:
        path = await asyncio.to_thread(
            provider.generate,
            f"instrumental underscore for a short explainer about {timeline.topic}",
            timeline.narration_duration,
            out,
        )
    except Exception:
        logger.warning("music generation failed; continuing without a bed", exc_info=True)
        timeline.music_path = None
        return
    timeline.music_path = str(path)


async def _stage_plan(timeline: Timeline) -> Timeline:
    """Creative decisions. Runs after alignment because plans depend on real durations."""
    planner = factory.visual_planner()
    return await asyncio.to_thread(planner.plan, timeline)


async def _stage_render(timeline: Timeline, job_dir: Path, theme: Theme | None = None) -> Timeline:
    """One clip per scene, from the planned timeline.

    Delegated to the backend's batch entry point so scenes render concurrently and
    frame counts are rounded cumulatively — looping ``render_scene`` here rounded
    each scene independently, leaking up to half a frame per scene.

    ``theme`` goes to the backend's constructor, which is where ``FFmpegBackend`` reads
    it from — ``Timeline.theme`` carries the same palette for the planner and the debug
    trail, but the backend does not consult it.
    """
    backend = factory.video_backend(theme=theme)

    # Imported here, not at module scope: the API must boot even when the render
    # backend is unavailable (see factory's lazy resolution).
    from app.render.ffmpeg_backend import layout_region

    for scene in timeline.scenes:
        if scene.plan is None:
            raise RuntimeError(f"planner returned no VisualPlan for scene {scene.id}")
        # TITLE_CARD has no image region, so a missing image is valid there.
        if not scene.image_path and layout_region(scene.plan, timeline.profile):
            raise RuntimeError(f"scene {scene.id} has no image to render")

    rendered = await asyncio.to_thread(backend.render_all, timeline, job_dir)

    missing = [s.id for s in rendered.scenes if not s.clip_path]
    if missing:
        raise RuntimeError(f"backend returned no clip for scenes {missing}")
    return rendered


def _within_bullet_budget(role: SceneRole, texts: list[str]) -> list[str]:
    """Trim `texts` to what `role` may show on screen.

    The last line of defence for the video's shape. A title card renders one large heading
    and nothing else, so bullets on it are not a style choice but a broken slide — and the
    script provider is not the only way scenes reach a Timeline (a persisted job is
    rehydrated, a different provider may not plan at all).
    """
    budget = role.bullet_budget
    if len(texts) <= budget:
        return texts
    logger.info(
        "%s scene: dropping %d bullet(s) over the role's budget of %d",
        role.value,
        len(texts) - budget,
        budget,
    )
    return texts[:budget]


def _stage_bullets(timeline: Timeline) -> Timeline:
    """Time each scene's bullets against its own spoken words.

    Runs after planning (so ``plan.bullet_min_gap`` is known) and after alignment (so
    real word timings exist). Pure and cheap — no I/O, so no thread needed.

    Also the enforcement point for ``SceneRole.bullet_budget``: the budget is per role, not
    per video, so a title card gets none however many the script sent.
    """
    for scene in timeline.scenes:
        texts = _within_bullet_budget(scene.role, [b.text for b in scene.bullets])
        if not texts:
            scene.bullets = []
            continue
        min_gap = scene.plan.bullet_min_gap if scene.plan else 0.6
        scene.bullets = time_bullets(
            texts,
            scene.words,
            scene.start,
            scene.duration,
            min_gap=min_gap,
        )
        logger.info(
            "scene %d (%s): %d bullets at %s",
            scene.id,
            scene.role.value,
            len(scene.bullets),
            [round(b.appear_at, 2) for b in scene.bullets],
        )
    return timeline


async def _stage_assemble(timeline: Timeline, job_dir: Path, theme: Theme | None = None) -> Path:
    backend = factory.video_backend(theme=theme)
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
        # Resolved once, up front: a palette that cannot be resolved should not surface
        # as a surprise halfway through an expensive render.
        theme = job.resolved_theme()
        logger.info("job %s renders on theme %r", job_id, theme.name)

        await _set_stage(job_id, JobStatus.SCRIPTING)
        timeline = await _stage_script(job, job_dir, theme)
        await _save_timeline(job_id, timeline)

        # Two independent branches. Pictures need nothing from the soundtrack, and the
        # soundtrack needs nothing from the pictures — only the render needs both. Running
        # them in series wasted the whole of whichever branch finished first.
        #
        #   visual : images
        #   audio  : narrate -> align -> music     (strictly ordered: the bed is sized
        #                                           from the measured narration length)
        async def visual_branch() -> None:
            await _set_stage(job_id, JobStatus.IMAGING, timeline)
            await _stage_images(timeline, job_dir)

        async def audio_branch() -> None:
            await _stage_narrate(timeline, job_dir)
            # Audio is the clock: scene start/end only become real here.
            await _set_stage(job_id, JobStatus.ALIGNING, timeline)
            await _stage_align(timeline)
            if job.music:
                await _set_stage(job_id, JobStatus.SCORING, timeline)
                await _stage_music(timeline, job_dir)

        await _set_stage(job_id, JobStatus.NARRATING, timeline)
        await asyncio.gather(visual_branch(), audio_branch())
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.RENDERING, timeline)
        # Persist the plans before the expensive part, so a render crash is diagnosable.
        timeline = await _stage_plan(timeline)
        timeline = _stage_bullets(timeline)
        await _save_timeline(job_id, timeline)
        timeline = await _stage_render(timeline, job_dir, theme)
        await _save_timeline(job_id, timeline)

        await _set_stage(job_id, JobStatus.ASSEMBLING, timeline)
        video_path = await _stage_assemble(timeline, job_dir, theme)

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
        _discard_empty_job_dir(job_id)


def _discard_empty_job_dir(job_id: str) -> None:
    """Remove the job's output dir if it never received a file.

    ``settings.job_dir`` creates the directory up front, so a job that fails before
    writing anything leaves an empty husk behind. One per failed job accumulates into
    a directory listing where you can't tell real output from debris.
    """
    try:
        job_dir = get_settings().video_output_dir / job_id
        if job_dir.is_dir() and not any(job_dir.iterdir()):
            job_dir.rmdir()
            logger.info("removed empty job dir %s", job_dir)
    except OSError:
        logger.debug("could not tidy job dir for %s", job_id, exc_info=True)


def timeline_from_job(job: Job) -> Timeline | None:
    """Rehydrate the persisted Timeline, tolerating a half-written row."""
    if not job.timeline_json:
        return None
    try:
        return Timeline.model_validate(json.loads(job.timeline_json))
    except Exception:  # noqa: BLE001 - debug aid must not break a status response
        return None
