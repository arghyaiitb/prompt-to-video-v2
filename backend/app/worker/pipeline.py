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
from app.db.models import NO_LOGO_ID, Job
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


#: Aspect of the hero image panel (720x900 at 1080p). Stills are REQUESTED at this shape
#: so the renderer's cover-crop has nothing to throw away. Kept as a constant rather than
#: read from `text_overlay.image_region` because imaging runs before planning, and the
#: planner holds one hero layout per video anyway.
_HERO_ASPECT = 0.8


def _render_profile() -> RenderProfile:
    """The render profile for a new job, honouring the configured scene concurrency.

    Split out so a run with several jobs in flight can lower the per-job worker count:
    the parallelism is per-job, so leaving it on auto oversubscribes the box.
    """
    return RenderProfile(render_concurrency=get_settings().video_render_concurrency)


def _reused_scenes(source: Timeline) -> list[Scene]:
    """Clone a previous job's scenes, keeping content and dropping measurements.

    Kept: role, narration, ssml, heading, bullets' TEXT, image_prompt/clip_prompt, and the
    already-generated ``image_path``/``video_path`` — so a re-render is visually identical
    and costs no image or clip credits.

    Dropped: ``audio_path``, ``start``/``end``, ``words``, and every bullet's ``appear_at``.
    Those are measurements of one particular narration. A different voice speaks at a
    different pace, so they MUST be re-derived — audio is the clock. Carrying them over is
    exactly how you get bullets that fire against the wrong words.
    """
    return [
        Scene(
            id=s.id,
            role=s.role,
            narration=s.narration,
            ssml=s.ssml,
            heading=s.heading,
            image_prompt=s.image_prompt,
            clip_prompt=s.clip_prompt,
            image_path=s.image_path,
            video_path=s.video_path,
            bullets=[BulletPoint(text=b.text, emphasis=b.emphasis) for b in s.bullets],
        )
        for s in source.scenes
    ]


async def _stage_script(job: Job, job_dir: Path, theme: Theme | None = None) -> Timeline:
    if job.reuse_from:
        # A/B rendering: same words, same pictures, different voice. Without this, two
        # jobs on one topic produce two different scripts and nothing is comparable.
        source_job = await asyncio.to_thread(_load_job, job.reuse_from)
        source = timeline_from_job(source_job)
        if source is None or not source.scenes:
            raise RuntimeError(
                f"job {job.reuse_from} has no usable timeline to reuse "
                f"(status={source_job.status})"
            )
        logger.info(
            "reusing the script and imagery from job %s (%d scenes)",
            job.reuse_from,
            len(source.scenes),
        )
        return Timeline(
            job_id=job.id,
            topic=source.topic,
            title=source.title,
            scenes=_reused_scenes(source),
            voice=job.voice,
            language=source.language,
            profile=_render_profile(),
            theme=theme if theme is not None else source.theme,
            logo_path=resolve_job_logo(job),
        )

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
        profile=_render_profile(),
    )
    if theme is not None:
        # Stamped on the Timeline so the palette is persisted with the job's debug trail
        # and visible to the planner. The render backend still takes it by constructor.
        timeline.theme = theme
    timeline.logo_path = resolve_job_logo(job)
    return timeline


def resolve_job_logo(job: Job) -> str | None:
    """What `Timeline.logo_path` should be for this job. Same three states as `Job.logo_id`.

    Resolved to a *path* here, at script time, for the same reason the theme is stamped:
    the Timeline is the self-describing artifact, so a re-render from a persisted Timeline
    reproduces the branding without re-reading config or the upload store.

    A logo id that no longer resolves (the file was deleted out from under the job) falls
    back to the default mark with a warning rather than failing the render — branding is
    the last 1% of the frame, and losing it must not cost the other 99%.
    """
    choice = (job.logo_id or "").strip()
    if not choice:
        return None  # the configured default; None is what "say nothing" has always meant
    if choice.lower() == NO_LOGO_ID:
        # `resolve_logo_source` reads this spelling as an explicit opt-out, and persisting
        # the sentinel rather than a path keeps the Timeline honest about the *choice*.
        return NO_LOGO_ID

    # Imported here, not at module scope: the worker must not depend on the HTTP layer
    # booting, and this is the only place it needs the upload store.
    from app.api.logos import logo_render_path

    path = logo_render_path(choice)
    if path is None:
        logger.warning(
            "job %s names logo %r, which is no longer in the store; "
            "rendering with the default mark",
            job.id,
            choice,
        )
        return None
    return str(path)


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

    # Request the HERO PANEL's aspect, not the frame's. The panel is 4:5 portrait
    # (720x900); a 16:9 source gets centre-cropped to it, which throws away ~55% of the
    # width and lands wherever the middle happens to be. Measured consequence: a
    # perfectly good landscape photo of a phone on a table rendered as three abstract
    # bands, because the crop landed on the phone's edge. Any wide, centred subject is
    # gutted the same way.
    #
    # Images are generated before planning, so the exact per-scene layout is not known
    # yet — but the planner holds ONE hero layout for the whole body, so 4:5 is right for
    # every scene that has a panel.
    height = profile.height * profile.upscale_factor
    width = int(round(height * _HERO_ASPECT))

    async def one(scene: Scene) -> None:
        out = job_dir / f"scene_{scene.id:02d}.png"
        async with limit:
            path = await asyncio.to_thread(
                provider.generate, scene.image_prompt, out, width, height
            )
        scene.image_path = str(path)

    # A reused scene already has its picture. Regenerating would cost credits AND produce
    # a *different* image — image generation is not deterministic — which defeats the
    # whole point of reuse: comparing two renders that differ in exactly one thing.
    pending = [s for s in timeline.scenes if not _usable_asset(s.image_path)]
    if len(pending) < len(timeline.scenes):
        logger.info(
            "reusing %d existing image(s); generating %d",
            len(timeline.scenes) - len(pending),
            len(pending),
        )
    if pending:
        await _gather_scenes("image generation", [one(s) for s in pending])


def _usable_asset(path: str | None) -> bool:
    """True when a carried-over asset path still points at a real, non-empty file."""
    if not path:
        return False
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError:
        return False


def _clip_scene(timeline: Timeline) -> Scene | None:
    """The one scene that gets generated FOOTAGE instead of a still, or None.

    Exactly one clip per video, on the CLOSING scene (docs/DIRECTION.md §7):

    * content and summary scenes carry four bullets to read — moving footage beside text
      is split attention, and a still with a slow zoom is the right answer there, not a
      compromise;
    * the title card is pure type on a solid ground, which is what makes it read as a
      title, and a soft 720p upscale on the first thing a viewer sees is the worst
      possible placement;
    * the closing has two bullets and little to read, so the one thing that changes at
      the end is that the picture comes alive.
    """
    for scene in reversed(timeline.scenes):
        if scene.role is SceneRole.CLOSING and scene.clip_prompt:
            return scene
    return None


async def _stage_clips(timeline: Timeline, job_dir: Path) -> None:
    """Generate the closing scene's footage. Best-effort: a still is a fine fallback.

    Veo returns a FIXED ~8s clip, so the renderer covers any shortfall itself (measured:
    a crossfaded loop leaves 1.5% duplicate frames, against 60.7% for freezing the final
    frame). Gated on ``video_enable_veo`` — this is the most expensive call in the
    pipeline, so it stays opt-in.
    """
    provider = factory.video_clip_provider()
    if provider is None:
        logger.debug("clip generation disabled; every scene uses its still")
        return

    scene = _clip_scene(timeline)
    if scene is None:
        logger.info("no closing scene carries a clip_prompt; skipping footage")
        return
    if _usable_asset(scene.video_path):
        # Reused from a previous job — and Veo is the most expensive call here, so
        # regenerating would both cost the most and break the comparison.
        logger.info("scene %s reuses existing footage: %s", scene.id, scene.video_path)
        return

    out = job_dir / f"scene_{scene.id:02d}.clip.mp4"
    try:
        path = await asyncio.to_thread(
            provider.generate, scene.clip_prompt or "", scene.duration or 8.0, out
        )
    except Exception:
        logger.warning(
            "clip generation failed for scene %s; falling back to its still",
            scene.id,
            exc_info=True,
        )
        return
    scene.video_path = str(path)
    logger.info("scene %s uses generated footage: %s", scene.id, path)


#: Delivery pace per role, as a multiplier on the engine's configured speed.
#:
#: An engine without SSML cannot stress a phrase, but it can change PACE — so the
#: modulation we do have is spent where it reads: a title card delivered slower lands as
#: deliberate, and a closing slightly slower lands as an instruction rather than a
#: throwaway. Content and summary stay at the base rate; varying them would fight the
#: measured 135 wpm pacing target.
#: These deltas must be LARGE or they are placebo. Measured: Aura's run-to-run variance on
#: identical input is ~0.68s over a 7s take, so a 6% speed change is unmeasurable — the
#: first version of this table used 0.94/0.96 and the "slower" setting came out *shorter*
#: on average across three repeats. Only a change big enough to clear that noise floor is
#: real, which is why the title is ~11% slower than content rather than 6%.
_ROLE_SPEED: dict[SceneRole, float] = {
    SceneRole.TITLE: 0.89,
    SceneRole.CONTENT: 1.0,
    SceneRole.SUMMARY: 1.0,
    SceneRole.CLOSING: 0.93,
}


def _voice_modulation(synth: Any, scene: Scene, send_ssml: bool) -> dict[str, Any]:
    """Per-scene delivery kwargs the synthesizer actually accepts.

    Two levers, applied only to engines that expose them (inspected, not assumed — a
    provider that ignores modulation must keep working unchanged):

    * ``speed`` — scaled per :data:`_ROLE_SPEED`.
    * ``emphasize`` — the phrase of the emphasised bullet, spoken slower so the spoken
      stress lands on the words appearing on screen. Aura has no loudness control at all
      (measured across 7 techniques), so a ~1.55x duration stretch is the closest
      substitute this engine has.

    Skipped entirely when the engine takes SSML: Polly already carries its own prosody in
    the markup, and stacking a second mechanism on top would fight it.
    """
    if send_ssml:
        return {}
    try:
        accepts = inspect.signature(synth.synthesize).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return {}

    settings = get_settings()
    out: dict[str, Any] = {}
    if "speed" in accepts:
        base = getattr(settings, "video_deepgram_speed", 0.9)
        out["speed"] = round(base * _ROLE_SPEED.get(scene.role, 1.0), 3)
    if "emphasize" in accepts:
        # Only a phrase that appears verbatim can be split out, and only one: the
        # splice costs an extra request and a join per phrase.
        for bullet in scene.bullets:
            if bullet.emphasis and bullet.text.lower() in scene.narration.lower():
                out["emphasize"] = bullet.text
                break
    return out


async def _stage_narrate(timeline: Timeline, job_dir: Path, engine: str | None = None) -> None:
    """All narration at once — each take is independent of the others.

    SSML goes ONLY to an engine that declares it parses SSML. Deepgram Aura does not: it
    vocalises the tags, so `<break time="800ms"/>` is spoken as "break time equals eight
    hundred milliseconds" (measured). `getattr` with a False default rather than a direct
    attribute read — absent must mean plain text, never "assume it copes".
    """
    synth = factory.speech_synthesizer(engine=engine)
    send_ssml = getattr(synth, "supports_ssml", False)
    limit = asyncio.Semaphore(_api_concurrency())

    async def one(scene: Scene) -> None:
        out = job_dir / f"scene_{scene.id:02d}.mp3"
        spoken = scene.ssml if (send_ssml and scene.ssml) else scene.narration
        extra = _voice_modulation(synth, scene, send_ssml)
        async with limit:
            path = await asyncio.to_thread(
                functools.partial(synth.synthesize, spoken, timeline.voice, out, **extra)
            )
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


def _video_backend(timeline: Timeline, theme: Theme | None) -> Any:
    """The render backend, on this job's palette *and* this job's brand mark.

    ``FFmpegBackend`` takes the mark as a constructor argument (``logo_path=``) and resolves
    it exactly once via ``resolve_logo_source``, whose three states line up with
    ``Timeline.logo_path``: ``AUTO_LOGO`` for the configured default, ``"none"`` for no
    branding, a path for an upload. ``factory.video_backend`` does not forward a
    ``logo_path`` yet, so this passes it when the factory grows one and otherwise assigns
    the resolved source before any render call — which is equivalent, because the
    constructor does nothing else with it.
    """
    from app.render.ffmpeg_backend import AUTO_LOGO, resolve_logo_source

    configured = timeline.logo_path if timeline.logo_path is not None else AUTO_LOGO
    try:
        params = inspect.signature(factory.video_backend).parameters
    except (TypeError, ValueError):  # pragma: no cover - factory is a plain function
        params = {}
    if "logo_path" in params:
        return factory.video_backend(theme=theme, logo_path=configured)

    backend = factory.video_backend(theme=theme)
    if hasattr(backend, "logo_source"):
        backend.logo_source = resolve_logo_source(configured)
    elif timeline.logo_path is not None:
        logger.warning(
            "render backend %s takes no brand logo; this job's choice (%s) is being ignored",
            type(backend).__name__,
            timeline.logo_path,
        )
    return backend


async def _stage_render(timeline: Timeline, job_dir: Path, theme: Theme | None = None) -> Timeline:
    """One clip per scene, from the planned timeline.

    Delegated to the backend's batch entry point so scenes render concurrently and
    frame counts are rounded cumulatively — looping ``render_scene`` here rounded
    each scene independently, leaking up to half a frame per scene.

    ``theme`` goes to the backend's constructor, which is where ``FFmpegBackend`` reads
    it from — ``Timeline.theme`` carries the same palette for the planner and the debug
    trail, but the backend does not consult it.
    """
    backend = _video_backend(timeline, theme)

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
    # The watermark is composited here, in `assemble` — once over the finished chain, never
    # per scene — so this is the construction that actually has to carry the job's mark.
    backend = _video_backend(timeline, theme)
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
            # Footage rides the visual branch: it depends on nothing from the soundtrack,
            # and a ~60s Veo call overlaps the audio branch for free.
            await _stage_clips(timeline, job_dir)

        async def audio_branch() -> None:
            await _stage_narrate(timeline, job_dir, engine=job.tts_engine)
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
