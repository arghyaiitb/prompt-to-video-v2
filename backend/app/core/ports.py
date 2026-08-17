"""Swappable seams. Structural typing — no inheritance, no registration ceremony.

Any object with the right shape satisfies these. Swapping Deepgram for ElevenLabs
means writing one new class and changing one config value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.core.models import BulletPoint, RenderProfile, Script, Timeline, VisualPlan, Word

if TYPE_CHECKING:
    # Render-time artifact; imported lazily so core stays independent of app.render.
    from app.render.contracts import SceneText


@runtime_checkable
class ScriptProvider(Protocol):
    """Verbatim passthrough or LLM generation — callers can't tell which."""

    def generate(
        self,
        topic: str,
        slide_count: int,
        *,
        bullets_per_slide: int = 4,
        tone: str | None = None,
    ) -> Script:
        """Write `slide_count` scenes about `topic`.

        ``bullets_per_slide`` is the on-screen point budget per scene (3-5 renders
        legibly). ``tone`` names the audience register — ``new_hires``, ``all_staff``,
        ``technical``, ``executives`` — or None to leave it to the provider.

        Both are keyword-only with defaults so a provider that ignores them still
        satisfies this Protocol; the caller passes them by keyword.
        """
        ...


@runtime_checkable
class ImageProvider(Protocol):
    def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path: ...


@runtime_checkable
class VideoClipProvider(Protocol):
    """Generated moving footage, as an alternative visual source to a still.

    Measured against Veo 3.1 on this key, so implementations must cope with:
      * a FIXED ~8s clip regardless of the duration asked for
      * 1280x720 at 24 fps (fine for a hero region, short of full-bleed 1080p)
      * an audio track we do not want — narration is authoritative, strip it

    ``target_duration`` is therefore a request, not a guarantee. The renderer decides how
    to cover a longer scene (hold the last frame, loop, or slow down); the provider's job
    is to return the clip and report what it actually got.
    """

    def generate(self, prompt: str, target_duration: float, out_path: Path) -> Path: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Text (or SSML) to audio.

    ``supports_ssml`` is NOT advisory. Deepgram Aura does not parse SSML — it VOCALISES
    the tags: feeding it ``<break time="800ms"/>`` produces the spoken words "break time
    equals eight hundred milliseconds" (measured, not theorised). So a caller must pass
    SSML only to an engine that declares support, and plain text to everything else.
    """

    supports_ssml: bool

    def synthesize(self, text: str, voice: str, out_path: Path) -> Path: ...


@runtime_checkable
class Aligner(Protocol):
    """Word-level timings. Deliberately separate from SpeechSynthesizer.

    Some vendors return timings with synthesis, some need a second STT pass.
    Fusing the two would mean swapping TTS silently breaks captions.
    """

    def align(self, audio_path: Path, reference_text: str) -> list[Word]: ...


@runtime_checkable
class MusicProvider(Protocol):
    """Must satisfy `target_duration` even if the model emits fixed-length clips
    (Lyria returns ~30s) — implementations loop with crossfade to fill.
    """

    def generate(self, mood: str, target_duration: float, out_path: Path) -> Path: ...


@runtime_checkable
class VisualPlanner(Protocol):
    """Pure and deterministic. No network, no ffmpeg — trivially unit-testable."""

    def plan(self, timeline: Timeline) -> Timeline: ...


@runtime_checkable
class VideoBackend(Protocol):
    """ffmpeg today, HTML/GSAP tomorrow. Consumes VisualPlan, emits clips."""

    def render_scene(
        self,
        image_path: Path | str | None,
        plan: VisualPlan,
        heading: str,
        duration: float,
        out_path: Path,
        profile: RenderProfile,
        *,
        bullets: list[BulletPoint] | None = None,
        scene_text: SceneText | None = None,
    ) -> Path:
        """Render one slide.

        ``image_path`` is optional: ``SlideLayout.TITLE_CARD`` has no image region and
        renders from the theme colour alone.
        """
        ...

    def render_all(self, timeline: Timeline, clip_dir: Path) -> Timeline:
        """Render every scene, returning a Timeline with ``clip_path`` populated.

        Batch rather than per-scene so each backend owns its own parallelism and
        cross-scene frame accounting. Callers should prefer this over looping
        ``render_scene`` themselves.
        """
        ...

    def assemble(self, timeline: Timeline, out_path: Path) -> Path: ...
