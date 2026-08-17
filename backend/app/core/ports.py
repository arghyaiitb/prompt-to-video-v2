"""Swappable seams. Structural typing — no inheritance, no registration ceremony.

Any object with the right shape satisfies these. Swapping Deepgram for ElevenLabs
means writing one new class and changing one config value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.models import RenderProfile, Script, Timeline, VisualPlan, Word


@runtime_checkable
class ScriptProvider(Protocol):
    """Verbatim passthrough or LLM generation — callers can't tell which."""

    def generate(self, topic: str, slide_count: int) -> Script: ...


@runtime_checkable
class ImageProvider(Protocol):
    def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
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
        image_path: Path,
        plan: VisualPlan,
        heading: str,
        duration: float,
        out_path: Path,
        profile: RenderProfile,
    ) -> Path: ...

    def render_all(self, timeline: Timeline, clip_dir: Path) -> Timeline:
        """Render every scene, returning a Timeline with ``clip_path`` populated.

        Batch rather than per-scene so each backend owns its own parallelism and
        cross-scene frame accounting. Callers should prefer this over looping
        ``render_scene`` themselves.
        """
        ...

    def assemble(self, timeline: Timeline, out_path: Path) -> Path: ...
