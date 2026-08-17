"""Canonical data model. Every module reads and writes these types.

The Timeline is the load-bearing artifact: script, audio, and visuals all agree on it.
Audio is the clock — scene boundaries are derived from real word timings, never guessed.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Motion(str, Enum):
    """Camera move applied to a still. Chosen by the planner, executed by the renderer."""

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    STATIC = "static"


class Transition(str, Enum):
    FADE = "fade"
    DISSOLVE = "dissolve"
    SLIDE_LEFT = "slideleft"
    WIPE_RIGHT = "wiperight"
    CUT = "cut"


class TextPosition(str, Enum):
    CENTER = "center"
    LOWER_THIRD = "lower_third"
    UPPER_THIRD = "upper_third"


class Word(BaseModel):
    """One aligned word. Shape mirrors Deepgram's verified response."""

    word: str
    start: float
    end: float
    confidence: float = 1.0
    punctuated_word: str | None = None

    @property
    def display(self) -> str:
        """Original punctuation is authoritative for on-screen text."""
        return self.punctuated_word or self.word


class SceneScript(BaseModel):
    """LLM output for one slide. Matches the Gemini responseSchema."""

    id: int
    narration: str
    heading: str = Field(description="Short on-screen title burned over the image")
    image_prompt: str
    motion: Motion = Motion.ZOOM_IN


class Script(BaseModel):
    topic: str
    title: str
    scenes: list[SceneScript]


class VisualPlan(BaseModel):
    """Creative decisions live here — deterministic, testable, renderer-agnostic.

    Both the ffmpeg backend and any future HTML backend consume this unchanged.
    """

    motion: Motion = Motion.ZOOM_IN
    zoom_from: float = 1.0
    zoom_to: float = 1.12
    easing: Literal["linear", "ease_in_out"] = "ease_in_out"
    transition_in: Transition = Transition.DISSOLVE
    transition_duration: float = 0.5
    text_position: TextPosition = TextPosition.LOWER_THIRD
    scrim_opacity: float = 0.45
    """Dark overlay behind text so it stays legible over any image."""


class Scene(BaseModel):
    id: int
    narration: str
    heading: str
    image_prompt: str

    image_path: str | None = None
    audio_path: str | None = None
    clip_path: str | None = None

    start: float = 0.0
    end: float = 0.0
    words: list[Word] = Field(default_factory=list)
    plan: VisualPlan | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class RenderProfile(BaseModel):
    """Draft vs final. Same plan, different cost."""

    name: Literal["draft", "final"] = "final"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_codec: str = "libx264"
    crf: int = 18
    upscale_factor: int = 4
    """zoompan truncates x/y to integers; pre-upscaling removes the visible stepping."""

    render_concurrency: int = 0
    """Scenes rendered at once. 0 = auto (derived from CPU count).

    Scene clips are independent by construction, so this is safe to raise; the
    ceiling is CPU, not correctness.
    """

    encoder_threads: int | None = None
    """Threads per ffmpeg process. None = let the encoder decide.

    Set automatically when rendering concurrently: N processes each grabbing every
    core oversubscribes the box and thrashes. Total threads stay near the core count.
    """

    def resolve_concurrency(self, scene_count: int) -> tuple[int, int]:
        """Return ``(workers, threads_per_worker)`` for a parallel render.

        libx264 scales sublinearly with threads, so several narrower processes beat
        one wide one for independent clips.
        """
        cpu = os.cpu_count() or 4
        workers = self.render_concurrency or min(4, max(1, cpu // 3))
        workers = max(1, min(workers, scene_count))
        threads = self.encoder_threads or (max(1, cpu // workers) if workers > 1 else 0)
        return workers, threads

    @classmethod
    def draft(cls) -> RenderProfile:
        return cls(
            name="draft",
            width=960,
            height=540,
            fps=24,
            video_codec="h264_videotoolbox",
            upscale_factor=2,
        )


class Timeline(BaseModel):
    job_id: str
    topic: str
    title: str
    scenes: list[Scene]
    voice: str
    music_path: str | None = None
    profile: RenderProfile = Field(default_factory=RenderProfile)

    @property
    def narration_duration(self) -> float:
        return max((s.end for s in self.scenes), default=0.0)

    def final_duration(self) -> float:
        """xfade consumes overlap: total shrinks by one transition per boundary.

        Getting this wrong desyncs narration by ~0.3s per scene.
        """
        overlap = sum(
            s.plan.transition_duration
            for s in self.scenes[1:]
            if s.plan and s.plan.transition_in != Transition.CUT
        )
        return self.narration_duration - overlap


class JobStatus(str, Enum):
    QUEUED = "queued"
    SCRIPTING = "scripting"
    IMAGING = "imaging"
    NARRATING = "narrating"
    ALIGNING = "aligning"
    SCORING = "scoring"
    RENDERING = "rendering"
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"
