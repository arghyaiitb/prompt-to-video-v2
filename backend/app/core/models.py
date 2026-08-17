"""Canonical data model. Every module reads and writes these types.

The Timeline is the load-bearing artifact: script, audio, and visuals all agree on it.
Audio is the clock — scene boundaries are derived from real word timings, never guessed.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Motion(StrEnum):
    """Camera move applied to a still. Chosen by the planner, executed by the renderer."""

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    STATIC = "static"


class Transition(StrEnum):
    FADE = "fade"
    DISSOLVE = "dissolve"
    SLIDE_LEFT = "slideleft"
    WIPE_RIGHT = "wiperight"
    CUT = "cut"


class TextPosition(StrEnum):
    CENTER = "center"
    LOWER_THIRD = "lower_third"
    UPPER_THIRD = "upper_third"
    LEFT_PANEL = "left_panel"
    """Heading top-left, bullets stacked beneath — the corporate-training layout."""


class TextAnimation(StrEnum):
    """How a text layer enters. Executed by the renderer, chosen by the planner."""

    NONE = "none"
    FADE_IN = "fade_in"
    SLIDE_UP = "slide_up"
    SLIDE_LEFT = "slide_left"
    POP = "pop"
    TYPEWRITER = "typewriter"


class Language(StrEnum):
    """Narration + on-screen language. Measured constraints per language, not assumptions.

    English and Spanish are straightforward. Hindi is not, and the differences are
    load-bearing rather than cosmetic:

    * **No Deepgram TTS voice exists for Hindi** (0 of 102). Its only path is AWS Polly
      via ``Aditi``/``Kajal``, which are ``en-IN`` voices that carry ``hi-IN`` in
      ``AdditionalLanguageCodes`` — so the request must set ``LanguageCode`` explicitly.
    * **Devanagari needs a shaping engine.** Rendering फ़िशिंग through plain freetype
      drops the nukta and anusvara and yields फिशिग. Text must go through Pango
      (HarfBuzz), and the font must be a Devanagari face — Arial has no such glyphs.
    * **Word-level alignment must be verified per language.** Bullet anchoring depends on
      the aligner returning word timings; without it, bullets fall back to proportional
      placement and stop matching the narration.
    """

    EN = "en"
    ES = "es"
    HI = "hi"

    @property
    def script(self) -> Literal["latin", "devanagari"]:
        return "devanagari" if self is Language.HI else "latin"

    @property
    def needs_shaping(self) -> bool:
        """True when glyph shaping is required and naive rendering corrupts the text."""
        return self.script != "latin"


class SceneRole(StrEnum):
    """What a scene is FOR. Drives duration, layout, type scale and bullet budget.

    A training video has a shape: it announces itself, teaches, recaps, then tells you
    what to do. Without roles every scene is the same 19-second slab of four bullets,
    which is what "all over the place" actually means — no structure, just a queue.
    """

    TITLE = "title"
    """Opener. SHORT (3-6s), one large title, no bullets. Announces the subject."""

    CONTENT = "content"
    """The teaching body. Full bullet budget, longest scenes."""

    SUMMARY = "summary"
    """Recap of the key points already made. No new information."""

    CLOSING = "closing"
    """What to do next. Short, few points, ends the video cleanly."""

    @property
    def target_duration(self) -> tuple[float, float]:
        """(min, max) seconds. Numbers from ``docs/DIRECTION.md``.

        The content ceiling is 19s, not 24s: every measured content scene already sat at
        17.9-20.1s and was still rejected as a stall, so 24s of a static fully-revealed
        slide IS the defect. The floors are the arithmetic minimum a role's bullets need
        at a 1.6s stagger plus the 2.6s motionless dwell text requires to be read.
        """
        return {
            SceneRole.TITLE: (4.0, 6.5),
            SceneRole.CONTENT: (11.0, 19.0),
            SceneRole.SUMMARY: (9.0, 14.0),
            SceneRole.CLOSING: (6.0, 9.0),
        }[self]

    @property
    def bullet_budget(self) -> int:
        """Bullets this role may show. The opener earns its impact by having none.

        Content is 4, not 5: at the 11s floor the usable reveal window is 6.27s and five
        bullets need 6.4s. Every content slide carrying the same four points is itself
        part of the uniformity.
        """
        return {
            SceneRole.TITLE: 0,
            SceneRole.CONTENT: 4,
            SceneRole.SUMMARY: 4,
            SceneRole.CLOSING: 2,
        }[self]

    @property
    def heading_scale(self) -> float:
        """Multiplier on the base heading size. Exactly TWO sizes exist in a video.

        Summary and closing sit at 1.0 deliberately: four heading sizes is the opposite
        of the uniformity being asked for, a 7.8px difference is undetectable in
        isolation, and the shifted bullet baseline it causes *is* detectable.

        Title is 1.35, not 1.8. Once the base heading clears the legibility floor at
        78px, 1.8x = 140px fits only ~20 chars per line — a 46-character title would
        need three lines. 1.35 x 78 = 105px fits 25 x 2 = 50 characters.
        """
        return {
            SceneRole.TITLE: 1.35,
            SceneRole.CONTENT: 1.0,
            SceneRole.SUMMARY: 1.0,
            SceneRole.CLOSING: 1.0,
        }[self]


class SlideLayout(StrEnum):
    """How the frame is divided. Corporate decks are mostly NOT full-bleed photo.

    The image is a supporting element inside a designed frame, not the frame itself.
    """

    TITLE_CARD = "title_card"
    """Solid background, large centred title. No image, or a small centred one."""

    HERO_RIGHT = "hero_right"
    """Text column left, image panel right. The workhorse training layout."""

    HERO_LEFT = "hero_left"
    IMAGE_BAND = "image_band"
    """Image as a horizontal band, text in the solid area above or below."""

    FULL_BLEED = "full_bleed"
    """Image fills the frame with a scrim. Use sparingly — opener/closer."""


class Theme(BaseModel):
    """Brand palette. Solid backgrounds mean colour choice carries the design.

    Not dark-only: ``bg`` and ``text`` are independent, so a light palette is just a
    light ``bg`` with dark ``text``. Anything that depends on polarity should ask
    :meth:`is_light` rather than assuming white-on-dark.

    Every preset must clear WCAG AA (4.5:1) for ``text`` on ``bg`` — see
    :meth:`contrast_report`. A palette that fails that is not a style choice, it's a bug.
    """

    name: str = "midnight"

    bg: str = "#0B1220"
    surface: str = "#131F35"
    """Panel/card fill, slightly lifted from bg."""

    text: str = "#F8FAFC"
    muted: str = "#94A3B8"
    accent: str = "#F5A524"
    """Graphic accents — heading rule, bullet markers. Not used for text when
    ``uniform_text`` is set."""

    @staticmethod
    def _luminance(hex_colour: str) -> float:
        """WCAG relative luminance."""
        raw = hex_colour.lstrip("#")
        channels = []
        for i in (0, 2, 4):
            c = int(raw[i : i + 2], 16) / 255
            channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = channels
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def contrast(cls, a: str, b: str) -> float:
        la, lb = cls._luminance(a), cls._luminance(b)
        hi, lo = max(la, lb), min(la, lb)
        return (hi + 0.05) / (lo + 0.05)

    @property
    def is_light(self) -> bool:
        """True for light palettes. Drives scrim colour and shadow direction."""
        return self._luminance(self.bg) > 0.5

    @property
    def scrim_colour(self) -> str:
        """Legibility wash over photography — opposite polarity to the text."""
        return "#FFFFFF" if self.is_light else "#000000"

    def contrast_report(self) -> dict[str, float]:
        """Ratios a caller can assert on. Keys are ``pair`` names."""
        return {
            "text_on_bg": self.contrast(self.text, self.bg),
            "text_on_surface": self.contrast(self.text, self.surface),
            "muted_on_bg": self.contrast(self.muted, self.bg),
            "accent_on_bg": self.contrast(self.accent, self.bg),
        }

    image_radius: int = 24
    """Corner rounding on hero images, in 1080p pixels. Scales with frame width."""

    marker: Literal["disc", "ring", "chevron", "dash", "none"] = "dash"
    """ONE bullet-marker shape for the entire video.

    Previously emphasis switched a marker from ring to disc, intending non-chromatic
    hierarchy. It reads as inconsistency, not hierarchy — a viewer sees two kinds of
    bullet and assumes the deck is sloppy. Emphasis is now weight only, and the marker
    never varies within a video.
    """

    uniform_text: bool = True
    """All text renders in ``text``, including emphasised bullets.

    Mixed text colours read as inconsistent branding. Emphasis is carried
    non-chromatically instead — heavier weight and a filled marker — so the hierarchy
    survives without a second text colour. ``accent`` still colours graphic elements
    (the heading rule, bullet markers), which are not text.
    """

    logo_height_fraction: float = 0.045
    """Watermark height as a fraction of frame height (~49px at 1080p)."""

    logo_margin_fraction: float = 0.028
    """Inset from the bottom-left corner, as a fraction of frame *width*."""

    logo_opacity: float = 0.85
    """Slightly under full so branding sits behind the content, not on top of it."""


class BulletPoint(BaseModel):
    """One on-screen point, revealed while the narration says it.

    ``appear_at`` is relative to the scene's own start, not the global timeline.
    """

    text: str
    appear_at: float = 0.0
    emphasis: bool = False
    """Renders in the accent colour — for the one point that matters most."""


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
    bullets: list[str] = Field(
        default_factory=list,
        description="3-5 short on-screen points, each echoing a phrase in the narration",
    )
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

    layout: SlideLayout = SlideLayout.HERO_RIGHT
    """Motion applies to the image within its region, not the whole frame."""

    motion: Motion = Motion.ZOOM_IN
    zoom_from: float = 1.0
    zoom_to: float = 1.12
    easing: Literal["linear", "ease_in_out"] = "ease_in_out"
    transition_in: Transition = Transition.DISSOLVE
    transition_duration: float = 0.5
    text_position: TextPosition = TextPosition.LOWER_THIRD
    scrim_opacity: float = 0.45
    """Dark overlay behind text so it stays legible over any image."""

    heading_animation: TextAnimation = TextAnimation.SLIDE_UP
    bullet_animation: TextAnimation = TextAnimation.SLIDE_LEFT
    anim_duration: float = 0.45
    bullet_min_gap: float = 0.6
    """Floor on the spacing between bullet reveals, even if narration is faster.

    Bullets landing closer than this read as a single flash rather than a sequence.
    """


class Scene(BaseModel):
    id: int
    role: SceneRole = SceneRole.CONTENT
    narration: str
    heading: str
    image_prompt: str

    clip_prompt: str | None = None
    """Motion prompt for a generated video clip, when the visual is Veo rather than a
    still. Distinct from ``image_prompt``: it must describe movement, not composition."""

    ssml: str | None = None
    """Marked-up narration for engines that parse SSML (Polly does; Deepgram does not).

    ``narration`` stays the plain-text source of truth. It is what gets displayed, and
    critically what the ALIGNER matches against — the aligner compares audio to reference
    text, so handing it SSML would corrupt every bullet anchor.
    """

    image_path: str | None = None
    video_path: str | None = None
    """A generated clip standing in for the still. Mutually exclusive with using
    ``image_path`` as the visual source; the renderer prefers this when set."""

    audio_path: str | None = None
    clip_path: str | None = None

    start: float = 0.0
    end: float = 0.0
    words: list[Word] = Field(default_factory=list)
    bullets: list[BulletPoint] = Field(default_factory=list)
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
    language: Language = Language.EN
    """Drives script generation, voice selection, font choice and text shaping."""

    music_path: str | None = None
    profile: RenderProfile = Field(default_factory=RenderProfile)
    theme: Theme = Field(default_factory=Theme)
    """Palette for this video. Persisted so a re-render reproduces the same branding."""

    logo_path: str | None = None
    """Brand mark for this video, overriding ``settings.video_logo_path``.

    Persisted on the Timeline rather than read from config at render time so a re-render
    reproduces the same branding even after the default or the upload store changes.
    None means fall back to the configured default; the renderer skips branding entirely
    if neither resolves.
    """

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


class JobStatus(StrEnum):
    """Pipeline stages. The values are contractual — the frontend stepper renders them."""

    QUEUED = "queued"
    SCRIPTING = "scripting"
    IMAGING = "imaging"
    NARRATING = "narrating"
    ALIGNING = "aligning"

    SCORING = "scoring"
    """Composing the MUSIC bed — "scoring" as in a film score.

    NOT quality scoring. ``app/evaluate/`` is a separate, deliberately offline tool run
    via ``scripts/evaluate_job.py``; nothing in the pipeline imports it. The name has
    misled readers into thinking each render is graded automatically. It is not.
    """

    RENDERING = "rendering"
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"
