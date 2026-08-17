"""The scorecard data model.

Everything the evaluator produces is one :class:`VideoScore`, and it is deliberately
serialisable: `score.json` next to `video.mp4` is the artifact a later run diffs against
to prove a fix worked.

Two separations matter here:

*Measurements vs scores.* :class:`SceneMetrics` / :class:`VideoMetrics` hold raw physical
numbers (contrast ratios, LUFS, duplicate-frame ratios). :class:`SceneScore` /
:class:`VideoScore` hold the 0-10 judgements derived from them. Keeping the raw numbers
means a threshold can be re-tuned later without re-rendering or re-calling the model.

*Scores vs recommendations.* A score says how bad it is; a :class:`Recommendation` says
what to do. ``auto_fixable`` is the important field — it is the difference between a
report a human reads and a queue the pipeline can drain.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- enums


class Severity(StrEnum):
    """How much a finding should block shipping.

    Ordered worst-first so ``sorted(..., key=SEVERITY_ORDER.get)`` puts blockers on top.
    """

    BLOCKER = "blocker"
    """The video is wrong, not just imperfect. An off-topic image is a blocker."""

    MAJOR = "major"
    """A viewer will notice and it degrades the message."""

    MINOR = "minor"
    """Polish. Worth fixing before publishing, not worth a re-render on its own."""

    NIT = "nit"
    """Observation. Recorded so a trend is visible across jobs."""


SEVERITY_ORDER: dict[str, int] = {
    Severity.BLOCKER: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.NIT: 3,
}


class Dimension(StrEnum):
    """What is being judged. One dimension, one weight, one owner in the pipeline."""

    LEGIBILITY = "legibility"
    """Can you read the burned-in heading. Measured (WCAG) and judged (vision)."""

    RELEVANCE = "relevance"
    """Does the image belong to this narration and this topic. Vision only."""

    COMPOSITION = "composition"
    """Is there clean space for the text; is the subject placed well."""

    PROFESSIONALISM = "professionalism"
    """Does it read as corporate training rather than stock-photo filler."""

    MOTION = "motion"
    """Smoothness of the camera move — duplicate frames mean visible stepping."""

    PACING = "pacing"
    """Words per minute, and whether one scene is wildly longer than its siblings."""

    TIMING = "timing"
    """Bullet reveal sanity and narration continuity inside the scene."""

    AUDIO = "audio"
    """Loudness, true peak, narration-vs-music balance, dead air."""

    SCRIPT = "script"
    """Narrative flow, clarity, actionability. Vision/LLM pass over the text."""

    TECHNICAL = "technical"
    """Container conformance: duration drift vs the Timeline, resolution, fps."""


class Grade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


def grade_for(overall: float) -> Grade:
    """Standard 90/80/70/60 cutoffs. Documented here so nothing else invents its own."""
    if overall >= 90:
        return Grade.A
    if overall >= 80:
        return Grade.B
    if overall >= 70:
        return Grade.C
    if overall >= 60:
        return Grade.D
    return Grade.F


# ----------------------------------------------------------------- recommendations


class Recommendation(BaseModel):
    """One concrete change, with enough structure to be executed rather than read.

    ``action`` + ``params`` are the machine-readable half: when ``auto_fixable`` is true
    the pipeline can dispatch on ``action`` and needs nothing else. ``problem``/``fix``
    are the human half and always populated, because an auto-fix that fires wrongly still
    has to be explainable.
    """

    severity: Severity
    dimension: Dimension
    scene_id: int | None = None
    """``None`` means the finding is about the whole video."""

    problem: str
    fix: str
    auto_fixable: bool = False
    """True only when the fix is mechanical: a parameter change, a filter, or a
    regeneration with a prompt we already hold. Anything needing taste is false."""

    action: str | None = None
    """Stable identifier for the mechanical fix, e.g. ``raise_scrim_opacity``."""

    params: dict[str, float | int | str] = Field(default_factory=dict)
    """Arguments for ``action``. Concrete values, already computed."""

    evidence: str | None = None
    """The measurement that triggered this, verbatim. Keeps the report auditable."""


# --------------------------------------------------------------------- measurements


class ContrastMeasure(BaseModel):
    """WCAG contrast between the heading's white fill and what sits behind it."""

    ratio: float
    """The flagged number: contrast against the *brighter quartile* of the background.

    Not the mean, and not the median. Illegibility is caused by the bright patches the
    glyphs cross, and both mean and median are dragged down by the dark majority of a
    scrim band — a heading can measure 8:1 on average while a sunlit window renders a
    third of it unreadable. See :mod:`app.evaluate.metrics` for the calibration.
    """

    ratio_median: float
    """Contrast against the median background. The typical-case reading."""

    background_level: int = 0
    """0-255 luma of the brighter-quartile background, for debugging."""

    frames_sampled: int = 0
    region: str = ""
    """``WxH+X+Y`` of the measured glyph box, so a bad reading can be eyeballed."""

    alt_position: str | None = None
    """The opposite ``TextPosition`` band, measured for comparison."""

    alt_ratio: float | None = None
    """Contrast the heading *would* have there, with the same scrim applied.

    This is what turns "your text is hard to read" into a one-parameter fix: if the other
    third of the frame is much darker, moving the heading costs nothing and needs no
    regeneration.
    """


class LoudnessMeasure(BaseModel):
    integrated_lufs: float
    true_peak_dbfs: float
    loudness_range_lu: float = 0.0


class BalanceMeasure(BaseModel):
    """Narration level vs whatever is playing underneath it."""

    speech_dbfs: float
    bed_dbfs: float
    """Level measured inside narration gaps — the music bed on its own."""

    separation_db: float
    windows_sampled: int = 0
    measured: bool = True
    """False when the narration had no gap long enough to measure the bed in."""


class SceneMetrics(BaseModel):
    """Raw deterministic measurements for one scene. No opinions, no thresholds."""

    scene_id: int
    heading: str = ""
    duration: float = 0.0
    contrast: ContrastMeasure | None = None
    duplicate_frame_ratio: float | None = None
    """Fraction of sampled frames that are near-identical to their predecessor."""

    words_per_minute: float | None = None
    narration_gaps: list[tuple[float, float]] = Field(default_factory=list)
    """``(start, end)`` holes in the word timings, scene-relative."""

    silence_windows: list[tuple[float, float]] = Field(default_factory=list)
    """Detected silence in the narration asset itself, scene-relative."""

    bullet_issues: list[str] = Field(default_factory=list)
    bullet_count: int = 0
    duration_deviation: float | None = None
    """Signed fraction by which this scene's duration differs from the sibling median."""


class VideoMetrics(BaseModel):
    """Raw deterministic measurements for the whole video."""

    duration: float = 0.0
    expected_duration: float = 0.0
    duration_drift_frames: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str | None = None
    audio_codec: str | None = None
    loudness: LoudnessMeasure | None = None
    balance: BalanceMeasure | None = None
    duplicate_frame_ratio: float | None = None
    profile_mismatch: list[str] = Field(default_factory=list)
    """Ways the file disagrees with the RenderProfile it claims to have been made with."""


# --------------------------------------------------------------------------- scores


class SceneScore(BaseModel):
    """Per-dimension 0-10 plus the weighted 0-100 roll-up for one scene."""

    scene_id: int
    heading: str = ""
    duration: float = 0.0

    legibility: float | None = None
    relevance: float | None = None
    composition: float | None = None
    professionalism: float | None = None
    motion: float | None = None
    pacing: float | None = None
    timing: float | None = None

    overall: float = 0.0
    grade: Grade = Grade.F
    issues: list[str] = Field(default_factory=list)
    """Short strings from the vision pass, kept verbatim."""

    suggested_image_prompt: str | None = None
    metrics: SceneMetrics | None = None

    def dimension_scores(self) -> dict[Dimension, float]:
        """Only the dimensions that were actually assessed. ``None`` is not zero."""
        pairs = {
            Dimension.LEGIBILITY: self.legibility,
            Dimension.RELEVANCE: self.relevance,
            Dimension.COMPOSITION: self.composition,
            Dimension.PROFESSIONALISM: self.professionalism,
            Dimension.MOTION: self.motion,
            Dimension.PACING: self.pacing,
            Dimension.TIMING: self.timing,
        }
        return {k: v for k, v in pairs.items() if v is not None}


class VideoScore(BaseModel):
    """The scorecard. Serialised to ``out/<job_id>/score.json``."""

    job_id: str
    topic: str = ""
    title: str = ""
    video_path: str = ""
    evaluated_at: str = ""
    evaluator_version: str = "1"
    """Bump when weights or thresholds change, so old score.json files are comparable
    only against their own version."""

    vision_used: bool = False
    vision_model: str | None = None

    overall: float = 0.0
    grade: Grade = Grade.F
    grade_capped: bool = False
    """True when a blocker held the grade below what the weighted average alone gave.

    A strong average must not be allowed to hide one broken scene: four good slides and
    one off-topic slide is a video you cannot publish, not a B.
    """

    scene_average: float = 0.0
    audio_score: float | None = None
    script_score: float | None = None
    technical_score: float | None = None

    narrative_flow: float | None = None
    clarity: float | None = None
    actionability: float | None = None
    bullets_echo_narration: bool | None = None

    scenes: list[SceneScore] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    metrics: VideoMetrics | None = None
    notes: list[str] = Field(default_factory=list)
    """What could not be assessed and why. An evaluator that hides its blind spots is
    worse than one that reports them."""

    def by_severity(self) -> list[Recommendation]:
        return sorted(
            self.recommendations,
            key=lambda r: (SEVERITY_ORDER.get(r.severity, 9), r.scene_id or 0),
        )

    def auto_fixable(self) -> list[Recommendation]:
        return [r for r in self.by_severity() if r.auto_fixable]

    def worst_scene(self) -> SceneScore | None:
        return min(self.scenes, key=lambda s: s.overall, default=None)
