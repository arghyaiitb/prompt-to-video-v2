"""Deterministic measurement of a rendered video. No model, no network, no opinion.

Every number in here comes out of ffmpeg/ffprobe and is reproducible on the same file,
which is the whole point: the vision pass in :mod:`app.evaluate.vision` can drift between
runs, so the metrics have to be the part you can regression-test.

Three calibrations in here were established by experiment against the real render at
``out/43859ea1-.../video.mp4`` rather than taken from documentation, and each is recorded
where it is used because the intuitive choice is wrong in every case:

*Contrast.* The naive reading — mean luminance of the text band — says the worst slide in
that video scores 6.5:1 and passes. It does not pass; a third of its heading sits on a
sunlit window. The band mean is dragged down by the dark majority of the scrim, and the
band's *median* and *percentiles* are contaminated by the glyphs themselves (~27% pure
white fill, a similar share of black outline). So the background is sampled from a skirt
of rows immediately above and below the glyphs, split into column strips about one cap
height wide, and the reported ratio is the *worst strip* — text has to be legible
everywhere it appears, not on average.

*Motion.* Default ``mpdecimate`` thresholds are useless here: measured on this repo's own
output they report 42/120 duplicates for a correctly pre-upscaled zoom and 36/120 for a
juddering one, i.e. the metric is inverted. ``hi=64*2:lo=64*1:frac=0.05`` reports 0/120
and 17/120 on the same pair. See :data:`MPDECIMATE`.

*Audio balance.* Integrated loudness over the mixed track cannot see whether the music is
drowning the voice. The level is measured separately inside narration gaps (music alone)
and inside speech runs, using the word timings as the window source.
"""

from __future__ import annotations

import logging
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.models import Scene, TextPosition, Timeline, Transition
from app.evaluate.models import (
    BalanceMeasure,
    ContrastMeasure,
    LoudnessMeasure,
    SceneMetrics,
    VideoMetrics,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- thresholds

TARGET_LUFS = -16.0
"""Web/streaming convention for spoken-word content. YouTube normalises to about -14,
podcast platforms to -16; -16 is the compromise this pipeline aims at. The reference
render sits at -20.3, which is audibly quiet against anything else in a browser tab."""

MAX_TRUE_PEAK_DBFS = -1.0
"""Headroom below full scale. Lossy re-encode downstream can push a -0.1 dBFS peak over
0 and clip, so -1 is the practical ceiling."""

MIN_CONTRAST_RATIO = 4.5
"""WCAG 2.1 AA for normal text. Headings here are large (>=24px bold), for which AA is
only 3:1 — 4.5 is deliberately the stricter target, because these frames are also viewed
downscaled in a feed, at which point the large-text allowance no longer applies."""

MIN_WPM = 110.0
MAX_WPM = 170.0
"""Comfortable narration band. Below 110 sounds sedated, above 170 loses non-native
listeners. Measured from real word timings, never estimated from text length."""

IDEAL_WPM = (135.0, 155.0)

MIN_BULLET_GAP = 0.6
"""Floor on the spacing between bullet reveals. Mirrors ``VisualPlan.bullet_min_gap``;
duplicated as a constant so the evaluator can still judge a Timeline with no plan."""

MAX_DURATION_DRIFT_FRAMES = 2.0
MIN_SPEECH_BED_SEPARATION_DB = 10.0
"""How far the narration should sit above the music. Below ~6 dB the bed competes with
consonants; the pipeline's own duck target is -18 dB, so 10 is a generous floor."""

MPDECIMATE = "mpdecimate=hi=128:lo=64:frac=0.05"
"""Near-identical-frame detector, calibrated on this repo's output (2026-08-17).

``hi``/``lo`` are 8x8-block SAD thresholds (ffmpeg's docs express the defaults as
``64*12`` and ``64*5``); ``frac`` is the share of blocks allowed to exceed ``lo``.

    source                                   default    this setting
    zoompan, no pre-upscale (juddering)      36/120     17/120
    zoompan, upscale_factor=4 (smooth)       42/120      0/120
    real clip_01..04, upscale_factor=4        —          6,9,7,12 /120

The defaults rank the good synthetic render *worse* than the juddering one, because a slow
smooth zoom moves every pixel by a sub-threshold amount on every frame. This setting fixes
the inversion — but the last row is the important caveat: the real 4x-upscaled clips, which
are genuinely smooth, still register 5-10%. That is this measurement's noise floor on
encoded content, so :data:`DUPLICATE_NOISE_FLOOR` is where scoring starts to care.

Tightening further does not help and was tried: at ``hi=64:lo=32:frac=0.02`` the juddering
render reports 0/120 while the smooth real clips report 1-4/120, i.e. the discrimination
disappears entirely. So this metric reliably catches *gross* stepping and should not be
trusted to resolve small differences.
"""

DUPLICATE_NOISE_FLOOR = 0.12
"""Duplicate ratio below which the reading is indistinguishable from encoder noise.

Set above the 0.10 measured on the reference render's own correctly upscaled clips, and
below the 0.14 measured on a deliberately non-upscaled zoom. The margin is thin — see
:data:`MPDECIMATE`.
"""

MIN_NARRATION_GAP = 1.0
"""Silence between two words that counts as a hole rather than punctuation.

0.6s was tried first and flagged ordinary sentence breaks: the reference render's
inter-sentence pauses measure 0.32-0.64s, which is prosody, not a defect.
"""

DUPLICATE_SAMPLE_SECONDS = 4.0
"""Window per scene. 120 frames at 30fps — enough for a stable ratio, short enough that
a five-scene video is measured in a couple of seconds."""

CONTRAST_FRAME_SAMPLES = 3
NEAR_WHITE = 240
"""Luma at or above this is treated as heading fill, not background."""

GLYPH_ROW_FRACTION = 0.03
"""A skirt row with more near-white pixels than this probably clipped a glyph; drop it."""

SKIRT_PAD = 8
"""Rows of clearance between the glyph band and the sampled background — enough to clear
the outline's antialiasing (``borderw`` is ~3px at 1080p, ImageMagick's stroke ~6px)."""

SKIRT_ROWS = 26
"""Depth of the sampled background band above and below the glyphs."""

SAFE_AREA = 0.90
AVG_GLYPH_RATIO = 0.52
"""Mean advance width / point size for a bold sans. Only ever used to bound a crop."""


class MeasurementError(RuntimeError):
    """A measurement could not be taken. Always names the file and the tool."""


# ------------------------------------------------------------------------ binaries


def ffmpeg_bin() -> str:
    """Resolve ffmpeg the same way the renderer does, without importing it.

    :mod:`app.render.ffmpeg` owns this logic, but the evaluator has to keep working when
    the render package is mid-edit, so the two-line version is duplicated deliberately.
    """
    import os

    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def ffprobe_bin() -> str:
    import os

    if env := os.environ.get("FFPROBE_BIN"):
        return env
    sibling = Path(ffmpeg_bin()).with_name("ffprobe")
    return str(sibling) if sibling.exists() else (shutil.which("ffprobe") or "ffprobe")


def _run(argv: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - argv list, no shell
        argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False
    )


def _stderr(argv: list[str], *, timeout: float = 600.0) -> str:
    proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-1200:]
        raise MeasurementError(f"{Path(argv[0]).name} exited {proc.returncode}: {tail}")
    return (proc.stderr or b"").decode("utf-8", "replace")


def probe(path: str | Path) -> dict:
    proc = _run(
        [ffprobe_bin(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    )
    if proc.returncode != 0:
        raise MeasurementError(f"ffprobe failed on {path}: {proc.stderr[-600:]!r}")
    import json

    return json.loads(proc.stdout or b"{}")


def _stream(data: dict, kind: str) -> dict:
    for s in data.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return {}


# ----------------------------------------------------------------- WCAG arithmetic


def relative_luminance(level: int | float) -> float:
    """WCAG 2.1 relative luminance of a neutral grey at 8-bit sRGB ``level``.

    Greys only, which is all we need: the coefficients sum to 1, so for R=G=B the
    weighted sum collapses to the single linearised channel.
    """
    c = max(0.0, min(1.0, level / 255.0))
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def contrast_ratio(level_a: int | float, level_b: int | float) -> float:
    """WCAG contrast ratio between two 8-bit greys. Always >= 1.0."""
    la, lb = relative_luminance(level_a), relative_luminance(level_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def contrast_against_white(level: int | float) -> float:
    """Contrast of white heading fill against a background at ``level``."""
    return contrast_ratio(255, level)


def level_for_contrast(ratio: float) -> float:
    """The 8-bit grey that gives exactly ``ratio`` against white. Inverse of the above.

    Needed to turn a measured ratio back into a pixel level so a scrim opacity can be
    solved for, rather than nudged by trial and error.
    """
    target = max(0.0, 1.05 / max(1.0, ratio) - 0.05)
    if target <= 0.00304:
        return target * 12.92 * 255.0
    return (1.055 * target ** (1 / 2.4) - 0.055) * 255.0


CONTRAST_TARGET_LEVEL = level_for_contrast(MIN_CONTRAST_RATIO)
"""~118/255. Any background at or below this clears 4.5:1 against white text."""


# ------------------------------------------------------------------ text geometry


@dataclass(frozen=True)
class TextBox:
    """Where the heading was burned in, in final-frame pixels."""

    x: int
    y: int
    width: int
    height: int
    scrim_top: int
    scrim_bottom: int
    lines: int

    def as_geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.x}+{self.y}"


def text_box(scene: Scene, timeline: Timeline) -> TextBox | None:
    """Resolve the heading's pixel box, preferring the renderer's own layout code.

    ``app.render.text_overlay.layout_heading`` is the authority — it is what actually
    placed the text — so it is used when importable. It belongs to another module that is
    under active edit, so an ImportError falls back to a local reconstruction of the same
    geometry rather than failing the whole evaluation.
    """
    plan = scene.plan
    if plan is None:
        return None
    profile = timeline.profile
    try:
        from app.render.text_overlay import layout_heading

        layout = layout_heading(scene.heading, plan, profile)
        lines = layout.lines or [scene.heading]
        widest = max(len(line) for line in lines)
        text_w = min(
            int(profile.width * SAFE_AREA),
            max(40, int(widest * layout.font_size * AVG_GLYPH_RATIO)),
        )
        return TextBox(
            x=(profile.width - text_w) // 2,
            y=layout.block_top,
            width=text_w,
            height=layout.block_height,
            scrim_top=layout.scrim_top,
            scrim_bottom=layout.scrim_top + layout.scrim_height,
            lines=len(lines),
        )
    except Exception:  # noqa: BLE001 - never let a layout import sink the evaluation
        logger.debug("layout_heading unavailable; using local geometry", exc_info=True)

    font_size = max(14, round(profile.height * 0.058))
    line_height = round(font_size * 1.22)
    text = " ".join(scene.heading.split())
    usable = int(profile.width * SAFE_AREA)
    per_line = max(8, int(usable / (font_size * AVG_GLYPH_RATIO)))
    n_lines = max(1, -(-len(text) // per_line))
    block_h = line_height * n_lines
    margin = round(profile.height * (1.0 - SAFE_AREA))
    text_w = min(usable, max(40, int(min(len(text), per_line) * font_size * AVG_GLYPH_RATIO)))

    if plan.text_position is TextPosition.UPPER_THIRD:
        block_top, scrim_top = margin, 0
        scrim_bottom = block_top + block_h + round(font_size * 1.2)
    elif plan.text_position is TextPosition.CENTER:
        block_top = max(0, (profile.height - block_h) // 2)
        scrim_top, scrim_bottom = 0, profile.height
    else:
        block_top = max(0, profile.height - margin - block_h)
        scrim_top, scrim_bottom = max(0, block_top - round(font_size * 1.2)), profile.height
    return TextBox(
        x=(profile.width - text_w) // 2,
        y=int(block_top),
        width=int(text_w),
        height=int(block_h),
        scrim_top=int(scrim_top),
        scrim_bottom=int(scrim_bottom),
        lines=n_lines,
    )


# ------------------------------------------------------------------ frame sampling


@dataclass(frozen=True)
class FrameSource:
    """Which file to read a scene's frames out of, and at what timestamp."""

    path: Path
    offset: float
    """Timestamp in ``path`` of the scene's first frame."""

    duration: float
    local: bool
    """True when ``path`` is the scene's own clip (clean), False for the final video
    (whose scene boundaries are shifted by every preceding xfade)."""


def timeline_offsets(timeline: Timeline) -> list[float]:
    """Cumulative xfade overlap consumed before each scene, in seconds.

    ``final_duration`` shrinks by one transition per boundary, so a scene's timestamp in
    the assembled file is its narration ``start`` minus this. Getting it wrong samples
    frames from the neighbouring slide, which quietly poisons every visual metric.
    """
    out: list[float] = []
    total = 0.0
    for index, scene in enumerate(timeline.scenes):
        if index and scene.plan and scene.plan.transition_in != Transition.CUT:
            total += scene.plan.transition_duration
        out.append(total)
    return out


def frame_source(scene: Scene, timeline: Timeline, video_path: Path) -> FrameSource:
    """Prefer the per-scene clip: no transition contamination, no offset arithmetic."""
    if scene.clip_path and Path(scene.clip_path).exists():
        return FrameSource(Path(scene.clip_path), 0.0, scene.duration, True)
    offsets = timeline_offsets(timeline)
    index = next((i for i, s in enumerate(timeline.scenes) if s.id == scene.id), 0)
    return FrameSource(
        video_path, max(0.0, scene.start - offsets[index]), scene.duration, False
    )


def sample_timestamps(source: FrameSource, count: int) -> list[float]:
    """Evenly spaced instants, kept clear of the transition at either end."""
    duration = max(0.2, source.duration)
    inset = min(1.0, duration * 0.2)
    usable = max(0.05, duration - 2 * inset)
    if count <= 1:
        return [source.offset + inset + usable / 2]
    step = usable / (count - 1)
    return [source.offset + inset + i * step for i in range(count)]


def gray_crop(
    path: Path, at: float, x: int, y: int, width: int, height: int
) -> tuple[bytes, int, int]:
    """One frame's luma plane for a crop, as raw bytes plus the realised size.

    ffmpeg silently clamps a crop that runs off the edge, so the caller gets the actual
    dimensions back rather than assuming its own.
    """
    width, height = max(2, width), max(2, height)
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{max(0.0, at):.3f}", "-i", str(path), "-frames:v", "1",
        "-vf", f"crop={width}:{height}:{max(0, x)}:{max(0, y)},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    proc = _run(argv, timeout=120)
    buf = proc.stdout or b""
    if proc.returncode != 0 or not buf:
        raise MeasurementError(f"could not read frame at {at:.2f}s from {path.name}")
    if len(buf) % width:
        raise MeasurementError(f"unexpected {len(buf)} bytes for width {width}")
    return buf, width, len(buf) // width


# ------------------------------------------------------------------- text contrast


def _strip_contrasts(rows: list[bytes], row_width: int, strip_width: int) -> list[float]:
    """Contrast of white against the mean background of each column strip.

    Strips are about one cap height wide because that is the scale at which a bright
    patch actually costs you a word — averaging across the whole heading hides it, and
    going per-pixel turns antialiasing into noise.
    """
    if not rows or row_width <= 0:
        return []
    strip_width = max(8, min(strip_width, row_width))
    count = max(1, row_width // strip_width)
    out: list[float] = []
    for index in range(count):
        lo = index * strip_width
        hi = row_width if index == count - 1 else lo + strip_width
        total = pixels = 0
        for row in rows:
            chunk = row[lo:hi]
            total += sum(chunk)
            pixels += len(chunk)
        if pixels:
            out.append(contrast_against_white(total / pixels))
    return out


def _skirt_rows(
    buf: bytes, width: int, height: int, glyph_top: int, glyph_bottom: int
) -> list[bytes]:
    """Background rows bracketing the glyph band, with any glyph-contaminated row dropped.

    Rows are taken from *both* sides so a two-line heading still contributes the interline
    background, and each candidate is rejected if it carries more than
    :data:`GLYPH_ROW_FRACTION` near-white pixels — cheap insurance against the geometry
    being a few pixels out.
    """
    rows = [buf[i * width : (i + 1) * width] for i in range(height)]
    bands = [
        range(max(0, glyph_top - SKIRT_PAD - SKIRT_ROWS), max(0, glyph_top - SKIRT_PAD)),
        range(
            min(height, glyph_bottom + SKIRT_PAD),
            min(height, glyph_bottom + SKIRT_PAD + SKIRT_ROWS),
        ),
    ]
    keep: list[bytes] = []
    for band in bands:
        for i in band:
            row = rows[i]
            if not row:
                continue
            near_white = sum(row.count(g) for g in range(NEAR_WHITE, 256))
            if near_white / len(row) <= GLYPH_ROW_FRACTION:
                keep.append(row)
    return keep


def measure_text_contrast(
    scene: Scene,
    timeline: Timeline,
    video_path: Path,
    *,
    samples: int = CONTRAST_FRAME_SAMPLES,
) -> ContrastMeasure | None:
    """WCAG contrast for one scene's burned-in heading.

    Returns ``None`` when the scene has no plan (so no known text geometry) or no frames
    could be read — an absent measurement, never a passing one.
    """
    box = text_box(scene, timeline)
    if box is None:
        return None
    source = frame_source(scene, timeline, video_path)
    if not source.path.exists():
        return None

    font_size = max(14, round(timeline.profile.height * 0.058))
    # Extend the crop past the glyphs by the skirt depth, but never outside the scrim:
    # unscrimmed pixels are brighter than what the text actually sits on.
    top = max(box.scrim_top, box.y - SKIRT_PAD - SKIRT_ROWS)
    bottom = min(box.scrim_bottom, box.y + box.height + SKIRT_PAD + SKIRT_ROWS)
    if bottom - top < 8:
        return None

    worst: list[float] = []
    typical: list[float] = []
    levels: list[float] = []
    read = 0
    for at in sample_timestamps(source, samples):
        try:
            buf, width, height = gray_crop(source.path, at, box.x, top, box.width, bottom - top)
        except MeasurementError:
            logger.debug("contrast sample failed at %.2fs", at, exc_info=True)
            continue
        read += 1
        rows = _skirt_rows(buf, width, height, box.y - top, box.y - top + box.height)
        if not rows:
            continue
        strips = _strip_contrasts(rows, width, font_size)
        if not strips:
            continue
        worst.append(min(strips))
        typical.append(statistics.median(strips))
        levels.append(statistics.median([sum(r) / len(r) for r in rows]))

    if not worst:
        return None
    alt_position, alt_ratio = _alternative_position_contrast(scene, timeline, source, box)
    return ContrastMeasure(
        ratio=round(statistics.median(worst), 2),
        ratio_median=round(statistics.median(typical), 2),
        background_level=round(statistics.median(levels)),
        frames_sampled=read,
        region=box.as_geometry(),
        alt_position=alt_position,
        alt_ratio=alt_ratio,
    )


_OPPOSITE = {
    TextPosition.LOWER_THIRD: TextPosition.UPPER_THIRD,
    TextPosition.UPPER_THIRD: TextPosition.LOWER_THIRD,
}


def _alternative_position_contrast(
    scene: Scene, timeline: Timeline, source: FrameSource, box: TextBox
) -> tuple[str | None, float | None]:
    """What contrast the heading would get in the opposite third of the frame.

    That band carries no scrim in the rendered frame, so the scrim is applied
    arithmetically: ``drawbox``/``overlay`` alpha-blend on the encoded values, so a black
    scrim at opacity *a* scales a code level by ``1 - a``. Same worst-strip rule as the
    real measurement, so the two numbers are directly comparable.
    """
    plan = scene.plan
    if plan is None or plan.text_position not in _OPPOSITE:
        return None, None
    other = _OPPOSITE[plan.text_position]
    height = timeline.profile.height
    margin = round(height * (1.0 - SAFE_AREA))
    top = margin if other is TextPosition.UPPER_THIRD else max(0, height - margin - box.height)
    font_size = max(14, round(height * 0.058))
    try:
        buf, width, rows = gray_crop(
            source.path, sample_timestamps(source, 1)[0], box.x, top, box.width, box.height
        )
    except MeasurementError:
        return other.value, None
    scaled = 1.0 - plan.scrim_opacity
    strips = _strip_contrasts(
        [bytes(round(v * scaled) for v in buf[i * width : (i + 1) * width]) for i in range(rows)],
        width,
        font_size,
    )
    if not strips:
        return other.value, None
    return other.value, round(min(strips), 2)


# ---------------------------------------------------------------- motion smoothness


def _frame_count(path: Path, *, start: float, duration: float, vf: str | None = None) -> int:
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.1, duration):.3f}",
        "-i", str(path), "-an",
    ]  # fmt: skip
    if vf:
        argv += ["-vf", vf, "-fps_mode", "passthrough"]
    argv += ["-f", "null", "-", "-stats"]
    text = _stderr(argv)
    matches = re.findall(r"frame=\s*(\d+)", text)
    return int(matches[-1]) if matches else 0


def duplicate_frame_ratio(
    path: Path, *, start: float = 0.0, duration: float | None = None
) -> float | None:
    """Fraction of frames in the window that repeat their predecessor.

    0.0 is a genuinely smooth move. The reference render's ``upscale_factor=4`` scenes all
    measure 0.0; the same zoom without pre-upscaling measures ~0.14. See
    :data:`MPDECIMATE` for why the filter's defaults cannot be used.
    """
    window = (
        DUPLICATE_SAMPLE_SECONDS
        if duration is None
        else min(duration, DUPLICATE_SAMPLE_SECONDS)
    )
    try:
        total = _frame_count(path, start=start, duration=window)
        kept = _frame_count(path, start=start, duration=window, vf=MPDECIMATE)
    except MeasurementError:
        logger.debug("mpdecimate failed on %s", path, exc_info=True)
        return None
    if total <= 1:
        return None
    return round(max(0, total - kept) / total, 4)


# -------------------------------------------------------------------------- audio

_EBUR128_I = re.compile(r"^\s*I:\s*(-?[\d.]+)\s*LUFS", re.MULTILINE)
_EBUR128_LRA = re.compile(r"^\s*LRA:\s*(-?[\d.]+)\s*LU", re.MULTILINE)
_EBUR128_PEAK = re.compile(r"^\s*Peak:\s*(-?[\d.]+)\s*dBFS", re.MULTILINE)


def measure_loudness(path: Path) -> LoudnessMeasure | None:
    """Integrated loudness, loudness range and true peak from one ``ebur128`` pass.

    Only the trailing ``Summary:`` block is parsed — the running ``I:`` lines it prints
    per 100ms are partial integrations and matching those instead gives you the loudness
    of the first tenth of a second.
    """
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
    ]  # fmt: skip
    try:
        text = _stderr(argv)
    except MeasurementError:
        logger.debug("ebur128 failed on %s", path, exc_info=True)
        return None
    summary = text.split("Summary:")[-1] if "Summary:" in text else ""
    integrated = _EBUR128_I.search(summary)
    if not integrated:
        return None
    peak = _EBUR128_PEAK.search(summary)
    lra = _EBUR128_LRA.search(summary)
    return LoudnessMeasure(
        integrated_lufs=float(integrated.group(1)),
        true_peak_dbfs=float(peak.group(1)) if peak else 0.0,
        loudness_range_lu=float(lra.group(1)) if lra else 0.0,
    )


_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")


def _window_level_dbfs(path: Path, windows: list[tuple[float, float]]) -> float | None:
    """Mean level over an arbitrary set of time windows, in one ffmpeg call.

    ``aselect`` with a sum of ``between()`` terms keeps this to a single decode instead of
    one process per window, which matters when there are a dozen narration gaps.
    """
    usable = [(a, b) for a, b in windows if b - a > 0.05]
    if not usable:
        return None
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in usable)
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", f"aselect='{expr}',volumedetect", "-vn", "-f", "null", "-",
    ]  # fmt: skip
    try:
        text = _stderr(argv)
    except MeasurementError:
        return None
    found = _MEAN_VOLUME.search(text)
    return float(found.group(1)) if found else None


def speech_windows(timeline: Timeline, *, min_run: float = 0.6) -> list[tuple[float, float]]:
    """Continuous runs of narration, in final-video time."""
    offsets = timeline_offsets(timeline)
    runs: list[tuple[float, float]] = []
    for index, scene in enumerate(timeline.scenes):
        shift = offsets[index]
        start = end = None
        for word in scene.words:
            if start is None:
                start, end = word.start, word.end
            elif word.start - (end or 0.0) <= 0.25:
                end = max(end or 0.0, word.end)
            else:
                if (end or 0.0) - start >= min_run:
                    runs.append((start - shift, (end or 0.0) - shift))
                start, end = word.start, word.end
        if start is not None and (end or 0.0) - start >= min_run:
            runs.append((start - shift, (end or 0.0) - shift))
    return [(max(0.0, a), b) for a, b in runs if b > a]


def narration_gap_windows(
    timeline: Timeline, *, min_gap: float = 0.35
) -> list[tuple[float, float]]:
    """Holes between words, in final-video time, trimmed inward.

    The trim matters: a gap's edges contain the tail of one word and the attack of the
    next, so sampling them measures speech, not the bed underneath it.
    """
    offsets = timeline_offsets(timeline)
    out: list[tuple[float, float]] = []
    for index, scene in enumerate(timeline.scenes):
        shift = offsets[index]
        cursor = scene.start
        for word in scene.words:
            if word.start - cursor >= min_gap:
                out.append((cursor + 0.12 - shift, word.start - 0.08 - shift))
            cursor = max(cursor, word.end)
    return [(max(0.0, a), b) for a, b in out if b - a > 0.1]


def measure_balance(video_path: Path, timeline: Timeline) -> BalanceMeasure | None:
    """How far the narration sits above whatever is playing underneath it."""
    speech = speech_windows(timeline)
    gaps = narration_gap_windows(timeline)
    if not speech:
        return None
    speech_level = _window_level_dbfs(video_path, speech)
    if speech_level is None:
        return None
    if not gaps:
        return BalanceMeasure(
            speech_dbfs=round(speech_level, 2),
            bed_dbfs=0.0,
            separation_db=0.0,
            windows_sampled=0,
            measured=False,
        )
    bed_level = _window_level_dbfs(video_path, gaps)
    if bed_level is None:
        return BalanceMeasure(
            speech_dbfs=round(speech_level, 2),
            bed_dbfs=0.0,
            separation_db=0.0,
            windows_sampled=0,
            measured=False,
        )
    return BalanceMeasure(
        speech_dbfs=round(speech_level, 2),
        bed_dbfs=round(bed_level, 2),
        separation_db=round(speech_level - bed_level, 2),
        windows_sampled=len(gaps),
    )


_SILENCE = re.compile(r"silence_start:\s*(-?[\d.]+)|silence_end:\s*(-?[\d.]+)")


def detect_silence(
    path: Path, *, noise_db: float = -40.0, min_duration: float = 0.7
) -> list[tuple[float, float]]:
    """Silent windows in an audio asset, as ``(start, end)``.

    Run against the *narration* asset, not the mixed video: music fills every gap, so
    ``silencedetect`` on the final mix reports nothing however broken the voice track is.
    """
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}", "-vn", "-f", "null", "-",
    ]  # fmt: skip
    try:
        text = _stderr(argv)
    except MeasurementError:
        return []
    starts: list[float] = []
    out: list[tuple[float, float]] = []
    for match in _SILENCE.finditer(text):
        if match.group(1) is not None:
            starts.append(float(match.group(1)))
        elif starts:
            out.append((starts.pop(0), float(match.group(2))))
    return out


# ------------------------------------------------------------------ script timing


def words_per_minute(scene: Scene) -> float | None:
    """Pace from measured word timings.

    Divided by the *spoken* span, not the scene duration: a scene padded with a second of
    silence at the end is a pacing problem, but it is not a slow delivery, and conflating
    them sends the fix to the wrong place.
    """
    if not scene.words:
        return None
    span = max(w.end for w in scene.words) - min(w.start for w in scene.words)
    if span <= 0:
        return None
    return round(len(scene.words) / (span / 60.0), 1)


def narration_gaps(
    scene: Scene, *, min_gap: float = MIN_NARRATION_GAP
) -> list[tuple[float, float]]:
    """Holes in the narration inside a scene, scene-relative.

    A trailing pad before the transition is expected (the pipeline adds one deliberately),
    so only *interior* gaps are reported.
    """
    out: list[tuple[float, float]] = []
    cursor = scene.start
    for word in scene.words:
        if word.start - cursor >= min_gap:
            out.append((round(cursor - scene.start, 2), round(word.start - scene.start, 2)))
        cursor = max(cursor, word.end)
    return out


def bullet_issues(scene: Scene, *, min_gap: float = MIN_BULLET_GAP) -> list[str]:
    """Every way a bullet reveal can be wrong, as human-readable strings.

    ``appear_at`` is scene-relative. Older jobs have no bullets at all, which is not a
    defect — an empty list means "nothing to check", not "everything passed".
    """
    bullets = scene.bullets or []
    if not bullets:
        return []
    floor = scene.plan.bullet_min_gap if scene.plan else min_gap
    duration = scene.duration
    issues: list[str] = []
    previous: float | None = None
    for index, bullet in enumerate(bullets, start=1):
        at = bullet.appear_at
        label = f"bullet {index} ({bullet.text[:32]!r})"
        if at < 0:
            issues.append(f"{label} appears at {at:.2f}s, before the scene starts")
        if duration > 0 and at > duration:
            issues.append(f"{label} appears at {at:.2f}s, after the scene ends ({duration:.2f}s)")
        elif duration > 0 and at > duration - 0.5:
            issues.append(
                f"{label} appears at {at:.2f}s, under 0.5s before the scene ends ({duration:.2f}s)"
            )
        if previous is not None:
            if at < previous:
                issues.append(
                    f"{label} appears at {at:.2f}s, out of order "
                    f"(previous {previous:.2f}s)"
                )
            elif at - previous < floor:
                issues.append(
                    f"{label} appears {at - previous:.2f}s after the previous bullet, "
                    f"under the {floor:.2f}s floor"
                )
        previous = max(previous, at) if previous is not None else at
    return issues


def duration_deviation(scene: Scene, timeline: Timeline) -> float | None:
    """Signed fraction by which this scene differs from the median sibling duration."""
    durations = [s.duration for s in timeline.scenes if s.duration > 0]
    if len(durations) < 3 or scene.duration <= 0:
        return None
    median = statistics.median(durations)
    if median <= 0:
        return None
    return round((scene.duration - median) / median, 3)


# -------------------------------------------------------------------- orchestration


def measure_scene(scene: Scene, timeline: Timeline, video_path: Path) -> SceneMetrics:
    """Every deterministic number for one scene. Failures become ``None``, never zero."""
    source = frame_source(scene, timeline, video_path)
    silence: list[tuple[float, float]] = []
    if scene.audio_path and Path(scene.audio_path).exists():
        silence = [
            (round(a, 2), round(b, 2))
            for a, b in detect_silence(Path(scene.audio_path))
            # Trailing silence is the deliberate breath before the transition.
            if b < max((w.end - scene.start for w in scene.words), default=0.0) + 0.1
        ]
    dupes = (
        duplicate_frame_ratio(source.path, start=source.offset, duration=source.duration)
        if source.path.exists()
        else None
    )
    return SceneMetrics(
        scene_id=scene.id,
        heading=scene.heading,
        duration=round(scene.duration, 3),
        contrast=measure_text_contrast(scene, timeline, video_path),
        duplicate_frame_ratio=dupes,
        words_per_minute=words_per_minute(scene),
        narration_gaps=narration_gaps(scene),
        silence_windows=silence,
        bullet_issues=bullet_issues(scene),
        bullet_count=len(scene.bullets or []),
        duration_deviation=duration_deviation(scene, timeline),
    )


def measure_video(timeline: Timeline, video_path: Path) -> VideoMetrics:
    """Container conformance plus whole-file audio."""
    data = probe(video_path)
    video = _stream(data, "video")
    audio = _stream(data, "audio")
    fps = 0.0
    if rate := video.get("r_frame_rate"):
        num, _, den = rate.partition("/")
        denominator = float(den or 1)
        fps = float(num) / denominator if denominator else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    expected = timeline.final_duration()
    frame_time = 1.0 / fps if fps else 1.0 / max(1, timeline.profile.fps)

    profile = timeline.profile
    mismatch: list[str] = []
    width, height = int(video.get("width", 0) or 0), int(video.get("height", 0) or 0)
    if width and (width, height) != (profile.width, profile.height):
        mismatch.append(f"{width}x{height} != profile {profile.width}x{profile.height}")
    if fps and abs(fps - profile.fps) > 0.05:
        mismatch.append(f"{fps:.3f}fps != profile {profile.fps}fps")
    if not audio:
        mismatch.append("no audio stream")

    return VideoMetrics(
        duration=round(duration, 3),
        expected_duration=round(expected, 3),
        duration_drift_frames=round(abs(duration - expected) / frame_time, 2) if expected else 0.0,
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name"),
        loudness=measure_loudness(video_path),
        balance=measure_balance(video_path, timeline),
        duplicate_frame_ratio=duplicate_frame_ratio(video_path),
        profile_mismatch=mismatch,
    )
