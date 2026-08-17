"""ffmpeg implementation of ``VideoBackend``.

Architecture: **one intermediate clip per scene**, then a single assemble pass. A
monolithic filter_complex for a 12-scene video is unreadable, impossible to debug
from an error message, and forces a full re-render when one image changes. Per-scene
clips are near-lossless (x264 crf 12) so chaining them costs nothing visible.

Four things in here are load-bearing and easy to get wrong:

*The frame is designed, not photographed.* Every slide starts as a solid
``Theme.bg`` colour and the image is placed **inside a region** (see
:func:`layout_region`). ``SlideLayout.FULL_BLEED`` is the exception, not the rule.

*zoompan jitter.* ``zoompan`` truncates its ``x``/``y`` expressions to integers, so a
slow pan advances 0px for several frames and then jumps 1px — a visible stutter. The
cure has two halves, and each fixes a different cause; see
:func:`motion_canvas` and :func:`eased_progress`.

*Generated clips.* A scene's visual may be a Veo clip rather than a still
(``Scene.video_path``). It is fitted into the same region, converted to the timeline's
fps, looped with a crossfade at the seam to cover a scene longer than the clip, and
stripped of its own audio. No zoompan is applied on top of moving footage. See
:meth:`FFmpegBackend._clip_chain`.

*Text animation without drawtext.* This build has no ``drawtext`` (no libfreetype),
so text arrives as pre-rasterised RGBA PNGs via :mod:`app.render.contracts`. Each
layer is animated by ``fade=...:alpha=1`` (invisible before ``appear_at`` — verified,
see tests) plus a time-varying ``overlay`` x/y expression with smoothstep easing.

*xfade timing.* ``xfade`` **consumes** its overlap: two 5s clips with a 0.5s crossfade
produce 9.5s, not 10s. Every offset is therefore cumulative over already-shortened
output, the narration has to be shifted by the same amount, and the final duration is
``sum(durations) - sum(transitions)``. :meth:`assemble` checks its own output against
``Timeline.final_duration()`` and refuses to lie about it.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from app.core.config import REPO_ROOT, get_settings
from app.core.models import (
    BulletPoint,
    Motion,
    RenderProfile,
    Scene,
    SlideLayout,
    TextAnimation,
    Theme,
    Timeline,
    Transition,
    VisualPlan,
)
from app.render import captions as captions_mod
from app.render import ffmpeg as ff
from app.render import text_overlay as tx
from app.render.contracts import SceneText, TextLayer

logger = logging.getLogger(__name__)

INTERMEDIATE_CRF = 12
"""Near-lossless: concatenation must not compound generational loss."""

ASPECT_TOLERANCE = 0.25
"""Beyond 25% aspect mismatch, centre-cropping throws away too much of the image, so
we switch to a blurred fill instead. Stretching is never an option."""

MIN_PAN_ZOOM = 1.06
"""A pan needs headroom to travel across; at zoom 1.0 there is nowhere to go."""

# ------------------------------------------------------- Ken Burns smoothness
#
# Two independent causes make a slow move step, and a fixed pre-upscale only ever
# addressed the first one.
#
# 1. *Quantisation.* zoompan's x/y (and its crop size) are integers in **input canvas**
#    pixels, so the finest move it can make is one canvas pixel, which lands on screen as
#    ``zoom / U`` output pixels where ``U = canvas_width / region_width``. Pre-upscaling
#    raises U. The factor was a fixed 4, calibrated when zoompan filled the frame
#    (1920 wide); Ken Burns now runs inside the image region (``hero_right`` is 856x816),
#    so a 4x canvas is 3424 wide instead of 7680 and the same move has less than half the
#    headroom it was tuned for. :func:`motion_canvas` derives the canvas from the region, the
#    distance the move actually travels and the frame count instead.
#
# 2. *Easing.* ``ease_in_out`` was a pure smoothstep, whose velocity is **exactly zero**
#    at both ends. The opening of every move was therefore genuinely stationary, and no
#    finite upscale can fix a zero. :func:`eased_progress` keeps the smoothstep character
#    but blends in enough linear ramp to guarantee a velocity floor. This is free, and on
#    the measured cases it is the larger of the two effects.

EASE_VELOCITY_FLOOR = 0.20
"""Fraction of the mean rate that an ``ease_in_out`` move still travels at its slowest.

0 is a pure smoothstep and is what shipped: velocity ``3u^2-2u^3`` differentiates to
``6u(1-u)``, which is zero at both endpoints, so the first and last frames of every move
held position no matter how much headroom zoompan was given. Blending in a linear ramp puts
a floor under it, and this is the single largest effect in the whole fix — it costs nothing.

0.20 is measured, not guessed. Sweeping it over the seven real layout/motion pairs, with
everything else held at its final value:

    floor   hero pan   hero zoom   hero zoom_out   bleed pan   bleed zoom
    0.00     51.67%      31.67%        27.50%        30.83%       9.17%
    0.15      7.50%      15.00%        11.67%         3.33%      10.00%
    0.20      4.17%      13.33%        10.83%         1.67%       6.67%
    0.25      4.17%      12.50%        10.83%         0.00%       4.17%

0.15 left two pairs above the evaluator's 12% noise floor; 0.20 leaves one, and it is the
pathological one (a 43px zoom spread over 20 seconds — see the module report). 0.25 buys
0.8% on that single case and flattens the ease further for nothing else, so it is not worth
it. At 0.20 the move's fastest frame is still 7x its slowest, which reads unmistakably as
an ease; a viewer cannot tell "starts at 20% speed" from "starts at 0% speed", but
``mpdecimate`` can, and so can the eye that was seeing the stutter.
"""

CANVAS_STEP_TARGET = 1.0
"""Canvas pixels the move must advance per frame at its **slowest** point.

1.0 is not a tuning knob, it is the condition itself: zoompan truncates to whole canvas
pixels, so a move that advances less than one of them per frame produces a byte-identical
frame. Below this the render *will* step; above it, it cannot.

An earlier draft of this used 0.35, fitted to the ``hero_right`` case where the anamorphic
stretch is wide enough that a fractional shift still perturbs every output pixel a little.
It does not generalise — on a small isotropic canvas the same 0.35 left 30% duplicates.
The target is the physics; :data:`CANVAS_PIXEL_BUDGET` is where the compromise lives, and
it is honest about being one.
"""

CANVAS_PIXEL_BUDGET = 24_000_000
"""Per-frame resample budget for zoompan's canvas, in pixels, at ``upscale_factor=4``.

The cost of a Ken Burns scene is dominated by zoompan cropping its canvas and lanczos-
scaling that crop down to the region, once per output frame. So the thing to budget is
canvas **area** — not the upscale factor, which says nothing about cost until you know how
big the region is. A 16x factor on an 856-wide panel and a 16x factor on a 1920-wide frame
differ by more than five times the work.

24M pixels is the measured break-even: it is the size of the ``hero_right`` canvas
(14552x1632) that rendered in 7.6s against the 7.7s the old fixed 4x canvas took, i.e. the
fix pays for itself there. Scaled by ``RenderProfile.upscale_factor / 4``, so ``draft``
buys half the smoothness for half the time and the field stays meaningful.

Where the budget binds before :data:`CANVAS_STEP_TARGET` is satisfied — the slowest moves
over the largest regions, ``full_bleed`` worst — some stepping survives by construction.
The cure there is a larger zoom span or a shorter scene, both of which are the planner's,
and :func:`slowest_step` is the number to argue with.
"""

UPSCALE_BUDGET_BASIS = 4
"""The ``upscale_factor`` that :data:`CANVAS_PIXEL_BUDGET` was measured at."""

ZOOM_CROSS_UPSCALE = 4
"""Cross-axis oversampling for a zoom, which moves *both* axes and changes the crop's size.

A pan gets no such constant: it holds a fixed zoom, so the crop's size and its cross-axis
offset are the same integers on every frame and that axis needs resolution only for
*detail*. Its cross factor is therefore exactly :func:`detail_upscale` and not a pixel more.
That is worth stating because spending anything there is spending it twice over — the area
budget is shared, so two wasted cross-axis pixels halve the travel-axis precision that
actually removes the artefact. Measured on ``full_bleed``/``pan_left``, where the source
cannot fill 1080p so detail is 1: a cross factor of 2 gave a 9600x2160 canvas and 19.17%
duplicates, while dropping it to detail gave 21120x1080 — the same 23M pixels, spent on the
axis that moves — and **3.33%**.

A zoom cannot do that. Its binding constraint is whichever canvas axis steps *last*, so
starving one axis stalls the whole frame however fine the other is: at a cross factor of 2 a
35280x1800 canvas still measured 25.83%, worse than a 6480x3600 one at 15.00%. 4 is the
measured knee — past it the ratio stops improving and the time keeps rising.
"""

MAX_DETAIL_UPSCALE = 4
MAX_SOURCE_UPSCALE = 1.25
"""How far past its own resolution the source may be lanczos-upscaled before zoompan.

This is the "source pixels vs region output size" half of the derivation. Fitting a
2752x1536 still into an 856x816 panel already needs a 0.53x scale, so a 2x canvas is
about the source's own resolution and a 4x canvas is inventing detail with a lanczos
kernel and then throwing it away — measurably softer *and* slower. Detail therefore stops
here; positional precision beyond it is bought with a cheap anamorphic stretch instead.
"""

STROBE_STEP_PIXELS = 3.0
"""Peak output pixels per frame past which a move reads as strobing rather than moving.

The opposite failure to stepping, and the reason the derived upscale has an upper clamp
rather than being maximised: there is no point buying sub-pixel precision for a move that
is already skipping three pixels a frame. Warned about, not corrected — the cure is a
smaller zoom span or a longer scene, both of which are the planner's call.
"""

AUDIO_RATE = 48_000

LOUDNESS_TARGET_LUFS = -16.0
"""Integrated loudness every render is normalised to.

The web convention, and what the evaluator scores against: an un-normalised assemble lands
near -20.3 LUFS, which reads as quiet next to anything else in a browser tab and cost every
video the same "+4.3 dB" note. Broadcast would want -23; this is not broadcast.
"""

LOUDNESS_TRUE_PEAK_DB = -1.0
"""True-peak ceiling, in dBFS. One dB of headroom survives lossy re-encoding downstream."""

LOUDNESS_RANGE = 11.0
"""Target loudness range. Narration plus a ducked bed is not dynamic material; a wider
target lets ``loudnorm`` leave quiet passages quiet and undoes the point of normalising."""

# ------------------------------------------------------------ generated clips

CLIP_SEAM_CROSSFADE = 0.5
"""Crossfade, in seconds, hiding the seam where a generated clip loops.

A Veo clip is a fixed 8.000s and a content scene runs 14-24s, so one clip cannot fill a
scene and something has to cover the shortfall. Of the three candidates, measured:

* *hold the final frame* — 12s of freeze on a 20s scene. 60% of the scene stops dead;
  the duplicate-frame ratio goes to ~60% and it reads as a hung player.
* *slow the clip down* — 2.5x on ``setpts`` turns 24fps source into ~9.6 unique frames a
  second, so every frame is held for three. Juddery, and it re-introduces exactly the
  duplicate-frame defect the rest of this module exists to remove.
* *loop with a crossfade at the seam* — motion never stops, no frame is ever held, and
  the only artefact is a half-second dissolve back to the opening composition.

The third wins on both the metric and the look, so it is what this does. 0.5s matches the
scene-to-scene transition duration: long enough to read as a dissolve rather than a cut,
short enough that it does not eat an eighth of the clip.
"""

CLIP_LOOP_SAFETY = 0.5
"""Extra seconds of cloned tail appended to a looped clip.

Belt and braces for frame-exactness. An image input is infinite (``-loop 1``) so
``-frames:v`` can always be satisfied; a clip is finite, and a filtergraph that comes up
one frame short would silently write a shorter file. ``tpad`` guarantees the graph can
always over-deliver, and ``-frames:v`` does the actual cutting.
"""

CLICK_FADE = 0.02
MUSIC_FADE_IN = 1.5
MUSIC_FADE_OUT = 2.0
FINAL_FADE_OUT = 0.6

# ------------------------------------------------------------- slide geometry

MARGIN_FRACTION = 0.045
"""Frame margin around image panels, as a fraction of frame *width*."""

HERO_WIDTH_FRACTION = 0.465
"""Image panel width for ``hero_left``/``hero_right``. Text gets the rest."""

BAND_HEIGHT_FRACTION = 0.45
"""Image band height for ``image_band``, as a fraction of frame height."""

POP_OVERSHOOT_C1 = 1.70158
"""Standard "back out" easing constant: overshoots ~10% then settles."""

POP_FADE_FRACTION = 0.6
"""A pop reads as snappy only if the opacity ramp beats the movement."""

MAX_TEXT_LAYERS = 8
"""1 background/scrim + 1 heading + up to 5 bullets, with one spare.

Each layer is an ffmpeg input and an overlay stage; past this the filtergraph stops
being reviewable and the render cost stops being worth it.
"""

_HEX_COLOUR = re.compile(r"^#?([0-9A-Fa-f]{6})$")

AUTO_LOGO = "auto"
"""Sentinel for ``FFmpegBackend(logo_path=...)``: resolve from settings.

``None`` disables branding outright; a path uses that file. A three-way knob rather than
two because "the caller said nothing" and "the caller said no logo" are different answers.
"""

DEFAULT_LOGO_RELATIVE = Path("frontend/public/favicon.svg")
"""Fallback mark, relative to the repo root: the app's own favicon."""

_DISABLED_PATHS = frozenset({"", ".", "none", "off"})
"""``VIDEO_LOGO_PATH=`` in ``.env`` arrives as ``Path('.')``, which is a directory, not a
mark. Treat the empty-ish spellings as an explicit "no branding"."""


class RenderError(RuntimeError):
    pass


class DurationMismatchError(RenderError):
    """The assembled output does not match ``Timeline.final_duration()``."""


@dataclass(frozen=True)
class Region:
    """Where the image lives inside the frame, in output pixels."""

    x: int
    y: int
    width: int
    height: int


def _even(value: float, *, minimum: int = 0) -> int:
    """Round down to an even number — yuv420p subsamples chroma 2x2.

    ``minimum=2`` for dimensions (a zero-width filter is a parse error); the default 0
    is for offsets, where 0 is both legal and extremely common.
    """
    whole = int(round(value))
    return max(minimum, whole - whole % 2)


def layout_region(
    plan: VisualPlan, profile: RenderProfile, *, theme: Theme | None = None
) -> Region | None:
    """The image's box, or ``None`` when the slide has no image (a title card).

    ``text_overlay.slide_geometry`` is the single source of truth here: it decides the
    text column *and* the image panel from the same numbers, so asking it rather than
    recomputing is what stops type and picture overlapping. :func:`fallback_region`
    covers a ``text_overlay`` that predates that function.

    Every edge is forced even — an odd overlay offset shifts the subsampled chroma
    plane by half a pixel and visibly softens the panel edge.
    """
    geometry = getattr(tx, "slide_geometry", None)
    if geometry is None:
        return fallback_region(plan.layout, profile)
    region = geometry(plan, profile, theme=theme or Theme()).image_region
    if region is None:
        return None
    return Region(
        _even(region.x),
        _even(region.y),
        _even(region.width, minimum=2),
        _even(region.height, minimum=2),
    )


def fallback_region(layout: SlideLayout, profile: RenderProfile) -> Region | None:
    """Proportional region maths, used when ``text_overlay`` cannot supply one.

    Same design intent as :func:`layout_region`, expressed in fractions of the frame
    so draft and final renders are the same slide at different sizes.
    """
    width, height = profile.width, profile.height
    margin = _even(width * MARGIN_FRACTION, minimum=2)

    if layout is SlideLayout.TITLE_CARD:
        return None
    if layout is SlideLayout.FULL_BLEED:
        return Region(0, 0, _even(width, minimum=2), _even(height, minimum=2))
    if layout is SlideLayout.IMAGE_BAND:
        band_w = _even(width - 2 * margin, minimum=2)
        band_h = _even(height * BAND_HEIGHT_FRACTION, minimum=2)
        return Region(margin, _even(height - margin - band_h), band_w, band_h)

    panel_w = _even(width * HERO_WIDTH_FRACTION, minimum=2)
    panel_h = _even(height - 2 * margin, minimum=2)
    x = margin if layout is SlideLayout.HERO_LEFT else _even(width - margin - panel_w)
    return Region(x, margin, panel_w, panel_h)


def corner_radius(
    plan: VisualPlan, region: Region, profile: RenderProfile, theme: Theme | None = None
) -> int:
    """Rounding for the image panel, in output pixels. 0 means square corners.

    Deferred to ``slide_geometry`` for the same reason as the region: a rounded panel
    and the card outline behind it have to agree. A full-bleed image *is* the frame,
    so it is never rounded.
    """
    theme = theme or Theme()
    if plan.layout is SlideLayout.FULL_BLEED:
        return 0
    geometry = getattr(tx, "slide_geometry", None)
    radius = (
        geometry(plan, profile, theme=theme).image_radius
        if geometry is not None
        else int(round(theme.image_radius * profile.width / 1920))
    )
    return max(0, min(radius, min(region.width, region.height) // 2))


def motion_travel(plan: VisualPlan, region: Region) -> float:
    """How far the move travels, in **output** pixels. 0 for a static shot.

    This is the number a fixed upscale factor ignored, and it is what makes 4x right for
    one shot and hopelessly short for another. A pan's travel is the slack the zoom buys
    it; a zoom's is how far the outermost pixel is displaced. Both are measured at the
    region's scale, because that — not the frame's — is what zoompan now emits.
    """
    if plan.motion in (Motion.PAN_LEFT, Motion.PAN_RIGHT):
        zoom = max(plan.zoom_from, MIN_PAN_ZOOM)
        return region.width * (1.0 - 1.0 / zoom)
    if plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
        span = abs(plan.zoom_to - plan.zoom_from)
        base = max(1e-6, min(plan.zoom_from, plan.zoom_to))
        return region.width / 2.0 * span / base
    return 0.0


def slowest_step(plan: VisualPlan, region: Region, frames: int) -> float:
    """Output pixels the move advances on its **slowest** frame.

    The slowest frame is the one that decides whether anything repeats, and with easing it
    is not the average. A pure smoothstep makes this exactly zero, which is why
    :data:`EASE_VELOCITY_FLOOR` exists — without a floor there is no finite answer to
    "how much headroom does this need".
    """
    travel = motion_travel(plan, region)
    if travel <= 0.0 or frames <= 1:
        return 0.0
    floor = EASE_VELOCITY_FLOOR if plan.easing == "ease_in_out" else 1.0
    return floor * travel / frames


def peak_step(plan: VisualPlan, region: Region, frames: int) -> float:
    """Output pixels the move advances on its **fastest** frame — the strobing end.

    Smoothstep peaks at 1.5x the mean rate; the blended curve peaks a little lower.
    """
    travel = motion_travel(plan, region)
    if travel <= 0.0 or frames <= 1:
        return 0.0
    gain = 1.0
    if plan.easing == "ease_in_out":
        gain = 1.5 * (1.0 - EASE_VELOCITY_FLOOR) + EASE_VELOCITY_FLOOR
    return gain * travel / frames


def detail_upscale(src_size: tuple[int, int], region: Region) -> int:
    """Isotropic factor for the lanczos fit — how much *detail* the canvas carries.

    Capped by the source: scaling a still past :data:`MAX_SOURCE_UPSCALE` of its own
    resolution invents pixels that the downscale inside zoompan then discards, which costs
    time and measurably softens the panel (lanczos ringing on the way up).
    """
    src_w, src_h = src_size
    if src_w <= 0 or src_h <= 0:
        return 1
    cover = max(region.width / src_w, region.height / src_h)
    if cover <= 0:
        return 1
    return max(1, min(MAX_DETAIL_UPSCALE, int(MAX_SOURCE_UPSCALE / cover)))


def plan_zoom_ceiling(plan: VisualPlan) -> float:
    """The largest zoom the move reaches — the factor zoompan's crop is divided by."""
    if plan.motion in (Motion.PAN_LEFT, Motion.PAN_RIGHT):
        return max(plan.zoom_from, MIN_PAN_ZOOM)
    if plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
        return max(plan.zoom_from, plan.zoom_to, 1.0)
    return 1.0


@dataclass(frozen=True)
class MotionCanvas:
    """What zoompan reads, and what was lanczos-fitted to get there."""

    fit: tuple[int, int]
    """Isotropic cover-and-crop target: ``region * detail``. Where lanczos runs."""

    canvas: tuple[int, int]
    """What zoompan actually reads — ``fit`` after a cheap anamorphic stretch."""

    detail: int

    @property
    def stretched(self) -> bool:
        return self.canvas != self.fit


def motion_canvas(
    plan: VisualPlan,
    region: Region,
    frames: int,
    profile: RenderProfile,
    src_size: tuple[int, int] = (0, 0),
) -> MotionCanvas:
    """Size zoompan's canvas for this move.

    Three requirements, and they pull in different directions:

    1. **No resolution loss.** zoompan crops ``canvas/zoom`` and scales it to the region, so
       ``canvas >= region * zoom`` on *both* axes or the crop is an upscale and the panel is
       measurably softer than the still it came from. This is the one an earlier draft of
       this function got wrong: it worked in integer multiples of the region, so the only
       values available were 1x (which upscales, because ``zoom > 1``) and 2x (which costs
       twice the area). Sizing the canvas in *pixels* instead makes 1.08x reachable.
    2. **Positional precision.** ``canvas/region`` on the travel axis is how many sub-steps
       zoompan's integer x/y gets per output pixel; :data:`CANVAS_STEP_TARGET` says how many
       it needs.
    3. **Cost.** Area, not factor — see :data:`CANVAS_PIXEL_BUDGET`.

    So: the cross axis takes the least it can get away with (1, plus the zoom, plus whatever
    detail the source justified) and the travel axis spends everything left over.

    Nothing is distorted by this, whatever the numbers. The still is cover-fitted to the
    region's aspect first, so its net scale through the stretch and back out of zoompan is
    ``region * zoom / fit`` on each axis — and ``fit`` has the region's aspect, so the two
    are equal by construction.

    Measured on ``hero_right``/``pan_right`` at 1080p over 604 frames, in the evaluator's
    own 4s window:

        canvas                       duplicate ratio   render
        2880x3600   (old, fixed 4x4)      51.67%        5.70s
        11520x14400 (isotropic 16x)       ~13%          ~64s
        13332x1800  (this)                 4.17%        9.02s

    The isotropic canvas that reaches the same smoothness is seven times the pixels and
    seven times the time. And because the budget is an *area*, the derivation makes
    ``full_bleed`` **cheaper** than the fixed 4x it replaces (24 vs 33 Mpixels) while taking
    it from 30.83% duplicates to under 2%.
    """
    region_size = (region.width, region.height)
    if plan.motion is Motion.STATIC or frames <= 1:
        return MotionCanvas(fit=region_size, canvas=region_size, detail=1)

    detail = detail_upscale(src_size, region)
    fit = (region.width * detail, region.height * detail)
    zoom = plan_zoom_ceiling(plan)

    # (1) Never let zoompan's own crop become an upscale, on either axis.
    floor_w = _even(region.width * zoom, minimum=2)
    floor_h = _even(region.height * zoom, minimum=2)

    # The cross axis takes the minimum it can. A pan does not move it, so "the minimum" is
    # all it will ever need; a zoom does move it, so it also buys precision there.
    zooming = plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT)
    canvas_h = max(fit[1], floor_h)
    if zooming:
        canvas_h = max(canvas_h, _even(region.height * ZOOM_CROSS_UPSCALE, minimum=2))

    # (3) Whatever area is left goes to the travel axis...
    budget = CANVAS_PIXEL_BUDGET * max(1, profile.upscale_factor) / UPSCALE_BUDGET_BASIS
    affordable = _even(budget / max(1, canvas_h), minimum=2)

    # ...up to (2) what the slowest frame of the move actually needs.
    step = slowest_step(plan, region, frames)
    wanted = affordable if step <= 0.0 else _even(
        region.width * CANVAS_STEP_TARGET / step, minimum=2
    )

    canvas_w = max(fit[0], floor_w, min(wanted, affordable))
    return MotionCanvas(fit=fit, canvas=(canvas_w, canvas_h), detail=detail)


def clip_seam(clip_duration: float) -> float:
    """Crossfade at a looped clip's seam, clamped so it cannot swallow the clip."""
    return max(0.0, min(CLIP_SEAM_CROSSFADE, clip_duration / 4.0))


def clip_loop_count(clip_duration: float, needed: float) -> int:
    """How many passes of the clip cover ``needed`` seconds, allowing for the seams.

    ``n`` passes crossfaded at ``seam`` yield ``n*clip - (n-1)*seam`` seconds, because an
    xfade consumes its overlap exactly as it does between scenes.
    """
    if clip_duration <= 0:
        return 1
    if needed <= clip_duration:
        return 1
    seam = clip_seam(clip_duration)
    period = clip_duration - seam
    if period <= 0:
        return 1
    return 1 + math.ceil((needed - clip_duration) / period)


def clip_loop_span(clip_duration: float, loops: int) -> float:
    """Seconds a crossfaded loop of ``loops`` passes actually produces."""
    if loops <= 1:
        return clip_duration
    return loops * clip_duration - (loops - 1) * clip_seam(clip_duration)


def eased_progress(expression: str, easing: str) -> str:
    """0..1 progress with the requested easing, as an ffmpeg expression.

    ``ease_in_out`` is a smoothstep blended with a linear ramp so that its velocity never
    reaches zero — see :data:`EASE_VELOCITY_FLOOR`. Both endpoints are still exact (the
    blend of two curves that pass through 0 and 1 also passes through 0 and 1), so a move
    still starts and finishes precisely where the plan says.
    """
    if easing != "ease_in_out":
        return expression
    smooth = f"({expression}*{expression}*(3-2*{expression}))"
    weight = 1.0 - EASE_VELOCITY_FLOOR
    return f"({weight:g}*{smooth}+{EASE_VELOCITY_FLOOR:g}*{expression})"


def resolve_logo_source(configured: Path | str | None = AUTO_LOGO) -> Path | None:
    """The brand mark to composite, or ``None`` for no branding.

    Three inputs, three answers, and none of them can fail a render:

    * :data:`AUTO_LOGO` — take ``settings.video_logo_path``; if that is unset, fall back to
      ``<repo>/frontend/public/favicon.svg`` when it exists, else no branding.
    * ``None`` or an empty-ish path — no branding, silently. That is what
      ``VIDEO_LOGO_PATH=`` in ``.env`` means.
    * a real path — use it, or warn and skip if it is missing.

    A missing or unreadable logo is never an error. Branding is the last 1% of the frame;
    losing it must not cost the other 99%.
    """
    if configured is None:
        return None
    if configured == AUTO_LOGO:
        configured = get_settings().video_logo_path
        if configured is None:
            fallback = REPO_ROOT / DEFAULT_LOGO_RELATIVE
            return fallback if fallback.is_file() else None
    if str(configured).strip().lower() in _DISABLED_PATHS:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        logger.warning("configured brand logo does not exist, skipping branding: %s", path)
        return None
    return path


def ffmpeg_colour(value: str) -> str:
    """``#0B1220`` -> ``0x0B1220``. ``#`` is a comment character in filter scripts."""
    match = _HEX_COLOUR.match(value.strip())
    if not match:
        raise RenderError(f"theme colour must be #RRGGBB, got {value!r}")
    return f"0x{match.group(1).upper()}"


class FFmpegBackend:
    """Satisfies ``VideoBackend``. All creative choices arrive via ``VisualPlan``."""

    def __init__(
        self,
        *,
        text_mode: str = "auto",
        music_duck_db: int | None = None,
        final_fade_out: bool = True,
        burn_captions: bool = False,
        strict_duration: bool = True,
        theme: Theme | None = None,
        logo_path: Path | str | None = AUTO_LOGO,
    ) -> None:
        self.text_mode = tx.resolve_text_mode(text_mode)
        self.music_duck_db = (
            music_duck_db if music_duck_db is not None else get_settings().video_music_duck_db
        )
        self.final_fade_out = final_fade_out
        self.burn_captions = burn_captions
        self.strict_duration = strict_duration
        self.theme = theme or Theme()
        self.logo_source = resolve_logo_source(logo_path)
        """Brand mark, or ``None``. Resolved once so a render cannot change its mind."""
        self._font = tx.find_font() if self.text_mode in ("drawtext", "png") else None

    # ------------------------------------------------------------ scene clip

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
        video_path: Path | str | None = None,
    ) -> Path:
        """Render one slide — solid background, image panel, animated text — to a clip.

        The clip is exactly ``round(duration * fps)`` frames so it matches its
        narration segment. Verified with ffprobe before returning.

        ``video_path`` makes the visual a generated clip rather than a still and takes
        precedence over ``image_path`` when both are given, which is what
        ``Scene.video_path`` means. The frame count is identical either way: the visual
        source changes, the timing contract does not.

        ``scene_text`` may be supplied by the caller (tests, or a pipeline that
        rasterises text elsewhere); otherwise it is built here from ``heading`` and
        ``bullets``. Everything written lands under ``out_path``'s stem, which is
        unique per scene — that is what makes concurrent scene renders safe.
        """
        out_path = Path(out_path)
        if duration <= 0:
            raise RenderError(f"scene duration must be positive, got {duration}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        region = layout_region(plan, profile, theme=self.theme)
        image_path = Path(image_path) if image_path else None
        video_path = Path(video_path) if video_path else None
        clip_duration = clip_fps = 0.0

        if region is not None:
            source = video_path or image_path
            if source is None:
                raise RenderError(f"layout {plan.layout.value} needs an image, none was given")
            if not source.is_file():
                kind = "clip" if video_path else "image"
                raise RenderError(f"scene {kind} not found: {source}")
            if video_path is not None:
                summary = ff.probe_summary(video_path)
                clip_duration, clip_fps = summary["duration"], summary["fps"]
                if clip_duration <= 0:
                    raise RenderError(f"generated clip has no usable duration: {video_path}")
                src_size = (summary["width"], summary["height"])
                if summary["audio_codec"]:
                    logger.debug(
                        "generated clip %s carries a %s track; the graph reads [0:v] only "
                        "and the encode is -an, so it is discarded",
                        video_path.name, summary["audio_codec"],
                    )
            else:
                src_size = ff.probe_image_size(image_path)
        else:
            src_size = (0, 0)

        frames = max(1, int(round(duration * profile.fps)))

        inputs: list[str | Path] = []
        if region is not None:
            if video_path is not None:
                # No -loop and no -framerate: the clip carries its own timing, and the
                # graph resamples it. -an here is the first of two defences against the
                # provider's audio track; the second is that nothing references [0:a].
                inputs += ["-an", "-i", video_path]
            else:
                assert image_path is not None
                inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", image_path]

        text_layout = tx.layout_heading(heading, plan, profile)
        if scene_text is None and self.text_mode == "png":
            scene_text = self._build_scene_text(
                heading, bullets or [], plan, profile, out_path, image_path=image_path
            )

        text_png: Path | None = None
        if scene_text is not None:
            if len(scene_text.layers) > MAX_TEXT_LAYERS:
                raise RenderError(
                    f"{len(scene_text.layers)} text layers exceeds the {MAX_TEXT_LAYERS} "
                    "the filtergraph is designed for"
                )
            for png in scene_text.inputs:
                inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", png]
        elif self.text_mode == "png" and heading.strip():
            # Legacy single-PNG heading: still supported so a text_overlay without
            # build_scene_text keeps working.
            text_png = out_path.with_suffix(".text.png")
            tx.render_text_png(
                heading, plan, profile, text_png, font=self._font, layout=text_layout
            )
            inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", text_png]

        graph = self._scene_graph(
            src_size=src_size,
            plan=plan,
            profile=profile,
            frames=frames,
            text_layout=text_layout,
            heading=heading,
            has_text_input=text_png is not None,
            scene_text=scene_text,
            has_image_input=region is not None,
            clip_duration=clip_duration if video_path else 0.0,
            clip_fps=clip_fps if video_path else 0.0,
        )

        ff.ffmpeg(
            [
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[vout]",
                "-frames:v",
                str(frames),
                "-r",
                str(profile.fps),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                str(INTERMEDIATE_CRF),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                *self._thread_args(profile),
                out_path,
            ]
        )

        actual = ff.probe_duration(out_path)
        expected = frames / profile.fps
        if abs(actual - expected) > 1.5 / profile.fps:
            raise DurationMismatchError(
                f"{out_path.name}: wanted {expected:.4f}s ({frames} frames), got {actual:.4f}s"
            )
        logger.debug(
            "scene clip %s: %d frames, %.4fs (target %.4fs), motion=%s",
            out_path.name,
            frames,
            actual,
            duration,
            plan.motion.value,
        )
        return out_path

    def render_all(self, timeline: Timeline, clip_dir: Path) -> Timeline:
        """Render every scene, assigning ``clip_path``.

        Frame counts are rounded *cumulatively* (``round(end*fps) - round(start*fps)``)
        rather than per scene. Per-scene rounding leaks up to half a frame each time
        and the error accumulates; cumulative rounding keeps the clip boundaries on the
        narration's own frame grid.

        Scenes render concurrently — each clip is an independent ffmpeg process
        writing its own file, so there is no shared state to guard. Results are
        reassembled in scene order regardless of completion order.
        """
        out = timeline.model_copy(deep=True)
        fps = out.profile.fps
        clip_dir = Path(clip_dir)
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Validate everything up front: failing fast beats discovering a bad scene
        # after paying for three renders.
        for scene in out.scenes:
            if scene.plan is None:
                raise RenderError(f"scene {scene.id} has no VisualPlan; run the planner first")
            # A title card is solid colour plus type: it has no image region, so
            # demanding an image would fail a slide that is by design image-free.
            needs_image = layout_region(scene.plan, out.profile, theme=self.theme) is not None
            if needs_image and not (scene.video_path or scene.image_path):
                raise RenderError(
                    f"scene {scene.id} has no image_path or video_path but layout is "
                    f"{scene.plan.layout.value}"
                )

        workers, threads = out.profile.resolve_concurrency(len(out.scenes))
        profile = out.profile.model_copy(update={"encoder_threads": threads})

        jobs: list[tuple[int, Scene, Path, float]] = []
        for idx, scene in enumerate(out.scenes):
            frames = max(1, int(round(scene.end * fps)) - int(round(scene.start * fps)))
            jobs.append((idx, scene, clip_dir / f"scene_{scene.id:03d}.mp4", frames / fps))

        if workers == 1:
            for _, scene, clip, dur in jobs:
                self._render_one(scene, clip, dur, profile)
            return out

        logger.info(
            "rendering %d scenes with %d workers x %s threads",
            len(jobs),
            workers,
            threads or "auto",
        )
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
            futures = {
                pool.submit(self._render_one, scene, clip, dur, profile): (idx, scene)
                for idx, scene, clip, dur in jobs
            }
            errors: list[str] = []
            for fut in as_completed(futures):
                _, scene = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
                    errors.append(f"scene {scene.id}: {exc}")
        if errors:
            raise RenderError(
                f"{len(errors)} of {len(jobs)} scenes failed to render:\n" + "\n".join(errors)
            )
        logger.info("rendered %d scenes in %.1fs", len(jobs), time.monotonic() - started)
        return out

    def _render_one(
        self, scene: Scene, clip: Path, duration: float, profile: RenderProfile
    ) -> None:
        """Render one scene and record its clip. Runs on a worker thread."""
        assert scene.plan is not None  # validated by render_all
        self.render_scene(
            scene.image_path,
            scene.plan,
            scene.heading,
            duration,
            clip,
            profile,
            bullets=scene.bullets,
            video_path=scene.video_path,
        )
        scene.clip_path = str(clip)

    def _build_scene_text(
        self,
        heading: str,
        bullets: list[BulletPoint],
        plan: VisualPlan,
        profile: RenderProfile,
        out_path: Path,
        *,
        image_path: Path | None = None,
    ) -> SceneText | None:
        """Rasterise this scene's text via the :mod:`app.render.contracts` seam.

        ``text_overlay.build_scene_text`` is the contract; if this build of
        ``text_overlay`` predates it we fall back to the legacy single-PNG heading
        rather than dropping the text on the floor.

        The work directory is derived from ``out_path``, which is unique per scene —
        no fixed temp names, so concurrent scene renders cannot collide.
        """
        builder = getattr(tx, "build_scene_text", None)
        if builder is None:
            logger.debug("text_overlay has no build_scene_text; using the legacy heading PNG")
            return None
        if not heading.strip() and not bullets:
            return None
        workdir = out_path.parent / f"{out_path.stem}.text"
        workdir.mkdir(parents=True, exist_ok=True)
        scene_text = builder(
            heading,
            bullets,
            plan,
            profile,
            workdir,
            image_path=image_path,
            theme=self.theme,
            font=self._font,
        )
        if scene_text is None or not scene_text.layers:
            return None
        return scene_text

    # -------------------------------------------------------- scene subgraph

    def _scene_graph(
        self,
        *,
        src_size: tuple[int, int],
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        text_layout: tx.TextLayout,
        heading: str,
        has_text_input: bool = False,
        scene_text: SceneText | None = None,
        has_image_input: bool = True,
        clip_duration: float = 0.0,
        clip_fps: float = 0.0,
    ) -> str:
        """The whole per-scene filtergraph, ending in ``[vout]``.

        Built back to front: solid background, image panel (with its own Ken Burns, or a
        fitted generated clip when ``clip_duration`` is set), then one overlay stage per
        animated text layer.
        """
        parts = self._background_chain(plan, profile)
        base = "bg"

        region = layout_region(plan, profile, theme=self.theme)
        if region is not None and has_image_input:
            if clip_duration > 0:
                parts += self._clip_chain(
                    src_size=src_size,
                    clip_duration=clip_duration,
                    clip_fps=clip_fps,
                    plan=plan,
                    profile=profile,
                    frames=frames,
                    region=region,
                )
            else:
                parts += self._image_chain(
                    src_size=src_size, plan=plan, profile=profile, frames=frames, region=region
                )
            parts.append(f"[{base}][hero]overlay=x={region.x}:y={region.y}:format=auto[base]")
            base = "base"

        if scene_text is not None:
            first_input = 1 if (region is not None and has_image_input) else 0
            layer_parts, base = self._text_chain(
                scene_text, base=base, first_input=first_input, fps=profile.fps
            )
            parts += layer_parts
            parts.append(f"[{base}]format=yuv420p[vout]")
            return ";".join(parts)

        if not heading.strip():
            parts.append(f"[{base}]format=yuv420p[vout]")
        elif has_text_input:
            index = 1 if (region is not None and has_image_input) else 0
            parts.append(f"[{base}][{index}:v]overlay=0:0:format=auto,format=yuv420p[vout]")
        elif self.text_mode == "drawtext":
            parts.append(f"[{base}]{tx.drawtext_filters(text_layout, font=self._font)}"
                         ",format=yuv420p[vout]")
        else:
            parts.append(f"[{base}]{tx.scrim_filter(text_layout)},format=yuv420p[vout]")

        return ";".join(parts)

    def _background_chain(self, plan: VisualPlan, profile: RenderProfile) -> list[str]:
        """The solid brand-colour canvas every slide is built on.

        A ``color`` *source* inside filter_complex rather than a CLI input: it keeps
        the input indices stable (image is 0, text layers follow) and needs no file.
        A full-bleed image covers this entirely, but generating it anyway costs one
        allocation and keeps every layout on the same code path.
        """
        colour = ffmpeg_colour(self.theme.bg)
        return [f"color=c={colour}:s={profile.width}x{profile.height}:r={profile.fps}[bg]"]

    def _image_chain(
        self,
        *,
        src_size: tuple[int, int],
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        region: Region,
    ) -> list[str]:
        """Fit, animate and round the image *inside its region*, ending in ``[hero]``.

        The pre-upscale is relative to the **region**, not the frame, and is derived rather
        than fixed — see :func:`motion_canvas` for why, and for why the canvas it asks
        for is usually anamorphic.
        """
        static = plan.motion is Motion.STATIC
        sizing = motion_canvas(plan, region, frames, profile, src_size=src_size)

        parts = self._fit_chain(
            src_size,
            (region.width, region.height),
            sizing.fit,
            blur_fill=plan.layout is SlideLayout.FULL_BLEED,
        )

        if static:
            parts.append("[fit]null[moved]")
        else:
            if sizing.stretched:
                # Bilinear on purpose: this stretch adds no information, only a finer
                # integer grid for zoompan's x/y to land on, and lanczos here would cost
                # real time to ring a canvas that is about to be resampled back down.
                parts.append(
                    f"[fit]scale={sizing.canvas[0]}:{sizing.canvas[1]}:flags=bilinear[canvas]"
                )
                source = "canvas"
            else:
                source = "fit"
            self._warn_if_strobing(plan, region, frames)
            z, x, y = self._zoompan_expressions(plan, frames)
            parts.append(
                f"[{source}]zoompan=z='{z}':x='{x}':y='{y}'"
                f":d={frames}:fps={profile.fps}:s={region.width}x{region.height}[moved]"
            )
            logger.debug(
                "ken burns %s/%s: region %dx%d, travel %.1fpx over %d frames, fit %dx%d "
                "(detail %dx), canvas %dx%d (%.1f Mpx), slowest %.4fpx/frame, "
                "canvas step %.2fpx/frame, peak %.3fpx/frame",
                plan.layout.value, plan.motion.value, region.width, region.height,
                motion_travel(plan, region), frames, sizing.fit[0], sizing.fit[1],
                sizing.detail, sizing.canvas[0], sizing.canvas[1],
                sizing.canvas[0] * sizing.canvas[1] / 1e6,
                slowest_step(plan, region, frames),
                slowest_step(plan, region, frames) * sizing.canvas[0] / max(1, region.width),
                peak_step(plan, region, frames),
            )

        return parts + self._corner_chain(plan, region, profile)

    def _warn_if_strobing(self, plan: VisualPlan, region: Region, frames: int) -> None:
        """The opposite failure to stepping: a move fast enough to strobe.

        Reported rather than corrected. Slowing it down here would silently contradict the
        plan; the honest fixes — a smaller zoom span or a longer scene — are the planner's.
        """
        peak = peak_step(plan, region, frames)
        if peak > STROBE_STEP_PIXELS:
            logger.warning(
                "%s/%s moves %.1f output px per frame at its fastest (over %.1f is visible "
                "strobing): %.0fpx of travel across a %dx%d region in %d frames. Reduce the "
                "zoom span or lengthen the scene.",
                plan.layout.value, plan.motion.value, peak, STROBE_STEP_PIXELS,
                motion_travel(plan, region), region.width, region.height, frames,
            )

    def _corner_chain(
        self, plan: VisualPlan, region: Region, profile: RenderProfile
    ) -> list[str]:
        """Round the panel's corners: ``[moved]`` -> ``[hero]``. Shared by stills and clips."""
        radius = corner_radius(plan, region, profile, self.theme)
        if radius <= 0:
            return ["[moved]null[hero]"]

        # The mask is constant, so evaluate geq for exactly one frame and let `loop`
        # repeat it forever. Per-frame geq over a 900x900 panel doubles the CPU of the
        # whole clip; this way it costs one frame's worth, once.
        return [
            f"color=c=white:s={region.width}x{region.height}"
            f":r={profile.fps}:d={1.0 / profile.fps:.6f}[maskraw]",
            f"[maskraw]format=gray,geq=lum='{self._round_rect_expr(radius)}'"
            ",loop=loop=-1:size=1:start=0[mask]",
            "[moved]format=rgba[heroa];[heroa][mask]alphamerge[hero]",
        ]

    def _clip_chain(
        self,
        *,
        src_size: tuple[int, int],
        clip_duration: float,
        clip_fps: float,
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        region: Region,
    ) -> list[str]:
        """Fit a generated clip into its region, ending in ``[hero]``.

        Four things this does differently from a still, each for a stated reason:

        *No zoompan.* The footage already moves. A camera move on top of moving footage
        reads as seasick, so a clip scene is framed statically and the clip supplies all
        the movement — regardless of what ``plan.motion`` asks for.

        *fps conversion.* ``fps`` resamples the clip's 24 to the timeline's 30 by repeating
        one frame in five. That is a 20% duplicate-frame floor which no amount of care
        removes without motion interpolation (``minterpolate``, which costs more than the
        rest of the scene put together and invents motion that was never shot). Preserving
        real-time speed is worth a 4:5 cadence; see the report in ``scratchpad/motionfix``.

        *Looping.* The clip is shorter than the scene, so it repeats with a crossfade at
        the seam — see :data:`CLIP_SEAM_CROSSFADE` for the three options and the numbers.

        *No audio.* Only ``[0:v]`` is ever referenced, so the clip's own AAC track cannot
        reach the output even if the provider forgot to strip it.
        """
        fit = self._fit_chain(
            src_size,
            (region.width, region.height),
            (region.width, region.height),
            blur_fill=plan.layout is SlideLayout.FULL_BLEED,
        )
        # Normalise the cadence *before* the fit's label is consumed, so every looped
        # branch is already on the output frame rate and the xfade offsets below are exact.
        parts = [f"[0:v]fps={profile.fps},setpts=PTS-STARTPTS[clipsrc]"]
        parts += [part.replace("[0:v]", "[clipsrc]", 1) for part in fit]

        self._warn_if_clip_is_upscaled(src_size, region, plan)

        need = frames / profile.fps
        loops = clip_loop_count(clip_duration, need)
        if loops <= 1:
            parts.append("[fit]null[looped]")
        else:
            seam = clip_seam(clip_duration)
            branches = "".join(f"[cl{index}]" for index in range(loops))
            parts.append(f"[fit]split={loops}{branches}")
            accumulator = "cl0"
            for index in range(1, loops):
                offset = index * (clip_duration - seam)
                label = f"cx{index}"
                parts.append(
                    f"[{accumulator}][cl{index}]xfade=transition=fade"
                    f":duration={seam:.6f}:offset={offset:.6f}[{label}]"
                )
                accumulator = label
            parts.append(f"[{accumulator}]null[looped]")
            logger.info(
                "clip scene: %.3fs of footage covering %.3fs — %d passes crossfaded %.2fs "
                "at each seam (%d seams)",
                clip_duration, need, loops, seam, loops - 1,
            )

        # tpad can only ever add frames past the end, and -frames:v cuts to the exact
        # count, so the clip path lands on the same frame grid as the still path.
        parts.append(
            f"[looped]tpad=stop_mode=clone:stop_duration={CLIP_LOOP_SAFETY:.3f},"
            f"setpts=PTS-STARTPTS,setsar=1[moved]"
        )
        return parts + self._corner_chain(plan, region, profile)

    @staticmethod
    def _warn_if_clip_is_upscaled(
        src_size: tuple[int, int], region: Region, plan: VisualPlan
    ) -> None:
        """Flag a region the clip cannot fill at its native resolution.

        A 1280x720 clip has plenty of pixels for an ~857px-wide hero panel and not enough
        for a 1080p ``full_bleed``, where covering the frame is a 1.5x upscale. Worth
        saying out loud, because it is invisible in a filtergraph and obvious on screen.
        """
        src_w, src_h = src_size
        if src_w <= 0 or src_h <= 0:
            return
        cover = max(region.width / src_w, region.height / src_h)
        if cover > 1.0:
            logger.warning(
                "generated clip is %dx%d but %s needs %dx%d: covering the region is a %.2fx "
                "upscale, so the footage will be softer than a still would be",
                src_w, src_h, plan.layout.value, region.width, region.height, cover,
            )

    @staticmethod
    def _round_rect_expr(radius: int) -> str:
        """Luma mask for a rounded rectangle: 255 inside, 0 in the clipped corners.

        Distance is measured only from the corner arc centres, so the straight edges
        stay hard and only the corners curve.
        """
        dx = f"max(0,max({radius}-X,X-(W-1-{radius})))"
        dy = f"max(0,max({radius}-Y,Y-(H-1-{radius})))"
        return f"255*lte(pow({dx},2)+pow({dy},2),{radius}*{radius})"

    @staticmethod
    def _fit_chain(
        src: tuple[int, int],
        out: tuple[int, int],
        work: tuple[int, int],
        *,
        blur_fill: bool = True,
    ) -> list[str]:
        """Fit the still to the canvas without ever distorting it.

        Close aspect ratios: scale to *cover* and centre-crop the excess. Far apart
        (a portrait image in a landscape frame): blurred fill — a cover-cropped,
        blurred copy behind a fully contained foreground.

        ``blur_fill=False`` forces cover-crop regardless of mismatch. That is the right
        answer for a *bounded* image panel: the panel already sits on the brand
        background, so letterboxing it inside a blurred copy of itself just adds mush
        where the design wants a clean edge. Cropping a wide photo to a tall panel is
        the intent, not a compromise.
        """
        src_w, src_h = src
        out_w, out_h = out
        work_w, work_h = work
        src_ar = src_w / max(1, src_h)
        out_ar = out_w / max(1, out_h)
        mismatch = abs(src_ar - out_ar) / out_ar

        if not blur_fill or mismatch <= ASPECT_TOLERANCE:
            return [
                f"[0:v]scale={work_w}:{work_h}:force_original_aspect_ratio=increase"
                f":flags=lanczos,crop={work_w}:{work_h},setsar=1[fit]"
            ]

        sigma = max(6, round(out_w / 45))
        # Labels are prefixed `fit` because the slide's own solid background already
        # owns [bg] -- duplicate labels are a filtergraph parse error, not a warning.
        return [
            "[0:v]split=2[fitbgsrc][fitfgsrc]",
            # Blur at output size (cheap) and only then scale up to the work canvas.
            f"[fitbgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
            f":flags=lanczos,crop={out_w}:{out_h},gblur=sigma={sigma}:steps=2,"
            f"scale={work_w}:{work_h}[fitbg]",
            f"[fitfgsrc]scale={work_w}:{work_h}:force_original_aspect_ratio=decrease"
            f":flags=lanczos[fitfg]",
            "[fitbg][fitfg]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[fit]",
        ]

    @staticmethod
    def _zoompan_expressions(plan: VisualPlan, frames: int) -> tuple[str, str, str]:
        """Build the ``z``/``x``/``y`` expressions.

        Easing lives *in the expression* — see :func:`eased_progress` — because zoompan has
        no easing of its own and a linear ramp starts and stops abruptly. Every expression
        is emitted inside single quotes by the caller so commas and colons in future
        expressions cannot break the filtergraph.
        """
        last = max(1, frames - 1)
        t = f"(on/{last})" if frames > 1 else "(0)"
        progress = eased_progress(t, plan.easing)

        centre_x, centre_y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"

        if plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
            delta = round(plan.zoom_to - plan.zoom_from, 6)
            z = f"({plan.zoom_from}+({delta})*{progress})"
            return z, centre_x, centre_y

        held = max(plan.zoom_from, MIN_PAN_ZOOM)
        if plan.motion is Motion.PAN_RIGHT:
            return f"({held})", f"((iw-iw/zoom)*{progress})", centre_y
        if plan.motion is Motion.PAN_LEFT:
            return f"({held})", f"((iw-iw/zoom)*(1-{progress}))", centre_y
        return "(1)", "(0)", "(0)"

    # -------------------------------------------------------- text animation

    def _text_chain(
        self, scene_text: SceneText, *, base: str, first_input: int, fps: int
    ) -> tuple[list[str], str]:
        """One prep filter + one overlay stage per layer, back to front.

        Returns the filter list and the label of the composited result.
        ``sorted_layers()`` decides z-order: the scrim/background panel first so it
        lands *under* the type it exists to make legible.
        """
        parts: list[str] = []
        previous = base
        for index, layer in enumerate(scene_text.sorted_layers()):
            source = f"{first_input + index}:v"
            prepped = f"tl{index}"
            parts.append(f"[{source}]{self._layer_prep(layer, fps=fps)}[{prepped}]")

            x_expr, y_expr = self._anim_position(layer)
            animated = "(" in x_expr or "(" in y_expr
            options = [
                f"x='{x_expr}'",
                f"y='{y_expr}'",
                "format=auto",
                f"eval={'frame' if animated else 'init'}",
                f"enable='{self._visibility_expr(layer, fps=fps)}'",
            ]
            label = f"ov{index}"
            parts.append(f"[{previous}][{prepped}]overlay={':'.join(options)}[{label}]")
            previous = label
        return parts, previous

    @classmethod
    def _layer_prep(cls, layer: TextLayer, *, fps: int) -> str:
        """Filters applied to the layer's own PNG stream before it is overlaid.

        ``fade=...:alpha=1`` is the load-bearing part: with ``t=in`` and ``st>0`` the
        alpha channel is held at **zero** for every frame before ``st``, so the layer
        genuinely does not exist until ``appear_at`` — the classic bug here is the PNG
        being visible from frame 0. ``format=rgba`` first, because fade can only touch
        an alpha channel that is actually there.
        """
        frame = 1.0 / max(1, fps)
        chain = ["format=rgba"]

        if layer.animation is TextAnimation.TYPEWRITER and ff.has_filter("geq"):
            chain.append(cls._wipe_filter(layer))

        chain.append(
            f"fade=t=in:st={layer.appear_at:.4f}:d={cls._fade_in(layer, fps=fps):.4f}:alpha=1"
        )
        if layer.disappear_at is not None:
            out_d = max(frame, min(layer.anim_duration, 0.5))
            chain.append(f"fade=t=out:st={layer.disappear_at:.4f}:d={out_d:.4f}:alpha=1")
        return ",".join(chain)

    @staticmethod
    def _fade_in(layer: TextLayer, *, fps: int) -> float:
        """How long the opacity ramp lasts.

        NONE and TYPEWRITER get a single frame — long enough to guarantee the
        pre-``appear_at`` transparency, short enough to read as instant (NONE) or to
        leave the reveal to the wipe (TYPEWRITER). POP fades faster than it moves,
        which is what makes it feel like a snap rather than a drift.
        """
        frame = 1.0 / max(1, fps)
        if layer.animation in (TextAnimation.NONE, TextAnimation.TYPEWRITER):
            return frame
        span = layer.anim_duration
        if layer.animation is TextAnimation.POP:
            span *= POP_FADE_FRACTION
        return max(frame, span)

    @staticmethod
    def _visibility_expr(layer: TextLayer, *, fps: int) -> str:
        """``enable`` gate for the overlay stage.

        Purely an optimisation and a belt-and-braces guard: ``enable`` cannot
        interpolate, so the fade above is what actually animates. Skipping the
        overlay entirely while a layer is invisible matters when a slide has seven
        of them.
        """
        if layer.disappear_at is None:
            return f"gte(t,{layer.appear_at:.4f})"
        tail = layer.disappear_at + max(1.0 / max(1, fps), min(layer.anim_duration, 0.5))
        return f"between(t,{layer.appear_at:.4f},{tail:.4f})"

    @classmethod
    def _anim_position(cls, layer: TextLayer) -> tuple[str, str]:
        """``overlay`` x/y for the layer — a constant, or an expression in ``t``.

        Every expression is clamped so it lands *exactly* on the final position once
        the animation is over and stays there; the layer's resting place is never a
        function of how it got there.
        """
        x, y = str(layer.x), str(layer.y)
        progress = cls._progress_expr(layer)

        if layer.animation is TextAnimation.SLIDE_UP:
            eased = cls._smoothstep(progress)
            return x, f"({layer.y}+({layer.slide_distance})*(1-{eased}))"
        if layer.animation is TextAnimation.SLIDE_LEFT:
            eased = cls._smoothstep(progress)
            return f"({layer.x}+({layer.slide_distance})*(1-{eased}))", y
        if layer.animation is TextAnimation.POP:
            # No scale animation is available (see module notes), so the overshoot is
            # spatial: a short rise that goes slightly past the mark and settles back.
            offset = max(6, round(layer.slide_distance / 3))
            eased = cls._back_out(progress)
            return x, f"({layer.y}+({offset})*(1-{eased}))"
        return x, y

    @staticmethod
    def _progress_expr(layer: TextLayer) -> str:
        """0 before ``appear_at``, 1 after the animation, linear in between.

        The clamp is what pins the layer to its final position afterwards -- an
        unclamped ramp keeps travelling for the rest of the scene.
        """
        duration = max(1e-3, layer.anim_duration)
        return f"min(1,max(0,(t-{layer.appear_at:.4f})/{duration:.4f}))"

    @staticmethod
    def _smoothstep(progress: str) -> str:
        """``3p^2-2p^3``: zero velocity at both ends. A linear slide looks mechanical."""
        return f"({progress}*{progress}*(3-2*{progress}))"

    @staticmethod
    def _back_out(progress: str) -> str:
        """Overshoots ~10% past 1 around p=0.75 and returns to exactly 1 at p=1."""
        c1 = POP_OVERSHOOT_C1
        c3 = c1 + 1
        shifted = f"({progress}-1)"
        return f"(1+{c3:.5f}*pow({shifted},3)+{c1:.5f}*pow({shifted},2))"

    @staticmethod
    def _wipe_filter(layer: TextLayer) -> str:
        """Left-to-right reveal of the layer's alpha channel.

        This is a **wipe, not a true typewriter**: without ``drawtext`` there is no
        glyph-level clock to step through, so the rasterised line is uncovered by a
        moving vertical edge. It reads as typing for short lines and as a wipe for
        long ones. ``geq`` is evaluated per pixel per frame, which is why only the
        alpha plane is computed and why the layer PNGs are line-sized, not full-frame.
        """
        duration = max(1e-3, layer.anim_duration)
        reveal = f"W*min(1,max(0,(T-{layer.appear_at:.4f})/{duration:.4f}))"
        return (
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
            f":a='if(lt(X,{reveal}),alpha(X,Y),0)'"
        )

    # -------------------------------------------------------------- branding
    #
    # The watermark is composited **once, over the finished chain** — never per scene.
    # That is the whole design point. A logo burnt into each scene clip is an input to
    # every `xfade`, so at each boundary the outgoing copy fades out while the incoming
    # one fades in; because the two are pixel-identical and `xfade` is a linear blend of
    # *frames*, not of layers, the result still dips wherever the crossfade curve does not
    # sum to one. On screen that is a logo that pulses at every cut. Applying it after the
    # last xfade means there is exactly one copy and nothing to blend it against.

    def logo_png(self, profile: RenderProfile, workdir: Path) -> Path | None:
        """The rasterised, opacity-baked watermark for this profile, or ``None``.

        Cached in ``workdir`` (the job directory) and keyed on the source file, the target
        height and the opacity, so a re-render or a second assemble reuses it.
        """
        if self.logo_source is None:
            return None
        return tx.rasterise_logo(
            self.logo_source,
            tx.logo_height(profile, self.theme),
            self.theme.logo_opacity,
            Path(workdir),
        )

    def logo_region(self, profile: RenderProfile, png: Path | None = None) -> Region:
        """The watermark's box in output pixels.

        The *offsets* are forced even for the same reason the image panel's are: an odd
        overlay offset lands the layer on a half-pixel of the subsampled chroma plane and
        softens its edges. The *size* is left exactly as rasterised — unlike the image
        panel this is not a scale target, it is a measurement of a file that already
        exists, and rounding it down under-reports the box the collision check needs.
        """
        size: tuple[int, int] | None = None
        if png is not None:
            try:
                size = ff.probe_image_size(png)
            except (ff.FFmpegError, OSError, KeyError, ValueError):
                size = None
        rect = tx.logo_rect(profile, self.theme, size=size)
        return Region(_even(rect.x), _even(rect.y), max(1, rect.width), max(1, rect.height))

    def logo_conflicts(self, timeline: Timeline, region: Region) -> list[str]:
        """Scenes whose *ink* would overlap the watermark. One string per collision.

        Reported rather than resolved. Nudging the logo per scene would make it move, which
        is exactly what a persistent brand mark must not do, and shrinking the text column
        for one slide would break the deck's rhythm. So the answer is to say so loudly and
        let the geometry be corrected upstream.

        Compared against :meth:`~app.render.text_overlay.SlidePlan.ink_rects`, not the
        layer canvases: a centred bullet's canvas spans the whole column while its words sit
        in the middle, and comparing canvases would cry wolf on every full-bleed slide.
        """
        box = tx.Rect(region.x, region.y, region.width, region.height)
        conflicts: list[str] = []
        for scene in timeline.scenes:
            if scene.plan is None:
                continue
            try:
                slide = tx.layout_slide(
                    scene.heading,
                    scene.bullets,
                    scene.plan,
                    timeline.profile,
                    theme=self.theme,
                    font=self._font,
                )
            except (tx.FontNotFoundError, OSError) as exc:
                logger.debug("could not lay out scene %s for the logo check (%s)", scene.id, exc)
                continue
            hits = [rect for rect in slide.ink_rects() if rect.intersects(box)]
            if hits:
                conflicts.append(
                    f"scene {scene.id} ({scene.plan.layout.value}/"
                    f"{scene.plan.text_position.value}): text at "
                    f"{[r.as_tuple() for r in hits]} overlaps the logo at {box.as_tuple()}"
                )
        return conflicts

    # --------------------------------------------------------------- assemble

    def assemble(self, timeline: Timeline, out_path: Path) -> Path:
        """Chain the scene clips with xfade, lock narration to them, duck the music."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scenes = timeline.scenes
        if not scenes:
            raise RenderError("timeline has no scenes")

        clips: list[Path] = []
        for scene in scenes:
            if not scene.clip_path or not Path(scene.clip_path).is_file():
                raise RenderError(f"scene {scene.id} has no rendered clip; call render_scene first")
            clips.append(Path(scene.clip_path))

        profile = timeline.profile
        # Real clip durations, not nominal ones: xfade offsets must line up with the
        # frames that actually exist, otherwise the crossfades creep.
        clip_durations = [ff.probe_duration(c) for c in clips]

        inputs: list[str | Path] = []
        for clip in clips:
            inputs += ["-i", clip]

        audio_index: dict[int, int] = {}
        next_index = len(clips)
        for position, scene in enumerate(scenes):
            if scene.audio_path and Path(scene.audio_path).is_file():
                inputs += ["-i", scene.audio_path]
                audio_index[position] = next_index
                next_index += 1
            elif scene.audio_path:
                logger.warning("scene %s narration missing: %s", scene.id, scene.audio_path)

        music_index: int | None = None
        if timeline.music_path and Path(timeline.music_path).is_file():
            inputs += ["-stream_loop", "-1", "-i", timeline.music_path]
            music_index = next_index
            next_index += 1
        elif timeline.music_path:
            logger.warning("music track missing: %s", timeline.music_path)

        # The watermark is a *single-frame* input on purpose: `overlay`'s default
        # `eof_action=repeat` holds that one frame for the whole video, so the logo is
        # present on every frame without introducing an unbounded `-loop 1` stream into a
        # pass that has no `-frames:v` to stop it.
        logo = self.logo_png(profile, out_path.parent)
        logo_input: int | None = None
        logo_box: Region | None = None
        if logo is not None:
            inputs += ["-i", logo]
            logo_input = next_index
            next_index += 1
            logo_box = self.logo_region(profile, logo)
            for conflict in self.logo_conflicts(timeline, logo_box):
                logger.warning("brand logo collides with slide text: %s", conflict)
            logger.info(
                "branding %s with %s at %dx%d+%d+%d, opacity %.2f",
                out_path.name,
                logo.name,
                logo_box.width,
                logo_box.height,
                logo_box.x,
                logo_box.y,
                self.theme.logo_opacity,
            )

        video_parts, starts, video_length = self._video_chain(
            timeline, clip_durations, logo_input=logo_input, logo_box=logo_box
        )
        graph = list(video_parts)

        audio_parts, audio_label = self._audio_chain(
            timeline,
            starts=starts,
            clip_durations=clip_durations,
            audio_index=audio_index,
            music_index=music_index,
            total=video_length,
        )
        graph += audio_parts

        maps: list[str | Path] = ["-map", "[vout]"]
        if audio_label:
            maps += ["-map", audio_label]

        expected = timeline.final_duration()
        logger.info(
            "assemble: %d clips, sum(clips)=%.4fs, overlap=%.4fs, "
            "chain length=%.4fs, Timeline.final_duration()=%.4fs",
            len(clips),
            sum(clip_durations),
            sum(clip_durations) - video_length,
            video_length,
            expected,
        )

        ff.ffmpeg(
            [
                *inputs,
                "-filter_complex",
                ";".join(graph),
                *maps,
                "-r",
                str(profile.fps),
                *self._video_encode_args(profile),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(AUDIO_RATE),
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )

        actual = ff.probe_duration(out_path)
        tolerance = max(0.1, (len(scenes) + 2) / profile.fps)
        drift = actual - expected
        message = (
            f"assembled {out_path.name}: actual={actual:.4f}s "
            f"final_duration()={expected:.4f}s drift={drift:+.4f}s "
            f"(tolerance {tolerance:.4f}s)"
        )
        if abs(drift) > tolerance:
            if self.strict_duration:
                raise DurationMismatchError(message)
            logger.warning(message)
        else:
            logger.info(message)
        return out_path

    def _video_chain(
        self,
        timeline: Timeline,
        clip_durations: list[float],
        *,
        logo_input: int | None = None,
        logo_box: Region | None = None,
    ) -> tuple[list[str], list[float], float]:
        """xfade/concat chain. Returns (filters, per-scene start times, total length).

        ``starts[i]`` is where scene *i* begins **in the assembled video**, which is
        also exactly the xfade offset used to bring it in. The narration is delayed by
        the same value, which is what keeps voice and picture together.

        The watermark, when there is one, is overlaid as the very last stage — after the
        fades, not before them. Before them it would dip with the opening fade-up and the
        closing fade-out, and "constant" is the requirement. It adds no filter that has a
        duration, so the chain length this returns is unchanged by it.
        """
        scenes = timeline.scenes
        profile = timeline.profile
        parts: list[str] = []
        for index in range(len(scenes)):
            parts.append(
                f"[{index}:v]fps={profile.fps},scale={profile.width}:{profile.height},"
                f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS[c{index}]"
            )

        acc = "c0"
        length = clip_durations[0]
        starts = [0.0]
        for index in range(1, len(scenes)):
            plan = scenes[index].plan
            transition = plan.transition_in if plan else Transition.DISSOLVE
            duration = plan.transition_duration if plan else 0.0
            label = f"x{index}"
            starts.append(length if transition is Transition.CUT else length - duration)
            if transition is Transition.CUT or duration <= 0:
                parts.append(f"[{acc}][c{index}]concat=n=2:v=1:a=0[{label}]")
                length += clip_durations[index]
            else:
                offset = length - duration
                parts.append(
                    f"[{acc}][c{index}]xfade=transition={transition.value}"
                    f":duration={duration:.6f}:offset={offset:.6f}[{label}]"
                )
                length += clip_durations[index] - duration
            acc = label

        # Scene 0's transition_in is a fade up from black, not a crossfade between
        # clips, so it costs no duration -- final_duration() deliberately ignores it.
        tail: list[str] = []
        first_plan = scenes[0].plan
        if first_plan and first_plan.transition_in in (Transition.FADE, Transition.DISSOLVE):
            fade_in = min(max(first_plan.transition_duration, 0.3), max(0.05, length / 4))
            tail.append(f"fade=t=in:st=0:d={fade_in:.3f}")
        if self.final_fade_out and length > 2 * FINAL_FADE_OUT:
            tail.append(f"fade=t=out:st={length - FINAL_FADE_OUT:.3f}:d={FINAL_FADE_OUT}")
        if self.burn_captions:
            tail.append(self._caption_filter(timeline))

        if logo_input is None or logo_box is None:
            tail.append("format=yuv420p")
            parts.append(f"[{acc}]" + ",".join(tail) + "[vout]")
            return parts, starts, length

        if not tail:
            # A filterchain cannot be empty, and with no fades and no captions it would be.
            tail.append("null")
        parts.append(f"[{acc}]" + ",".join(tail) + "[faded]")
        # Scaled and pre-multiplied by ImageMagick at rasterisation time, so there is
        # nothing left to do here but place it. `eof_action=repeat` is the default and is
        # what makes one PNG frame cover the whole timeline.
        parts.append(f"[{logo_input}:v]format=rgba,setsar=1[logo]")
        parts.append(
            f"[faded][logo]overlay=x={logo_box.x}:y={logo_box.y}"
            ":format=auto:eval=init:eof_action=repeat,format=yuv420p[vout]"
        )
        return parts, starts, length

    def _caption_filter(self, timeline: Timeline) -> str:
        if not ff.has_filter("subtitles"):
            raise RenderError(
                f"burn_captions=True but {ff.ffmpeg_bin()} has no 'subtitles' filter "
                "(built without libass)"
            )
        ass_path = Path(get_settings().job_dir(timeline.job_id)) / "captions.ass"
        captions_mod.write_ass(timeline, ass_path)
        return captions_mod.burn_filter(ass_path)

    def _audio_chain(
        self,
        timeline: Timeline,
        *,
        starts: list[float],
        clip_durations: list[float],
        audio_index: dict[int, int],
        music_index: int | None,
        total: float,
    ) -> tuple[list[str], str | None]:
        """Narration locked to the *video* clock, plus a ducked music bed.

        Each narration segment is trimmed/silence-padded to its scene's clip length and
        delayed to that scene's position in the assembled video. Segments therefore
        overlap by exactly the crossfade duration, which is what we want: the voice
        stays glued to the picture instead of drifting later with every transition.
        """
        parts: list[str] = []
        labels: list[str] = []
        common = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"

        for position, input_index in sorted(audio_index.items()):
            segment = clip_durations[position]
            delay_ms = int(round(starts[position] * 1000))
            chain = [
                common,
                f"atrim=0:{segment:.6f}",
                f"apad=whole_dur={segment:.6f}",
            ]
            if segment > 4 * CLICK_FADE:
                chain.append(f"afade=t=out:st={segment - CLICK_FADE:.6f}:d={CLICK_FADE}")
            if delay_ms > 0:
                chain.append(f"adelay=delays={delay_ms}:all=1")
            label = f"a{position}"
            parts.append(f"[{input_index}:a]" + ",".join(chain) + f"[{label}]")
            labels.append(label)

        def master(parts: list[str], label: str | None) -> tuple[list[str], str | None]:
            """Append the master bus, so every path is normalised the same way."""
            if label is None:
                return parts, None
            chain, out = self._master_chain(label, total=total)
            return parts + chain, out

        narration: str | None = None
        if len(labels) == 1:
            parts.append(f"[{labels[0]}]atrim=0:{total:.6f},apad=whole_dur={total:.6f}[narr]")
            narration = "narr"
        elif labels:
            joined = "".join(f"[{label}]" for label in labels)
            parts.append(
                f"{joined}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
                f"atrim=0:{total:.6f},apad=whole_dur={total:.6f}[narr]"
            )
            narration = "narr"

        if music_index is None:
            return master(parts, narration)

        fade_in = min(MUSIC_FADE_IN, max(0.1, total / 4))
        fade_out = min(MUSIC_FADE_OUT, max(0.1, total / 4))
        music_chain = [
            common,
            f"atrim=0:{total:.6f}",
            f"apad=whole_dur={total:.6f}",
            f"volume={self.music_duck_db}dB",
            f"afade=t=in:st=0:d={fade_in:.3f}",
            f"afade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out:.3f}",
        ]
        parts.append(f"[{music_index}:a]" + ",".join(music_chain) + "[music]")

        if narration is None:
            return master(parts, "music")

        voice, bed = narration, "music"
        if ff.has_filter("sidechaincompress"):
            # Static -18 dB alone still fights the voice on louder passages; the
            # sidechain dips the bed only while narration is actually present.
            parts.append(f"[{narration}]asplit=2[narr_out][narr_key]")
            parts.append(
                "[music][narr_key]sidechaincompress=threshold=0.03:ratio=8"
                ":attack=20:release=400:makeup=1:detection=rms:level_sc=1[music_ducked]"
            )
            voice, bed = "narr_out", "music_ducked"
        else:
            logger.warning("no sidechaincompress filter; music uses static ducking only")

        # Summing (normalize=0) preserves the narration level but can clip on peaks; the
        # limiter that catches that now lives on the master bus, after loudnorm, so it is
        # the last thing to touch the signal rather than the second-to-last.
        parts.append(f"[{voice}][{bed}]amix=inputs=2:normalize=0:duration=first[aout]")
        return master(parts, "aout")

    def _master_chain(self, label: str, *, total: float) -> tuple[list[str], str]:
        """Normalise the finished mix to :data:`LOUDNESS_TARGET_LUFS` and cap its peaks.

        Single pass. ``loudnorm`` can also be run twice — measure, then apply the measured
        offsets — which is exact; one pass is approximate but lands within a fraction of a
        dB and does not require decoding the whole programme twice for a value nothing
        downstream reads. Approximate is the right trade here.

        The ``atrim``/``apad`` pair is not decoration, and on its own it is not enough.
        ``loudnorm`` runs a lookahead buffer and hands back a stream whose timestamps it has
        re-based off its own internal block clock rather than passing the input's through.
        ``atrim`` selects on *timestamps*, so once loudnorm has moved them the trim window
        no longer lines up with the samples and the clamp leaks. Measured on the reference
        render: ``loudnorm`` alone put **+0.066992s (+2.01 frames at 30fps)** past a
        ``atrim=0:74.633008``, and because a container is as long as its longest stream that
        landed as the whole file's 2.4-frame drift. ``alimiter`` is duration-neutral and the
        logo overlay was innocent — both were checked separately.

        ``asetpts=N/SR/TB`` regenerates timestamps from the running sample count, which puts
        the stream back on a sample-exact clock before it is clamped. With it the same graph
        lands on 74.633000s, i.e. exactly the length asked for.
        """
        chain: list[str] = []
        if ff.has_filter("loudnorm"):
            chain.append(
                f"loudnorm=I={LOUDNESS_TARGET_LUFS:g}"
                f":TP={LOUDNESS_TRUE_PEAK_DB:g}"
                f":LRA={LOUDNESS_RANGE:g}"
            )
        else:
            logger.warning(
                "no loudnorm filter in %s; shipping un-normalised audio", ff.ffmpeg_bin()
            )
        if ff.has_filter("alimiter"):
            # Matched to the loudnorm ceiling rather than left at a nominal 0.97, so the
            # two agree about what "true peak" means instead of one undoing the other.
            limit = 10 ** (LOUDNESS_TRUE_PEAK_DB / 20)
            chain.append(f"alimiter=limit={limit:.4f}:level=disabled")
        # Order is load-bearing: rebase the clock, *then* clamp against it.
        chain += [
            "asetpts=N/SR/TB",
            f"atrim=0:{total:.6f}",
            f"apad=whole_dur={total:.6f}",
        ]
        return [f"[{label}]" + ",".join(chain) + "[amaster]"], "[amaster]"

    # ----------------------------------------------------------------- encode

    @staticmethod
    def _thread_args(profile: RenderProfile) -> list[str]:
        """Per-process thread cap. Empty list means "let ffmpeg decide".

        ``render_all`` divides the machine between its workers and records the share in
        ``profile.encoder_threads``; the scene encode is the process that actually runs
        concurrently, so this is where the cap has to land. Without it four x264
        instances each claim every core and spend their time contending.
        """
        if not profile.encoder_threads:
            return []
        return ["-threads", str(profile.encoder_threads)]

    def _video_encode_args(self, profile: RenderProfile) -> list[str]:
        codec = profile.video_codec
        if not ff.has_encoder(codec):
            logger.warning("encoder %s unavailable in %s; using libx264", codec, ff.ffmpeg_bin())
            codec = "libx264"

        if codec.endswith("videotoolbox"):
            # VideoToolbox ignores crf; map it onto its 1-100 constant-quality scale.
            quality = max(20, min(100, int(round(100 - profile.crf * 1.8))))
            return [
                "-c:v",
                codec,
                "-q:v",
                str(quality),
                "-allow_sw",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        args = ["-c:v", codec, "-crf", str(profile.crf), "-pix_fmt", "yuv420p"]
        if codec in ("libx264", "libx265"):
            args += ["-preset", "medium", "-profile:v", "high" if codec == "libx264" else "main"]
        return args + self._thread_args(profile)


def frames_for(duration: float, fps: int) -> int:
    """Frame count for a duration — exported so callers can reason about exactness."""
    return max(1, int(round(duration * fps)))


def expected_assembled_duration(timeline: Timeline) -> float:
    """Mirror of ``Timeline.final_duration()`` computed from ``math.fsum`` for tests."""
    overlap = math.fsum(
        scene.plan.transition_duration
        for scene in timeline.scenes[1:]
        if scene.plan and scene.plan.transition_in is not Transition.CUT
    )
    return timeline.narration_duration - overlap
