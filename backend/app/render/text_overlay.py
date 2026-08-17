"""Rasterise slide text — headings, bullets, background, scrim — to positioned PNGs.

This module owns *what the text looks like and where it sits*. It hands
:class:`~app.render.contracts.SceneText` to the filtergraph builder, which owns *how
each layer enters the frame*. The seam is deliberately narrow: a PNG, a rectangle in
output-frame pixels, a time, and an animation name.

Three problems drive the design.

*No text filter.* ffmpeg's ``drawtext`` only exists when the build has libfreetype, and
the current Homebrew bottle on macOS does not — its formula dropped the dependency. So
every glyph here is rasterised by **ImageMagick** and composited with ``overlay``. The
``drawtext`` code path is kept for portability (see :func:`drawtext_filters`) but nothing
in the slide layout depends on it.

*Corporate slides are not full-bleed photos.* A training deck is a *designed frame*:
solid brand background, text column, and the image as a bounded hero element. That makes
:class:`~app.core.models.SlideLayout` — not the image — the thing that decides geometry.
:func:`slide_geometry` resolves one layout into rectangles; everything downstream reads
those rectangles.

*Legibility is measurable, so measure it.* On a solid background the contrast ratio of
text against ``Theme.bg`` is known exactly and no scrim is needed — a scrim over a flat
brand colour just reads as a smudge. Only ``full_bleed`` puts text over unknown pixels,
and there the scrim opacity is *derived* from the image: sample the luminance of the
region the text will actually occupy and solve for the opacity that clears WCAG AA
(4.5:1). A fixed opacity is what made white headings marginal over a sunlit sky.

*Uniformity is a hard invariant, not a preference.* The first render of this module was
rejected on sight for looking "all over the place", and the specific thing the viewer named
was two marker shapes in one list (a filled disc for the emphasised bullet, a hollow ring
for the rest). They were right, and the lesson generalises: within one video there is **one
type scale** (see ``TYPE_SCALE_RATIO``), **one marker shape** (``Theme.marker``), **one text
colour** (``Theme.uniform_text``), one rule width, one bullet pitch, and one fixed grid that
the heading and the first bullet land on whatever the copy does. ``docs/DIRECTION.md`` is
the normative source for those numbers; where a constant here disagreed with it, DIRECTION
won, and the constant says so.

Layer emission
--------------
Every scene gets exactly one ``kind="scrim"`` layer, and it is always full-frame RGBA:

* solid layouts — the brand background, with a **rounded-rect transparent hole** cut out
  where the hero image belongs. Overlaying it *on top of* the image therefore both paints
  the background and masks the image into a rounded card, whatever order the filtergraph
  composites in and however the image moves inside its region.
* ``full_bleed`` — a dark adaptive gradient over the text region only, transparent
  elsewhere.

Then one ``kind="heading"`` layer, one ``kind="bullet"`` layer per bullet, and — on a title
card only — one ``kind="kicker"``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.models import (
    BulletPoint,
    RenderProfile,
    SceneRole,
    SlideLayout,
    TextAnimation,
    TextPosition,
    Theme,
    VisualPlan,
)
from app.render import ffmpeg as ff
from app.render.contracts import SceneText, TextLayer

logger = logging.getLogger(__name__)

FONT_CANDIDATES: tuple[str, ...] = (
    # Plain .ttf first: some freetype builds choke picking a face out of a .ttc.
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux / container fallbacks.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)

HEAVIER_FACE_SUFFIXES: tuple[tuple[str, str], ...] = (
    # Arial Bold (700) -> Arial Black (900). The one substitution that matters on macOS,
    # because `find_font` already picks a Bold face as the *base* weight.
    ("Bold.ttf", "Black.ttf"),
    ("-Bold.ttf", "-Black.ttf"),
    ("Bold.ttc", "Black.ttc"),
    ("Regular.ttf", "Bold.ttf"),
    ("-Regular.ttf", "-Bold.ttf"),
    (".ttf", "-Bold.ttf"),
    (".ttf", " Bold.ttf"),
)
"""Filename rewrites tried, in order, to find a genuinely heavier face of the same family.

Weight is the *only* emphasis signal left (see :data:`EMPHASIS_MODE`), so it has to be a
real face: ImageMagick has no synthetic-bold switch, and faking it with a fill-coloured
stroke (see ``EMPHASIS_FAUX_BOLD_RATIO``) softens the letterforms. A file rewrite is used
rather than fontconfig because :func:`find_font` already addresses fonts by path.
"""

SAFE_AREA = 0.90
"""Text stays inside 90% of the frame — never touching an edge."""

HEADING_SIZE_RATIO = 0.058
"""Base cap height as a fraction of frame height (1080p -> ~63px)."""

MIN_HEADING_SIZE_RATIO = 0.034
LINE_SPACING = 1.22
AVG_GLYPH_RATIO = 0.52
"""Mean advance width / point size for a bold sans. Only used to pick a size and a
wrap point, so being a few percent off is harmless."""

MAX_LINES = 2
ELLIPSIS = "…"


# --------------------------------------------------------------- slide metrics
#
# Every size is a fraction of a 1920x1080 reference frame and is resolved against
# ``profile.width`` / ``profile.height``. A 960x540 draft is then a pure 0.5 scale of the
# 1080p final: same wrap points, same rhythm, same look.

REF_WIDTH = 1920
REF_HEIGHT = 1080

# ------------------------------------------------------------------ type scale
#
# One modular scale, keyed off `profile.width`, is the whole type system. Every point size
# on every slide is `base * ratio ** step` — so a 540p draft is a pure 0.5 scale of the
# 1080p final (same design, cheaper), and the role sizes are steps on one ladder instead of
# four unrelated constants that happen to look alright next to each other.
#
# Ratio **1.333** (a perfect fourth), base = the bullet. Numbers from `docs/DIRECTION.md`
# §2, which derives them from published legibility floors rather than from taste:
#
#     step -1    33px   kicker      (DIRECTION says 32; the ladder says 33, see below)
#     step  0    44px   bullet      >= the 40px 1080p body floor and the BBC 44px HD floor
#     step  1    59px   (unused)
#     step  2    78px   heading     78/44 = 1.77, clearing "titles >= 50% larger than body"
#     step  3   104px   ~ title     (SceneRole.TITLE is 1.35 * 78 = 105px)
#
# The previous scale had the bullet at 36px, which is under *both* published floors, and
# the heading at 63px from an unrelated constant. The title used a third constant,
# TITLE_HEADING_W_RATIO (86px), which DIRECTION §9 correctly calls two sources of truth for
# one number: it is deleted, and the title is `heading * SceneRole.heading_scale`.
#
# The 1px kicker discrepancy: DIRECTION's table gives 32px (ratio 0.727) where a pure
# 1/1.333 step is 33px. The ladder wins because being *on* the scale is the whole point of
# having one, and 1px at 1080p is under a third of a stroke width.

TYPE_SCALE_RATIO = 4 / 3
"""Step ratio of the modular scale. A perfect fourth. DIRECTION §2."""

TYPE_BASE_W_RATIO = 44 / REF_WIDTH
"""Step 0 of the scale, as a fraction of frame width. 44px at 1920, 22px at 960."""

TYPE_STEP_BODY = 0
"""Bullets and body copy."""

TYPE_STEP_HEADING = 2
"""Slide heading at ``SceneRole.CONTENT``. Roles multiply this by ``heading_scale``."""

TYPE_STEP_KICKER = -1
"""The title card's eyebrow. See :data:`TITLE_KICKER`."""

MIN_FONT_PX = 11
SHRINK_STEPS = (1.0, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64, 0.58)
"""Uniform down-scales tried, in order, until the whole stack fits its column.

A safety net against overflow, not a design tool — DIRECTION §2 is right that a bullet
shrunk to 21px is unreadable. The real fix is upstream: 22 characters for a heading and 34
for a bullet, asserted in the script provider's schema, at which point this ladder is never
walked past its first rung. Weakening the net here would trade an unreadable slide for a
clipped one, which is worse.
"""

MAX_HEADING_LINES = 2
"""DIRECTION §2.1 wants 1, enforced by capping generated headings at 22 characters.

Two, not one, until that cap exists: today's headings run to 27 characters, and at
``max_lines=1`` :func:`wrap_to_width` would ellipsise them — silently deleting words from
the slide's most important line. Two lines is the honest ceiling for uncapped copy, and it
is what the fixed heading box (see :func:`layout_slide`) is sized for, so the rule and the
first bullet do not move when a heading happens to wrap.
"""

MAX_BULLET_LINES = 2
"""Beyond this a "bullet" is a paragraph; ellipsise rather than wreck the rhythm.

Two rather than four: a bullet is capped at 34 characters (DIRECTION §2.1), so a third line
can only come from copy that already broke the rule.
"""

# ---- the grid (DIRECTION §6.2), as fractions of a 1920x1080 reference frame ----
#
# Every value on a hero slide is fixed. Nothing depends on the bullet count or the heading
# length, because "the text block is somewhere else on this slide" is a large part of what
# "all over the place" meant.

SOLID_MARGIN_X_RATIO = 104 / REF_WIDTH
SOLID_MARGIN_Y_RATIO = 90 / REF_HEIGHT
SOLID_GUTTER_RATIO = 88 / REF_WIDTH
TEXT_COLUMN_SHARE = 904 / 1624
"""Text column / (width left after margins and gutter). 904px of 1624 at 1920.

The image gets the other 720px, making the hero region **4:5** (720x900). That is a
requirement on the *image provider*, not a crop preference: stills generated at 16:9 lose
55% of their width to this region, which is why framing never matches the prompt
(DIRECTION §6.2, defect 16).
"""

HERO_TEXT_TOP_RATIO = 226 / REF_HEIGHT
"""Top of the hero text stack. Fixed, so nothing below it can move between scenes.

The stack is *top-anchored* rather than centred: a centred stack shifts every time the
bullet count or a wrap changes, which is what put the text block at three different heights
across four slides in the rejected video.

226px is the top of the two-line heading *box*, not of the heading's ink. The heading is
bottom-aligned inside that box (see :func:`layout_slide`), so a one-line heading — the normal
case — puts its cap-top at DIRECTION §6.2's y=330, its rule at y=450 (spec: 442) and the
first bullet at y=514 (spec: 494). A two-line heading grows *upward* into the air above,
leaving the rule and the whole bullet stack exactly where they were.
"""

HERO_TEXT_BOTTOM_RATIO = 990 / REF_HEIGHT
"""Bottom of the hero text column: the same baseline the hero image ends on."""

TITLE_MARGIN_X_RATIO = 0.1150
TITLE_MARGIN_Y_RATIO = 0.1500
TITLE_OPTICAL_CENTRE_RATIO = 0.48
"""The title block is centred on 0.48 of the frame height, not 0.50.

Optical centre sits slightly above geometric centre; a block centred on 540 reads as
hanging low. DIRECTION §6.3.
"""

TITLE_KICKER = "TRAINING MODULE"
"""Eyebrow above the title, or ``""`` to suppress it. DIRECTION §1.3.

A constant rather than a script field: it is a label for the *format*, not for the topic,
so there is nothing to generate and nothing to get wrong. It is the cheapest of the three
things that make a title card read as a title card (kicker, big centred title, rule) and
the only one that says "this is a training module" before a word is spoken.
"""

TITLE_KICKER_TRACKING_EM = 0.12
"""Letter-spacing on the kicker, in em. The only tracked text in the deck.

Uppercase at 33px needs it — tightly-set caps read as a single shape. Applied via
ImageMagick's ``-kerning``, which adds a constant advance between glyphs.
"""

BAND_HEIGHT_RATIO = 0.40
"""``image_band``: image occupies the top 40% of the frame, text the solid area below."""

LEFT_PANEL_SHARE = 0.55
"""``full_bleed`` + ``LEFT_PANEL``: text lives in the left 55%, subject stays right."""

THIRD_HEIGHT_RATIO = 0.45

BULLET_GAP_RATIO = 30 / 44
"""Gap between bullet blocks, in bullet point sizes. 30px at a 44px bullet.

With ``LINE_SPACING`` this sets the pitch: 54px line + 30px gap = **84px** between two
single-line bullets, and 54px more per wrapped line (DIRECTION §3.3). The pitch is measured
on the *type*, not on the canvas — see :func:`_bullet_pitch`.
"""

HEADING_GAP_RATIO = 1.15
"""Accent rule to the first bullet's cap-top, in bullet point sizes. 51px at 44px."""

RULE_GAP_RATIO = 0.44
"""Heading to its accent rule, in *heading* point sizes.

Keyed to the heading so one constant serves both sizes of heading the video has: 34px under
a 78px content heading and 46px under a 105px title, which is DIRECTION §6.2's 34px and
§6.3's 48px to within 2px.
"""

RULE_W_RATIO = 88 / REF_WIDTH
"""Accent rule width as a fraction of *frame* width — a fixed 88px at 1920.

Not a fraction of the text column, which is what made the same graphic element 266px wide
on the title card and 137px wide on a hero slide (DIRECTION §6.2, defect 14).
"""

TITLE_RULE_W_RATIO = 120 / REF_WIDTH
"""...and 120px under the title, which is the one slide allowed to differ. DIRECTION §6.3."""

RULE_HEIGHT_PX = 4
"""Rule thickness at 1920x1080, scaled by ``geometry.scale``."""
DESCENDER_PAD_RATIO = 0.34
"""Extra canvas below the last line, as a fraction of the point size.

See :func:`_descender_pad`. Three things live in this margin and all three were measured
rather than guessed: the font's internal leading pushes the first line down ~0.11em, the
descender drops below the last baseline, and the dark outline stroke adds its own width
under that. Getting this wrong clips the bottom row of a wrapped bullet — which showed up
at 540p first, because the floors on stroke width make small type relatively fatter.
"""

MARKER_SHAPES: tuple[str, ...] = ("disc", "ring", "chevron", "dash", "none")
"""Every shape :attr:`~app.core.models.Theme.marker` may name. One is used per video."""

MARKER_RATIO = 16 / 44
"""Disc diameter in bullet point sizes — 16px at a 44px bullet. DIRECTION §3.2.

Was 0.30, i.e. an 11px dot: small enough to read as dirt on the screen, large enough to be
noticed. 16px is the smallest mark that reads as a deliberate one at 1080p.
"""

BULLET_GUTTER_EM = 1.0
"""Marker gutter width in bullet point sizes — a fixed 44px. DIRECTION §3.2.

Fixed, not "marker width + gap": the text edge then lands on the same x for every bullet of
every scene whatever shape the theme picked, and a wrapped line hangs to that same edge.
"""

MARKER_DASH_W_RATIO = 20 / 44
"""``dash`` ink width in bullet point sizes — 20px at 44px."""

MARKER_DASH_H_RATIO = 2 / 44
"""``dash`` ink height — 2px at 44px, floored at 2px so it survives a 540p draft."""

MARKER_RING_RATIO = 0.26
"""Ring thickness, as a fraction of the marker diameter — used only by ``marker="ring"``.

This is a property of the *shape*, not of emphasis. It used to be the emphasis cue: a
hollow ring for a normal bullet, a filled disc for the emphasised one. Rejected on sight
by the first person to watch a render — two marker shapes in one list reads as sloppiness,
not as hierarchy, and it costs more credibility than the hierarchy was ever worth. The
shape now comes from :attr:`~app.core.models.Theme.marker` and is the same for every
bullet in the video.
"""

MARKER_CHEVRON_RATIO = 0.20
"""Stroke width of ``marker="chevron"``, as a fraction of its diameter."""

HEADING_STROKE_RATIO = 0.075
BULLET_STROKE_RATIO = 0.060

EMPHASIS_MODE = os.environ.get("VIDEO_EMPHASIS", "off").strip().lower() or "off"
"""``off`` | ``weight`` — how an ``emphasis=True`` bullet differs from its neighbours.

**Default off, on the evidence.** Colour is off the table (``uniform_text``) and so is
shape (one marker per video), which leaves weight. Proofed at 1920x1080 on a real slide:
Arial Bold vs Arial Black at 36px is a 22% difference in ink mass, but in a stack where
the two bullets are 60px apart vertically and no two words are the same, nothing announces
which one is heavier — it reads as one line rendered slightly differently, i.e. as a
rendering wobble. It only became *visible* in the old build because it also carried a size
bump and a fatter halo, and those are exactly what made the list look non-uniform.

So a uniform list that reads cleanly wins over a hierarchy nobody notices. ``weight``
remains available for one line of copy that genuinely has to lead — set ``VIDEO_EMPHASIS``
or pass ``emphasis_mode=`` — and it is weight *only*: same size, same outline, same marker.
"""

EMPHASIS_MODES = ("off", "weight")

EMPHASIS_FAUX_BOLD_RATIO = 0.030
"""Fill-coloured stroke used to fake a heavier weight, as a fraction of the point size.

Only applied in ``weight`` mode when :func:`heavier_font` finds nothing. Stroking in the
*fill* colour is the only way to genuinely thicken a letterform in ImageMagick; it costs a
little crispness at the joins, which is why a real face is always preferred.
"""

SLIDE_DISTANCE_RATIO = 12 / REF_WIDTH
"""Pixels the *heading* travels on a SLIDE_* entrance. 12px at 1920. DIRECTION §4.1.

Was 60px. 60px is a swipe, not an entrance — and on a left-aligned list a horizontal one
sweeps the text through the marker gutter. A short rise is a settle: present, not noticed,
which is the right trade for material that has to be read rather than admired.
"""

BULLET_SLIDE_DISTANCE_RATIO = 8 / REF_WIDTH
"""...and 8px for a bullet, which is smaller type and should move less. DIRECTION §4.1."""

KICKER_ANIM_DURATION = 0.35
"""Ceiling on the kicker's fade, in seconds. DIRECTION §4.1."""

BULLET_ANIM_DURATION = 0.28
"""Ceiling on a bullet's entrance, in seconds. DIRECTION §4.1.

``VisualPlan`` carries one ``anim_duration`` for the whole scene, but a 44px bullet moving
for as long as a 78px heading is slower per pixel and reads as sluggish. The layer contract
is per-layer, so the split is made here: the heading keeps the plan's duration, a bullet is
capped at 0.28s — eight frames at 30fps, the shortest that still eases perceptibly.
"""

FIRST_REVEAL_EARLIEST = 1.15
"""Floor on the first bullet's reveal, in seconds from scene start. DIRECTION §4.2.

The heading's entrance (0.45s image fade + 0.40s heading) has to finish before anything
else moves, plus 0.30s to let the heading be read. A bullet arriving at 0.4s lands while the
heading is still travelling and both movements are lost.
"""

WCAG_AA = 4.5
"""Target text-to-background contrast ratio. AA for normal-size text."""

SCRIM_PROBE_SIGMA = 1.0
"""Luminance probe = mean +/- 1 sigma. A sunlit image has a high mean *and* a fat tail;
solving against the mean alone is exactly how a fixed 0.45 scrim ended up marginal.

Which tail binds depends on the palette's polarity. Light text is killed by the *bright*
tail, so a dark scrim solves against ``mean + sigma``; dark text on a light palette is
killed by the *dark* tail, so a white scrim solves against ``mean - sigma``. Using the
bright tail for both is how a light theme would silently ship illegible text.
"""

SCRIM_OPACITY_FLOOR = 0.0
SCRIM_OPACITY_CEIL = 0.90

SCRIM_MIN_TINT = 0.12
"""Floor applied to a *solved* opacity, as policy rather than arithmetic.

A genuinely dark still needs no scrim at all, but letting the opacity go to zero makes
the tint pop between consecutive scenes, and it leaves nothing in reserve for the local
highlight that a mean-plus-sigma probe still under-weights. A slight constant tint costs
nothing and keeps the deck looking like one deck.
"""
SCRIM_FEATHER_RATIO = 0.12
"""Width of the gradient that feathers the scrim out, as a fraction of frame width."""

CARD_FRAME_RATIO = 0.0028
"""Surface-coloured frame drawn around the hero image hole. ~5px at 1920."""


class FontNotFoundError(RuntimeError):
    pass


# ------------------------------------------------------------------ type scale


def type_size(step: int, width: int) -> int:
    """Point size for ``step`` of the modular scale at a frame ``width``, in pixels.

    The single source of every point size in the module. ``step`` 0 is body copy; see the
    ladder in the constants above.
    """
    return max(MIN_FONT_PX, round(width * TYPE_BASE_W_RATIO * TYPE_SCALE_RATIO**step))


def heading_size_for(role: SceneRole, width: int) -> int:
    """Heading point size for one role: the scale's heading step times ``heading_scale``.

    ``SceneRole.heading_scale`` is the contract (``app.core.models``), so it is read from
    there rather than restated here.
    """
    return max(MIN_FONT_PX, round(type_size(TYPE_STEP_HEADING, width) * role.heading_scale))


# ------------------------------------------------------------------ role styling
#
# What a role does to a slide, beyond the two numbers the contract already carries
# (`heading_scale`, `bullet_budget`). Everything here is about the *frame*: how the stack
# is aligned, how much air it gets, and how long the accent rule is. A title card is not a
# content slide with bigger type — it is centred, has nothing under the rule, and gets its
# breathing room from a wider margin.


@dataclass(frozen=True)
class RoleStyle:
    """Per-role treatment. One entry per :class:`~app.core.models.SceneRole`.

    Note how little varies. Three of the four roles are byte-identical, and the fourth is
    the title card. That is the design: a summary and a closing earn their difference from
    having fewer bullets and different words, not from a different frame. Four heading sizes
    and four rule widths in one video is the opposite of what was asked for.
    """

    align: str | None = None
    """``left`` | ``center``, or ``None`` to keep whatever the layout chose."""

    vertical_anchor: str | None = None
    rule_w_ratio: float = RULE_W_RATIO
    """Rule width as a fraction of frame width."""

    kicker: bool = False
    """Emit :data:`TITLE_KICKER` above the heading."""

    optical_centre_ratio: float | None = None
    """Centre the stack on this fraction of the frame height instead of on the column."""


ROLE_STYLES: dict[SceneRole, RoleStyle] = {
    # Centred, optically centred, a wider rule, and a kicker. Those four things plus 105px
    # type are what make it read as a title card rather than as a content slide whose image
    # failed to generate — which is exactly how the rejected opener read.
    SceneRole.TITLE: RoleStyle(
        align="center",
        vertical_anchor="center",
        rule_w_ratio=TITLE_RULE_W_RATIO,
        kicker=True,
        optical_centre_ratio=TITLE_OPTICAL_CENTRE_RATIO,
    ),
    SceneRole.CONTENT: RoleStyle(),
    SceneRole.SUMMARY: RoleStyle(),
    SceneRole.CLOSING: RoleStyle(),
}


def role_style(role: SceneRole) -> RoleStyle:
    return ROLE_STYLES.get(role, ROLE_STYLES[SceneRole.CONTENT])


def role_for_layout(layout: SlideLayout) -> SceneRole:
    """The role a bare ``SlideLayout`` implies, for callers that have no scene.

    Only ``title_card`` is unambiguous — a title card *is* a title — so everything else
    falls back to ``CONTENT``. A caller holding a :class:`~app.core.models.Scene` should
    pass ``role=scene.role`` instead of relying on this; ``summary`` and ``closing`` can
    only be known from the scene.
    """
    return SceneRole.TITLE if layout is SlideLayout.TITLE_CARD else SceneRole.CONTENT


# --------------------------------------------------------------------- fonts


@lru_cache(maxsize=4)
def find_font(candidates: tuple[str, ...] = FONT_CANDIDATES) -> str:
    """First font file that actually exists on this machine.

    ``VIDEO_FONT_FILE`` overrides. We verify with ``os.path.isfile`` rather than
    hardcoding, because a missing fontfile makes drawtext fail at *render* time.
    """
    override = os.environ.get("VIDEO_FONT_FILE")
    ordered = (override, *candidates) if override else candidates
    for path in ordered:
        if path and Path(path).is_file():
            return path
    raise FontNotFoundError(
        "no usable font file found; set VIDEO_FONT_FILE. tried: " + ", ".join(c for c in ordered)
    )


def available_fonts(candidates: tuple[str, ...] = FONT_CANDIDATES) -> list[str]:
    """Every candidate that exists — the fallback order for a fontfile failure."""
    return [p for p in candidates if Path(p).is_file()]


@lru_cache(maxsize=8)
def heavier_font(font: str) -> str | None:
    """A genuinely heavier face of the same family as ``font``, or ``None``.

    This is what makes emphasis survive without a second colour. The base face is already
    bold (see :data:`FONT_CANDIDATES`), so the useful jump is Bold -> Black — a 700/900
    pair, which at a 36px bullet is an unmistakable difference in stroke mass.

    ``None`` is a normal answer, not a failure: :func:`layout_slide` falls back to a
    larger size plus a faux-bold stroke, and says so in the block it produces.
    """
    path = Path(font)
    for old, new in HEAVIER_FACE_SUFFIXES:
        if not path.name.endswith(old) or old == new:
            continue
        candidate = path.with_name(path.name[: -len(old)] + new)
        if candidate.is_file() and candidate != path:
            return str(candidate)
    return None


# ------------------------------------------------------------------ escaping


FILTERGRAPH_SPECIALS = "\\':,;[]="
"""Characters that must be escaped inside a filter option value.

Determined by experiment against ffmpeg 8.1, not by reading the docs, because the
intuitive readings are wrong in two important ways:

* Wrapping the value in ``'...'`` does **not** protect ``:`` or ``'``. It only helps
  for ``,`` ``;`` ``[`` ``]``.
* ``,`` splits the filter *chain* before option quoting is applied at all, so a
  heading containing a comma silently produces "No option name near ..." — the
  filtergraph fails to parse rather than rendering wrong text.
"""

_ESCAPE_PREFIX = "\\\\\\"
"""Three backslashes.

A filtergraph is unescaped twice: once as a filter description, once as an option
value, and each pass turns ``\\X`` into ``X``. So a literal special character needs
``\\\\`` (surviving pass one as ``\\``) followed by ``\\X`` — three backslashes in
total. A literal backslash needs four, which falls out of the same rule.
"""


def escape_filtergraph(value: str) -> str:
    """Escape an arbitrary string for use as an *unquoted* filter option value.

    Over-escaping is harmless (``\\\\\\a`` still yields ``a``), so this escapes every
    candidate character rather than trying to be clever about context.
    """
    return "".join(
        _ESCAPE_PREFIX + char if char in FILTERGRAPH_SPECIALS else char for char in value
    )


def escape_drawtext(text: str) -> str:
    """Escape a heading for ``drawtext``'s ``text=`` option.

    ``%`` is deliberately *not* escaped here: it is not special to the filtergraph
    parser, only to drawtext's own text expansion, which we switch off with
    ``expansion=none``. (``textfile=`` would sidestep escaping entirely, at the cost of
    a temp file per line.)
    """
    return escape_filtergraph(text)


def escape_filter_path(path: str) -> str:
    """Escape a filesystem path for an unquoted filtergraph option value."""
    return escape_filtergraph(path)


# ImageMagick has its own, completely separate escaping rules, and they bite:
#
# * a text argument beginning with ``@`` means *read the text from this file*, so any
#   string a writer can type ("@channel, read this") is a file-read injection;
# * ``%`` in an *inline* text argument is a format escape — verified against IM 7.1.2,
#   where ``-annotate ... "100%% sure"`` renders "100% sure" and ``%w`` would render the
#   image width. Headings genuinely contain "40%".
#
# Both vanish if the text is *always* passed as ``@file``: file content is taken
# verbatim, with no percent expansion (also verified). So no string this module renders
# is ever interpolated into an ImageMagick argument — only a path we generated is.


def imagemagick_text_arg(text_file: Path) -> str:
    """The only safe way to hand arbitrary text to ImageMagick: ``@`` + our own path."""
    return f"@{text_file}"


def write_text_file(directory: Path, name: str, lines: Sequence[str]) -> Path:
    """Write pre-wrapped lines for ``-annotate``. No trailing newline.

    A trailing newline makes ImageMagick allocate an extra, empty line — which silently
    shifts every vertical measurement by one line height.
    """
    path = directory / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def imagemagick_colour(colour: str) -> str:
    """Normalise a colour to ``#RRGGBB`` before it reaches an ImageMagick argument.

    ImageMagick accepts named colours and expressions; a brand palette arriving from
    JSON should never be able to become one.
    """
    r, g, b = parse_hex(colour)
    return f"#{r:02X}{g:02X}{b:02X}"


def parse_hex(colour: str) -> tuple[int, int, int]:
    """``#F5A524`` / ``f5a524`` / ``#fa2`` -> ``(245, 165, 36)``."""
    raw = colour.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(char * 2 for char in raw)
    if len(raw) != 6:
        raise ValueError(f"not a hex colour: {colour!r}")
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"not a hex colour: {colour!r}") from exc


# ---------------------------------------------------------------- colorimetry


def srgb_to_linear(channel: float) -> float:
    """One sRGB-encoded channel (0..1) to linear light. The WCAG transfer function."""
    channel = _clamp(channel, 0.0, 1.0)
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value: float) -> float:
    """Inverse of :func:`srgb_to_linear` — needed to solve for a scrim opacity."""
    value = _clamp(value, 0.0, 1.0)
    return value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def relative_luminance(colour: str) -> float:
    """WCAG relative luminance of a hex colour."""
    r, g, b = (channel / 255 for channel in parse_hex(colour))
    return 0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)


def contrast_ratio(luminance_a: float, luminance_b: float) -> float:
    """WCAG contrast ratio between two relative luminances. 1.0 .. 21.0."""
    lighter, darker = max(luminance_a, luminance_b), min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


def colour_contrast(colour_a: str, colour_b: str) -> float:
    """Contrast ratio between two hex colours — the exact answer for a solid layout."""
    return contrast_ratio(relative_luminance(colour_a), relative_luminance(colour_b))


def encoded_grey(colour: str) -> float:
    """Luma of a hex colour in the sRGB-*encoded* domain, 0..1.

    The domain a scrim actually blends in (see :class:`Luminance`), so this — not
    :func:`relative_luminance` — is what the opacity solver needs for the wash colour.
    Black gives 0.0 and white 1.0, which are the only two values
    :attr:`~app.core.models.Theme.scrim_colour` produces today.
    """
    r, g, b = (channel / 255 for channel in parse_hex(colour))
    return _clamp(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ------------------------------------------------------------------ rectangles


@dataclass(frozen=True)
class Rect:
    """A box in output-frame pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def centre_x(self) -> int:
        return self.x + self.width // 2

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def inflate(self, pad: int) -> Rect:
        return Rect(self.x - pad, self.y - pad, self.width + 2 * pad, self.height + 2 * pad)

    def clamp_to(self, bounds: Rect) -> Rect:
        x = max(bounds.x, min(self.x, bounds.right))
        y = max(bounds.y, min(self.y, bounds.bottom))
        return Rect(x, y, max(1, min(self.right, bounds.right) - x),
                    max(1, min(self.bottom, bounds.bottom) - y))

    def intersects(self, other: Rect) -> bool:
        """True when the two boxes share at least one pixel. Touching edges do not count."""
        return (
            self.x < other.right
            and other.x < self.right
            and self.y < other.bottom
            and other.y < self.bottom
        )

    @staticmethod
    def union(rects: Iterable[Rect]) -> Rect | None:
        boxes = list(rects)
        if not boxes:
            return None
        left = min(r.x for r in boxes)
        top = min(r.y for r in boxes)
        return Rect(left, top, max(r.right for r in boxes) - left,
                    max(r.bottom for r in boxes) - top)


# -------------------------------------------------------------- slide geometry


@dataclass(frozen=True)
class SlideGeometry:
    """Where the text column and the image live, for one :class:`SlideLayout`.

    ``image_region`` is the contract with the filtergraph builder: cover-fit and animate
    the image inside exactly this box. ``None`` means the slide has no image at all.
    """

    layout: SlideLayout
    frame: Rect
    text_column: Rect
    image_region: Rect | None
    align: str
    """``left`` or ``center`` — horizontal alignment inside ``text_column``."""

    vertical_anchor: str
    """``top`` | ``center`` | ``bottom`` — where the text stack sits in the column."""

    heading_size: int
    bullet_size: int
    scale: float
    """``profile.width / 1920``. Every pixel constant here is a 1080p reference value."""

    over_image: bool
    """True only for ``full_bleed``: text sits on unknown pixels and needs a scrim."""

    image_radius: int

    role: SceneRole = SceneRole.CONTENT
    """What the scene is for. Sets ``heading_size`` and the treatment in
    :data:`ROLE_STYLES`; see :func:`heading_size_for`."""

    @property
    def style(self) -> RoleStyle:
        return role_style(self.role)


def slide_geometry(
    plan: VisualPlan,
    profile: RenderProfile,
    *,
    theme: Theme | None = None,
    role: SceneRole | None = None,
) -> SlideGeometry:
    """Resolve one ``SlideLayout`` into rectangles, in output-frame pixels.

    Pure arithmetic — no fonts, no subprocesses — so the filtergraph builder can call it
    to find ``image_region`` without paying for a rasterisation.

    ``role`` sets the type scale and the treatment; ``None`` infers it from the layout (see
    :func:`role_for_layout`). It never moves ``image_region``, so a caller that only wants
    the image rectangle can keep ignoring it.
    """
    theme = theme or Theme()
    width, height = int(profile.width), int(profile.height)
    frame = Rect(0, 0, width, height)
    scale = width / REF_WIDTH
    layout = getattr(plan, "layout", SlideLayout.HERO_RIGHT)
    role = role if role is not None else role_for_layout(layout)
    style = role_style(role)

    heading_size = heading_size_for(role, width)
    bullet_size = type_size(TYPE_STEP_BODY, width)
    radius = max(0, round(theme.image_radius * scale))

    if layout is SlideLayout.TITLE_CARD:
        margin_x = round(width * TITLE_MARGIN_X_RATIO)
        margin_y = round(height * TITLE_MARGIN_Y_RATIO)
        return SlideGeometry(
            layout=layout,
            frame=frame,
            text_column=Rect(margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y),
            image_region=None,
            align=style.align or "center",
            vertical_anchor=style.vertical_anchor or "center",
            heading_size=heading_size,
            bullet_size=bullet_size,
            scale=scale,
            over_image=False,
            image_radius=radius,
            role=role,
        )

    if layout in (SlideLayout.HERO_RIGHT, SlideLayout.HERO_LEFT):
        margin_x = round(width * SOLID_MARGIN_X_RATIO)
        margin_y = round(height * SOLID_MARGIN_Y_RATIO)
        gutter = round(width * SOLID_GUTTER_RATIO)
        usable = width - 2 * margin_x - gutter
        text_w = round(usable * TEXT_COLUMN_SHARE)
        image_w = usable - text_w
        image_h = height - 2 * margin_y
        # The image keeps the full column height (4:5); the text starts lower, at a fixed y,
        # so the heading lands on the same line whatever the scene contains.
        text_y = round(height * HERO_TEXT_TOP_RATIO)
        text_h = max(1, round(height * HERO_TEXT_BOTTOM_RATIO) - text_y)
        if layout is SlideLayout.HERO_RIGHT:
            text_x, image_x = margin_x, margin_x + text_w + gutter
        else:
            image_x, text_x = margin_x, margin_x + image_w + gutter
        return SlideGeometry(
            layout=layout,
            frame=frame,
            text_column=Rect(text_x, text_y, text_w, text_h),
            image_region=Rect(image_x, margin_y, image_w, image_h),
            align=style.align or "left",
            vertical_anchor=style.vertical_anchor or "top",
            heading_size=heading_size,
            bullet_size=bullet_size,
            scale=scale,
            over_image=False,
            image_radius=radius,
            role=role,
        )

    if layout is SlideLayout.IMAGE_BAND:
        band_h = round(height * BAND_HEIGHT_RATIO)
        margin_x = round(width * SOLID_MARGIN_X_RATIO)
        gap = round(height * 0.074)
        column_top = band_h + gap
        return SlideGeometry(
            layout=layout,
            frame=frame,
            text_column=Rect(
                margin_x, column_top, width - 2 * margin_x, max(1, height - column_top - gap)
            ),
            image_region=Rect(0, 0, width, band_h),
            align=style.align or "left",
            vertical_anchor=style.vertical_anchor or "top",
            heading_size=heading_size,
            bullet_size=bullet_size,
            scale=scale,
            over_image=False,
            image_radius=0,  # a full-width band reads as a band; rounding fights it
            role=role,
        )

    # full_bleed: the image is the frame, so TextPosition picks the column.
    margin_x = round(width * (1.0 - SAFE_AREA) / 2)
    margin_y = round(height * (1.0 - SAFE_AREA) / 2)
    position = plan.text_position
    third_h = round(height * THIRD_HEIGHT_RATIO)
    if position is TextPosition.LEFT_PANEL:
        column = Rect(
            margin_x,
            round(height * 0.115),
            max(1, round(width * LEFT_PANEL_SHARE) - margin_x),
            max(1, height - 2 * round(height * 0.115)),
        )
        align, anchor = "left", "top"
    elif position is TextPosition.UPPER_THIRD:
        column = Rect(margin_x, margin_y, width - 2 * margin_x, third_h)
        align, anchor = "center", "top"
    elif position is TextPosition.CENTER:
        column = Rect(margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        align, anchor = "center", "center"
    else:  # LOWER_THIRD
        column = Rect(margin_x, height - margin_y - third_h, width - 2 * margin_x, third_h)
        align, anchor = "center", "bottom"

    return SlideGeometry(
        layout=SlideLayout.FULL_BLEED,
        frame=frame,
        text_column=column,
        image_region=frame,
        align=style.align or align,
        vertical_anchor=style.vertical_anchor or anchor,
        heading_size=heading_size,
        bullet_size=bullet_size,
        scale=scale,
        over_image=True,
        image_radius=0,
        role=role,
    )


def image_region(
    plan: VisualPlan, profile: RenderProfile, *, theme: Theme | None = None
) -> Rect | None:
    """Where the filtergraph builder should cover-fit the scene image. ``None`` = no image."""
    return slide_geometry(plan, profile, theme=theme).image_region


def _inset_x(rect: Rect, inset: int, symmetric: bool) -> Rect:
    """Shrink a rect horizontally by ``inset``, on both sides when ``symmetric``."""
    inset = max(0, min(inset, rect.width // 2 - 1)) if rect.width > 2 else 0
    width = rect.width - (2 * inset if symmetric else inset)
    return Rect(rect.x + inset, rect.y, max(1, width), rect.height)


LOGO_RESERVED_ASPECT = 1.6
"""Width/height assumed for the watermark before it has been rasterised.

Generous on purpose: the reserved box only feeds the collision check, and a box that is
too *narrow* is the one that lets the logo land on a word.
"""


def logo_height(profile: RenderProfile, theme: Theme | None = None) -> int:
    """Watermark height in output pixels. A fraction of the frame *height*, so the mark
    keeps its apparent size at any resolution."""
    theme = theme or Theme()
    return max(1, round(int(profile.height) * theme.logo_height_fraction))


def logo_rect(
    profile: RenderProfile, theme: Theme | None = None, *, size: tuple[int, int] | None = None
) -> Rect:
    """Where the brand watermark sits, in output-frame pixels. Bottom-left, always.

    The inset is a fraction of the frame *width* on both axes, so the corner reads as
    square rather than as a wider gap on one side.

    ``size`` is the rasterised PNG's real ``(width, height)`` when the caller has it; the
    fallback reserves :data:`LOGO_RESERVED_ASPECT` so a pure-geometry caller still gets a
    box that cannot under-report.

    Lives in ``text_overlay`` rather than in the backend because it is the same kind of
    fact as :func:`slide_geometry`: pure arithmetic that both the compositor and the
    collision check have to agree on.
    """
    theme = theme or Theme()
    width, height = int(profile.width), int(profile.height)
    if size is not None:
        logo_w, logo_h = max(1, int(size[0])), max(1, int(size[1]))
    else:
        logo_h = logo_height(profile, theme)
        logo_w = max(1, round(logo_h * LOGO_RESERVED_ASPECT))
    margin = max(0, round(width * theme.logo_margin_fraction))
    return Rect(margin, max(0, height - margin - logo_h), logo_w, logo_h)


# -------------------------------------------------------------- text measuring
#
# Wrapping needs real advance widths: the analytic "0.52 * point size per character"
# estimate is fine for picking a heading size but wrong enough to overflow a 47%-wide
# column. ImageMagick will report the rendered width of a string, and widths are additive
# across words to ~0.2% (measured: "Handgloves mixed CASE 12345" is 1501px; the sum of
# its word widths plus three spaces is 1504px). So one batched call per new vocabulary
# gives exact-enough wrapping, and the cache means a re-render costs nothing.

_SPACE_KEY = "\x00space"
_advance_cache: dict[tuple[str, int, str], float] = {}
_advance_lock = threading.Lock()

TextMeasurer = Callable[[str], float]


def _measure_batch(strings: Sequence[str], font: str, size: int, binary: str) -> list[float] | None:
    """Rendered pixel width of each string, in one ImageMagick invocation."""
    if not strings:
        return []
    with tempfile.TemporaryDirectory(prefix="tw-") as tmp:
        directory = Path(tmp)
        argv = [binary, "-font", font, "-pointsize", str(size), "-background", "none"]
        for index, value in enumerate(strings):
            path = directory / f"w{index}.txt"
            path.write_text(value, encoding="utf-8")
            argv.append(f"label:{imagemagick_text_arg(path)}")
        argv += ["-format", "%w\n", "info:"]
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                argv, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("ImageMagick text measurement failed (%s); estimating widths", exc)
            return None
    values = [line for line in proc.stdout.split() if line]
    if proc.returncode != 0 or len(values) != len(strings):
        logger.warning(
            "ImageMagick text measurement returned %d widths for %d strings; estimating",
            len(values),
            len(strings),
        )
        return None
    try:
        return [float(value) for value in values]
    except ValueError:
        return None


def _estimated_width(text: str, size: int) -> float:
    return len(text) * size * AVG_GLYPH_RATIO


_advance_height_cache: dict[tuple[str, int], int | None] = {}


def font_line_advance(font: str, size: int, *, binary: str | None = None) -> int | None:
    """Baseline-to-baseline advance ImageMagick uses at ``-interline-spacing 0``, in pixels.

    Not ``size``, and not ``size * LINE_SPACING``: it is the *face's* own line height, and
    it varies a lot between weights. Measured against IM 7.1.2 on macOS, at 36pt:
    Arial Bold advances 41px, Arial Black 52px — 27% further for the same point size.

    That difference is why this exists. Drawing a heavier face with the interline spacing
    computed for a lighter one silently spaces its lines a third further apart than the
    layout declared, which both overflows the canvas the layout sized (clipping the last
    line) and desynchronises the block from the stack rhythm around it.

    Measured by differencing the height of a two-line ``label:`` against a one-line one,
    which is exactly the advance and costs one invocation per ``(font, size)``. ``None``
    when ImageMagick cannot answer, which leaves the caller on the historical assumption.
    """
    key = (font, int(size))
    with _advance_lock:
        if key in _advance_height_cache:
            return _advance_height_cache[key]
    binary = binary if binary is not None else imagemagick_bin()
    value: int | None = None
    if binary:
        value = _measure_line_advance(font, int(size), binary)
    with _advance_lock:
        _advance_height_cache[key] = value
    return value


def _measure_line_advance(font: str, size: int, binary: str) -> int | None:
    with tempfile.TemporaryDirectory(prefix="adv-") as tmp:
        directory = Path(tmp)
        one = write_text_file(directory, "one.txt", ["Hxg"])
        two = write_text_file(directory, "two.txt", ["Hxg", "Hxg"])
        argv = [
            binary,
            "-font", font,
            "-pointsize", str(size),
            # Zero it explicitly: any inherited spacing would be counted twice.
            "-interline-spacing", "0",
            "-background", "none",
            f"label:{imagemagick_text_arg(one)}",
            f"label:{imagemagick_text_arg(two)}",
            "-format", "%h\n",
            "info:",
        ]
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                argv, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("line-advance probe for %s failed (%s)", font, exc)
            return None
    values = [line for line in proc.stdout.split() if line]
    if proc.returncode != 0 or len(values) != 2:
        logger.warning("line-advance probe for %s returned %r", font, proc.stdout)
        return None
    try:
        advance = int(values[1]) - int(values[0])
    except ValueError:
        return None
    return advance if advance > 0 else None


def interline_spacing(line_height: int, size: int, *, advance: int | None) -> int:
    """The ``-interline-spacing`` that makes the rendered advance equal ``line_height``.

    With a measured ``advance`` this is exact — verified by rendering two lines and
    differencing their ink, for both Arial Bold and Arial Black. ``None`` falls back to
    ``line_height - size``, which is what every caller assumed before the advance was
    measurable; it leaves the base face's spacing exactly as it has always shipped.
    """
    return line_height - (size if advance is None else advance)


def text_measurer(font: str, size: int, *, binary: str | None = None) -> TextMeasurer:
    """A ``str -> pixel width`` function for one font at one size.

    Word advances are cached process-wide and looked up under a lock, so the 4 render
    threads share one vocabulary instead of each shelling out for "phishing".
    """
    binary = binary if binary is not None else imagemagick_bin()
    if not binary:
        return lambda text: _estimated_width(text, size)

    def measure(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        needed = list(dict.fromkeys([*words, _SPACE_KEY]))
        with _advance_lock:
            missing = [w for w in needed if (font, size, w) not in _advance_cache]
        if missing:
            probes = [w for w in missing if w != _SPACE_KEY]
            # "x x" minus "xx" isolates the space advance, which `label:` trims away.
            widths = _measure_batch([*probes, "x x", "xx"], font, size, binary)
            if widths is None:
                return _estimated_width(text, size)
            space = max(0.0, widths[-2] - widths[-1])
            with _advance_lock:
                for word, value in zip(probes, widths[:-2], strict=True):
                    _advance_cache[(font, size, word)] = value
                _advance_cache[(font, size, _SPACE_KEY)] = space
        with _advance_lock:
            try:
                total = sum(_advance_cache[(font, size, word)] for word in words)
                total += _advance_cache[(font, size, _SPACE_KEY)] * (len(words) - 1)
            except KeyError:
                return _estimated_width(text, size)
        return total

    return measure


# ------------------------------------------------------------------ wrapping


def wrap_text(text: str, max_chars: int, max_lines: int = MAX_LINES) -> list[str] | None:
    """Wrap into at most ``max_lines`` lines of ``max_chars``; ``None`` if impossible.

    For the two-line case it picks the *balanced* split rather than greedy-filling,
    because a long first line over a two-word second line looks like an accident.
    """
    words = text.split()
    if not words:
        return []
    if len(text) <= max_chars:
        return [text]
    if max_lines < 2 or any(len(w) > max_chars for w in words):
        return None

    if max_lines == 2:
        best: tuple[tuple[int, int], list[str]] | None = None
        for i in range(1, len(words)):
            head, tail = " ".join(words[:i]), " ".join(words[i:])
            if len(head) <= max_chars and len(tail) <= max_chars:
                # Shortest longest line, then the most even split of the ties.
                score = (max(len(head), len(tail)), abs(len(head) - len(tail)))
                if best is None or score < best[0]:
                    best = (score, [head, tail])
        return best[1] if best else None

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                return None
    lines.append(current)
    return lines if len(lines) <= max_lines else None


def _truncate(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Absolute last resort for a heading that will not fit at any size."""
    budget = max_chars * max_lines - 1
    clipped = text[:budget].rstrip() + ELLIPSIS
    return wrap_text(clipped, max_chars, max_lines) or [clipped[:max_chars]]


def wrap_to_width(
    text: str, max_px: float, measure: TextMeasurer, *, max_lines: int = MAX_BULLET_LINES
) -> list[str]:
    """Greedy-wrap ``text`` to ``max_px``, hard-breaking any single word that overflows.

    Always returns at least one line, and never more than ``max_lines`` — the last line
    is ellipsised instead. Line *count* is authoritative: the caller sizes its canvas
    from it and renders these exact lines, so the rasteriser cannot disagree about how
    tall the block is. That is what keeps a wrapped bullet from landing on the next one.
    """
    words = text.split()
    if not words:
        return []
    max_px = max(1.0, max_px)

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if measure(candidate) <= max_px or not current:
            if not current and measure(word) > max_px:
                pieces = _break_word(word, max_px, measure)
                lines.extend(pieces[:-1])
                current = pieces[-1]
                continue
            current = candidate
        else:
            lines.append(current)
            current = word
            if measure(current) > max_px:
                pieces = _break_word(current, max_px, measure)
                lines.extend(pieces[:-1])
                current = pieces[-1]
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        kept = lines[: max_lines - 1] if max_lines > 1 else []
        tail = " ".join(lines[max_lines - 1 :])
        kept.append(_ellipsise(tail, max_px, measure))
        lines = kept
    if len(lines) == 2:
        lines = _balance_two(words, max_px, measure) or lines
    return lines


def _break_word(word: str, max_px: float, measure: TextMeasurer) -> list[str]:
    """Split an unbreakable token (a URL, a German compound) at the pixel limit."""
    pieces: list[str] = []
    current = ""
    for char in word:
        if current and measure(current + char) > max_px:
            pieces.append(current)
            current = char
        else:
            current += char
    pieces.append(current)
    return pieces or [word]


def _balance_two(
    words: Sequence[str], max_px: float, measure: TextMeasurer
) -> list[str] | None:
    """Even out a two-line wrap. Greedy leaves a long line over a two-word orphan."""
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(words)):
        head, tail = " ".join(words[:i]), " ".join(words[i:])
        head_w, tail_w = measure(head), measure(tail)
        if head_w <= max_px and tail_w <= max_px:
            score = max(head_w, tail_w) + abs(head_w - tail_w) * 0.5
            if best is None or score < best[0]:
                best = (score, [head, tail])
    return best[1] if best else None


def _ellipsise(text: str, max_px: float, measure: TextMeasurer) -> str:
    if measure(text) <= max_px:
        return text
    clipped = text
    while clipped and measure(clipped + ELLIPSIS) > max_px:
        clipped = clipped[:-1]
    return (clipped.rstrip() + ELLIPSIS) if clipped else ELLIPSIS


# ------------------------------------------------------------ heading layout
# (legacy single-heading path, kept so the `drawtext` strategy still works)


@dataclass(frozen=True)
class TextLayout:
    """Resolved pixel geometry for one heading. Shared by every strategy."""

    lines: list[str]
    font_size: int
    line_height: int
    block_top: int
    block_height: int
    pad: int
    scrim_top: int
    scrim_height: int
    scrim_opacity: float
    position: TextPosition
    width: int
    height: int

    @property
    def block_bottom(self) -> int:
        return self.block_top + self.block_height

    def scrim_bands(self) -> list[tuple[int, int, str]]:
        """``(y, height, kind)`` bands making up the scrim.

        The band directly behind the text is *solid* at full opacity — that is the part
        doing the legibility work. A softer gradient band feathers the edge so the
        scrim does not read as a grey rectangle pasted across the picture.
        """
        feather = min(round(self.height * 0.14), max(1, self.pad * 3))
        if self.position is TextPosition.CENTER:
            return [(0, self.height, "solid")]
        if self.position is TextPosition.UPPER_THIRD:
            solid_end = min(self.height, self.block_bottom + self.pad)
            bands = [(0, solid_end, "solid")]
            if solid_end < self.height:
                bands.append((solid_end, min(feather, self.height - solid_end), "fade_down"))
            return bands
        solid_start = max(0, self.block_top - self.pad)
        bands = [(solid_start, self.height - solid_start, "solid")]
        if solid_start > 0:
            top = max(0, solid_start - feather)
            bands.append((top, solid_start - top, "fade_up"))
        return bands

    def line_y(self, index: int) -> int:
        return self.block_top + index * self.line_height


def layout_heading(
    heading: str,
    plan: VisualPlan,
    profile: RenderProfile,
    *,
    max_lines: int = MAX_LINES,
) -> TextLayout:
    """Pick a font size, wrap to <=``max_lines``, and place the block + scrim."""
    text = " ".join(heading.split())
    usable = profile.width * SAFE_AREA
    base = max(14, round(profile.height * HEADING_SIZE_RATIO))
    floor = max(12, round(profile.height * MIN_HEADING_SIZE_RATIO))

    size = floor
    lines: list[str] | None = None
    for candidate_size in range(base, floor - 1, -2):
        max_chars = max(8, int(usable / (candidate_size * AVG_GLYPH_RATIO)))
        lines = wrap_text(text, max_chars, max_lines)
        if lines is not None:
            size = candidate_size
            break
    if lines is None:
        max_chars = max(8, int(usable / (floor * AVG_GLYPH_RATIO)))
        lines = _truncate(text, max_chars, max_lines)
        size = floor

    line_height = round(size * LINE_SPACING)
    block_height = max(line_height, line_height * len(lines))
    pad = round(size * 0.6)
    margin = round(profile.height * (1.0 - SAFE_AREA))

    if plan.text_position is TextPosition.UPPER_THIRD:
        block_top = margin
        scrim_top = 0
        scrim_height = min(profile.height, block_top + block_height + 2 * pad)
        opacity = plan.scrim_opacity
    elif plan.text_position is TextPosition.CENTER:
        block_top = max(0, (profile.height - block_height) // 2)
        scrim_top = 0
        scrim_height = profile.height
        opacity = round(plan.scrim_opacity * 0.8, 4)  # full-frame, so go lighter
    else:  # LOWER_THIRD / LEFT_PANEL
        block_top = max(0, profile.height - margin - block_height)
        scrim_top = max(0, block_top - 2 * pad)
        scrim_height = profile.height - scrim_top
        opacity = plan.scrim_opacity

    return TextLayout(
        lines=list(lines),
        font_size=size,
        line_height=line_height,
        pad=int(pad),
        block_top=int(block_top),
        block_height=int(block_height),
        scrim_top=int(scrim_top),
        scrim_height=int(scrim_height),
        scrim_opacity=float(opacity),
        position=plan.text_position,
        width=profile.width,
        height=profile.height,
    )


# ---------------------------------------------------------- filtergraph text


def _wash_name(theme: Theme | None) -> str:
    """``black`` or ``white`` — the wash/outline name ffmpeg and ImageMagick both accept.

    Named rather than hex so the default output is byte-identical to the pre-light-theme
    filtergraphs, which several tests assert on verbatim.
    """
    return "white" if (theme or Theme()).is_light else "black"


def scrim_filter(layout: TextLayout, *, theme: Theme | None = None) -> str:
    """The legibility scrim behind the text, as ``drawbox`` filters.

    ``drawbox`` cannot do a gradient, so the feather band is approximated with a
    single half-opacity box. Visually close enough to the PNG path, and ``drawbox``
    exists in every ffmpeg build.

    The wash follows the palette's polarity: black under light text, white under dark.
    """
    wash = _wash_name(theme)
    parts = []
    for y, band_height, kind in layout.scrim_bands():
        if band_height <= 0:
            continue
        opacity = layout.scrim_opacity if kind == "solid" else round(layout.scrim_opacity / 2, 4)
        parts.append(
            f"drawbox=x=0:y={y}:w={layout.width}:h={band_height}:color={wash}@{opacity}:t=fill"
        )
    return ",".join(parts)


def drawtext_filters(
    layout: TextLayout, *, font: str | None = None, theme: Theme | None = None
) -> str:
    """Comma-joined ``drawbox`` + one ``drawtext`` per line.

    One filter per line (instead of an embedded ``\\n``) keeps control of the line
    spacing and lets each line be centred independently.

    Kept for portability only — this ffmpeg has no ``drawtext``. It still honours the
    palette, so a light theme does not render white-on-white if it is ever reached.
    """
    theme = theme or Theme()
    font_path = escape_filter_path(font or find_font())
    parts = [scrim_filter(layout, theme=theme)]
    border = max(2, round(layout.font_size * 0.045))
    fill = ffmpeg_hex(theme.text)
    outline = _wash_name(theme)
    for index, line in enumerate(layout.lines):
        parts.append(
            f"drawtext=fontfile={font_path}"
            f":text={escape_drawtext(line)}"
            f":expansion=none"
            f":fontsize={layout.font_size}"
            f":fontcolor={fill}"
            f":borderw={border}:bordercolor={outline}@0.9"
            f":x=(w-text_w)/2:y={layout.line_y(index)}"
        )
    return ",".join(parts)


def ffmpeg_hex(colour: str) -> str:
    """``#F8FAFC`` -> ``0xF8FAFC``. ``#`` starts a comment in an ffmpeg filter script."""
    r, g, b = parse_hex(colour)
    return f"0x{r:02X}{g:02X}{b:02X}"


# ------------------------------------------------------------- ImageMagick I/O


def imagemagick_bin() -> str | None:
    """``magick`` (IM7) or ``convert`` (IM6), or None."""
    return (
        os.environ.get("IMAGEMAGICK_BIN")
        or shutil.which("magick")
        or shutil.which("convert")
    )


def require_imagemagick() -> str:
    binary = imagemagick_bin()
    if binary is None:
        raise RuntimeError(
            "ImageMagick not found and this ffmpeg has no drawtext filter; "
            "install `magick` or set IMAGEMAGICK_BIN"
        )
    return binary


def _rgba(opacity: float, colour: str = "#000000") -> str:
    """An ImageMagick ``rgba()`` literal. ``colour`` is the wash, black unless a light
    palette asks for :attr:`~app.core.models.Theme.scrim_colour` to flip."""
    r, g, b = parse_hex(colour)
    return f"rgba({r},{g},{b},{round(opacity, 4)})"


# Cache + concurrency.
#
# Scene clips render on 4 threads, so two threads can want the same PNG at the same
# instant. Two defences, both needed: a per-path lock so the second thread waits instead
# of duplicating the work, and a write to a unique temp file followed by `os.replace`, so
# a reader never sees a half-written PNG even across processes.

_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = _path_locks[key] = threading.Lock()
        return lock


def cache_key(*parts: object) -> str:
    """Stable short hash of everything that changes the pixels."""
    blob = "\x1f".join(repr(part) for part in parts)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=10).hexdigest()


def cache_path(workdir: Path, kind: str, *parts: object) -> Path:
    return Path(workdir) / "text" / f"{kind}-{cache_key(kind, *parts)}.png"


def render_cached_png(argv_tail: Sequence[str], out_path: Path, *, binary: str) -> Path:
    """Run ImageMagick into ``out_path``, once, atomically, reusing an existing file."""
    if out_path.exists():
        return out_path
    with _lock_for(out_path):
        if out_path.exists():
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=out_path.parent, prefix=f".{out_path.stem}.", suffix=".png"
        )
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            ff.run([binary, *argv_tail, f"PNG32:{tmp}"], timeout=180)
            os.replace(tmp, out_path)
        finally:
            tmp.unlink(missing_ok=True)
    return out_path


def render_text_png(
    heading: str,
    plan: VisualPlan,
    profile: RenderProfile,
    out_path: Path,
    *,
    font: str | None = None,
    layout: TextLayout | None = None,
    theme: Theme | None = None,
) -> Path:
    """Rasterise scrim + heading to a transparent RGBA PNG via ImageMagick.

    Legacy single-heading path, retained for callers that have not moved to
    :func:`build_scene_text`. The scrim is a *gradient* band here rather than a
    hard-edged box — a visible scrim edge across the frame is more distracting than the
    scrim itself.

    Both the wash and the outline follow the palette's polarity, so a light theme gets a
    white wash under dark type instead of a dark wash that erases it.
    """
    binary = require_imagemagick()
    theme = theme or Theme()
    layout = layout or layout_heading(heading, plan, profile)
    font_path = font or find_font()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wash = theme.scrim_colour
    argv: list[str] = ["-size", f"{layout.width}x{layout.height}", "xc:none"]

    transparent, dark = _rgba(0.0, wash), _rgba(layout.scrim_opacity, wash)
    for y, band_height, kind in layout.scrim_bands():
        if band_height <= 0:
            continue
        if kind == "solid":
            source = f"xc:{dark}"
        elif kind == "fade_up":  # transparent at the top, dark where the text begins
            source = f"gradient:{transparent}-{dark}"
        else:  # fade_down
            source = f"gradient:{dark}-{transparent}"
        argv += ["(", "-size", f"{layout.width}x{band_height}", source, ")",
                 "-geometry", f"+0+{y}", "-composite"]

    with tempfile.TemporaryDirectory(prefix="heading-") as tmp:
        text_file = write_text_file(Path(tmp), "heading.txt", layout.lines)
        text_arg = imagemagick_text_arg(text_file)
        argv += [
            "-font", font_path,
            "-pointsize", str(layout.font_size),
            "-interline-spacing", str(layout.line_height - layout.font_size),
            "-gravity", "north",
            # Pass 1: the outline silhouette. Pass 2: the text face on top.
            "-stroke", imagemagick_colour(wash),
            "-strokewidth", str(max(3, round(layout.font_size * 0.09))),
            "-fill", imagemagick_colour(wash),
            "-annotate", f"+0+{layout.block_top}", text_arg,
            "-stroke", "none",
            "-fill", imagemagick_colour(theme.text),
            "-annotate", f"+0+{layout.block_top}", text_arg,
            "-strip",
        ]
        ff.run([binary, *argv, f"PNG32:{out_path}"], timeout=180)
    return out_path


# -------------------------------------------------------------- luminance probe


@dataclass(frozen=True)
class Luminance:
    """Luminance statistics of one region of one image, in the sRGB-*encoded* domain.

    Encoded, not linear, because that is the domain the scrim actually blends in:
    ``overlay`` composites gamma-encoded values, so black at opacity ``a`` scales an
    encoded channel by ``1 - a``. Solving for ``a`` in the encoded domain and only then
    converting to relative luminance keeps the arithmetic honest.
    """

    mean: float
    stddev: float
    probe: float
    """``mean + SCRIM_PROBE_SIGMA * stddev``, clamped — the bright tail that ruins light text."""

    probe_low: float = 0.0
    """``mean - SCRIM_PROBE_SIGMA * stddev``, clamped — the dark tail that ruins dark text."""

    @classmethod
    def from_stats(cls, mean: float, stddev: float) -> Luminance:
        """Build both tails from a mean and a standard deviation."""
        mean = _clamp(mean, 0.0, 1.0)
        stddev = _clamp(stddev, 0.0, 1.0)
        return cls(
            mean=mean,
            stddev=stddev,
            probe=_clamp(mean + SCRIM_PROBE_SIGMA * stddev, 0.0, 1.0),
            probe_low=_clamp(mean - SCRIM_PROBE_SIGMA * stddev, 0.0, 1.0),
        )

    @property
    def mean_relative(self) -> float:
        return srgb_to_linear(self.mean)

    @property
    def probe_relative(self) -> float:
        return srgb_to_linear(self.probe)

    @property
    def probe_low_relative(self) -> float:
        return srgb_to_linear(self.probe_low)

    def tail(self, *, low: bool) -> float:
        """The encoded tail a wash has to defend against.

        ``low=False`` for a dark wash under light text (the highlights are the enemy),
        ``low=True`` for a light wash under dark text (the shadows are).
        """
        return self.probe_low if low else self.probe

    def after_scrim(self, opacity: float, *, scrim_encoded: float = 0.0) -> Luminance:
        """This region after ``opacity`` of a ``scrim_encoded``-valued wash.

        ``overlay`` composites gamma-encoded values, so every statistic moves toward the
        wash colour by the opacity. The spread shrinks by ``1 - opacity`` regardless of
        which way it moved — blending toward *any* constant is a contraction.
        """
        alpha = _clamp(opacity, 0.0, 1.0)
        keep = 1.0 - alpha

        def blend(value: float) -> float:
            return _clamp(value + alpha * (scrim_encoded - value), 0.0, 1.0)

        return Luminance(
            mean=blend(self.mean),
            stddev=self.stddev * keep,
            probe=blend(self.probe),
            probe_low=blend(self.probe_low),
        )


def measure_region_luminance(
    image_path: str | Path,
    region: Rect,
    frame: Rect,
    *,
    binary: str | None = None,
) -> Luminance | None:
    """Sample the image where the text will sit. ``None`` if it cannot be measured.

    The image is first cover-fitted to ``frame`` exactly as the renderer will fit it, so
    ``region`` — which is in output-frame pixels — lands on the same pixels the viewer
    will see behind the text.
    """
    binary = binary if binary is not None else imagemagick_bin()
    if not binary or not Path(image_path).is_file():
        return None
    box = region.clamp_to(frame)
    argv = [
        binary,
        str(image_path),
        "-alpha", "off",
        "-resize", f"{frame.width}x{frame.height}^",
        "-gravity", "center",
        "-extent", f"{frame.width}x{frame.height}",
        # `-crop` honours gravity too, so the centring used for `-extent` has to be
        # cleared or the probe silently samples a region offset from the frame origin.
        "-gravity", "none",
        "-crop", f"{box.width}x{box.height}+{box.x}+{box.y}",
        "+repage",
        "-format",
        "%[fx:0.2126*mean.r+0.7152*mean.g+0.0722*mean.b] %[fx:standard_deviation]",
        "info:",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, no shell
            argv, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("luminance probe of %s failed (%s)", image_path, exc)
        return None
    fields = proc.stdout.split()
    if proc.returncode != 0 or len(fields) < 2:
        logger.warning("luminance probe of %s produced %r", image_path, proc.stdout)
        return None
    try:
        mean, stddev = float(fields[0]), float(fields[1])
    except ValueError:
        return None
    return Luminance.from_stats(mean, stddev)


def required_scrim_opacity(
    probe_encoded: float,
    *,
    text_luminance: float = 1.0,
    target_ratio: float = WCAG_AA,
    floor: float = SCRIM_OPACITY_FLOOR,
    ceil: float = SCRIM_OPACITY_CEIL,
    scrim_encoded: float = 0.0,
) -> float:
    """Smallest scrim opacity that gets ``target_ratio`` against the probe.

    Closed form, and symmetric in the palette's polarity. ``overlay`` blends encoded
    values linearly, so after a wash of encoded value ``s`` at opacity ``a`` the
    background sits at ``v + a*(s - v)``. The contrast ratio is monotonic in the
    background luminance, so invert it for the *worst* background we will tolerate,
    encode that back to sRGB, and ``a`` follows directly.

    A **dark** wash (``scrim_encoded`` below the text) serves light text and drives the
    background down toward the brightest tolerable value. A **light** wash serves dark
    text and drives it *up* toward the darkest tolerable value. Assuming the first case
    is what makes a light palette render invisible type.
    """
    darkening = scrim_encoded < linear_to_srgb(text_luminance)
    if darkening:
        allowed_relative = (text_luminance + 0.05) / target_ratio - 0.05
        if allowed_relative <= 0.0:
            return ceil
    else:
        allowed_relative = target_ratio * (text_luminance + 0.05) - 0.05
        if allowed_relative >= 1.0:
            return ceil

    allowed_encoded = linear_to_srgb(allowed_relative)
    already_legible = (
        probe_encoded <= allowed_encoded if darkening else probe_encoded >= allowed_encoded
    )
    if already_legible:
        return _clamp(floor, 0.0, 1.0)
    span = scrim_encoded - probe_encoded
    if span == 0.0:  # the wash is the background: it can never help
        return ceil
    return _clamp((allowed_encoded - probe_encoded) / span, floor, ceil)


def quantise_opacity(opacity: float) -> float:
    """Snap a solved opacity *up* onto the 8-bit alpha grid.

    Two reasons, both about the reported number being true. Rounding to 4 decimals rounds
    the requirement *down* and lands at 4.4994:1 — a hair under AA. And an RGBA PNG can
    only store ``n/255`` anyway, so any other value is a number we report but do not
    render. Ceiling to the next representable step fixes both at once.
    """
    return math.ceil(_clamp(opacity, 0.0, 1.0) * 255) / 255


@dataclass(frozen=True)
class ContrastReport:
    """What the text/background contrast actually is, and how we got there."""

    source: str
    """``theme`` (computed exactly from the palette) or ``image`` (measured pixels)."""

    text_colour: str
    accent_colour: str
    scrim_opacity: float
    ratio_before: float
    ratio_after: float
    accent_ratio_after: float
    background_luminance_before: float
    background_luminance_after: float
    meets_aa: bool
    detail: str = ""

    scrim_colour: str = "#000000"
    """The wash actually used — ``#000000`` for a dark palette, ``#FFFFFF`` for a light one."""

    def summary(self) -> str:
        return (
            f"{self.source}: text {self.text_colour} contrast "
            f"{self.ratio_before:.2f}:1 -> {self.ratio_after:.2f}:1 "
            f"(scrim {self.scrim_colour}@{self.scrim_opacity:.3f}, "
            f"accent {self.accent_ratio_after:.2f}:1, "
            f"AA={'pass' if self.meets_aa else 'FAIL'})"
        )


# ------------------------------------------------------------- slide text plan


@dataclass(frozen=True)
class BulletBlock:
    """One laid-out bullet: marker + wrapped lines, with its own box in the frame.

    **Uniformity is the invariant here.** Inside one slide — and, because both come from
    the theme and the scale, inside one video — every block shares ``marker_shape``,
    ``marker_diameter``, ``marker_colour``, ``text_colour``, ``size``, ``indent`` and
    ``stroke_ratio``. The only field an emphasised bullet may change is ``font`` (plus
    ``faux_bold`` when no heavier face exists), and only in ``weight`` mode; see
    :data:`EMPHASIS_MODE`.
    """

    lines: list[str]
    rect: Rect
    size: int
    line_height: int
    marker_diameter: int
    indent: int
    emphasis: bool
    appear_at: float
    text_colour: str
    marker_colour: str
    offset_x: int = 0
    """Marker x inside ``rect``. Shared by every bullet on the slide, so a centred stack
    is centred *as a block* — centring each bullet on its own gives a ragged left edge
    that reads as a mistake rather than as centred text."""

    font: str = ""
    """Face this block is set in. A heavier file than the slide's base font when the block
    is emphasised and one exists — see :func:`heavier_font`."""

    stroke_ratio: float = BULLET_STROKE_RATIO
    """Dark-outline width as a fraction of ``size``. The same for every bullet."""

    faux_bold: float = 0.0
    """Fill-coloured stroke, as a fraction of ``size``. Non-zero only when emphasis had to
    be faked because no heavier face was installed."""

    marker_shape: str = "disc"
    """``disc`` | ``ring`` | ``chevron`` | ``dash`` | ``none``, from
    :attr:`~app.core.models.Theme.marker`.

    One shape for the whole video, and it never varies by emphasis, position or scene. It
    used to: a filled disc marked the emphasised point and a hollow ring the rest. The
    intent was non-chromatic hierarchy; what a viewer actually sees is two kinds of bullet
    in one list, and they conclude the deck was assembled carelessly.
    """

    outline_colour: str = "#000000"
    """Colour of the outline pass. Flips to white on a light palette — a dark halo around
    dark type on a pale background thickens the glyph into mud instead of separating it."""

    interline_spacing: int | None = None
    """``-interline-spacing`` override, so ``line_height`` really is the rendered advance.

    Set only for a block drawn in a face other than the slide's base one, where the face's
    own line height differs enough to matter (see :func:`font_line_advance`). ``None``
    keeps the historical ``line_height - size``, which is what the base face has always
    shipped — introducing a heavier weight must not move type that already looks right.
    """

    @property
    def marker_ring_width(self) -> int:
        """Stroke width of the ``ring`` shape. Ignored by every other shape."""
        return max(2, round(self.marker_diameter * MARKER_RING_RATIO))


@dataclass(frozen=True)
class Emphasis:
    """How an emphasised bullet differs from a normal one, resolved once per slide.

    Three states, and two of them do nothing:

    * ``mode="off"`` (the default — see :data:`EMPHASIS_MODE`) — nothing differs. Every
      bullet in the list is set identically, which is the look the design asks for.
    * ``mode="weight"`` — a genuinely heavier *face* at the same point size, or a faux-bold
      stroke where the family has no heavier file. Size, outline, marker and colour are
      untouched, because varying any of those is what made the list look inconsistent.
    * ``uniform_text=False`` — :meth:`chromatic`, the pre-brand-rule behaviour: the accent
      colour carries emphasis and the geometry is identical. Kept switchable, not default.

    Nothing here can change the marker. The marker comes from
    :attr:`~app.core.models.Theme.marker` and is one shape for the whole video.
    """

    uniform: bool
    font: str | None
    """Heavier face, or ``None`` when weight emphasis is off or unavailable."""

    faux_bold: float
    """Fill-coloured stroke used to fake weight. Zero whenever ``font`` is available."""

    @classmethod
    def resolve(cls, theme: Theme, base_font: str, mode: str | None = None) -> Emphasis:
        mode = (mode or EMPHASIS_MODE).strip().lower()
        if mode not in EMPHASIS_MODES:
            logger.warning("unknown emphasis mode %r; falling back to 'off'", mode)
            mode = "off"
        if not theme.uniform_text:
            return cls.chromatic()
        if mode == "off":
            return cls(uniform=True, font=None, faux_bold=0.0)
        heavier = heavier_font(base_font)
        if heavier:
            return cls(uniform=True, font=heavier, faux_bold=0.0)
        logger.debug(
            "no heavier face for %s; emphasising with a faux-bold stroke instead", base_font
        )
        return cls(uniform=True, font=None, faux_bold=EMPHASIS_FAUX_BOLD_RATIO)

    @classmethod
    def chromatic(cls) -> Emphasis:
        """The pre-``uniform_text`` behaviour: accent colour, identical metrics."""
        return cls(uniform=False, font=None, faux_bold=0.0)

    def text_colour(self, theme: Theme, *, emphasis: bool) -> str:
        """One colour for every word on the slide, unless ``uniform_text`` is off."""
        if self.uniform:
            return theme.text
        return theme.accent if emphasis else theme.text

    def font_for(self, base_font: str, *, emphasis: bool) -> str:
        return self.font if (emphasis and self.font) else base_font

    def faux_bold_for(self, *, emphasis: bool) -> float:
        return self.faux_bold if emphasis else 0.0


@dataclass(frozen=True)
class _WrappedBullet:
    """One bullet after wrapping, carrying the face and size it was wrapped at."""

    point: BulletPoint
    lines: list[str]
    size: int
    font: str
    stroke_ratio: float = BULLET_STROKE_RATIO

    @property
    def line_height(self) -> int:
        return round(self.size * LINE_SPACING)

    @property
    def height(self) -> int:
        """Canvas height: the type, plus slack for descenders and the outline stroke."""
        return len(self.lines) * self.line_height + _descender_pad(self.size, self.stroke_ratio)

    @property
    def pitch(self) -> int:
        """Distance to the *next* block's top. The canvas is taller than this.

        Measured on the type, not on the canvas: :func:`_descender_pad` is transparent slack
        that exists so the rasteriser cannot clip a descender, and counting it toward the
        rhythm inflates a single-line bullet's pitch from 84px to 107px. The pad is smaller
        than :data:`BULLET_GAP_RATIO`'s gap, so consecutive canvases still never share ink.
        """
        return len(self.lines) * self.line_height + round(self.size * BULLET_GAP_RATIO)


@dataclass(frozen=True)
class SlidePlan:
    """Everything about a slide's text except the pixels. Pure data, fully testable."""

    geometry: SlideGeometry
    heading_lines: list[str]
    heading_size: int
    heading_line_height: int
    heading_rect: Rect
    rule_rect: Rect | None
    heading_offset_y: int = 0
    """Air reserved *above* the heading's type inside ``heading_rect``.

    Non-zero when the box reserves :data:`MAX_HEADING_LINES` and the heading used fewer: the
    type is bottom-aligned, so a one-line and a two-line heading share a last baseline and
    everything under them stays put.
    """

    kicker: str = ""
    """The title card's eyebrow, or ``""``. Its own layer: different size, own entrance."""

    kicker_size: int = 0
    kicker_rect: Rect | None = None
    kicker_tracking: int = 0
    """``-kerning`` in pixels. Non-zero only for the kicker."""

    bullets: list[BulletBlock] = field(default_factory=list)
    scrim_opacity: float = 0.0
    scrim_region: Rect | None = None
    contrast: ContrastReport | None = None
    theme: Theme = field(default_factory=Theme)
    font: str = ""
    ink_pad: int = 0
    """Stroke bleed reserved inside every text canvas, in pixels.

    The dark outline extends half a stroke width *outside* the glyph, so drawing at x=0
    shaves the left edge of the first letter's halo. Reserving the bleed inside the
    column fixes that without moving the canvas outside the safe area — and one pad
    shared by the heading and the bullets keeps their left edges aligned, which a
    per-size pad would not.
    """

    @property
    def text_rects(self) -> list[Rect]:
        rects = [self.kicker_rect] if self.kicker_rect is not None else []
        if self.heading_lines:
            rects.append(self.heading_rect)
        return rects + [bullet.rect for bullet in self.bullets]

    def ink_rects(self) -> list[Rect]:
        """``text_rects`` narrowed to where ink can actually land.

        A layer's canvas spans the whole column even when the type inside it is centred,
        so ``text_rects`` massively over-reports how far left the words reach. ``offset_x``
        is exactly the inset the rasteriser applies, and for a centred block the right
        inset is the same by construction — which makes this tight, not a guess.

        Only used for collision checks (the watermark); layout still works in
        ``text_rects``, because that is what the PNGs are sized to.
        """
        rects: list[Rect] = []
        if self.kicker_rect is not None:
            rects.append(
                _inset_x(self.kicker_rect, self.ink_pad, self.geometry.align == "center")
            )
        if self.heading_lines:
            inset = self.ink_pad
            if self.geometry.align == "center" and self.rule_rect is not None:
                inset = max(inset, self.rule_rect.x - self.heading_rect.x)
            rects.append(_inset_x(self.heading_rect, inset, self.geometry.align == "center"))
        for bullet in self.bullets:
            rects.append(
                _inset_x(bullet.rect, bullet.offset_x, self.geometry.align == "center")
            )
        return rects

    def stack_height(self) -> int:
        box = Rect.union(self.text_rects)
        return box.height if box else 0

    def fits_column(self) -> bool:
        box = Rect.union(self.text_rects)
        if box is None:
            return True
        column = self.geometry.text_column
        return box.y >= column.y and box.bottom <= column.bottom and box.width <= column.width

    def within_safe_area(self) -> bool:
        """No text may touch a frame edge."""
        frame = self.geometry.frame
        margin_x = round(frame.width * (1.0 - SAFE_AREA) / 2)
        margin_y = round(frame.height * (1.0 - SAFE_AREA) / 2)
        return all(
            r.x >= margin_x
            and r.y >= margin_y
            and r.right <= frame.width - margin_x
            and r.bottom <= frame.height - margin_y
            for r in self.text_rects
        )


def bullet_times(
    bullets: Sequence[BulletPoint], plan: VisualPlan, *, first_at: float = FIRST_REVEAL_EARLIEST
) -> list[float]:
    """Reveal times, forced monotonic and at least ``plan.bullet_min_gap`` apart.

    Word timings can put two bullets 0.1s apart, which reads as one flash rather than a
    sequence, so the gap is a floor rather than a suggestion. ``first_at`` is the other
    floor — see :data:`FIRST_REVEAL_EARLIEST`.
    """
    times: list[float] = []
    previous = first_at - plan.bullet_min_gap
    for bullet in bullets:
        moment = max(float(bullet.appear_at), first_at, previous + plan.bullet_min_gap)
        times.append(round(moment, 3))
        previous = moment
    return times


def layout_slide(
    heading: str,
    bullets: Sequence[BulletPoint],
    plan: VisualPlan,
    profile: RenderProfile,
    *,
    theme: Theme | None = None,
    font: str | None = None,
    image_path: str | Path | None = None,
    measure_factory: Callable[[str, int], TextMeasurer] | None = None,
    role: SceneRole | None = None,
    emphasis_mode: str | None = None,
) -> SlidePlan:
    """Resolve heading + bullets into boxes, colours, timings and a scrim opacity.

    The one non-obvious step is the fit loop. Wrapping depends on the font size and the
    stack height depends on the wrapping, so the two are solved together: try the design
    size, and if the stack overflows its column, scale *both* sizes down uniformly and
    re-wrap. Uniform scaling is what keeps the heading/bullet size relationship — the
    thing that makes a slide look designed — constant.

    ``role`` drives the type scale (``SceneRole.heading_scale``), the treatment
    (:data:`ROLE_STYLES`) and the bullet count: the list is truncated to
    ``role.bullet_budget``, which is what makes a title card have no bullets even if the
    script wrote some. ``None`` infers the role from the layout — see
    :func:`role_for_layout`.

    An emphasised bullet is wrapped at its own face, which is why the wrapped bullets carry
    a font each; every other metric is shared, so the stack has one rhythm.
    """
    theme = theme or Theme()
    geometry = slide_geometry(plan, profile, theme=theme, role=role)
    role = geometry.role
    style = geometry.style
    font = font or find_font()
    factory = measure_factory or (lambda f, s: text_measurer(f, s))
    emphasis = Emphasis.resolve(theme, font, emphasis_mode)
    marker_shape = _marker_shape(theme)
    column = geometry.text_column
    heading_text = " ".join(heading.split())
    kicker_text = TITLE_KICKER.strip().upper() if style.kicker else ""
    # The budget is a hard cap, not a suggestion: a title card with a bullet on it is the
    # single clearest way to make an opener look like a content slide.
    points = [b for b in bullets if b.text.strip()][: max(0, role.bullet_budget)]

    best: tuple[list[str], int, int, list[_WrappedBullet]] | None = None
    for step in SHRINK_STEPS:
        heading_size = max(MIN_FONT_PX, round(geometry.heading_size * step))
        bullet_size = max(MIN_FONT_PX, round(geometry.bullet_size * step))
        # Wrap into the column minus the stroke bleed on both sides, or the outline of
        # the last word on a full line gets shaved by the canvas edge.
        pad = _ink_pad(heading_size)
        usable = max(1, column.width - 2 * pad)
        heading_lines = (
            wrap_to_width(
                heading_text,
                usable,
                factory(font, heading_size),
                max_lines=MAX_HEADING_LINES,
            )
            if heading_text
            else []
        )
        indent = _bullet_indent(bullet_size, marker_shape)
        wrapped = [
            _WrappedBullet(
                point=point,
                lines=wrap_to_width(
                    " ".join(point.text.split()),
                    usable - indent,
                    factory(emphasis.font_for(font, emphasis=point.emphasis), bullet_size),
                ),
                size=bullet_size,
                font=emphasis.font_for(font, emphasis=point.emphasis),
            )
            for point in points
        ]
        total = _stack_height(
            heading_lines, heading_size, wrapped, bullet_size, geometry, kicker=kicker_text
        )
        best = (heading_lines, heading_size, bullet_size, wrapped)
        if total <= column.height:
            break

    heading_lines, heading_size, bullet_size, wrapped = best or (
        [],
        geometry.heading_size,
        max(
            MIN_FONT_PX,
            round(geometry.bullet_size),
        ),
        [],
    )
    heading_line_height = round(heading_size * LINE_SPACING)
    ink_pad = _ink_pad(heading_size)
    total = _stack_height(
        heading_lines, heading_size, wrapped, bullet_size, geometry, kicker=kicker_text
    )

    if style.optical_centre_ratio is not None:
        # Optical centre, not the column's: a block centred on the geometric middle of the
        # frame reads as hanging slightly low.
        cursor = round(geometry.frame.height * style.optical_centre_ratio) - total // 2
        cursor = max(column.y, min(cursor, column.bottom - total))
    elif geometry.vertical_anchor == "center":
        cursor = column.y + max(0, (column.height - total) // 2)
    elif geometry.vertical_anchor == "bottom":
        cursor = max(column.y, column.bottom - total)
    else:
        cursor = column.y

    kicker_size = 0
    kicker_rect: Rect | None = None
    kicker_tracking = 0
    if kicker_text:
        kicker_size = type_size(TYPE_STEP_KICKER, geometry.frame.width)
        kicker_tracking = max(1, round(kicker_size * TITLE_KICKER_TRACKING_EM))
        kicker_height = round(kicker_size * KICKER_LINE_SPACING) + _descender_pad(
            kicker_size, BULLET_STROKE_RATIO
        )
        kicker_rect = Rect(column.x, cursor, column.width, kicker_height)
        cursor += _kicker_pitch(kicker_size)

    heading_rect = Rect(column.x, cursor, column.width, 0)
    rule_rect: Rect | None = None
    heading_offset_y = 0
    if heading_lines:
        # A *fixed* box wherever the stack is top-anchored: the rule and the first bullet
        # then land on the same y whether the heading took one line or two. Letting the box
        # follow the line count moves everything below it by a whole line between scenes,
        # which is the single largest contributor to "all over the place" (DIRECTION §2.1).
        reserved = MAX_HEADING_LINES if _fixed_heading_box(geometry) else len(heading_lines)
        # The gap is measured from the bottom of the *type*, not from the bottom of the
        # canvas: `_descender_pad` is transparent slack for the rasteriser, and adding it in
        # pushed the rule ~50px further from a 105px title than the spec's 48px.
        lines_shown = max(reserved, len(heading_lines))
        type_height = lines_shown * heading_line_height
        heading_offset_y = (lines_shown - len(heading_lines)) * heading_line_height
        rule_gap = round(heading_size * RULE_GAP_RATIO)
        rule_height = max(2, round(RULE_HEIGHT_PX * geometry.scale))
        # Fixed width, off the frame — not a share of the column, which gave the same
        # graphic element two widths in one video.
        rule_width = max(rule_height * 6, round(geometry.frame.width * style.rule_w_ratio))
        rule_y = type_height + rule_gap
        # Keep the rule off the canvas edge: a layer whose ink is flush with its own
        # boundary has nowhere to put antialiasing, and a SLIDE_* entry shows the seam.
        canvas_height = (
            max(
                type_height + _descender_pad(heading_size, HEADING_STROKE_RATIO),
                rule_y + rule_height,
            )
            + ink_pad
        )
        heading_rect = Rect(column.x, cursor, column.width, canvas_height)
        rule_x = column.x + (
            (column.width - rule_width) // 2 if geometry.align == "center" else ink_pad
        )
        rule_rect = Rect(rule_x, cursor + rule_y, rule_width, rule_height)
        cursor += canvas_height + round(bullet_size * HEADING_GAP_RATIO)

    # One gap, one gutter, one size for the whole stack.
    times = bullet_times([w.point for w in wrapped], plan)
    indent = _bullet_indent(bullet_size, marker_shape)
    offset_x = ink_pad
    if geometry.align == "center" and wrapped:
        widest = max(
            (
                factory(w.font, w.size)(line)
                for w in wrapped
                for line in w.lines
            ),
            default=0.0,
        )
        offset_x = max(ink_pad, round((column.width - (indent + widest)) / 2))

    outline = theme.scrim_colour
    blocks: list[BulletBlock] = []
    for wrapped_bullet, appear_at in zip(wrapped, times, strict=True):
        point = wrapped_bullet.point
        # Only a block in a *different* face needs its advance corrected; the base face
        # keeps the spacing it has always rendered with.
        spacing = None
        if wrapped_bullet.font != font:
            spacing = interline_spacing(
                wrapped_bullet.line_height,
                wrapped_bullet.size,
                advance=font_line_advance(wrapped_bullet.font, wrapped_bullet.size),
            )
        blocks.append(
            BulletBlock(
                lines=list(wrapped_bullet.lines),
                rect=Rect(column.x, cursor, column.width, wrapped_bullet.height),
                size=wrapped_bullet.size,
                line_height=wrapped_bullet.line_height,
                marker_diameter=_marker_diameter(bullet_size),
                indent=indent,
                emphasis=point.emphasis,
                appear_at=appear_at,
                text_colour=emphasis.text_colour(theme, emphasis=point.emphasis),
                marker_colour=theme.accent,
                offset_x=offset_x,
                font=wrapped_bullet.font,
                stroke_ratio=wrapped_bullet.stroke_ratio,
                faux_bold=emphasis.faux_bold_for(emphasis=point.emphasis),
                marker_shape=marker_shape,
                outline_colour=outline,
                interline_spacing=spacing,
            )
        )
        cursor += wrapped_bullet.pitch

    scrim_opacity, scrim_region, contrast = _resolve_scrim(
        geometry=geometry,
        theme=theme,
        plan=plan,
        rects=([kicker_rect] if kicker_rect else [])
        + ([heading_rect] if heading_lines else [])
        + [b.rect for b in blocks],
        image_path=image_path,
    )

    return SlidePlan(
        geometry=geometry,
        heading_lines=list(heading_lines),
        heading_size=heading_size,
        heading_line_height=heading_line_height,
        heading_rect=heading_rect,
        rule_rect=rule_rect,
        heading_offset_y=heading_offset_y,
        kicker=kicker_text,
        kicker_size=kicker_size,
        kicker_rect=kicker_rect,
        kicker_tracking=kicker_tracking,
        bullets=blocks,
        scrim_opacity=scrim_opacity,
        scrim_region=scrim_region,
        contrast=contrast,
        theme=theme,
        font=font,
        ink_pad=ink_pad,
    )


def _stroke_px(size: int, ratio: float) -> int:
    """Outline width. The floor of 2 is what makes small type relatively fatter."""
    return max(2, round(size * ratio))


def _descender_pad(size: int, stroke_ratio: float = BULLET_STROKE_RATIO) -> int:
    """Canvas slack below the last line: descender + outline + a pixel of antialiasing.

    The stroke term is not decorative. Because :func:`_stroke_px` has a 2px floor, an 18px
    draft bullet carries the same absolute outline as a 36px final one, so a purely
    proportional pad is ~4px short at draft resolution and clips the descenders.
    """
    return round(size * DESCENDER_PAD_RATIO) + 2 * _stroke_px(size, stroke_ratio) + 2


def _ink_pad(heading_size: int) -> int:
    """Horizontal stroke bleed, sized for the widest stroke on the slide (the heading's)."""
    return _stroke_px(heading_size, HEADING_STROKE_RATIO)


def _marker_shape(theme: Theme) -> str:
    """The one marker shape for this video, validated. Unknown values fall back to disc."""
    shape = str(getattr(theme, "marker", "disc")).strip().lower()
    if shape not in MARKER_SHAPES:
        logger.warning("unknown theme.marker %r; using 'disc'", shape)
        return "disc"
    return shape


def _marker_diameter(size: int) -> int:
    """Marker size, derived from the bullet size. One diameter for the whole slide.

    No emphasis term: a marker that grows for one bullet is a second marker, which is the
    inconsistency this design removed.
    """
    return max(4, round(size * MARKER_RATIO))


def _bullet_indent(size: int, shape: str = "disc") -> int:
    """Marker gutter: a fixed 1.0 em, whatever the shape is.

    Fixed rather than "marker width + gap" so the text edge is at the same x on every bullet
    of every scene — including under ``marker="none"``, where the gutter still has to exist
    for a wrapped line to hang to it. ``shape`` is accepted so the signature says out loud
    that the answer does not depend on it.
    """
    del shape
    return max(4, round(size * BULLET_GUTTER_EM))


def _fixed_heading_box(geometry: SlideGeometry) -> bool:
    """Should the heading canvas reserve :data:`MAX_HEADING_LINES` regardless of content?

    Yes wherever the stack is anchored to a fixed top, which is every body slide: the point
    is that the rule and the first bullet do not move between scenes. No for a centred block
    (the title card), where reserving an unused line would visibly push the title off centre
    and there is only one such slide to be consistent with anyway.
    """
    return geometry.vertical_anchor == "top"


KICKER_LINE_SPACING = 1.30
"""Line height multiple for the kicker. DIRECTION §2 — looser than body text, as caps want."""


def _kicker_pitch(kicker_size: int) -> int:
    """Kicker top to heading top. One line of caps plus its own size in air."""
    return round(kicker_size * (KICKER_LINE_SPACING + 1.2))


def _stack_height(
    heading_lines: Sequence[str],
    heading_size: int,
    wrapped: Sequence[_WrappedBullet],
    bullet_size: int,
    geometry: SlideGeometry,
    *,
    kicker: str = "",
) -> int:
    """Total height of the stack, using exactly the arithmetic :func:`layout_slide` uses."""
    total = 0
    if kicker:
        total += _kicker_pitch(type_size(TYPE_STEP_KICKER, geometry.frame.width))
    if heading_lines:
        reserved = MAX_HEADING_LINES if _fixed_heading_box(geometry) else len(heading_lines)
        type_height = max(reserved, len(heading_lines)) * round(heading_size * LINE_SPACING)
        rule_y = type_height + round(heading_size * RULE_GAP_RATIO)
        total += (
            max(
                type_height + _descender_pad(heading_size, HEADING_STROKE_RATIO),
                rule_y + max(2, round(RULE_HEIGHT_PX * geometry.scale)),
            )
            + _ink_pad(heading_size)
        )
        if wrapped:
            total += round(bullet_size * HEADING_GAP_RATIO)
    # Pitch between blocks, then the last block's full canvas — which is taller than its
    # pitch by the descender pad, and that slack has to be inside the column.
    for bullet in wrapped[:-1]:
        total += bullet.pitch
    if wrapped:
        total += wrapped[-1].height
    return total


def _resolve_scrim(
    *,
    geometry: SlideGeometry,
    theme: Theme,
    plan: VisualPlan,
    rects: Sequence[Rect],
    image_path: str | Path | None,
) -> tuple[float, Rect | None, ContrastReport]:
    """Decide the scrim, and report the contrast either way.

    On a solid background there is nothing to measure and nothing to hide behind: the
    palette gives the answer exactly, and a scrim over a flat brand colour just looks
    like a smudge. Only ``full_bleed`` needs the pixels sampled.

    The wash colour comes from :attr:`~app.core.models.Theme.scrim_colour`, so a light
    palette gets a *white* wash and the solver defends the dark tail rather than the
    bright one. Hardcoding a dark wash is how light-themed text disappears.
    """
    scrim_colour = theme.scrim_colour
    scrim_encoded = encoded_grey(scrim_colour)
    text_ratio = colour_contrast(theme.text, theme.bg)
    accent_ratio = colour_contrast(theme.accent, theme.bg)
    if not geometry.over_image:
        report = ContrastReport(
            source="theme",
            text_colour=theme.text,
            accent_colour=theme.accent,
            scrim_opacity=0.0,
            ratio_before=text_ratio,
            ratio_after=text_ratio,
            accent_ratio_after=accent_ratio,
            background_luminance_before=relative_luminance(theme.bg),
            background_luminance_after=relative_luminance(theme.bg),
            meets_aa=min(text_ratio, accent_ratio) >= WCAG_AA,
            detail=f"solid {theme.bg}; no scrim",
            scrim_colour=scrim_colour,
        )
        if not report.meets_aa:
            logger.warning(
                "theme contrast below WCAG AA: text %.2f:1, accent %.2f:1 on %s",
                text_ratio,
                accent_ratio,
                theme.bg,
            )
        return 0.0, None, report

    box = Rect.union(rects)
    region = (box.inflate(round(geometry.frame.width * 0.012)) if box else geometry.text_column)
    region = region.clamp_to(geometry.frame)
    text_luminance = relative_luminance(theme.text)
    accent_luminance = relative_luminance(theme.accent)
    probe = measure_region_luminance(image_path, region, geometry.frame) if image_path else None

    if probe is None:
        opacity = float(plan.scrim_opacity)
        detail = (
            "no image measured; using plan.scrim_opacity"
            if image_path is None
            else "luminance probe failed; using plan.scrim_opacity"
        )
        return (
            opacity,
            region,
            ContrastReport(
                source="image",
                text_colour=theme.text,
                accent_colour=theme.accent,
                scrim_opacity=opacity,
                ratio_before=float("nan"),
                ratio_after=float("nan"),
                accent_ratio_after=float("nan"),
                background_luminance_before=float("nan"),
                background_luminance_after=float("nan"),
                meets_aa=False,
                detail=detail,
                scrim_colour=scrim_colour,
            ),
        )

    # Which tail binds is a property of the palette, not of the image: a dark wash is
    # fighting the highlights, a light wash is fighting the shadows.
    darkening = scrim_encoded < linear_to_srgb(text_luminance)
    before_encoded = probe.tail(low=not darkening)

    # Both inks have to clear AA. Under a dark palette amber is the dimmer of the two and
    # binds; the accent stays in the constraint either way because it still colours the
    # heading rule and the bullet markers, which have to be seen.
    needed_text = required_scrim_opacity(
        before_encoded, text_luminance=text_luminance, scrim_encoded=scrim_encoded
    )
    needed_accent = required_scrim_opacity(
        before_encoded, text_luminance=accent_luminance, scrim_encoded=scrim_encoded
    )
    opacity = quantise_opacity(max(needed_text, needed_accent, SCRIM_MIN_TINT))
    after = probe.after_scrim(opacity, scrim_encoded=scrim_encoded)
    after_encoded = after.tail(low=not darkening)
    before_relative = srgb_to_linear(before_encoded)
    after_relative = srgb_to_linear(after_encoded)
    report = ContrastReport(
        source="image",
        text_colour=theme.text,
        accent_colour=theme.accent,
        scrim_opacity=opacity,
        ratio_before=contrast_ratio(text_luminance, before_relative),
        ratio_after=contrast_ratio(text_luminance, after_relative),
        accent_ratio_after=contrast_ratio(accent_luminance, after_relative),
        background_luminance_before=before_relative,
        background_luminance_after=after_relative,
        meets_aa=min(
            contrast_ratio(text_luminance, after_relative),
            contrast_ratio(accent_luminance, after_relative),
        )
        >= WCAG_AA,
        detail=(
            f"probe mean={probe.mean:.4f} sd={probe.stddev:.4f} "
            f"{'p+' if darkening else 'p-'}={before_encoded:.4f} (encoded sRGB) "
            f"over {region.as_tuple()}, wash {scrim_colour}"
        ),
        scrim_colour=scrim_colour,
    )
    return opacity, region, report


# ------------------------------------------------------------- rasterisation


def _background_argv(slide: SlidePlan) -> list[str]:
    """Full-frame background: brand fill with a rounded hole where the image goes.

    The hole is what makes ordering irrelevant. The filtergraph can composite this over
    the (moving, zooming) image and get a rounded hero card on a solid background; it
    never has to know that the background is "behind" anything.
    """
    geometry = slide.geometry
    frame = geometry.frame
    theme = slide.theme
    argv = ["-size", f"{frame.width}x{frame.height}", f"xc:{imagemagick_colour(theme.bg)}"]
    region = geometry.image_region
    if region is None:
        return [*argv, "-strip"]

    radius = geometry.image_radius
    inset = max(0, round(frame.width * CARD_FRAME_RATIO))
    if inset:
        # A hair of `surface` around the hole so the hero reads as a deliberate card.
        frame_box = region.inflate(inset).clamp_to(frame)
        argv += [
            "-fill", imagemagick_colour(theme.surface),
            "-draw",
            _round_rect(frame_box, radius + inset),
        ]
    # White = keep, black = punch through. CopyOpacity turns the mask into alpha.
    argv += [
        "(",
        "-size", f"{frame.width}x{frame.height}", "xc:white",
        "-fill", "black",
        "-draw", _round_rect(region, radius),
        ")",
        "-alpha", "off",
        "-compose", "CopyOpacity",
        "-composite",
        "-strip",
    ]
    return argv


def _round_rect(box: Rect, radius: int) -> str:
    x2, y2 = box.right - 1, box.bottom - 1
    if radius <= 0:
        return f"rectangle {box.x},{box.y} {x2},{y2}"
    limit = max(0, min(radius, box.width // 2 - 1, box.height // 2 - 1))
    if limit <= 0:
        return f"rectangle {box.x},{box.y} {x2},{y2}"
    return f"roundrectangle {box.x},{box.y} {x2},{y2} {limit},{limit}"


def _scrim_argv(slide: SlidePlan) -> list[str]:
    """Adaptive scrim for ``full_bleed``, feathered so it has no visible edge.

    The wash is :attr:`~app.core.models.Theme.scrim_colour`: black under a dark palette's
    light text, white under a light palette's dark text. The local names below say "dark"
    because that is the common case, but nothing here assumes it.
    """
    geometry = slide.geometry
    frame = geometry.frame
    opacity = slide.scrim_opacity
    argv = ["-size", f"{frame.width}x{frame.height}", "xc:none"]
    if opacity <= 0.0:
        return [*argv, "-strip"]
    wash = slide.theme.scrim_colour
    dark, clear = _rgba(opacity, wash), _rgba(0.0, wash)
    region = (slide.scrim_region or geometry.text_column).clamp_to(frame)
    feather = max(8, round(frame.width * SCRIM_FEATHER_RATIO))

    if geometry.align == "left":
        solid_w = min(frame.width, region.right)
        argv += ["(", "-size", f"{solid_w}x{frame.height}", f"xc:{dark}", ")",
                 "-geometry", "+0+0", "-composite"]
        band = min(feather, frame.width - solid_w)
        if band > 0:
            # `gradient:` runs top-to-bottom, so build it rotated and turn it upright.
            argv += ["(", "-size", f"{frame.height}x{band}", f"gradient:{dark}-{clear}",
                     "-rotate", "-90", ")", "-geometry", f"+{solid_w}+0", "-composite"]
        return [*argv, "-strip"]

    if geometry.vertical_anchor == "center":
        return [*argv, "(", "-size", f"{frame.width}x{frame.height}", f"xc:{dark}", ")",
                "-geometry", "+0+0", "-composite", "-strip"]

    if geometry.vertical_anchor == "top":
        solid_h = min(frame.height, region.bottom)
        argv += ["(", "-size", f"{frame.width}x{solid_h}", f"xc:{dark}", ")",
                 "-geometry", "+0+0", "-composite"]
        band = min(feather, frame.height - solid_h)
        if band > 0:
            argv += ["(", "-size", f"{frame.width}x{band}", f"gradient:{dark}-{clear}", ")",
                     "-geometry", f"+0+{solid_h}", "-composite"]
        return [*argv, "-strip"]

    solid_top = max(0, region.y)
    argv += ["(", "-size", f"{frame.width}x{frame.height - solid_top}", f"xc:{dark}", ")",
             "-geometry", f"+0+{solid_top}", "-composite"]
    band = min(feather, solid_top)
    if band > 0:
        argv += ["(", "-size", f"{frame.width}x{band}", f"gradient:{clear}-{dark}", ")",
                 "-geometry", f"+0+{solid_top - band}", "-composite"]
    return [*argv, "-strip"]


@dataclass(frozen=True)
class MarkerSpec:
    """The bullet marker, in canvas-local pixels.

    ``shape`` comes from :attr:`~app.core.models.Theme.marker` and is identical for every
    bullet in the video — it is a property of the brand, not of the sentence next to it. It
    is drawn in ``colour`` (the accent) because a marker is a graphic element, not text, so
    ``uniform_text`` says nothing about it.
    """

    cx: int
    cy: int
    radius: int
    colour: str
    shape: str = "disc"
    ring_width: int = 2
    outline: str = "#000000"


@dataclass(frozen=True)
class RuleSpec:
    """The heading's accent rule, in canvas-local pixels."""

    x: int
    y: int
    width: int
    height: int
    colour: str


def _marker_argv(marker: MarkerSpec, *, size: int) -> list[str]:
    """Draw the marker. Coordinates are absolute, so this must precede any ``-gravity``.

    ``radius`` is the **outer edge of the accent ink** in every shape, so all five occupy
    the same optical box and the gutter geometry does not depend on which one the theme
    picked.

    Every shape is drawn twice: an over-wide pass in the outline colour lays down the halo
    that keeps the mark visible over a photograph, then the accent ink goes on top. That is
    the only way to get "stroke outside the stroke" out of ImageMagick.
    """
    outline_w = max(1, round(size * 0.05))
    shape = marker.shape if marker.shape in MARKER_SHAPES else "disc"
    accent = imagemagick_colour(marker.colour)
    halo = imagemagick_colour(marker.outline)

    if shape == "none":
        return []

    if shape == "disc":
        return [
            "-fill", accent,
            "-stroke", halo,
            "-strokewidth", str(outline_w),
            "-draw", f"circle {marker.cx},{marker.cy} {marker.cx},{marker.cy - marker.radius}",
            "-stroke", "none",
        ]

    if shape == "ring":
        ring = max(2, marker.ring_width)
        # Pull the ring's *path* in by half its width so the band's outer edge lands on
        # `radius`, and give the halo the same reach beyond the ink as the disc's has.
        path_r = max(1, round(marker.radius - ring / 2))
        circle = f"circle {marker.cx},{marker.cy} {marker.cx},{marker.cy - path_r}"
        return [
            "-fill", "none",
            "-stroke", halo,
            "-strokewidth", str(ring + outline_w),
            "-draw", circle,
            "-stroke", accent,
            "-strokewidth", str(ring),
            "-draw", circle,
            "-stroke", "none",
            "-fill", "none",
        ]

    if shape == "dash":
        # A short accent rule, in the same graphic language as the heading's rule. There is
        # no "hollow dash", so the failure this design removed is structurally impossible.
        thickness = max(2, round(size * MARKER_DASH_H_RATIO))
        length = max(4, round(size * MARKER_DASH_W_RATIO))
        half = max(1, thickness // 2)
        x1 = marker.cx - length // 2
        x2 = x1 + length - 1
        bar = f"rectangle {x1},{marker.cy - half} {x2},{marker.cy - half + thickness - 1}"
        return [
            "-stroke", halo,
            "-strokewidth", str(outline_w * 2),
            "-fill", halo,
            "-draw", bar,
            "-stroke", "none",
            "-fill", accent,
            "-draw", bar,
        ]

    # chevron: a small ">" pointing into the text.
    stroke = max(2, round(marker.radius * 2 * MARKER_CHEVRON_RATIO))
    arm = max(2, round(marker.radius * 0.95))
    tip_x = marker.cx + arm
    back_x = marker.cx - arm
    polyline = (
        f"polyline {back_x},{marker.cy - arm} {tip_x},{marker.cy} {back_x},{marker.cy + arm}"
    )
    return [
        "-fill", "none",
        "-stroke", halo,
        "-strokewidth", str(stroke + outline_w),
        "-draw", polyline,
        "-stroke", accent,
        "-strokewidth", str(stroke),
        "-draw", polyline,
        "-stroke", "none",
        "-fill", "none",
    ]


def _text_argv(
    *,
    lines: Sequence[str],
    text_file: Path,
    canvas: Rect,
    size: int,
    line_height: int,
    colour: str,
    font: str,
    stroke_ratio: float,
    offset_x: int,
    align: str,
    marker: MarkerSpec | None = None,
    rule: RuleSpec | None = None,
    outline_colour: str = "#000000",
    faux_bold: float = 0.0,
    spacing: int | None = None,
    kerning: int = 0,
    offset_y: int = 0,
) -> list[str]:
    """Canvas the exact size of the block, with the text drawn twice (or three times).

    Pass one is a fat ``outline_colour`` stroke, pass two the ``colour`` face on top. That
    outline is what keeps text readable when the pixels behind it are busy — a plain fill
    disappears into any high-frequency image, scrim or not. Its colour has to be the
    *opposite polarity to the text*: black under a dark palette's light type, white under a
    light palette's dark type. A dark halo around dark type on a pale background does not
    separate the glyph, it thickens it into a smudge.

    ``faux_bold`` adds a third pass — a thin stroke in the *fill* colour, which genuinely
    fattens the letterform. It exists only for machines with no heavier face installed;
    see :class:`Emphasis`.

    ``offset_y`` pushes the block down inside its canvas, which is how a heading is
    bottom-aligned in a fixed-height box: the reserved-but-unused line ends up as air *above*
    the type instead of a hole between the type and the rule.

    ``-gravity northwest`` + ``-annotate +x+y`` puts the text block's top-left corner at
    ``(x, y)``. Between baselines it advances *the face's own line height* plus
    ``-interline-spacing`` — not ``pointsize`` plus it, which is what this code assumed
    until a 900-weight face made the gap impossible to ignore. ``spacing`` lets the caller
    pass a value measured for the actual face (see :func:`interline_spacing`) so the
    rendered advance equals ``line_height`` exactly; ``None`` keeps the historical
    ``line_height - size``.
    """
    argv = ["-size", f"{canvas.width}x{canvas.height}", "xc:none"]
    if marker is not None:
        argv += _marker_argv(marker, size=size)
    if rule is not None:
        argv += [
            "-fill", imagemagick_colour(rule.colour),
            "-stroke", "none",
            "-draw",
            f"rectangle {rule.x},{rule.y} {rule.x + rule.width - 1},{rule.y + rule.height - 1}",
        ]
    if not lines:
        return [*argv, "-strip"]

    text_arg = imagemagick_text_arg(text_file)
    gravity = "north" if align == "center" else "northwest"
    stroke = max(2, round(size * stroke_ratio))
    fill = imagemagick_colour(colour)
    argv += [
        "-font", font,
        "-pointsize", str(size),
        "-interline-spacing", str(line_height - size if spacing is None else spacing),
        # Tracking. Only the kicker asks for it; everything else must stay at the face's own
        # metrics, because *looser* type on one line is exactly the "different font" signal
        # that made the emphasised bullets read as a rendering fault.
        "-kerning", str(kerning),
        "-gravity", gravity,
        "-stroke", imagemagick_colour(outline_colour),
        "-strokewidth", str(stroke),
        "-fill", imagemagick_colour(outline_colour),
        "-annotate", f"+{offset_x}+{offset_y}", text_arg,
    ]
    if faux_bold > 0.0:
        argv += [
            "-stroke", fill,
            "-strokewidth", str(max(1, round(size * faux_bold))),
            "-fill", fill,
            "-annotate", f"+{offset_x}+{offset_y}", text_arg,
        ]
    argv += [
        "-stroke", "none",
        "-fill", fill,
        "-annotate", f"+{offset_x}+{offset_y}", text_arg,
        "-strip",
    ]
    return argv


def build_scene_text(
    heading: str,
    bullets: list[BulletPoint],
    plan: VisualPlan,
    profile: RenderProfile,
    workdir: Path,
    *,
    image_path: str | Path | None = None,
    theme: Theme | None = None,
    font: str | None = None,
    slide: SlidePlan | None = None,
    role: SceneRole | None = None,
    emphasis_mode: str | None = None,
) -> SceneText:
    """Rasterise one slide's text and return positioned, timed layers.

    ``image_path`` is only read for ``full_bleed``, to measure the luminance behind the
    text and size the scrim; solid layouts never touch it.

    ``role`` is ``scene.role`` — it sets the type scale, the treatment and the bullet
    budget. Both it and ``emphasis_mode`` are optional keywords with inferred defaults, so
    a caller that does not know about roles keeps working unchanged.

    Layers: one full-frame ``scrim`` (the brand background with the hero hole punched out,
    or the adaptive dark gradient for ``full_bleed``), one ``heading``, one ``bullet`` per
    point, and — on a title card only — one ``kicker``. Every ``x/y/width/height`` is in
    output-frame pixels and is the layer's *final* resting place.
    """
    binary = require_imagemagick()
    slide = slide or layout_slide(
        heading,
        bullets,
        plan,
        profile,
        theme=theme,
        font=font,
        image_path=image_path,
        role=role,
        emphasis_mode=emphasis_mode,
    )
    workdir = Path(workdir)
    geometry = slide.geometry
    scene = SceneText()
    slide_distance = max(4, round(profile.width * SLIDE_DISTANCE_RATIO))
    bullet_slide_distance = max(3, round(profile.width * BULLET_SLIDE_DISTANCE_RATIO))

    background = geometry.over_image
    if background:
        argv = _scrim_argv(slide)
        key: tuple[object, ...] = (
            "scrim",
            geometry.frame.as_tuple(),
            geometry.align,
            geometry.vertical_anchor,
            slide.scrim_opacity,
            (slide.scrim_region or geometry.text_column).as_tuple(),
        )
    else:
        argv = _background_argv(slide)
        key = (
            "bg",
            geometry.frame.as_tuple(),
            slide.theme.bg,
            slide.theme.surface,
            geometry.image_radius,
            geometry.image_region.as_tuple() if geometry.image_region else None,
        )
    scrim_png = cache_path(workdir, "scrim", *key)
    render_cached_png(argv, scrim_png, binary=binary)
    scene.layers.append(
        TextLayer(
            png_path=scrim_png,
            x=0,
            y=0,
            width=geometry.frame.width,
            height=geometry.frame.height,
            appear_at=0.0,
            animation=TextAnimation.NONE,
            anim_duration=0.0,
            kind="scrim",
        )
    )

    with tempfile.TemporaryDirectory(prefix="slide-text-") as tmp:
        tmpdir = Path(tmp)

        outline = slide.theme.scrim_colour
        if slide.kicker and slide.kicker_rect is not None:
            kicker_key = (
                slide.kicker,
                slide.kicker_size,
                slide.kicker_tracking,
                slide.kicker_rect.as_tuple(),
                slide.theme.text,
                outline,
                geometry.align,
                slide.font,
            )
            kicker_png = cache_path(workdir, "kicker", *kicker_key)
            if not kicker_png.exists():
                text_file = write_text_file(tmpdir, "kicker.txt", [slide.kicker])
                render_cached_png(
                    _text_argv(
                        lines=[slide.kicker],
                        text_file=text_file,
                        canvas=slide.kicker_rect,
                        size=slide.kicker_size,
                        line_height=round(slide.kicker_size * KICKER_LINE_SPACING),
                        # `theme.text`, not `theme.muted`: DIRECTION §6.3 asks for muted, but
                        # `uniform_text` says every word in the video is one colour and the
                        # rejection we are fixing was about inconsistency. A second text
                        # colour on the very first slide is not the place to spend that.
                        colour=slide.theme.text,
                        font=slide.font,
                        stroke_ratio=BULLET_STROKE_RATIO,
                        offset_x=0 if geometry.align == "center" else slide.ink_pad,
                        align=geometry.align,
                        outline_colour=outline,
                        kerning=slide.kicker_tracking,
                    ),
                    kicker_png,
                    binary=binary,
                )
            scene.layers.append(
                TextLayer(
                    png_path=kicker_png,
                    x=slide.kicker_rect.x,
                    y=slide.kicker_rect.y,
                    width=slide.kicker_rect.width,
                    height=slide.kicker_rect.height,
                    appear_at=0.0,
                    # A plain fade: the kicker is a label, and two elements rising together
                    # would read as one block moving.
                    animation=TextAnimation.FADE_IN,
                    anim_duration=min(plan.anim_duration, KICKER_ANIM_DURATION),
                    slide_distance=slide_distance,
                    # A fourth kind, which `SceneText.sorted_layers` ranks after the bullets.
                    # Harmless — the kicker's box overlaps nothing, so its composite order
                    # cannot matter — and it keeps `kind == "heading"` meaning exactly one
                    # layer for every existing caller.
                    kind="kicker",
                )
            )

        if slide.heading_lines:
            rule = None
            if slide.rule_rect is not None:
                rule = RuleSpec(
                    x=slide.rule_rect.x - slide.heading_rect.x,
                    y=slide.rule_rect.y - slide.heading_rect.y,
                    width=slide.rule_rect.width,
                    height=slide.rule_rect.height,
                    colour=slide.theme.accent,
                )
            heading_key = (
                tuple(slide.heading_lines),
                slide.heading_size,
                slide.heading_line_height,
                slide.heading_rect.as_tuple(),
                # The heading is always `theme.text`; `accent` is in the key because it
                # colours the rule, and the outline because it flips on a light palette.
                slide.theme.text,
                slide.theme.accent,
                outline,
                geometry.align,
                slide.font,
                slide.ink_pad,
                slide.heading_offset_y,
                rule,
            )
            heading_png = cache_path(workdir, "heading", *heading_key)
            if not heading_png.exists():
                text_file = write_text_file(tmpdir, "heading.txt", slide.heading_lines)
                render_cached_png(
                    _text_argv(
                        lines=slide.heading_lines,
                        text_file=text_file,
                        canvas=slide.heading_rect,
                        size=slide.heading_size,
                        line_height=slide.heading_line_height,
                        colour=slide.theme.text,
                        font=slide.font,
                        stroke_ratio=HEADING_STROKE_RATIO,
                        # Centred text is placed from the canvas centre, and the reduced
                        # wrap width already leaves the bleed clear on both sides.
                        offset_x=0 if geometry.align == "center" else slide.ink_pad,
                        align=geometry.align,
                        rule=rule,
                        outline_colour=outline,
                        offset_y=slide.heading_offset_y,
                    ),
                    heading_png,
                    binary=binary,
                )
            scene.layers.append(
                TextLayer(
                    png_path=heading_png,
                    x=slide.heading_rect.x,
                    y=slide.heading_rect.y,
                    width=slide.heading_rect.width,
                    height=slide.heading_rect.height,
                    appear_at=0.0,
                    animation=plan.heading_animation,
                    anim_duration=plan.anim_duration,
                    slide_distance=slide_distance,
                    kind="heading",
                )
            )

        for index, block in enumerate(slide.bullets):
            # Optical centre of the first line's cap height, measured against IM's own
            # `-annotate` advance (see _text_argv): ~0.52 em below the block top.
            marker_cy = round(block.size * 0.52)
            radius = max(2, block.marker_diameter // 2)
            # Centred in the gutter, so every mark in the stack sits on one vertical axis.
            marker_cx = block.offset_x + max(radius, block.indent // 2)
            marker = MarkerSpec(
                cx=marker_cx,
                cy=marker_cy,
                radius=radius,
                colour=block.marker_colour,
                shape=block.marker_shape,
                ring_width=block.marker_ring_width,
                outline=block.outline_colour,
            )
            font = block.font or slide.font
            bullet_key = (
                tuple(block.lines),
                block.size,
                block.line_height,
                block.rect.as_tuple(),
                block.text_colour,
                block.marker_colour,
                block.marker_diameter,
                block.indent,
                block.offset_x,
                font,
                # Everything that carries emphasis without colour has to be in the key,
                # or two bullets that differ only in weight share a PNG.
                block.stroke_ratio,
                block.faux_bold,
                block.marker_shape,
                block.outline_colour,
                block.interline_spacing,
            )
            bullet_png = cache_path(workdir, "bullet", *bullet_key)
            if not bullet_png.exists():
                text_file = write_text_file(tmpdir, f"bullet{index}.txt", block.lines)
                render_cached_png(
                    _text_argv(
                        lines=block.lines,
                        text_file=text_file,
                        canvas=block.rect,
                        size=block.size,
                        line_height=block.line_height,
                        colour=block.text_colour,
                        font=font,
                        stroke_ratio=block.stroke_ratio,
                        offset_x=block.offset_x + block.indent,
                        align="left",
                        marker=marker,
                        outline_colour=block.outline_colour,
                        faux_bold=block.faux_bold,
                        spacing=block.interline_spacing,
                    ),
                    bullet_png,
                    binary=binary,
                )
            scene.layers.append(
                TextLayer(
                    png_path=bullet_png,
                    x=block.rect.x,
                    y=block.rect.y,
                    width=block.rect.width,
                    height=block.rect.height,
                    appear_at=block.appear_at,
                    animation=plan.bullet_animation,
                    # Smaller type travels less and for less time than the heading — see
                    # BULLET_ANIM_DURATION. `VisualPlan` carries one duration for the scene,
                    # so the split has to happen at the layer, which is where it belongs.
                    anim_duration=min(plan.anim_duration, BULLET_ANIM_DURATION),
                    slide_distance=bullet_slide_distance,
                    kind="bullet",
                )
            )

    if slide.contrast is not None:
        logger.debug("scene text contrast: %s", slide.contrast.summary())
    return scene


# ------------------------------------------------------------------- watermark
#
# The brand mark is composited once over the *assembled* video (see
# `FFmpegBackend.assemble`), so all this module owes it is a correctly sized RGBA PNG.
#
# The catch is SVG. ImageMagick delegates SVG to `rsvg-convert` when that binary is on
# PATH; without it the built-in MSVG renderer takes over, and MSVG implements neither
# `<mask>` nor `<filter>`. On this machine that is not a subtle degradation — the app's
# own favicon, whose highlights live in a masked group of blurred ellipses, comes out as
# black blobs over the mark. So when there is no real renderer we rasterise only the
# root's direct `<path>` children: the flat vector shapes, which at a 49px watermark is
# every pixel a viewer could have resolved anyway.

SVG_SUFFIXES = frozenset({".svg", ".svgz"})

LOGO_MIN_ALPHA_COVERAGE = 0.02
"""Mean alpha below which a rasterised mark is treated as a failed render.

A blank or near-blank PNG means the rasteriser silently gave up. Branding is optional, so
the answer is to skip it and say so, never to composite an invisible layer.
"""


def _svg_delegate_available() -> bool:
    """True when ImageMagick will hand SVG to a renderer that implements the whole spec."""
    return shutil.which("rsvg-convert") is not None


def flatten_svg_paths(svg_text: str) -> str | None:
    """The root ``<svg>`` with only its direct ``<path>`` children kept, or ``None``.

    Drops ``<defs>``, ``<mask>`` and ``<g>`` — everything whose rendering needs the parts
    of the SVG spec ImageMagick's built-in renderer does not have. Parsed with
    ``ElementTree`` rather than a regex because an XML document deserves an XML parser and
    because a half-matched tag would produce a file that renders as nothing at all.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415 - only needed on the SVG path

    svg_ns = "http://www.w3.org/2000/svg"
    try:
        root = ET.fromstring(svg_text)  # noqa: S314 - our own repo asset, not user input
    except ET.ParseError as exc:
        logger.warning("logo SVG did not parse (%s)", exc)
        return None
    paths = [child for child in root if child.tag in ("path", f"{{{svg_ns}}}path")]
    if not paths:
        return None
    ET.register_namespace("", svg_ns)
    flat = ET.Element(root.tag, dict(root.attrib))
    flat.extend(paths)
    return ET.tostring(flat, encoding="unicode")


def _logo_source_argv(source: Path, workdir: Path) -> list[str]:
    """Input arguments for ``source``, pre-flattening an SVG when we must.

    ``-background none`` has to precede the input for the transparency to survive, which
    is why this returns the leading arguments rather than just a path.
    """
    if source.suffix.lower() not in SVG_SUFFIXES or _svg_delegate_available():
        return ["-background", "none", str(source)]
    flat = flatten_svg_paths(source.read_text(encoding="utf-8"))
    if flat is None:
        logger.warning(
            "no rsvg-convert and %s has no top-level <path>; rasterising it as-is, "
            "which ImageMagick's built-in SVG renderer may get wrong",
            source.name,
        )
        return ["-background", "none", str(source)]
    logger.debug(
        "no rsvg-convert on PATH; rasterising %s from its base paths only "
        "(masks and filters dropped)",
        source.name,
    )
    flat_path = workdir / f"{source.stem}.flat.svg"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    flat_path.write_text(flat, encoding="utf-8")
    return ["-background", "none", str(flat_path)]


def rasterise_logo(
    source: str | Path,
    height: int,
    opacity: float,
    workdir: Path,
    *,
    binary: str | None = None,
) -> Path | None:
    """Rasterise the brand mark to an RGBA PNG of exactly ``height`` pixels, or ``None``.

    Scaling *and* the opacity multiply happen here rather than in the filtergraph: an SVG
    rasterises natively at any size, so asking for the final height is strictly sharper
    than rasterising small and letting ffmpeg scale up, and a pre-multiplied PNG keeps the
    assemble filtergraph down to a single ``overlay``.

    Cached under ``workdir`` and written atomically (see :func:`render_cached_png`),
    keyed on everything that changes the pixels, so the render pays for it once.

    Returns ``None`` — never raises — whenever branding cannot be produced: no source, no
    ImageMagick, a rasteriser that failed, or output with no ink in it. Branding is a
    finishing touch and must not be able to fail a render.
    """
    source = Path(source)
    if not source.is_file():
        logger.warning("brand logo not found, skipping watermark: %s", source)
        return None
    binary = binary if binary is not None else imagemagick_bin()
    if not binary:
        logger.warning("no ImageMagick; skipping the brand watermark")
        return None

    height = max(1, int(height))
    opacity = _clamp(float(opacity), 0.0, 1.0)
    workdir = Path(workdir)
    try:
        stat = source.stat()
        argv = [
            *_logo_source_argv(source, workdir),
            "-resize", f"x{height}",
            # Resize ringing overshoots, and on a Q16 *HDRI* build the overshoot is kept
            # rather than clipped — alpha came out at 1.043, so a 0.85 opacity rendered as
            # 0.886. Clamp back into gamut before the multiply so `logo_opacity` is the
            # ceiling it claims to be.
            "-clamp",
            # Multiply the existing alpha rather than replacing it, so antialiased edges
            # stay antialiased instead of going fully opaque.
            "-channel", "A", "-evaluate", "multiply", f"{opacity:.4f}", "+channel",
            "-strip",
        ]
        out = cache_path(
            workdir, "logo", str(source), stat.st_mtime_ns, stat.st_size, height, opacity
        )
        render_cached_png(argv, out, binary=binary)
    except (OSError, ff.FFmpegError) as exc:
        logger.warning("could not rasterise the brand logo %s (%s); skipping", source, exc)
        return None

    coverage = _alpha_coverage(out, binary=binary)
    if coverage is not None and coverage < LOGO_MIN_ALPHA_COVERAGE:
        logger.warning(
            "rasterised brand logo %s is effectively blank (alpha coverage %.4f); skipping",
            source.name,
            coverage,
        )
        return None
    return out


def _alpha_coverage(path: Path, *, binary: str) -> float | None:
    """Mean alpha of a PNG, 0..1. ``None`` when it cannot be measured."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, no shell
            [binary, str(path), "-alpha", "extract", "-format", "%[fx:mean]", "info:"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


# ------------------------------------------------------------------ strategy


def resolve_text_mode(mode: str = "auto") -> str:
    """Pick a text strategy: ``drawtext`` | ``png`` | ``scrim``."""
    if mode != "auto":
        return mode
    if ff.has_filter("drawtext"):
        return "drawtext"
    if imagemagick_bin():
        logger.warning(
            "ffmpeg at %s has no drawtext filter (built without libfreetype); "
            "rendering headings via an ImageMagick PNG overlay instead",
            ff.ffmpeg_bin(),
        )
        return "png"
    logger.error(
        "no drawtext filter and no ImageMagick: slides will get a scrim but NO heading text"
    )
    return "scrim"
