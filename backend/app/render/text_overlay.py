"""Burn the heading over the slide image, legibly, on any image.

Two independent problems:

*Legibility.* A generated image can be any brightness. White text on a bright sky is
invisible, so every heading sits on a dark scrim and carries a dark outline. Without
the scrim this is the single most common way "image with text" output looks broken.

*Text rendering.* The preferred path is ffmpeg's ``drawtext``, but that filter only
exists when ffmpeg was built with libfreetype, and plenty of builds are not — notably
the current Homebrew ``ffmpeg`` bottle on macOS, whose formula no longer depends on
freetype at all. So text rendering is a strategy:

===========  =======================================================================
``drawtext`` scrim + text drawn entirely inside the filtergraph. Preferred.
``png``      scrim + text rasterised to an RGBA PNG by ImageMagick and composited
             with ``overlay``. Identical geometry, works on any ffmpeg build.
``scrim``    last resort: scrim only, no text. Loud warning; the slide is still usable.
===========  =======================================================================

Geometry (font size, wrapping, safe area, scrim band) is computed once in
:func:`layout_heading` and shared by every strategy so they stay visually identical.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.models import RenderProfile, TextPosition, VisualPlan
from app.render import ffmpeg as ff

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


class FontNotFoundError(RuntimeError):
    pass


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


# -------------------------------------------------------------------- layout


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
    else:  # LOWER_THIRD
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


def scrim_filter(layout: TextLayout) -> str:
    """The dark scrim behind the text, as ``drawbox`` filters.

    ``drawbox`` cannot do a gradient, so the feather band is approximated with a
    single half-opacity box. Visually close enough to the PNG path, and ``drawbox``
    exists in every ffmpeg build.
    """
    parts = []
    for y, band_height, kind in layout.scrim_bands():
        if band_height <= 0:
            continue
        opacity = layout.scrim_opacity if kind == "solid" else round(layout.scrim_opacity / 2, 4)
        parts.append(
            f"drawbox=x=0:y={y}:w={layout.width}:h={band_height}:color=black@{opacity}:t=fill"
        )
    return ",".join(parts)


def drawtext_filters(layout: TextLayout, *, font: str | None = None) -> str:
    """Comma-joined ``drawbox`` + one ``drawtext`` per line.

    One filter per line (instead of an embedded ``\\n``) keeps control of the line
    spacing and lets each line be centred independently.
    """
    font_path = escape_filter_path(font or find_font())
    parts = [scrim_filter(layout)]
    border = max(2, round(layout.font_size * 0.045))
    for index, line in enumerate(layout.lines):
        parts.append(
            f"drawtext=fontfile={font_path}"
            f":text={escape_drawtext(line)}"
            f":expansion=none"
            f":fontsize={layout.font_size}"
            f":fontcolor=white"
            f":borderw={border}:bordercolor=black@0.9"
            f":x=(w-text_w)/2:y={layout.line_y(index)}"
        )
    return ",".join(parts)


# ------------------------------------------------------------- png text layer


def imagemagick_bin() -> str | None:
    """``magick`` (IM7) or ``convert`` (IM6), or None."""
    return (
        os.environ.get("IMAGEMAGICK_BIN")
        or shutil.which("magick")
        or shutil.which("convert")
    )


def _rgba(opacity: float) -> str:
    return f"rgba(0,0,0,{round(opacity, 4)})"


def render_text_png(
    heading: str,
    plan: VisualPlan,
    profile: RenderProfile,
    out_path: Path,
    *,
    font: str | None = None,
    layout: TextLayout | None = None,
) -> Path:
    """Rasterise scrim + heading to a transparent RGBA PNG via ImageMagick.

    The scrim is a *gradient* band here rather than a hard-edged box — a visible
    scrim edge across the frame is more distracting than the scrim itself.
    """
    binary = imagemagick_bin()
    if binary is None:
        raise RuntimeError("ImageMagick not found; cannot render the PNG text layer")
    layout = layout or layout_heading(heading, plan, profile)
    font_path = font or find_font()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [binary, "-size", f"{layout.width}x{layout.height}", "xc:none"]

    transparent, dark = _rgba(0.0), _rgba(layout.scrim_opacity)
    for y, band_height, kind in layout.scrim_bands():
        if band_height <= 0:
            continue
        if kind == "solid":
            source = f"xc:{dark}"
        elif kind == "fade_up":  # transparent at the top, dark where the text begins
            source = f"gradient:{transparent}-{dark}"
        else:  # fade_down
            source = f"gradient:{dark}-{transparent}"
        argv += [
            "(",
            "-size",
            f"{layout.width}x{band_height}",
            source,
            ")",
            "-geometry",
            f"+0+{y}",
            "-composite",
        ]

    body = "\n".join(layout.lines)
    interline = layout.line_height - layout.font_size
    argv += [
        "-font",
        font_path,
        "-pointsize",
        str(layout.font_size),
        "-interline-spacing",
        str(interline),
        "-gravity",
        "north",
        # Pass 1: dark silhouette (outline). Pass 2: the white face on top.
        "-stroke",
        "black",
        "-strokewidth",
        str(max(3, round(layout.font_size * 0.09))),
        "-fill",
        "black",
        "-annotate",
        f"+0+{layout.block_top}",
        body,
        "-stroke",
        "none",
        "-fill",
        "white",
        "-annotate",
        f"+0+{layout.block_top}",
        body,
        "-strip",
        f"PNG32:{out_path}",
    ]
    ff.run(argv, timeout=120)
    return out_path


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
