"""Validated theme presets and the contrast gate that keeps them honest.

Solid backgrounds mean colour choice *is* the design, so a palette here is a product
decision, not decoration. Two rules follow from that:

1. Every preset in :data:`PRESETS` clears the thresholds in :data:`THRESHOLDS`. The
   parametrised test over every preset is the regression guard — a palette that fails is
   a bug, not a style.
2. Users get a colour picker, so the same gate is exposed as :func:`validate_theme` for
   arbitrary input, and :func:`suggest_fix` repairs a failing palette by moving lightness
   only, so the result still reads as the customer's brand.

**Why the text threshold is AAA (7.0), not the AA web minimum (4.5).**
WCAG ratios assume a reader an arm's length from a self-lit sRGB display showing exactly
the colours you authored. Training video breaks all three assumptions: it is watched on a
projector in a lit room or in a small browser tile, and it arrives through h.264 4:2:0
chroma subsampling at a bitrate that smears the edges of thin type. Delivered contrast is
always lower than authored contrast, so the authored number needs headroom. 7.0 buys
roughly the margin that survives a bad projector plus a bad bitrate.

**Why the accent threshold is 3.0, not 4.5.**
``accent`` is deliberately not a text colour (see ``Theme.uniform_text``): it paints the
heading rule and bullet markers. WCAG 2.1 SC 1.4.11 *Non-text Contrast* sets 3:1 for
graphical objects and UI components, and SC 1.4.3 sets the same 3:1 for large-scale text.
A 6px rule is a large graphical object by any reading, so 3.0 is the correct rule here —
not a relaxed version of 4.5. If a caller ever renders body copy in ``accent``, that
caller is wrong, and the 4.5 gate on ``muted``/``text`` is what protects readers.
"""

from __future__ import annotations

import colorsys
import re

from app.core.models import Theme

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

WCAG_AA_TEXT = 4.5
"""The web minimum for body text. Kept as a named constant because it is the floor we
deliberately exceed for ``text`` — see the module docstring."""

WCAG_AAA_TEXT = 7.0
WCAG_LARGE_OBJECT = 3.0
"""SC 1.4.11 non-text contrast / SC 1.4.3 large text. Applies to ``accent``."""

THRESHOLDS: dict[str, float] = {
    "text_on_bg": WCAG_AAA_TEXT,
    "text_on_surface": WCAG_AAA_TEXT,
    "muted_on_bg": WCAG_AA_TEXT,
    "accent_on_bg": WCAG_LARGE_OBJECT,
}
"""Minimum ratio per :meth:`Theme.contrast_report` key. Keys must stay in sync with it.

``muted`` sits at AA rather than AAA on purpose: it is a secondary tier (kickers, source
lines, dates) and pushing it to 7.0 collapses the tonal gap that makes it read as
secondary at all. Muted failing 4.5 is the single most common palette bug, which is why
it is gated at all rather than left to taste.
"""

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_COLOUR_FIELDS = ("bg", "surface", "text", "muted", "accent")

# Which colour a caller should move to fix each pair, and why. Drives the hint in
# validate_theme so the message is actionable rather than just a number.
_REMEDY: dict[str, str] = {
    "text_on_bg": "lighten/darken `text` away from `bg`",
    "text_on_surface": "move `surface` closer to `bg` in tone",
    "muted_on_bg": "raise `muted` toward `text` (it is too close to `bg`)",
    "accent_on_bg": "pick a lighter or darker tint of the same accent hue",
}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEFAULT_THEME_NAME = "midnight"

PRESETS: dict[str, Theme] = {
    # -- dark ---------------------------------------------------------------
    "midnight": Theme(
        name="midnight",
        bg="#0B1220",
        surface="#131F35",
        text="#F8FAFC",
        muted="#94A3B8",
        accent="#F5A524",
    ),
    "graphite": Theme(
        name="graphite",
        bg="#15171C",
        surface="#212530",
        text="#F4F6F8",
        muted="#A2A9B6",
        accent="#38BDF8",
    ),
    # Brand-anchored. #863BFF is the product logo purple and it appears here verbatim,
    # as `accent`. It cannot be text: on white it is 4.63:1 (fails AAA) and on any dark
    # ground it is ~3.7:1 (fails AA), because a saturated violet has low relative
    # luminance no matter how vivid it looks. As a graphical accent on a bg mixed from
    # the same hue it clears the 3.0 non-text bar with room, and the near-black plum
    # bg/surface are the brand hue taken down to L~7%, so the whole frame is on-brand
    # without ever putting the logo colour under a paragraph.
    #
    # 3.69:1 is the lowest accent in the library. A lifted tint (#9C63FF, 5.09:1) was
    # proofed as an alternative and rejected: it washes toward lavender and stops reading
    # as the logo, which defeats the point of a brand preset. If a projector ever proves
    # the rule too dim in the room, #9C63FF is the documented escape hatch — but the fix
    # belongs in marker/rule *weight*, which is the renderer's call, not in the hue.
    "halo": Theme(
        name="halo",
        bg="#140C24",
        surface="#20153A",
        text="#F6F2FF",
        muted="#B7A9DA",
        accent="#863BFF",
    ),
    # The ground is deliberately lifted off near-black (L~9%, not ~4%). A deeper pine
    # measures better — 16.5:1 rather than 15.5:1 — but on a projector it reads as plain
    # black and the palette loses its identity, and the surface panel stops separating
    # from bg. Proofed both; the lifted one is the better frame at the same AAA tier.
    "forest": Theme(
        name="forest",
        bg="#0C2118",
        surface="#143024",
        text="#F1F7F3",
        muted="#96B8A6",
        accent="#F0B429",
    ),
    # -- light --------------------------------------------------------------
    # Light presets lift `surface` *above* `bg` rather than below it: on light grounds,
    # elevation reads as "closer to white", the inverse of the dark presets. Both
    # directions still have to clear text_on_surface, which is what the gate checks.
    "daylight": Theme(
        name="daylight",
        bg="#F7F8FA",
        surface="#FFFFFF",
        text="#111827",
        muted="#55607A",
        accent="#1D4ED8",
        logo_opacity=0.92,
    ),
    "boardroom": Theme(
        name="boardroom",
        bg="#E8EDF4",
        surface="#F8FAFD",
        text="#0F2440",
        muted="#49597A",
        accent="#0E6BA8",
        image_radius=12,
        logo_opacity=0.92,
    ),
    "paper": Theme(
        name="paper",
        bg="#F6F1E7",
        surface="#FFFDF8",
        text="#211C15",
        muted="#5D5346",
        accent="#A8442A",
        image_radius=6,
        logo_opacity=0.92,
    ),
    "lilac": Theme(
        name="lilac",
        bg="#F5F1FF",
        surface="#FFFFFF",
        text="#1B1230",
        muted="#574A78",
        accent="#863BFF",
        logo_opacity=0.92,
    ),
}
"""Registry, ordered dark-first with the default at the head — :func:`list_themes`
preserves this order so the picker is stable across releases."""

THEME_META: dict[str, dict[str, str]] = {
    "midnight": {
        "name": "Midnight",
        "description": "Deep navy with an amber accent — the default; confident and neutral"
        " for policy, security and compliance modules.",
    },
    "graphite": {
        "name": "Graphite",
        "description": "Neutral charcoal with a cool blue accent — the safest dark option"
        " when the footage has to sit under someone else's brand.",
    },
    "halo": {
        "name": "Halo",
        "description": "Near-black plum carrying the product's own violet — use for"
        " first-party launch, onboarding and enablement content.",
    },
    "forest": {
        "name": "Forest",
        "description": "Deep pine with warm gold — for sustainability, operations and"
        " field-safety training that should feel grounded rather than technical.",
    },
    "daylight": {
        "name": "Daylight",
        "description": "Bright near-white with ink text — the clearest option for dense"
        " process walkthroughs and anything watched on a projector in a lit room.",
    },
    "boardroom": {
        "name": "Boardroom",
        "description": "Cool grey-blue with a corporate blue accent — for leadership,"
        " finance and formal announcement decks.",
    },
    "paper": {
        "name": "Paper",
        "description": "Warm off-white with brick red — an editorial, low-glare feel for"
        " long-form culture, ethics and HR narrative content.",
    },
    "lilac": {
        "name": "Lilac",
        "description": "Soft violet-tinted white behind the brand purple — the light"
        " counterpart to Halo, for customer-facing product education.",
    },
}
"""Picker metadata. Lives beside the registry rather than on ``Theme`` because ``Theme``
is the render contract and a marketing label has no business travelling into a timeline.
(If the UI ever needs the label after a round-trip, adding ``label``/``description`` to
``Theme`` would be the cleaner move — flagged, not done, since models.py is shared.)
"""


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------


def get_theme(name: str | None) -> Theme:
    """Resolve a preset id. Never raises — an unknown id falls back to the default.

    A bad theme id is not worth failing a render over: the caller is a query parameter or
    a stale bookmark, and a video in the default palette is a better outcome than a 500.
    Matching is case- and whitespace-insensitive so ``"Midnight "`` resolves.
    """
    if name:
        key = name.strip().lower()
        if key in PRESETS:
            return PRESETS[key]
    return PRESETS[DEFAULT_THEME_NAME]


def list_themes() -> list[dict]:
    """Picker payload: id, label, description, polarity, swatches and measured contrast.

    Ratios are computed, never transcribed, so the UI cannot drift from the palette.
    """
    out: list[dict] = []
    for key, theme in PRESETS.items():
        meta = THEME_META.get(key, {})
        out.append(
            {
                "id": key,
                "name": meta.get("name", theme.name),
                "description": meta.get("description", ""),
                "is_light": theme.is_light,
                "is_default": key == DEFAULT_THEME_NAME,
                "swatches": {field: getattr(theme, field) for field in _COLOUR_FIELDS},
                "contrast": {k: round(v, 2) for k, v in theme.contrast_report().items()},
            }
        )
    return out


def validate_theme(theme: Theme) -> list[str]:
    """Return human-readable failures for a palette. Empty list means it ships.

    Built for the custom colour picker, so every message names the pair, the measured
    ratio, the requirement and the move that fixes it. Malformed colours are reported
    instead of raising: ``Theme`` stores colours as plain ``str``, so a picker can hand us
    ``"red"`` or ``"#f0f"``, and ``Theme._luminance`` would blow up on both.
    """
    malformed = [
        f"{field} is {getattr(theme, field)!r}, which is not a 6-digit hex colour "
        f"like '#1A2B3C'"
        for field in _COLOUR_FIELDS
        if not _HEX_RE.match(str(getattr(theme, field)))
    ]
    if malformed:
        # Bail before measuring: ratios against an unparseable colour are meaningless.
        return malformed

    report = theme.contrast_report()
    problems = []
    for pair, minimum in THRESHOLDS.items():
        ratio = report[pair]
        if ratio < minimum:
            problems.append(
                f"{pair} is {ratio:.2f}:1, needs >= {minimum:.2f}:1 — {_REMEDY[pair]}"
            )
    return problems


REQUIRED_THRESHOLDS: dict[str, float] = {
    "text_on_bg": WCAG_AA_TEXT,
    "text_on_surface": WCAG_AA_TEXT,
    "accent_on_bg": WCAG_LARGE_OBJECT,
}
"""Pairs that BLOCK a user palette. Note ``muted_on_bg`` is deliberately absent.

``Theme.uniform_text`` means the renderer draws exactly one text colour, so ``muted``
never reaches a video frame — ``grep -rn '\\.muted' app/render/`` is empty by design, not
by omission. It is still used by the frontend (preview kicker, chrome) and is still held
to AA in ``THRESHOLDS`` for our own presets, so it stays validated as a *warning*.
Blocking someone's render over a colour we never render would be indefensible.
"""
"""The hard floor for a *user-supplied* palette: WCAG AA, the accessibility minimum.

``THRESHOLDS`` (AAA for text) is the bar our own presets are held to — we choose those,
so they should be excellent. Applying it to customer brand colours rejects legitimate
palettes: slate ``#64748B`` on white is 4.76:1, which is perfectly accessible and a
completely normal corporate grey, yet fails a 7.0 gate. Refusing to render someone's
actual brand is worse than rendering it at AA, so the picker blocks below AA and *warns*
between AA and AAA.
"""


def review_theme(theme: Theme) -> tuple[list[str], list[str]]:
    """``(failures, warnings)`` for a user-supplied palette.

    Failures are below WCAG AA and must block. Warnings clear AA but miss the AAA
    headroom that survives a projector and h.264 chroma subsampling — worth saying so,
    not worth refusing the job over.
    """
    malformed = [
        f"{field} is {getattr(theme, field)!r}, which is not a 6-digit hex colour "
        f"like '#1A2B3C'"
        for field in _COLOUR_FIELDS
        if not _HEX_RE.match(str(getattr(theme, field)))
    ]
    if malformed:
        return malformed, []

    report = theme.contrast_report()
    failures, warnings = [], []
    # Iterate THRESHOLDS, not REQUIRED_THRESHOLDS: every pair is *assessed*, but only the
    # pairs with a required floor can block. A pair absent from REQUIRED (muted_on_bg) is
    # advisory — dropping it from the loop entirely would silently stop reporting it.
    for pair, recommended in THRESHOLDS.items():
        ratio = report[pair]
        floor = REQUIRED_THRESHOLDS.get(pair)
        if floor is not None and ratio < floor:
            failures.append(
                f"{pair} is {ratio:.2f}:1, needs >= {floor:.2f}:1 — {_REMEDY[pair]}"
            )
        elif ratio < recommended:
            warnings.append(
                f"{pair} is {ratio:.2f}:1, under the {recommended:.1f}:1 we recommend "
                f"for video — thin type may smear on a projector or at low bitrate"
                + ("" if floor is None else " (it does clear WCAG AA)")
            )
    return failures, warnings


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
#
# suggest_fix moves *lightness only*, in HLS, keeping hue and saturation fixed. That is
# the whole point: a brand is mostly hue and chroma, so a palette repaired this way still
# looks like the customer's palette, just legible. Recolouring to a "safe" blue would pass
# the gate and get rejected by the brand owner, which means the video ships unchecked.
#
# Contrast is monotonic in HLS lightness for fixed hue and saturation (each RGB channel is
# non-decreasing in L, so relative luminance is too), so a binary search between the
# current lightness and the nearer of black/white finds the *smallest* move that clears
# the threshold. Smallest move = least brand drift.
#
# Foregrounds move first and ``bg`` is held, because ``bg`` is the colour a customer
# actually recognises. Only when a foreground driven all the way to pure white or pure
# black still fails — which happens with a mid-tone ground like #808080, where the best
# achievable ratio is about 5.3:1 — does the background move, and then only far enough.


def _hex_to_hls(colour: str) -> tuple[float, float, float]:
    raw = colour.lstrip("#")
    r, g, b = (int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, ell: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, min(1.0, max(0.0, ell)), s)
    return "#{:02X}{:02X}{:02X}".format(*(round(c * 255) for c in (r, g, b)))


def _with_lightness(colour: str, ell: float) -> str:
    h, _, s = _hex_to_hls(colour)
    return _hls_to_hex(h, ell, s)


def _solve_lightness(colour: str, against: str, target: float, lighten: bool) -> str:
    """Smallest lightness move on ``colour`` that reaches ``target`` against ``against``.

    Returns the clamped extreme (white or black at the original hue/saturation) if the
    target is unreachable, so callers can detect failure by re-measuring.
    """
    h, current, s = _hex_to_hls(colour)
    bound = 1.0 if lighten else 0.0
    if Theme.contrast(_hls_to_hex(h, bound, s), against) < target:
        return _hls_to_hex(h, bound, s)

    lo, hi = current, bound  # lo fails, hi passes
    for _ in range(24):
        mid = (lo + hi) / 2
        if Theme.contrast(_hls_to_hex(h, mid, s), against) >= target:
            hi = mid
        else:
            lo = mid
    return _hls_to_hex(h, hi, s)


def suggest_fix(theme: Theme) -> Theme:
    """Return the nearest palette that passes :func:`validate_theme`, hue/sat preserved.

    Lightness-only repair; see the block comment above for the approach and its limits.
    A palette that already passes is returned unchanged (as a copy). Malformed colours
    cannot be repaired — validate first; they are passed through untouched.
    """
    if any(not _HEX_RE.match(str(getattr(theme, f))) for f in _COLOUR_FIELDS):
        return theme.model_copy(deep=True)

    fixed = theme.model_copy(deep=True)
    # Push foregrounds away from the ground's own polarity: brighter on dark, darker on
    # light. Recomputed from the original bg, not per-step, so all tiers move together.
    lighten = not fixed.is_light

    text_min = THRESHOLDS["text_on_bg"]
    if Theme.contrast(fixed.text, fixed.bg) < text_min:
        candidate = _solve_lightness(fixed.text, fixed.bg, text_min, lighten)
        if Theme.contrast(candidate, fixed.bg) < text_min:
            # Mid-tone ground: no lightness of this hue can carry AAA text on it, so the
            # ground has to give. Move it the opposite way, then re-solve the text.
            fixed.bg = _solve_lightness(fixed.bg, candidate, text_min, not lighten)
            candidate = _solve_lightness(fixed.text, fixed.bg, text_min, lighten)
        fixed.text = candidate

    # surface is a ground, so it moves rather than the text that sits on it, and it moves
    # toward bg's polarity to stay a plausible sibling of bg.
    surface_min = THRESHOLDS["text_on_surface"]
    if Theme.contrast(fixed.text, fixed.surface) < surface_min:
        fixed.surface = _solve_lightness(fixed.surface, fixed.text, surface_min, not lighten)

    for field, pair in (("muted", "muted_on_bg"), ("accent", "accent_on_bg")):
        minimum = THRESHOLDS[pair]
        colour = getattr(fixed, field)
        if Theme.contrast(colour, fixed.bg) < minimum:
            setattr(fixed, field, _solve_lightness(colour, fixed.bg, minimum, lighten))

    return fixed


def contrast_table() -> str:
    """Fixed-width report of every preset and its four ratios. Used by tests and by eye."""
    header = (
        f"{'id':<10} {'pol':<5} {'text/bg':>8} {'text/surf':>10} "
        f"{'muted/bg':>9} {'accent/bg':>10}  {'verdict':<7}"
    )
    lines = [header, "-" * len(header)]
    for key, theme in PRESETS.items():
        r = theme.contrast_report()
        problems = validate_theme(theme)
        lines.append(
            f"{key:<10} {'light' if theme.is_light else 'dark':<5} "
            f"{r['text_on_bg']:>8.2f} {r['text_on_surface']:>10.2f} "
            f"{r['muted_on_bg']:>9.2f} {r['accent_on_bg']:>10.2f}  "
            f"{'PASS' if not problems else 'FAIL':<7}"
        )
    thresholds = "  ".join(f"{k} >= {v:.1f}" for k, v in THRESHOLDS.items())
    lines.append(f"thresholds: {thresholds}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - developer convenience
    print(contrast_table())
