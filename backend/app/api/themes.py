"""Theme catalogue for the palette picker, plus the contrast gate POST /api/jobs uses.

`app.core.themes` owns the presets and the WCAG rules. It is imported lazily in every
function here, and its absence degrades to a one-entry catalogue built from `Theme()`
rather than a 500 — the API must boot while that module is being written.

Contrast ratios ship with the catalogue on purpose: a picker that shows swatches but not
numbers invites someone to choose an unreadable palette, and the renderer burns text into
pixels where it cannot be fixed afterwards.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.core.models import Theme

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["themes"])

#: WCAG AA for body text. Mirrors the threshold `validate_theme` enforces; used only by
#: the local fallback check when `app.core.themes` is not importable.
AA_CONTRAST = 4.5

#: Colour fields a caller may override. `name` is metadata, the rest are the palette.
PALETTE_FIELDS = ("bg", "surface", "text", "muted", "accent")

HEX_COLOUR = r"^#[0-9a-fA-F]{6}$"
"""Six digits only: `Theme._luminance` slices fixed pairs, so `#fff` would crash it."""


class ThemeCustom(BaseModel):
    """A caller-supplied palette. Every colour is required — a half-palette mixed with
    preset defaults produces combinations nobody chose or checked.
    """

    model_config = ConfigDict(extra="forbid")

    bg: str = Field(pattern=HEX_COLOUR)
    surface: str = Field(pattern=HEX_COLOUR)
    text: str = Field(pattern=HEX_COLOUR)
    muted: str = Field(pattern=HEX_COLOUR)
    accent: str = Field(pattern=HEX_COLOUR)
    name: str = Field(default="custom", max_length=60)

    def to_theme(self) -> Theme:
        return Theme(**self.model_dump())


class ThemeOut(BaseModel):
    """One catalogue entry. Shape follows `app.core.themes.list_themes()`.

    ``extra="allow"`` on purpose: `app.core.themes` owns the catalogue, and a field it
    adds should reach the picker without a matching edit here. The fields below are the
    ones the UI is guaranteed to get.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    is_light: bool = False
    is_default: bool = False
    swatches: dict[str, str] = Field(default_factory=dict)
    contrast: dict[str, float] = Field(default_factory=dict)
    """Measured WCAG ratios, e.g. ``text_on_bg``. Shown next to the swatches so nobody
    picks a palette the renderer will burn in unreadably."""


def _core_themes() -> Any | None:
    """The themes module, or None while it does not exist yet."""
    try:
        from app.core import themes
    except ImportError:  # pragma: no cover - exercised only pre-merge
        logger.warning("app.core.themes is unavailable; serving the built-in default only")
        return None
    return themes


def _fallback_entry() -> dict[str, Any]:
    theme = Theme()
    return {
        "id": theme.name,
        "name": theme.name.title(),
        "description": "Built-in default palette.",
        "is_light": theme.is_light,
        "is_default": True,
        "swatches": {field: getattr(theme, field) for field in PALETTE_FIELDS},
        "contrast": {k: round(v, 2) for k, v in theme.contrast_report().items()},
    }


def theme_catalogue() -> list[dict[str, Any]]:
    """Presets with swatches and contrast ratios, for the picker."""
    module = _core_themes()
    if module is None:
        return [_fallback_entry()]
    try:
        return [dict(entry) for entry in module.list_themes()]
    except Exception:  # noqa: BLE001 - a broken catalogue must not take the endpoint down
        logger.exception("list_themes() failed; serving the built-in default only")
        return [_fallback_entry()]


def known_theme_ids() -> set[str]:
    """Preset ids. Empty set means "unknown" — callers must not treat it as "none valid"."""
    module = _core_themes()
    if module is None:
        return set()
    try:
        return set(module.PRESETS)
    except Exception:  # noqa: BLE001
        return set()


def default_theme_name() -> str:
    from app.db.models import default_theme_name as _default

    return _default()


def validate_palette(theme: Theme) -> list[str]:
    """Contrast failures, most important first. `[]` means the palette is renderable."""
    return review_palette(theme)[0]


def review_palette(theme: Theme) -> tuple[list[str], list[str]]:
    """``(failures, warnings)`` for a customer palette.

    Gated at WCAG AA, not the AAA bar our own presets meet. A 7.0 gate rejects ordinary
    brand colours — slate ``#64748B`` on white is 4.76:1, accessible and unremarkable —
    and refusing to render someone's real brand is worse than rendering it at AA. Below
    AA still blocks: the text is burned into the pixels and cannot be fixed afterwards.
    """
    module = _core_themes()
    if module is not None:
        review = getattr(module, "review_theme", None)
        if review is not None:
            try:
                failures, warnings = review(theme)
                return list(failures), list(warnings)
            except Exception:  # noqa: BLE001
                logger.exception("review_theme() failed; falling back to the local AA check")
        else:
            try:
                return list(module.validate_theme(theme)), []
            except Exception:  # noqa: BLE001
                logger.exception("validate_theme() failed; falling back to the local AA check")
    return _local_validate(theme), []


def _local_validate(theme: Theme) -> list[str]:
    """Minimal stand-in for `validate_theme`: body text must clear AA on both fills.

    Deliberately narrower than the real rules — it exists so a missing `app.core.themes`
    cannot turn the gate off entirely and let an unreadable palette through.
    """
    failures = []
    for pair, surface in (("bg", theme.bg), ("surface", theme.surface)):
        ratio = Theme.contrast(theme.text, surface)
        if ratio < AA_CONTRAST:
            failures.append(
                f"text on {pair} is {ratio:.2f}:1, below the WCAG AA minimum of "
                f"{AA_CONTRAST}:1 ({theme.text} on {surface})"
            )
    return failures


def suggest_palette_fix(theme: Theme) -> Theme:
    """A corrected palette the UI can offer as one click. Never raises."""
    module = _core_themes()
    if module is not None:
        try:
            return module.suggest_fix(theme)
        except Exception:  # noqa: BLE001
            logger.exception("suggest_fix() failed; falling back to a polarity flip")
    # Local last resort: keep the caller's background, take the text colour to the
    # extreme that contrasts with it.
    return theme.model_copy(
        update={"text": "#0B1220" if theme.is_light else "#F8FAFC"},
    )


@router.get("/themes", response_model=list[ThemeOut])
def list_themes() -> list[dict[str, Any]]:
    """Palette catalogue with swatches and measured contrast ratios."""
    return theme_catalogue()
