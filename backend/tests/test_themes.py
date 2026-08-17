"""Contrast is the only property of a palette that can be objectively wrong, so it is the
one this suite locks down. The parametrised sweep over every preset is the guard that
matters: it means nobody can add or tweak a palette without proving it is legible.
"""

from __future__ import annotations

import pytest

from app.core.models import Theme
from app.core.themes import (
    DEFAULT_THEME_NAME,
    PRESETS,
    THEME_META,
    THRESHOLDS,
    WCAG_AA_TEXT,
    contrast_table,
    get_theme,
    list_themes,
    suggest_fix,
    validate_theme,
)

# ---------------------------------------------------------------------------
# The regression guard: every preset, every threshold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_clears_every_threshold(preset_id: str) -> None:
    theme = PRESETS[preset_id]
    report = theme.contrast_report()
    failures = [
        f"{pair}={report[pair]:.2f} < {minimum:.2f}"
        for pair, minimum in THRESHOLDS.items()
        if report[pair] < minimum
    ]
    assert not failures, f"{preset_id} fails: {', '.join(failures)}"


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_every_preset_passes_its_own_validator(preset_id: str) -> None:
    """PRESETS and validate_theme must agree, or the gate is decorative."""
    assert validate_theme(PRESETS[preset_id]) == []


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_preset_text_beats_the_web_minimum_with_headroom(preset_id: str) -> None:
    """Video is compressed and watched at distance; AA alone is not enough. See module doc."""
    assert PRESETS[preset_id].contrast_report()["text_on_bg"] > WCAG_AA_TEXT


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_preset_name_matches_registry_key(preset_id: str) -> None:
    """get_theme(theme.name) must round-trip, or a persisted timeline can't be re-resolved."""
    theme = PRESETS[preset_id]
    assert theme.name == preset_id
    assert get_theme(theme.name) is theme


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_preset_has_metadata(preset_id: str) -> None:
    meta = THEME_META[preset_id]
    assert meta["name"].strip()
    assert len(meta["description"].strip()) > 20


# ---------------------------------------------------------------------------
# Library shape: the user asked for light options, so prove they exist.
# ---------------------------------------------------------------------------


def test_library_size_is_six_to_eight() -> None:
    assert 6 <= len(PRESETS) <= 8


def test_both_polarities_are_well_represented() -> None:
    light = [k for k, t in PRESETS.items() if t.is_light]
    dark = [k for k, t in PRESETS.items() if not t.is_light]
    assert len(light) >= 3, f"only {len(light)} light presets: {light}"
    assert len(dark) >= 3, f"only {len(dark)} dark presets: {dark}"


def test_light_presets_have_dark_text_and_flip_the_scrim() -> None:
    for key, theme in PRESETS.items():
        if theme.is_light:
            assert Theme._luminance(theme.text) < 0.5, f"{key} light bg needs dark text"
            assert theme.scrim_colour == "#FFFFFF"
        else:
            assert Theme._luminance(theme.text) > 0.5, f"{key} dark bg needs light text"
            assert theme.scrim_colour == "#000000"


def test_brand_purple_is_anchored_in_a_preset() -> None:
    """#863BFF is the product logo colour; at least one preset must use it verbatim."""
    branded = [k for k, t in PRESETS.items() if "#863BFF" in (t.accent.upper(), t.bg.upper())]
    assert branded, "no preset carries the brand purple"
    for key in branded:
        assert validate_theme(PRESETS[key]) == []


def test_brand_purple_would_fail_as_body_text_on_white() -> None:
    """Documents why the brand colour is an accent and not the text colour."""
    assert Theme.contrast("#863BFF", "#FFFFFF") < THRESHOLDS["text_on_bg"]


def test_swatches_are_all_six_digit_hex() -> None:
    for entry in list_themes():
        for field, value in entry["swatches"].items():
            assert value.startswith("#") and len(value) == 7, f"{entry['id']}.{field}={value}"


# ---------------------------------------------------------------------------
# get_theme
# ---------------------------------------------------------------------------


def test_default_theme_name_resolves() -> None:
    assert DEFAULT_THEME_NAME in PRESETS


@pytest.mark.parametrize("bad", [None, "", "   ", "nope", "MIDNIGHTX", "../../etc/passwd", "0"])
def test_get_theme_falls_back_and_never_raises(bad: str | None) -> None:
    assert get_theme(bad) is PRESETS[DEFAULT_THEME_NAME]


@pytest.mark.parametrize("given", ["midnight", "MIDNIGHT", " Midnight ", "MidNight"])
def test_get_theme_is_case_and_whitespace_insensitive(given: str) -> None:
    assert get_theme(given) is PRESETS["midnight"]


def test_get_theme_returns_the_shared_instance_not_a_copy() -> None:
    """Callers must not be able to mutate the registry by accident — so also assert the
    caller-side contract: copy before editing."""
    theme = get_theme("graphite")
    edited = theme.model_copy(update={"accent": "#FFFFFF"})
    assert PRESETS["graphite"].accent == "#38BDF8"
    assert edited.accent == "#FFFFFF"


# ---------------------------------------------------------------------------
# list_themes
# ---------------------------------------------------------------------------


def test_list_themes_shape_and_order() -> None:
    entries = list_themes()
    assert [e["id"] for e in entries] == list(PRESETS)
    assert entries[0]["id"] == DEFAULT_THEME_NAME
    assert sum(1 for e in entries if e["is_default"]) == 1
    for entry in entries:
        assert set(entry) == {
            "id",
            "name",
            "description",
            "is_light",
            "is_default",
            "swatches",
            "contrast",
        }
        assert set(entry["swatches"]) == {"bg", "surface", "text", "muted", "accent"}
        assert set(entry["contrast"]) == set(THRESHOLDS)


def test_list_themes_contrast_is_measured_not_transcribed() -> None:
    for entry in list_themes():
        live = PRESETS[entry["id"]].contrast_report()
        for pair, value in entry["contrast"].items():
            assert value == pytest.approx(round(live[pair], 2))


# ---------------------------------------------------------------------------
# validate_theme
# ---------------------------------------------------------------------------


def _bad_theme(**overrides: str) -> Theme:
    base = {
        "name": "custom",
        "bg": "#0B1220",
        "surface": "#131F35",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "accent": "#F5A524",
    }
    return Theme(**{**base, **overrides})


def test_validate_catches_a_deliberately_bad_palette() -> None:
    """Grey-on-grey: the palette someone builds in a colour picker at 6pm."""
    awful = Theme(
        name="awful",
        bg="#6B7280",
        surface="#6B7280",
        text="#9CA3AF",
        muted="#7C8592",
        accent="#6E7580",
    )
    problems = validate_theme(awful)
    assert len(problems) == len(THRESHOLDS), problems
    joined = " | ".join(problems)
    for pair in THRESHOLDS:
        assert pair in joined


def test_validate_message_quantifies_the_gap() -> None:
    theme = _bad_theme(muted="#5A6B80")  # ~3.1:1 on midnight's bg
    problems = validate_theme(theme)
    assert len(problems) == 1
    ratio = theme.contrast_report()["muted_on_bg"]
    assert 3.0 < ratio < 4.5
    assert problems[0].startswith(f"muted_on_bg is {ratio:.2f}:1, needs >= 4.50:1")
    assert "muted" in problems[0]


def test_validate_accepts_accent_between_three_and_four_five() -> None:
    """accent is a graphical object (SC 1.4.11): 3.0 is the correct bar, not 4.5."""
    theme = _bad_theme(accent="#6C4BD8")
    ratio = theme.contrast_report()["accent_on_bg"]
    assert 3.0 <= ratio < 4.5, ratio
    assert validate_theme(theme) == []


def test_validate_rejects_accent_below_three() -> None:
    theme = _bad_theme(accent="#2A2F45")
    assert theme.contrast_report()["accent_on_bg"] < 3.0
    problems = validate_theme(theme)
    assert len(problems) == 1 and problems[0].startswith("accent_on_bg is ")


@pytest.mark.parametrize("value", ["red", "#f0f", "0B1220", "#GGHHII", "", "#0B12200"])
def test_validate_reports_malformed_colours_instead_of_raising(value: str) -> None:
    problems = validate_theme(_bad_theme(bg=value))
    assert problems and "not a 6-digit hex colour" in problems[0]
    assert "bg is" in problems[0]


def test_validate_reports_every_malformed_field_at_once() -> None:
    problems = validate_theme(_bad_theme(bg="red", text="blue"))
    assert len(problems) == 2


def test_validate_lowercase_hex_is_fine() -> None:
    assert validate_theme(_bad_theme(bg="#0b1220")) == []


# ---------------------------------------------------------------------------
# suggest_fix
# ---------------------------------------------------------------------------


def _hue_sat(colour: str) -> tuple[float, float]:
    import colorsys

    raw = colour.lstrip("#")
    r, g, b = (int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return h, s


FAILING = {
    "muted too close to bg": _bad_theme(muted="#3A4657"),
    "accent too dark": _bad_theme(accent="#2A2F45"),
    "text below AAA on dark": _bad_theme(text="#66748C"),
    "surface too close to text": _bad_theme(surface="#C9D3E0"),
    "light palette, weak everything": Theme(
        name="weak-light",
        bg="#F4F4F6",
        surface="#FAFAFB",
        text="#8A8F98",
        muted="#B9BDC4",
        accent="#D8DCE2",
    ),
    "mid-tone ground, unfixable by foreground alone": Theme(
        name="mid",
        bg="#808080",
        surface="#808080",
        text="#8B8B8B",
        muted="#909090",
        accent="#888888",
    ),
    "saturated brand palette": Theme(
        name="brandy",
        bg="#2A1259",
        surface="#2C1560",
        text="#7A4BD0",
        muted="#5B3AA0",
        accent="#3A1E70",
    ),
}


@pytest.mark.parametrize("label", sorted(FAILING))
def test_suggest_fix_turns_a_failing_palette_into_a_passing_one(label: str) -> None:
    broken = FAILING[label]
    assert validate_theme(broken), f"{label} was supposed to fail"
    fixed = suggest_fix(broken)
    assert validate_theme(fixed) == [], f"{label} still fails: {validate_theme(fixed)}"


@pytest.mark.parametrize("label", sorted(FAILING))
def test_suggest_fix_preserves_hue_and_saturation(label: str) -> None:
    """The point of lightness-only repair: it still looks like their brand."""
    broken = FAILING[label]
    fixed = suggest_fix(broken)
    for field in ("bg", "surface", "text", "muted", "accent"):
        before = _hue_sat(getattr(broken, field))
        after = _hue_sat(getattr(fixed, field))
        moved_to_extreme = getattr(fixed, field).upper() in {"#000000", "#FFFFFF"}
        if moved_to_extreme:
            continue  # pure black/white has no meaningful hue or saturation
        assert after[0] == pytest.approx(before[0], abs=0.02), f"{label}.{field} hue moved"
        assert after[1] == pytest.approx(before[1], abs=0.02), f"{label}.{field} sat moved"


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_suggest_fix_is_a_no_op_on_a_passing_palette(preset_id: str) -> None:
    theme = PRESETS[preset_id]
    fixed = suggest_fix(theme)
    assert fixed == theme
    assert fixed is not theme, "must not hand back the registry instance"


def test_suggest_fix_makes_the_smallest_move_it_can() -> None:
    """A near-miss should be nudged, not blown out to pure white."""
    broken = _bad_theme(muted="#3A4657")
    fixed = suggest_fix(broken)
    assert fixed.muted.upper() != "#FFFFFF"
    ratio = fixed.contrast_report()["muted_on_bg"]
    assert THRESHOLDS["muted_on_bg"] <= ratio < THRESHOLDS["muted_on_bg"] + 0.2


def test_suggest_fix_keeps_the_background_when_foregrounds_can_carry_the_fix() -> None:
    broken = _bad_theme(muted="#3A4657", accent="#2A2F45")
    fixed = suggest_fix(broken)
    assert fixed.bg == broken.bg
    assert fixed.surface == broken.surface


def test_suggest_fix_moves_the_background_only_when_it_must() -> None:
    """#808080 cannot carry 7:1 text at any lightness of a grey hue, so the ground gives."""
    broken = FAILING["mid-tone ground, unfixable by foreground alone"]
    fixed = suggest_fix(broken)
    assert fixed.bg != broken.bg
    assert validate_theme(fixed) == []


def test_suggest_fix_preserves_non_colour_fields() -> None:
    broken = _bad_theme(muted="#3A4657")
    broken = broken.model_copy(update={"image_radius": 7, "logo_opacity": 0.5})
    fixed = suggest_fix(broken)
    assert (fixed.image_radius, fixed.logo_opacity, fixed.name) == (7, 0.5, broken.name)


def test_suggest_fix_passes_malformed_input_through_untouched() -> None:
    broken = _bad_theme(bg="red")
    assert suggest_fix(broken).bg == "red"


def test_suggest_fix_output_is_idempotent() -> None:
    for broken in FAILING.values():
        once = suggest_fix(broken)
        assert suggest_fix(once) == once


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


def test_contrast_table_covers_every_preset(capsys: pytest.CaptureFixture[str]) -> None:
    table = contrast_table()
    print(table)
    capsys.readouterr()
    for key in PRESETS:
        assert key in table
    assert "FAIL" not in table
