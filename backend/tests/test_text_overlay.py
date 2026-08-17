"""Tests for :mod:`app.render.text_overlay`.

Split three ways:

* **pure** — geometry, wrapping, timing, colorimetry. No subprocess, so these run
  everywhere and are the ones that pin the layout contract.
* **imagemagick** — actually shell out and inspect the PNGs. Skipped when ``magick`` is
  absent. These exist because the interesting bugs in this module (escaping, gravity,
  alpha) are only visible in pixels.
* **assets** — additionally need the two generated stills from a real render, one dark
  and one bright, to check that the adaptive scrim solves for the right opacity on real
  images rather than on synthetic gradients.
"""

from __future__ import annotations

import concurrent.futures
import subprocess
from pathlib import Path

import pytest

from app.core.models import (
    BulletPoint,
    RenderProfile,
    SlideLayout,
    TextAnimation,
    TextPosition,
    Theme,
    VisualPlan,
)
from app.render import text_overlay as tx

HD = RenderProfile(name="final", width=1920, height=1080)
DRAFT = RenderProfile(name="draft", width=960, height=540)

HEADING = "Spot a Phishing Email Before You Click"

LONG_BULLET = (
    "Hover every link and read the domain before you trust it, because a lookalike "
    "domain is the single most common tell in a targeted campaign"
)
HOSTILE = "It's 40%: fast, & cheap — 100% sure, [really]; @everyone"

BULLETS = [
    BulletPoint(text="Check the sender's real address, not the display name", appear_at=0.8),
    BulletPoint(text=LONG_BULLET, appear_at=1.9),
    BulletPoint(text=HOSTILE, appear_at=3.1, emphasis=True),
    BulletPoint(text="Report it; don't delete it", appear_at=4.2),
]

SOLID_LAYOUTS = (
    SlideLayout.TITLE_CARD,
    SlideLayout.HERO_RIGHT,
    SlideLayout.HERO_LEFT,
    SlideLayout.IMAGE_BAND,
)
ALL_POSITIONS = (
    TextPosition.LEFT_PANEL,
    TextPosition.CENTER,
    TextPosition.UPPER_THIRD,
    TextPosition.LOWER_THIRD,
)

ASSETS = Path("/Users/argo/ab/prompt-to-video-v2/out/43859ea1-a861-42ff-a4ec-548433c38ec0")
DARK_IMAGE = ASSETS / "scene_01.png"
BRIGHT_IMAGE = ASSETS / "scene_04.png"

needs_magick = pytest.mark.skipif(
    tx.imagemagick_bin() is None, reason="ImageMagick not installed"
)
needs_assets = pytest.mark.skipif(
    not (DARK_IMAGE.is_file() and BRIGHT_IMAGE.is_file()),
    reason="generated sample stills not present",
)


def full_bleed(position: TextPosition = TextPosition.LEFT_PANEL, **kwargs) -> VisualPlan:
    return VisualPlan(layout=SlideLayout.FULL_BLEED, text_position=position, **kwargs)


def ink(path: Path) -> int:
    """Count of non-transparent pixels — a cheap "is there text here" assertion."""
    out = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(path), "-alpha", "extract",
         "-format", "%[fx:mean*w*h]", "info:"],
        capture_output=True, text=True, check=True,
    )
    return round(float(out.stdout.strip()))


def png_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), "identify", "-format", "%w %h", str(path)],
        capture_output=True, text=True, check=True,
    )
    width, height = out.stdout.split()
    return int(width), int(height)


def pixel(path: Path, x: int, y: int) -> str:
    out = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(path), "-format", f"%[pixel:p{{{x},{y}}}]", "info:"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


# =========================================================== colour + contrast


def test_relative_luminance_anchors_at_black_and_white():
    assert tx.relative_luminance("#000000") == pytest.approx(0.0)
    assert tx.relative_luminance("#FFFFFF") == pytest.approx(1.0)
    assert tx.contrast_ratio(1.0, 0.0) == pytest.approx(21.0)


def test_hex_parsing_accepts_short_form_and_rejects_junk():
    assert tx.parse_hex("#F5A524") == (245, 165, 36)
    assert tx.parse_hex("fa2") == (255, 170, 34)
    for bad in ("", "#12345", "not-a-colour", "rgb(1,2,3)"):
        with pytest.raises(ValueError):
            tx.parse_hex(bad)


def test_colour_never_reaches_imagemagick_unnormalised():
    """A palette arriving from JSON must not be able to become an IM expression."""
    assert tx.imagemagick_colour("#f5a524") == "#F5A524"
    with pytest.raises(ValueError):
        tx.imagemagick_colour("gradient:red-blue")


def test_srgb_transfer_function_round_trips():
    for value in (0.0, 0.002, 0.04, 0.2, 0.5, 0.9, 1.0):
        assert tx.linear_to_srgb(tx.srgb_to_linear(value)) == pytest.approx(value, abs=1e-6)


def test_default_theme_clears_aa_on_its_own_background():
    theme = Theme()
    assert tx.colour_contrast(theme.text, theme.bg) >= tx.WCAG_AA
    assert tx.colour_contrast(theme.accent, theme.bg) >= tx.WCAG_AA


def test_required_scrim_opacity_solves_for_the_target_ratio():
    """The closed form must actually hit 4.5:1, not just move in the right direction."""
    for probe in (0.2, 0.45, 0.6, 0.8, 1.0):
        opacity = tx.required_scrim_opacity(probe, text_luminance=1.0)
        achieved = tx.contrast_ratio(1.0, tx.srgb_to_linear(probe * (1 - opacity)))
        assert achieved >= tx.WCAG_AA - 1e-6
        if opacity > tx.SCRIM_OPACITY_FLOOR:
            # And it must be the *smallest* such opacity: back it off and AA fails.
            slack = tx.contrast_ratio(1.0, tx.srgb_to_linear(probe * (1 - opacity + 0.02)))
            assert slack < tx.WCAG_AA


def test_required_scrim_opacity_is_zero_when_the_image_is_already_dark():
    assert tx.required_scrim_opacity(0.05, text_luminance=1.0) == tx.SCRIM_OPACITY_FLOOR


def test_required_scrim_opacity_rises_monotonically_with_brightness():
    opacities = [tx.required_scrim_opacity(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert opacities == sorted(opacities)
    assert opacities[-1] > opacities[0]


def test_amber_accent_needs_more_scrim_than_white_text():
    """The binding constraint is the *dimmest* ink on the slide, not the brightest."""
    theme = Theme()
    white = tx.required_scrim_opacity(0.8, text_luminance=tx.relative_luminance(theme.text))
    amber = tx.required_scrim_opacity(0.8, text_luminance=tx.relative_luminance(theme.accent))
    assert amber > white


# ===================================================== polarity: light palettes


LIGHT = Theme(
    name="paper", bg="#F7F7F5", surface="#E7E7E3", text="#111827",
    muted="#4B5563", accent="#7C3AED",
)


def test_a_light_palette_reports_itself_as_light_and_asks_for_a_white_wash():
    assert LIGHT.is_light and LIGHT.scrim_colour == "#FFFFFF"
    assert not Theme().is_light and Theme().scrim_colour == "#000000"
    assert tx.encoded_grey("#FFFFFF") == pytest.approx(1.0)
    assert tx.encoded_grey("#000000") == pytest.approx(0.0)


def test_a_white_wash_solves_by_driving_the_background_up():
    """The mirror image of the dark case, and the reason it cannot be hardcoded.

    A dark wash makes the background darker to serve light text; a white wash has to make
    it *lighter* to serve dark text. Solving the wrong direction returns the floor and
    ships illegible type.
    """
    dark_ink = tx.relative_luminance(LIGHT.text)
    for probe in (0.0, 0.1, 0.3, 0.5):
        opacity = tx.required_scrim_opacity(
            probe, text_luminance=dark_ink, scrim_encoded=1.0
        )
        after = probe + opacity * (1.0 - probe)
        assert after >= probe, "a white wash must not darken the background"
        achieved = tx.contrast_ratio(dark_ink, tx.srgb_to_linear(after))
        assert achieved >= tx.WCAG_AA - 1e-6, f"probe {probe} -> {achieved}"


def test_a_white_wash_is_a_no_op_on_an_already_pale_background():
    assert tx.required_scrim_opacity(
        0.98, text_luminance=tx.relative_luminance(LIGHT.text), scrim_encoded=1.0
    ) == tx.SCRIM_OPACITY_FLOOR


def test_the_wash_defends_the_tail_that_actually_threatens_the_text():
    """Light text is killed by highlights, dark text by shadows. Different tails."""
    probe = tx.Luminance.from_stats(mean=0.42, stddev=0.25)
    assert probe.probe > probe.mean > probe.probe_low
    assert probe.tail(low=False) == pytest.approx(0.67)
    assert probe.tail(low=True) == pytest.approx(0.17)


def test_after_scrim_moves_the_background_toward_the_wash_either_way():
    probe = tx.Luminance.from_stats(mean=0.5, stddev=0.2)
    darker = probe.after_scrim(0.5, scrim_encoded=0.0)
    lighter = probe.after_scrim(0.5, scrim_encoded=1.0)
    assert darker.mean == pytest.approx(0.25)
    assert lighter.mean == pytest.approx(0.75)
    # Blending toward any constant is a contraction, so the spread shrinks both ways.
    assert darker.stddev == pytest.approx(0.1)
    assert lighter.stddev == pytest.approx(0.1)


@needs_magick
@needs_assets
def test_a_light_theme_solves_for_a_white_scrim_over_a_bright_still():
    plan = full_bleed(TextPosition.LOWER_THIRD, scrim_opacity=0.45)
    light = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=LIGHT, image_path=BRIGHT_IMAGE)
    dark = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=Theme(), image_path=BRIGHT_IMAGE)

    assert light.contrast is not None and dark.contrast is not None
    assert light.contrast.scrim_colour == "#FFFFFF"
    assert dark.contrast.scrim_colour == "#000000"
    # The light palette brightens the plate; the dark one darkens it. Same image.
    assert (
        light.contrast.background_luminance_after
        > light.contrast.background_luminance_before
    )
    assert (
        dark.contrast.background_luminance_after < dark.contrast.background_luminance_before
    )
    for report in (light.contrast, dark.contrast):
        assert report.meets_aa, report.summary()
        assert report.ratio_after >= tx.WCAG_AA


@needs_magick
@needs_assets
def test_the_light_theme_scrim_png_is_white_not_black(tmp_path):
    """The failure this rules out: a dark wash under dark text erases the slide."""
    plan = full_bleed(TextPosition.LOWER_THIRD)
    slide = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=LIGHT, image_path=BRIGHT_IMAGE)
    scene = tx.build_scene_text(
        HEADING, BULLETS, plan, HD, tmp_path, image_path=BRIGHT_IMAGE, theme=LIGHT, slide=slide
    )
    scrim = next(x for x in scene.layers if x.kind == "scrim")
    assert slide.scrim_region is not None
    inside = pixel(
        scrim.png_path,
        slide.scrim_region.centre_x,
        slide.scrim_region.y + slide.scrim_region.height // 2,
    )
    assert inside.startswith("srgba(255,255,255,"), inside
    assert not inside.startswith("srgba(0,0,0,1")


@needs_magick
def test_a_light_theme_outlines_its_text_in_white(tmp_path):
    """A dark halo around dark type on a pale ground thickens it into mud."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    for theme, expected in ((LIGHT, "#FFFFFF"), (Theme(), "#000000")):
        slide = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=theme)
        assert all(b.outline_colour == expected for b in slide.bullets)
        scene = tx.build_scene_text(
            HEADING, BULLETS, plan, HD, tmp_path / theme.name, theme=theme, slide=slide
        )
        heading = next(x for x in scene.layers if x.kind == "heading")
        colours = subprocess.run(  # noqa: S603
            [tx.require_imagemagick(), str(heading.png_path), "-depth", "8",
             "-format", "%c", "histogram:info:"],
            capture_output=True, text=True, check=True,
        ).stdout.upper()
        assert f"{expected}FF" in colours, f"{theme.name} has no {expected} outline"
        assert f"{theme.text.upper()}FF" in colours


# ================================================================== geometry


@pytest.mark.parametrize("layout", list(SlideLayout))
def test_every_layout_resolves_to_a_column_inside_the_frame(layout):
    geometry = tx.slide_geometry(VisualPlan(layout=layout), HD)
    column = geometry.text_column
    assert column.width > 0 and column.height > 0
    assert 0 <= column.x and column.right <= HD.width
    assert 0 <= column.y and column.bottom <= HD.height


def test_title_card_has_no_image_region():
    assert tx.image_region(VisualPlan(layout=SlideLayout.TITLE_CARD), HD) is None


def test_full_bleed_image_region_is_the_whole_frame():
    region = tx.image_region(full_bleed(), HD)
    assert region is not None
    assert region.as_tuple() == (0, 0, HD.width, HD.height)


@pytest.mark.parametrize(
    ("layout", "text_side"),
    [(SlideLayout.HERO_RIGHT, "left"), (SlideLayout.HERO_LEFT, "right")],
)
def test_hero_layouts_put_text_and_image_on_opposite_sides(layout, text_side):
    geometry = tx.slide_geometry(VisualPlan(layout=layout), HD)
    column, region = geometry.text_column, geometry.image_region
    assert region is not None
    if text_side == "left":
        assert column.right <= region.x, "text column must not run into the image panel"
        assert column.right < HD.width * 0.5
    else:
        assert region.right <= column.x
        assert column.x > HD.width * 0.5


def test_hero_right_text_column_is_a_readable_fraction_of_the_frame():
    geometry = tx.slide_geometry(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    share = geometry.text_column.width / HD.width
    assert 0.35 <= share <= 0.50


def test_image_band_text_sits_entirely_below_the_band():
    geometry = tx.slide_geometry(VisualPlan(layout=SlideLayout.IMAGE_BAND), HD)
    assert geometry.image_region is not None
    assert geometry.text_column.y >= geometry.image_region.bottom


def test_left_panel_keeps_the_right_of_the_frame_clear_for_the_subject():
    geometry = tx.slide_geometry(full_bleed(TextPosition.LEFT_PANEL), HD)
    assert geometry.text_column.right <= HD.width * tx.LEFT_PANEL_SHARE
    assert geometry.align == "left"


def test_solid_layouts_are_not_marked_as_over_image():
    for layout in SOLID_LAYOUTS:
        assert not tx.slide_geometry(VisualPlan(layout=layout), HD).over_image
    assert tx.slide_geometry(full_bleed(), HD).over_image


def test_geometry_scales_proportionally_between_draft_and_final():
    """A 540p draft must be the 1080p slide at half size — same design, cheaper."""
    for layout in list(SlideLayout):
        plan = VisualPlan(layout=layout)
        final = tx.slide_geometry(plan, HD)
        draft = tx.slide_geometry(plan, DRAFT)
        assert draft.scale == pytest.approx(0.5)
        assert draft.heading_size == pytest.approx(final.heading_size / 2, abs=1)
        assert draft.bullet_size == pytest.approx(final.bullet_size / 2, abs=1)
        for attr in ("text_column", "image_region"):
            small, large = getattr(draft, attr), getattr(final, attr)
            if large is None:
                assert small is None
                continue
            for value_small, value_large in zip(
                small.as_tuple(), large.as_tuple(), strict=True
            ):
                assert value_small == pytest.approx(value_large / 2, abs=1)


def test_rect_union_and_inflate():
    a, b = tx.Rect(10, 10, 20, 20), tx.Rect(50, 5, 10, 10)
    assert tx.Rect.union([a, b]).as_tuple() == (10, 5, 50, 25)
    assert tx.Rect.union([]) is None
    assert a.inflate(5).as_tuple() == (5, 5, 30, 30)


def test_rect_clamp_stays_inside_bounds():
    frame = tx.Rect(0, 0, 100, 100)
    assert tx.Rect(-20, -20, 50, 50).clamp_to(frame).as_tuple() == (0, 0, 30, 30)
    assert tx.Rect(80, 80, 50, 50).clamp_to(frame).as_tuple() == (80, 80, 20, 20)


# ==================================================================== wrapping


def make_measurer(px_per_char: float = 10.0):
    """A deterministic stand-in for the ImageMagick metric."""
    return lambda text: len(text) * px_per_char


def test_wrap_to_width_keeps_every_line_inside_the_limit():
    measure = make_measurer()
    lines = tx.wrap_to_width(LONG_BULLET, 400, measure)
    assert len(lines) > 1
    assert all(measure(line) <= 400 for line in lines)
    assert " ".join(lines) == " ".join(LONG_BULLET.split())


def test_wrap_to_width_balances_a_two_line_wrap():
    measure = make_measurer()
    lines = tx.wrap_to_width("alpha beta gamma delta epsilon", 200, measure)
    assert len(lines) == 2
    # Greedy would leave a long line over an orphan; balanced keeps them close.
    assert abs(measure(lines[0]) - measure(lines[1])) <= 200 * 0.5


def test_wrap_to_width_hard_breaks_a_word_that_cannot_fit():
    measure = make_measurer()
    lines = tx.wrap_to_width("supercalifragilistic", 60, measure)
    assert len(lines) > 1
    assert all(measure(line) <= 60 for line in lines)


def test_wrap_to_width_ellipsises_rather_than_exceeding_max_lines():
    measure = make_measurer()
    lines = tx.wrap_to_width(LONG_BULLET, 100, measure, max_lines=2)
    assert len(lines) == 2
    assert lines[-1].endswith(tx.ELLIPSIS)
    assert all(measure(line) <= 100 for line in lines)


def test_wrap_to_width_handles_empty_and_whitespace():
    assert tx.wrap_to_width("", 100, make_measurer()) == []
    assert tx.wrap_to_width("   ", 100, make_measurer()) == []


# ===================================================================== timing


def test_bullet_times_are_monotonic_and_respect_the_minimum_gap():
    plan = VisualPlan(bullet_min_gap=0.6)
    crowded = [BulletPoint(text="a", appear_at=t) for t in (0.0, 0.05, 0.1, 0.15)]
    times = tx.bullet_times(crowded, plan)
    assert times == sorted(times)
    assert all(b - a >= 0.6 - 1e-6 for a, b in zip(times, times[1:], strict=False))


def test_bullet_times_respect_a_later_narration_cue():
    plan = VisualPlan(bullet_min_gap=0.6)
    times = tx.bullet_times(
        [BulletPoint(text="a", appear_at=0.0), BulletPoint(text="b", appear_at=5.0)], plan
    )
    assert times == [0.0, 5.0]


# ============================================================ slide layout plan


def layout(plan: VisualPlan, profile: RenderProfile = HD, **kwargs):
    return tx.layout_slide(HEADING, BULLETS, plan, profile, **kwargs)


@pytest.mark.parametrize("layout_kind", list(SlideLayout))
def test_layout_fits_its_column_and_the_safe_area(layout_kind):
    slide = layout(VisualPlan(layout=layout_kind))
    assert slide.fits_column(), "text overflows the column it was given"
    assert slide.within_safe_area(), "text touches the 90% safe area"


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_full_bleed_layout_fits_every_text_position(position):
    slide = layout(full_bleed(position))
    assert slide.fits_column()
    assert slide.within_safe_area()


def test_bullets_never_overlap_even_when_one_wraps():
    """The #1 layout bug: a wrapped bullet must push the stack down, not sit on it."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    rects = [b.rect for b in slide.bullets]
    assert any(len(b.lines) > 1 for b in slide.bullets), "test data must include a wrap"
    for above, below in zip(rects, rects[1:], strict=False):
        assert above.bottom <= below.y, f"{above.as_tuple()} overlaps {below.as_tuple()}"


def test_vertical_rhythm_is_even_regardless_of_wrapping():
    """Gaps between blocks must be constant; only block *heights* vary."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    rects = [b.rect for b in slide.bullets]
    gaps = {below.y - above.bottom for above, below in zip(rects, rects[1:], strict=False)}
    assert len(gaps) == 1, f"uneven bullet gaps: {gaps}"


def test_block_height_tracks_line_count_exactly():
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    for block in slide.bullets:
        expected = len(block.lines) * block.line_height
        assert expected <= block.rect.height <= expected + block.size


def test_heading_sits_above_every_bullet_and_is_larger():
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    assert slide.heading_size > slide.bullets[0].size * 1.4
    assert all(slide.heading_rect.bottom <= b.rect.y for b in slide.bullets)


def test_a_ludicrously_long_bullet_shrinks_the_stack_instead_of_overflowing():
    monster = [BulletPoint(text=LONG_BULLET * 3) for _ in range(5)]
    slide = tx.layout_slide(HEADING, monster, VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    assert slide.fits_column()
    assert slide.within_safe_area()
    assert slide.bullets[0].size < tx.slide_geometry(
        VisualPlan(layout=SlideLayout.HERO_RIGHT), HD
    ).bullet_size


def test_every_word_on_the_slide_is_one_colour():
    """The brand rule: mixed text colours read as inconsistent, so there is only one."""
    theme = Theme()
    assert theme.uniform_text, "uniform text is the default"
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    emphasised = [b for b in slide.bullets if b.emphasis]
    normal = [b for b in slide.bullets if not b.emphasis]
    assert emphasised and normal, "test data must contain both kinds of bullet"

    assert {b.text_colour for b in slide.bullets} == {theme.text}
    assert theme.accent not in {b.text_colour for b in slide.bullets}


def test_emphasis_survives_without_a_second_colour():
    """Same ink, different weight/size/marker. Every signal here is non-chromatic."""
    theme = Theme()
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    emphasised = [b for b in slide.bullets if b.emphasis][0]
    normal = [b for b in slide.bullets if not b.emphasis][0]

    assert emphasised.text_colour == normal.text_colour
    # Weight: a genuinely heavier face when one exists, otherwise a faux-bold stroke.
    if tx.heavier_font(slide.font):
        assert emphasised.font != normal.font
        assert emphasised.faux_bold == 0.0
    else:
        assert emphasised.faux_bold > 0.0
    assert emphasised.size > normal.size
    assert emphasised.stroke_ratio > normal.stroke_ratio
    # Shape: a filled disc against a hollow ring, in the same gutter.
    assert emphasised.marker_filled and not normal.marker_filled
    assert emphasised.marker_diameter > normal.marker_diameter


def test_markers_are_graphic_so_they_keep_the_accent_colour():
    """`accent` is still allowed — on the rule and the markers, which are not text."""
    theme = Theme()
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    assert {b.marker_colour for b in slide.bullets} == {theme.accent}
    assert slide.rule_rect is not None
    # The text indent is shared so the left edge lines up regardless of emphasis.
    assert len({b.indent for b in slide.bullets}) == 1


def test_uniform_text_is_switchable_back_to_accent_for_emphasis():
    """`uniform_text=False` must reproduce the pre-brand-rule behaviour exactly."""
    theme = Theme(uniform_text=False)
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    emphasised = [b for b in slide.bullets if b.emphasis]
    normal = [b for b in slide.bullets if not b.emphasis]
    assert emphasised and normal

    assert all(b.text_colour == theme.accent for b in emphasised)
    assert all(b.text_colour == theme.text for b in normal)
    # ...and the geometry is left alone: colour was doing all the work.
    assert {b.font for b in slide.bullets} == {slide.font}
    assert {b.size for b in slide.bullets} == {normal[0].size}
    assert all(b.marker_filled for b in slide.bullets)
    assert emphasised[0].marker_diameter > normal[0].marker_diameter


def test_centred_layouts_share_one_offset_so_the_stack_reads_as_a_block():
    slide = layout(VisualPlan(layout=SlideLayout.TITLE_CARD))
    assert slide.geometry.align == "center"
    assert len({b.offset_x for b in slide.bullets}) == 1
    assert slide.bullets[0].offset_x > 0


def test_left_aligned_layouts_offset_only_by_the_stroke_bleed():
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    assert {b.offset_x for b in slide.bullets} == {slide.ink_pad}
    assert 0 < slide.ink_pad < slide.bullets[0].size


def test_heading_and_bullet_markers_share_a_left_edge():
    """One ink pad for the whole slide, so type and markers line up on the same x."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    assert slide.rule_rect is not None
    assert slide.rule_rect.x == slide.heading_rect.x + slide.ink_pad
    assert all(b.rect.x + b.offset_x == slide.rule_rect.x for b in slide.bullets)


def test_empty_bullets_and_blank_heading_are_tolerated():
    slide = tx.layout_slide("", [], VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    assert slide.heading_lines == []
    assert slide.bullets == []
    assert slide.rule_rect is None
    blank = tx.layout_slide(
        HEADING, [BulletPoint(text="   ")], VisualPlan(layout=SlideLayout.HERO_RIGHT), HD
    )
    assert blank.bullets == []


def test_solid_layouts_skip_the_scrim_and_report_exact_theme_contrast():
    """No scrim over a flat brand colour, and the contrast is computed not measured."""
    for layout_kind in SOLID_LAYOUTS:
        slide = layout(VisualPlan(layout=layout_kind))
        assert slide.scrim_opacity == 0.0
        assert slide.scrim_region is None
        assert slide.contrast is not None
        assert slide.contrast.source == "theme"
        assert slide.contrast.meets_aa
        assert slide.contrast.ratio_after >= tx.WCAG_AA


def test_a_low_contrast_theme_is_reported_as_failing_rather_than_silently_shipped():
    theme = Theme(bg="#EEEEEE", text="#FFFFFF", accent="#FFFF00")
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    assert slide.contrast is not None
    assert not slide.contrast.meets_aa


def test_full_bleed_without_an_image_falls_back_to_the_planned_opacity():
    plan = full_bleed(scrim_opacity=0.45)
    slide = tx.layout_slide(HEADING, BULLETS, plan, HD, image_path=None)
    assert slide.scrim_opacity == pytest.approx(0.45)
    assert slide.contrast is not None
    assert slide.contrast.source == "image"


def test_scrim_region_covers_all_the_text_it_has_to_make_legible():
    slide = tx.layout_slide(HEADING, BULLETS, full_bleed(), HD, image_path=None)
    assert slide.scrim_region is not None
    for rect in slide.text_rects:
        assert slide.scrim_region.x <= rect.x
        assert slide.scrim_region.y <= rect.y
        assert slide.scrim_region.bottom >= rect.bottom


# =================================================================== escaping


def test_imagemagick_text_is_always_passed_as_an_at_path():
    assert tx.imagemagick_text_arg(Path("/tmp/x.txt")) == "@/tmp/x.txt"


def test_write_text_file_does_not_leave_a_trailing_newline():
    """A trailing newline makes ImageMagick allocate an extra, empty line."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = tx.write_text_file(Path(tmp), "t.txt", ["one", "two"])
        assert path.read_text(encoding="utf-8") == "one\ntwo"


def test_drawtext_escaping_still_covers_the_hostile_characters():
    escaped = tx.escape_drawtext(HOSTILE)
    for char in set(HOSTILE) & set(tx.FILTERGRAPH_SPECIALS):
        assert f"\\\\\\{char}" in escaped
    assert tx.escape_drawtext("40% off") == "40% off"


# =============================================================== ImageMagick


@needs_magick
def test_text_measurer_matches_real_rendered_width():
    font = tx.find_font()
    measure = tx.text_measurer(font, 100)
    # Sum-of-words is additive to within a percent (kerning across the space).
    assert measure("Handgloves mixed CASE 12345") == pytest.approx(1501, rel=0.02)
    assert measure("") == 0.0
    assert measure("iiiii") < measure("MMMMM")


@needs_magick
def test_text_measurer_distinguishes_percent_from_double_percent():
    """Proof that measurement goes through `@file` and not percent expansion."""
    measure = tx.text_measurer(tx.find_font(), 60)
    assert measure("100%") < measure("100%%")


@needs_magick
def test_measurement_falls_back_to_an_estimate_without_imagemagick():
    measure = tx.text_measurer("/nope.ttf", 40, binary="")
    assert measure("abcd") == pytest.approx(4 * 40 * tx.AVG_GLYPH_RATIO)


@needs_magick
def test_build_scene_text_emits_the_documented_layer_contract(tmp_path):
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path)
    layers = scene.sorted_layers()

    assert [layer.kind for layer in layers] == ["scrim", "heading", *["bullet"] * 4]
    assert layers[0].animation is TextAnimation.NONE and layers[0].appear_at == 0.0
    assert (layers[0].width, layers[0].height) == (HD.width, HD.height)
    assert layers[1].animation is plan.heading_animation
    assert all(layer.animation is plan.bullet_animation for layer in layers[2:])

    for layer in layers:
        assert layer.png_path.is_file()
        assert png_size(layer.png_path) == (layer.width, layer.height)
        assert 0 <= layer.x and layer.x + layer.width <= HD.width
        assert 0 <= layer.y and layer.y + layer.height <= HD.height
        assert layer.slide_distance > 0
        assert layer.disappear_at is None

    bullet_times = [layer.appear_at for layer in layers if layer.kind == "bullet"]
    assert bullet_times == sorted(bullet_times)


@needs_magick
def test_declared_layer_size_always_matches_the_rasterised_png(tmp_path):
    """The seam only works if the geometry and the pixels agree exactly."""
    for index, layout_kind in enumerate(SlideLayout):
        scene = tx.build_scene_text(
            HEADING, BULLETS, VisualPlan(layout=layout_kind), HD, tmp_path / str(index)
        )
        for layer in scene.layers:
            assert png_size(layer.png_path) == (layer.width, layer.height), layout_kind


def ink_bbox(path: Path) -> tuple[int, int, int, int]:
    """``(width, height, x, y)`` of the non-transparent ink, via IM's ``%@``."""
    out = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(path), "-format", "%@", "info:"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    size, _, offset = out.partition("+")
    width, height = (int(v) for v in size.split("x"))
    x, y = (int(v) for v in offset.split("+"))
    return width, height, x, y


@needs_magick
@pytest.mark.parametrize("profile", [HD, DRAFT], ids=["1080p", "540p"])
def test_no_glyph_is_clipped_by_its_own_canvas(tmp_path, profile):
    """Descenders plus the outline stroke must fit; at 540p they used to be cut off."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    slide = tx.layout_slide(HEADING, BULLETS, plan, profile)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, profile, tmp_path, slide=slide)
    assert any(len(b.lines) > 2 for b in slide.bullets), "need a multi-line wrap to test"

    for layer in scene.layers:
        if layer.kind == "scrim":
            continue
        width, height, x, y = ink_bbox(layer.png_path)
        assert y > 0, f"{layer.kind} ink touches the top of its canvas"
        assert x > 0, f"{layer.kind} ink touches the left of its canvas"
        assert y + height < layer.height, f"{layer.kind} ink clipped at the bottom"
        assert x + width < layer.width, f"{layer.kind} ink clipped on the right"


@needs_magick
def test_hostile_text_renders_glyphs_instead_of_failing_or_expanding(tmp_path):
    bullets = [
        BulletPoint(text=HOSTILE),
        BulletPoint(text="@/etc/passwd"),
        BulletPoint(text="%w x %h ~ %[fx:1]"),
        BulletPoint(text="C:\\path\\to\\file 'quoted' \"double\""),
    ]
    scene = tx.build_scene_text(
        "50% Off: Today's Deal, & More", bullets, VisualPlan(layout=SlideLayout.HERO_RIGHT),
        HD, tmp_path,
    )
    for layer in scene.layers:
        if layer.kind == "scrim":
            continue
        assert ink(layer.png_path) > 200, f"{layer.kind} rendered no glyphs"


@needs_magick
def test_percent_is_not_expanded_when_rasterising(tmp_path):
    """`100%` and `100%%` must produce different pixels; inline argv collapses them."""
    plan = VisualPlan(layout=SlideLayout.TITLE_CARD)
    one = tx.build_scene_text("100%", [], plan, HD, tmp_path / "a")
    two = tx.build_scene_text("100%%", [], plan, HD, tmp_path / "b")
    heading_one = next(x for x in one.layers if x.kind == "heading")
    heading_two = next(x for x in two.layers if x.kind == "heading")
    assert ink(heading_one.png_path) != ink(heading_two.png_path)


@needs_magick
def test_a_heading_starting_with_at_is_not_read_as_a_file(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    plan = VisualPlan(layout=SlideLayout.TITLE_CARD)
    scene = tx.build_scene_text(f"@{secret}", [], plan, HD, tmp_path / "out")
    heading = next(x for x in scene.layers if x.kind == "heading")
    # If IM had followed the @, the block would be one short word, not a long path.
    assert ink(heading.png_path) > 0
    plain = tx.build_scene_text("TOPSECRET", [], plan, HD, tmp_path / "plain")
    plain_heading = next(x for x in plain.layers if x.kind == "heading")
    assert ink(heading.png_path) != ink(plain_heading.png_path)


@needs_magick
def test_bullet_png_contains_a_marker_and_text(tmp_path):
    scene = tx.build_scene_text(
        HEADING, [BulletPoint(text="Report it")], VisualPlan(layout=SlideLayout.HERO_RIGHT),
        HD, tmp_path,
    )
    bullet = next(x for x in scene.layers if x.kind == "bullet")
    slide = tx.layout_slide(
        HEADING, [BulletPoint(text="Report it")], VisualPlan(layout=SlideLayout.HERO_RIGHT), HD
    )
    block = slide.bullets[0]
    # A marker was drawn in the indent gutter, left of where any glyph can start.
    gutter = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(bullet.png_path),
         "-crop", f"{block.indent}x{block.rect.height}+0+0", "+repage",
         "-alpha", "extract", "-format", "%[fx:maxima]", "info:"],
        capture_output=True, text=True, check=True,
    )
    assert float(gutter.stdout.strip()) > 0.5, "no bullet marker drawn"


@needs_magick
def test_solid_background_is_the_brand_colour_with_a_hole_for_the_hero(tmp_path):
    theme = Theme(bg="#0B1220")
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path, theme=theme)
    scrim = next(x for x in scene.layers if x.kind == "scrim")
    region = tx.image_region(plan, HD, theme=theme)
    assert region is not None

    assert pixel(scrim.png_path, 10, 10) == "srgba(11,18,32,1)"
    # Middle of the image panel is punched through so the picture shows.
    centre = pixel(scrim.png_path, region.centre_x, region.y + region.height // 2)
    assert centre.endswith(",0)"), f"hero hole is not transparent: {centre}"


@needs_magick
def test_title_card_background_has_no_hole_at_all(tmp_path):
    scene = tx.build_scene_text(
        HEADING, BULLETS, VisualPlan(layout=SlideLayout.TITLE_CARD), HD, tmp_path
    )
    scrim = next(x for x in scene.layers if x.kind == "scrim")
    alpha = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(scrim.png_path), "-alpha", "extract",
         "-format", "%[fx:minima]", "info:"],
        capture_output=True, text=True, check=True,
    )
    assert float(alpha.stdout.strip()) == pytest.approx(1.0)


@needs_magick
def test_full_bleed_scrim_is_transparent_away_from_the_text(tmp_path):
    plan = full_bleed(TextPosition.LEFT_PANEL, scrim_opacity=0.6)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path)
    scrim = next(x for x in scene.layers if x.kind == "scrim")
    assert pixel(scrim.png_path, 20, HD.height // 2).endswith(",0.6)")
    assert pixel(scrim.png_path, HD.width - 5, HD.height // 2).endswith(",0)")


# ====================================================== adaptive scrim on real images


@needs_magick
@needs_assets
def test_luminance_probe_separates_a_dark_image_from_a_bright_one():
    frame = tx.Rect(0, 0, HD.width, HD.height)
    region = tx.Rect(96, 124, 960, 700)
    dark = tx.measure_region_luminance(DARK_IMAGE, region, frame)
    bright = tx.measure_region_luminance(BRIGHT_IMAGE, region, frame)
    assert dark is not None and bright is not None
    assert bright.mean > dark.mean * 2
    assert 0.0 <= dark.mean <= dark.probe <= 1.0


@needs_magick
def test_luminance_probe_returns_none_for_a_missing_file():
    frame = tx.Rect(0, 0, 100, 100)
    assert tx.measure_region_luminance(Path("/nope/missing.png"), frame, frame) is None


@needs_magick
@needs_assets
def test_dark_image_gets_only_the_minimum_tint():
    plan = full_bleed(TextPosition.LEFT_PANEL, scrim_opacity=0.45)
    dark = tx.layout_slide(HEADING, BULLETS, plan, HD, image_path=DARK_IMAGE)
    assert dark.scrim_opacity == pytest.approx(tx.SCRIM_MIN_TINT, abs=1 / 255)


def test_solved_opacity_is_representable_in_an_8_bit_alpha_channel():
    for value in (0.0, 0.1234, 0.5, 0.6564, 0.9999, 1.0):
        snapped = tx.quantise_opacity(value)
        assert snapped >= value
        assert round(snapped * 255) == pytest.approx(snapped * 255)


@needs_magick
@needs_assets
def test_adaptive_scrim_is_heavier_on_the_bright_image_and_both_clear_aa():
    plan = full_bleed(TextPosition.LEFT_PANEL, scrim_opacity=0.45)
    dark = tx.layout_slide(HEADING, BULLETS, plan, HD, image_path=DARK_IMAGE)
    bright = tx.layout_slide(HEADING, BULLETS, plan, HD, image_path=BRIGHT_IMAGE)

    assert bright.scrim_opacity > dark.scrim_opacity
    # The defect this fixes: a fixed 0.45 was too little for the sunlit still.
    assert bright.scrim_opacity > 0.45
    # ...and needlessly heavy for the dark one.
    assert dark.scrim_opacity < 0.45

    for slide in (dark, bright):
        report = slide.contrast
        assert report is not None
        assert report.source == "image"
        assert report.meets_aa, report.summary()
        assert report.ratio_after >= tx.WCAG_AA
        assert report.accent_ratio_after >= tx.WCAG_AA
        assert report.background_luminance_after <= report.background_luminance_before

    assert bright.contrast.ratio_before < tx.WCAG_AA, "bright still should start illegible"
    assert bright.contrast.ratio_after > bright.contrast.ratio_before * 3


@needs_magick
@needs_assets
def test_the_modelled_scrim_matches_what_compositing_actually_produces(tmp_path):
    """Solve for the opacity, then burn it in and re-measure. Model vs pixels."""
    plan = full_bleed(TextPosition.LEFT_PANEL)
    slide = tx.layout_slide(HEADING, BULLETS, plan, HD, image_path=BRIGHT_IMAGE)
    scene = tx.build_scene_text(
        HEADING, BULLETS, plan, HD, tmp_path, image_path=BRIGHT_IMAGE, slide=slide
    )
    scrim = next(x for x in scene.layers if x.kind == "scrim")
    binary = tx.require_imagemagick()

    base = tmp_path / "base.png"
    subprocess.run(  # noqa: S603
        [binary, str(BRIGHT_IMAGE), "-resize", f"{HD.width}x{HD.height}^",
         "-gravity", "center", "-extent", f"{HD.width}x{HD.height}", str(base)], check=True
    )
    composited = tmp_path / "composited.png"
    subprocess.run(  # noqa: S603
        [binary, str(base), "-gravity", "none", str(scrim.png_path),
         "-geometry", "+0+0", "-composite", str(composited)], check=True
    )

    frame = tx.Rect(0, 0, HD.width, HD.height)
    assert slide.scrim_region is not None
    after = tx.measure_region_luminance(composited, slide.scrim_region, frame)
    assert after is not None
    modelled = slide.contrast.background_luminance_after
    assert after.probe_relative == pytest.approx(modelled, abs=0.02)
    text_luminance = tx.relative_luminance(slide.theme.text)
    assert tx.contrast_ratio(text_luminance, after.probe_relative) >= tx.WCAG_AA


# ====================================================================== caching


@needs_magick
def test_identical_input_reuses_the_cached_png(tmp_path):
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    first = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path)
    stamps = {layer.png_path: layer.png_path.stat().st_mtime_ns for layer in first.layers}
    second = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path)

    assert [x.png_path for x in second.layers] == [x.png_path for x in first.layers]
    for layer in second.layers:
        assert layer.png_path.stat().st_mtime_ns == stamps[layer.png_path], "re-rasterised"


@needs_magick
def test_changing_the_text_changes_the_cache_key(tmp_path):
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    a = tx.build_scene_text("Heading One", BULLETS, plan, HD, tmp_path)
    b = tx.build_scene_text("Heading Two", BULLETS, plan, HD, tmp_path)
    heading_a = next(x for x in a.layers if x.kind == "heading").png_path
    heading_b = next(x for x in b.layers if x.kind == "heading").png_path
    assert heading_a != heading_b
    assert heading_a.is_file() and heading_b.is_file()


@needs_magick
def test_emphasis_and_profile_are_part_of_the_cache_key(tmp_path):
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    plain = tx.build_scene_text(HEADING, [BulletPoint(text="Same")], plan, HD, tmp_path)
    accent = tx.build_scene_text(
        HEADING, [BulletPoint(text="Same", emphasis=True)], plan, HD, tmp_path
    )
    draft = tx.build_scene_text(HEADING, [BulletPoint(text="Same")], plan, DRAFT, tmp_path)
    paths = {
        next(x for x in scene.layers if x.kind == "bullet").png_path
        for scene in (plain, accent, draft)
    }
    assert len(paths) == 3


def test_cache_key_is_stable_and_sensitive():
    assert tx.cache_key("a", 1) == tx.cache_key("a", 1)
    assert tx.cache_key("a", 1) != tx.cache_key("a", 2)
    assert tx.cache_key("a", 1) != tx.cache_key("a", "1")


@needs_magick
def test_concurrent_builds_share_one_cache_without_corrupting_it(tmp_path):
    """Scene clips render on 4 threads; the cache has to survive that."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)

    def build(_: int):
        return tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        scenes = list(pool.map(build, range(8)))

    expected = [layer.png_path for layer in scenes[0].sorted_layers()]
    for scene in scenes:
        assert [layer.png_path for layer in scene.sorted_layers()] == expected

    for layer in scenes[0].sorted_layers():
        assert png_size(layer.png_path) == (layer.width, layer.height)
    # No half-written temp files survived the race.
    assert not list((tmp_path / "text").glob(".*.png"))
    assert len(list((tmp_path / "text").glob("*.png"))) == len(expected)


# ======================================================================= fonts


def test_find_font_returns_an_existing_file():
    assert Path(tx.find_font()).is_file()


def test_find_font_skips_missing_candidates():
    chosen = tx.find_font(("/nope/a.ttf", "/nope/b.ttf", *tx.FONT_CANDIDATES))
    assert Path(chosen).is_file()


def test_find_font_raises_when_nothing_exists():
    with pytest.raises(tx.FontNotFoundError):
        tx.find_font(("/nope/a.ttf", "/nope/b.ttf"))


def test_font_env_override_wins(monkeypatch, tmp_path):
    fake = tmp_path / "Brand.ttf"
    fake.write_bytes(b"not really a font")
    monkeypatch.setenv("VIDEO_FONT_FILE", str(fake))
    tx.find_font.cache_clear()
    try:
        assert tx.find_font() == str(fake)
    finally:
        tx.find_font.cache_clear()


def test_heavier_font_finds_a_real_heavier_face_or_says_so():
    """`None` is a normal answer; a path that does not exist is not."""
    heavier = tx.heavier_font(tx.find_font())
    if heavier is not None:
        assert Path(heavier).is_file()
        assert heavier != tx.find_font()
    assert tx.heavier_font("/nope/Whatever-Bold.ttf") is None


def test_heavier_font_maps_bold_onto_black_when_both_exist(tmp_path):
    bold = tmp_path / "Fake Bold.ttf"
    black = tmp_path / "Fake Black.ttf"
    bold.write_bytes(b"x")
    black.write_bytes(b"x")
    assert tx.heavier_font(str(bold)) == str(black)
    # ...and does not invent a file that is not there.
    lonely = tmp_path / "Only Bold.ttf"
    lonely.write_bytes(b"x")
    assert tx.heavier_font(str(lonely)) is None


# =============================================================== line advance


@needs_magick
def test_a_heavier_face_advances_lines_further_than_the_point_size_implies():
    """The measurement that stops a 900-weight bullet clipping its own canvas.

    ImageMagick advances by the *face's* line height plus interline-spacing, not by the
    point size plus it. Bold and Black disagree by a lot, so the layout cannot assume.
    """
    base = tx.find_font()
    heavier = tx.heavier_font(base)
    if heavier is None:
        pytest.skip("no heavier face installed")
    for size in (18, 36):
        light = tx.font_line_advance(base, size)
        heavy = tx.font_line_advance(heavier, size)
        assert light and heavy
        assert heavy > light, f"{size}px: {heavy} !> {light}"
        assert light > size, "even the base face advances more than its point size"


@needs_magick
def test_interline_spacing_makes_the_rendered_advance_match_the_declared_one():
    """Declared ``line_height`` has to be the real advance, or the canvas is a lie."""
    heavier = tx.heavier_font(tx.find_font())
    if heavier is None:
        pytest.skip("no heavier face installed")
    binary = tx.require_imagemagick()
    size = 36
    line_height = round(size * tx.LINE_SPACING)
    spacing = tx.interline_spacing(
        line_height, size, advance=tx.font_line_advance(heavier, size)
    )

    def ink_bottom(lines: list[str]) -> int:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            text_file = tx.write_text_file(Path(tmp), "t.txt", lines)
            out = subprocess.run(  # noqa: S603
                [binary, "-size", "900x400", "xc:none", "-font", heavier,
                 "-pointsize", str(size), "-interline-spacing", str(spacing),
                 "-gravity", "northwest", "-fill", "white", "-annotate", "+0+0",
                 tx.imagemagick_text_arg(text_file), "-format", "%@", "info:"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        size_part, _, offset = out.partition("+")
        _, height = (int(v) for v in size_part.split("x"))
        return int(offset.split("+")[0]) + height

    assert ink_bottom(["Hxg", "Hxg"]) - ink_bottom(["Hxg"]) == line_height


def test_interline_spacing_falls_back_to_the_historical_formula():
    """No measurement available means the base face keeps exactly what it always had."""
    assert tx.interline_spacing(44, 36, advance=None) == 44 - 36


@needs_magick
def test_only_a_block_in_a_different_face_gets_its_advance_corrected():
    """Introducing a heavier weight must not move type that already ships."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    for block in slide.bullets:
        if block.font == slide.font:
            assert block.interline_spacing is None
        elif tx.heavier_font(slide.font):
            assert block.interline_spacing is not None


# ================================================================== watermark


def test_logo_rect_sits_in_the_bottom_left_inside_the_frame():
    theme = Theme()
    rect = tx.logo_rect(HD, theme)
    assert rect.x == round(HD.width * theme.logo_margin_fraction)
    assert rect.height == round(HD.height * theme.logo_height_fraction)
    # Bottom-left: the same inset from both the left and the bottom edge.
    assert HD.height - rect.bottom == rect.x
    assert rect.x > 0 and rect.bottom < HD.height


def test_logo_rect_scales_with_the_profile_not_hardcoded():
    full, draft = tx.logo_rect(HD), tx.logo_rect(DRAFT)
    assert draft.height == pytest.approx(full.height / 2, abs=1)
    assert draft.x == pytest.approx(full.x / 2, abs=1)


def test_logo_rect_uses_the_real_size_when_the_caller_has_it():
    reserved = tx.logo_rect(HD)
    exact = tx.logo_rect(HD, size=(51, 49))
    assert (exact.width, exact.height) == (51, 49)
    # The unmeasured box must never be narrower than a real one, or the collision
    # check would miss an overlap.
    assert reserved.width >= exact.width


def test_rect_intersects_is_exclusive_at_the_edges():
    a = tx.Rect(0, 0, 10, 10)
    assert a.intersects(tx.Rect(9, 9, 5, 5))
    assert not a.intersects(tx.Rect(10, 0, 5, 5)), "touching edges are not an overlap"
    assert not a.intersects(tx.Rect(0, 10, 5, 5))
    assert a.intersects(tx.Rect(-5, -5, 20, 20))


def test_ink_rects_are_tighter_than_the_layer_canvases():
    """A centred stack's canvas spans the column; its words do not.

    The watermark collision check runs against these, so over-reporting would cry wolf on
    every centred slide.
    """
    centred = layout(VisualPlan(layout=SlideLayout.TITLE_CARD))
    assert centred.geometry.align == "center"
    for canvas, ink in zip(centred.text_rects, centred.ink_rects(), strict=True):
        assert ink.x > canvas.x
        assert ink.right < canvas.right
        assert ink.y == canvas.y and ink.height == canvas.height

    left = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    for canvas, ink in zip(left.text_rects, left.ink_rects(), strict=True):
        assert ink.x >= canvas.x
        assert ink.right <= canvas.right


def test_flatten_svg_paths_keeps_the_shapes_and_drops_the_effects():
    """ImageMagick's built-in SVG renderer implements neither <mask> nor <filter>."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="46">'
        '<path fill="#863bff" d="M0 0h4v4z"/>'
        '<defs><filter id="b"/></defs>'
        '<mask id="a"><path d="M0 0h1v1z"/></mask>'
        '<g mask="url(#a)"><ellipse cx="5" cy="5" rx="2" ry="2"/></g>'
        "</svg>"
    )
    flat = tx.flatten_svg_paths(svg)
    assert flat is not None
    assert "#863bff" in flat
    assert flat.count("<path") == 1, "only the root's own paths"
    for dropped in ("<mask", "<g", "ellipse", "filter"):
        assert dropped not in flat, dropped
    assert 'width="48"' in flat and 'height="46"' in flat


def test_flatten_svg_paths_gives_up_cleanly_on_junk():
    assert tx.flatten_svg_paths("not xml at all") is None
    assert tx.flatten_svg_paths('<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>') is None


REPO_LOGO = Path("/Users/argo/ab/prompt-to-video-v2/frontend/public/favicon.svg")
needs_logo = pytest.mark.skipif(not REPO_LOGO.is_file(), reason="repo logo not present")


@needs_magick
@needs_logo
def test_rasterise_logo_hits_the_requested_height_with_real_alpha(tmp_path):
    png = tx.rasterise_logo(REPO_LOGO, 49, 0.85, tmp_path)
    assert png is not None and png.is_file()
    width, height = png_size(png)
    assert height == 49
    # 48x46 source, so the width follows the aspect rather than being forced square.
    assert width == round(49 * 48 / 46)

    stats = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(png), "-alpha", "extract",
         "-format", "%[fx:mean] %[fx:maxima]", "info:"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    coverage, peak = float(stats[0]), float(stats[1])
    assert coverage > tx.LOGO_MIN_ALPHA_COVERAGE, "rasterised to nothing"
    # Opacity is baked in, so the most opaque pixel is the requested opacity.
    assert peak == pytest.approx(0.85, abs=2 / 255)


@needs_magick
@needs_logo
def test_rasterise_logo_keeps_the_brand_colour(tmp_path):
    """The favicon's fill is `#863bff`; a renderer that drops it is a silent failure."""
    png = tx.rasterise_logo(REPO_LOGO, 120, 1.0, tmp_path)
    assert png is not None
    colours = subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), str(png), "-alpha", "off", "-depth", "8",
         "-format", "%c", "histogram:info:"],
        capture_output=True, text=True, check=True,
    ).stdout.upper()
    assert "#863BFF" in colours


@needs_magick
@needs_logo
def test_rasterise_logo_is_cached_and_keyed_on_everything_that_matters(tmp_path):
    first = tx.rasterise_logo(REPO_LOGO, 49, 0.85, tmp_path)
    assert first is not None
    stamp = first.stat().st_mtime_ns
    again = tx.rasterise_logo(REPO_LOGO, 49, 0.85, tmp_path)
    assert again == first
    assert again.stat().st_mtime_ns == stamp, "re-rasterised instead of reusing"

    taller = tx.rasterise_logo(REPO_LOGO, 98, 0.85, tmp_path)
    fainter = tx.rasterise_logo(REPO_LOGO, 49, 0.40, tmp_path)
    assert len({first, taller, fainter}) == 3


def test_rasterise_logo_returns_none_rather_than_failing_a_render(tmp_path):
    """Branding is the last 1% of the frame; it must not cost the other 99%."""
    assert tx.rasterise_logo(tmp_path / "nope.svg", 49, 0.85, tmp_path) is None
    empty = tmp_path / "empty.svg"
    empty.write_text("", encoding="utf-8")
    assert tx.rasterise_logo(empty, 49, 0.85, tmp_path) is None
    # No ImageMagick at all: still an answer, not an exception.
    if REPO_LOGO.is_file():
        assert tx.rasterise_logo(REPO_LOGO, 49, 0.85, tmp_path, binary="") is None


@needs_magick
def test_a_blank_source_is_reported_as_no_logo(tmp_path):
    blank = tmp_path / "blank.png"
    subprocess.run(  # noqa: S603
        [tx.require_imagemagick(), "-size", "48x46", "xc:none", f"PNG32:{blank}"], check=True
    )
    assert tx.rasterise_logo(blank, 49, 0.85, tmp_path) is None


# ============================================== emphasis in the actual pixels


@needs_magick
def test_a_normal_marker_is_hollow_and_an_emphasised_one_is_solid(tmp_path):
    """The shape cue, measured: a ring has a transparent middle, a disc does not."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    scene = tx.build_scene_text(
        HEADING, BULLETS, VisualPlan(layout=SlideLayout.HERO_RIGHT), HD, tmp_path, slide=slide
    )
    layers = [x for x in scene.sorted_layers() if x.kind == "bullet"]
    binary = tx.require_imagemagick()

    for block, layer in zip(slide.bullets, layers, strict=True):
        cx = block.offset_x + max(block.marker_diameter // 2, block.indent // 2)
        cy = round(block.size * 0.52)
        centre = subprocess.run(  # noqa: S603
            [binary, str(layer.png_path), "-crop", f"1x1+{cx}+{cy}", "+repage",
             "-alpha", "extract", "-format", "%[fx:maxima]", "info:"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        ring_edge = subprocess.run(  # noqa: S603
            [binary, str(layer.png_path),
             "-crop", f"1x1+{cx}+{cy - max(2, block.marker_diameter // 2) + 1}", "+repage",
             "-alpha", "extract", "-format", "%[fx:maxima]", "info:"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert float(ring_edge) > 0.5, "no marker ink at the rim"
        if block.marker_filled:
            assert float(centre) > 0.5, "emphasised marker should be a solid disc"
        else:
            assert float(centre) < 0.5, "normal marker should be a hollow ring"


@needs_magick
def test_an_emphasised_bullet_carries_more_ink_than_a_normal_one(tmp_path):
    """Weight, measured on pixels rather than asserted from a font name."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT))
    emph = next(b for b in slide.bullets if b.emphasis)
    normal = next(b for b in slide.bullets if not b.emphasis)
    binary = tx.require_imagemagick()

    def ink_mass(block) -> float:
        text_file = tx.write_text_file(tmp_path, f"w{block.emphasis}.txt", ["Handgloves"])
        out = tmp_path / f"w{block.emphasis}.png"
        subprocess.run(  # noqa: S603
            [binary, "-size", "800x160", "xc:none", "-font", block.font,
             "-pointsize", str(block.size), "-gravity", "northwest", "-fill", "white",
             "-annotate", "+10+10", tx.imagemagick_text_arg(text_file), f"PNG32:{out}"],
            check=True,
        )
        return float(subprocess.run(  # noqa: S603
            [binary, str(out), "-alpha", "extract", "-format", "%[fx:mean*w*h]", "info:"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())

    heavy, light = ink_mass(emph), ink_mass(normal)
    # Same word, same colour: the only thing that changed is how much ink is on the page.
    assert heavy > light * 1.15, f"emphasis barely reads: {heavy} vs {light}"


def test_available_fonts_are_all_real_files():
    assert all(Path(path).is_file() for path in tx.available_fonts())
