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
import hashlib
import subprocess
from pathlib import Path

import pytest

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
        assert column.x < region.x
    else:
        assert region.right <= column.x
        assert column.x > region.x


def test_hero_grid_matches_the_direction_spec():
    """DIRECTION §6.2 fixes the grid in pixels. The type scale is derived from this column
    width (22 characters of a 78px heading), so the two cannot drift apart."""
    geometry = tx.slide_geometry(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    assert geometry.text_column.x == 104
    assert geometry.text_column.width == 904
    assert geometry.image_region is not None
    assert geometry.image_region.as_tuple() == (1096, 90, 720, 900)
    # 4:5 — a requirement on the image provider, not a crop preference.
    assert geometry.image_region.width / geometry.image_region.height == pytest.approx(0.8)


def test_hero_right_text_column_is_a_readable_fraction_of_the_frame():
    geometry = tx.slide_geometry(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    share = geometry.text_column.width / HD.width
    assert 0.40 <= share <= 0.50


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
    lines = tx.wrap_to_width(LONG_BULLET, 400, measure, max_lines=8)
    assert len(lines) > 1
    assert all(measure(line) <= 400 for line in lines)
    assert " ".join(lines) == " ".join(LONG_BULLET.split())


def test_a_bullet_wraps_to_at_most_two_lines_by_default():
    """DIRECTION §2.1: a bullet is 34 characters of copy. A third line means the copy rule
    was broken upstream, and ellipsising is a louder complaint than silently reflowing."""
    assert tx.MAX_BULLET_LINES == 2
    lines = tx.wrap_to_width(LONG_BULLET, 400, make_measurer())
    assert len(lines) == 2
    assert lines[-1].endswith(tx.ELLIPSIS)


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
    times = tx.bullet_times(crowded, plan, first_at=0.0)
    assert times == sorted(times)
    assert all(b - a >= 0.6 - 1e-6 for a, b in zip(times, times[1:], strict=False))


def test_bullet_times_respect_a_later_narration_cue():
    plan = VisualPlan(bullet_min_gap=0.6)
    times = tx.bullet_times(
        [BulletPoint(text="a", appear_at=0.0), BulletPoint(text="b", appear_at=5.0)],
        plan,
        first_at=0.0,
    )
    assert times == [0.0, 5.0]


def test_no_bullet_reveals_before_the_heading_has_finished_arriving():
    """DIRECTION §4.2. A bullet landing at 0.4s moves while the heading is still moving,
    and both entrances are lost."""
    plan = VisualPlan(bullet_min_gap=1.6)
    times = tx.bullet_times(
        [BulletPoint(text="a", appear_at=0.0), BulletPoint(text="b", appear_at=0.2)], plan
    )
    assert times[0] == pytest.approx(tx.FIRST_REVEAL_EARLIEST)
    assert times[1] - times[0] >= 1.6 - 1e-6


# ================================================================== type scale


def test_the_type_scale_is_one_modular_ladder():
    """DIRECTION §2: ratio 1.333, base = the bullet, and every size is a step on it."""
    assert tx.TYPE_SCALE_RATIO == pytest.approx(4 / 3)
    sizes = [tx.type_size(step, 1920) for step in (-1, 0, 1, 2, 3)]
    assert sizes == [33, 44, 59, 78, 104]
    for small, large in zip(sizes, sizes[1:], strict=False):
        assert large / small == pytest.approx(4 / 3, abs=0.02)


def test_the_bullet_clears_the_published_legibility_floors():
    """40px is the 1080p body-text floor and 44px is the BBC HD accessibility floor. The
    old 36px bullet was under both before ``SHRINK_STEPS`` was even involved."""
    assert tx.type_size(tx.TYPE_STEP_BODY, 1920) >= 44
    assert tx.type_size(tx.TYPE_STEP_BODY, 1920) / 1080 >= 1 / 27  # inside the EBU band


def test_the_heading_is_at_least_half_again_the_bullet():
    bullet = tx.type_size(tx.TYPE_STEP_BODY, 1920)
    heading = tx.type_size(tx.TYPE_STEP_HEADING, 1920)
    assert heading / bullet >= 1.5


def test_the_scale_is_the_same_design_at_540p():
    for step in (-1, 0, 1, 2, 3):
        assert tx.type_size(step, 960) == pytest.approx(tx.type_size(step, 1920) / 2, abs=1)


def test_a_video_has_exactly_two_heading_sizes():
    """DIRECTION §9: the title card, and everything else. Four heading sizes is the
    opposite of the uniformity that was asked for, and a 1.1x heading is a 7.8px difference
    nobody can see that still moves the bullet baseline, which they can."""
    sizes = {role: tx.heading_size_for(role, 1920) for role in SceneRole}
    assert sizes[SceneRole.TITLE] == 105
    assert (
        sizes[SceneRole.CONTENT] == sizes[SceneRole.SUMMARY] == sizes[SceneRole.CLOSING] == 78
    )
    assert len(set(sizes.values())) == 2


# ============================================================ slide layout plan


def layout(plan: VisualPlan, profile: RenderProfile = HD, **kwargs):
    return tx.layout_slide(HEADING, BULLETS, plan, profile, **kwargs)


def hero(role: SceneRole, heading: str, points: list[str], **kwargs):
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, text_position=TextPosition.LEFT_PANEL)
    bullets = [BulletPoint(text=t, appear_at=1.15 + 1.6 * i) for i, t in enumerate(points)]
    return tx.layout_slide(heading, bullets, plan, HD, role=role, **kwargs)


# DIRECTION-conforming copy: heading <= 22 chars, bullets <= 34 chars.
SHORT_HEADING = "Inspect the sender"
SHORT_BULLETS = [
    "Check the real sender address",
    "Hover a link before clicking",
    "Treat urgency as a warning",
    "Report it, never delete it",
]


def test_every_body_role_lands_on_the_same_grid():
    """The rule that makes the deck read as one deck: content, summary and closing differ
    only in how many bullets they carry. Nothing moves."""
    content = hero(SceneRole.CONTENT, SHORT_HEADING, SHORT_BULLETS)
    summary = hero(SceneRole.SUMMARY, "What to remember", SHORT_BULLETS)
    closing = hero(SceneRole.CLOSING, "If in doubt, report", SHORT_BULLETS[:2])

    for other in (summary, closing):
        assert other.heading_size == content.heading_size
        assert other.heading_rect.y == content.heading_rect.y
        assert other.rule_rect == content.rule_rect
        assert other.bullets[0].rect.y == content.bullets[0].rect.y
        assert other.bullets[0].size == content.bullets[0].size
    assert len(closing.bullets) == 2, "the closing earns its difference from having 2 points"


def test_the_bullet_pitch_is_constant_and_a_wrap_pushes_the_stack_down():
    slide = hero(SceneRole.CONTENT, SHORT_HEADING, SHORT_BULLETS)
    tops = [b.rect.y for b in slide.bullets]
    pitches = {b - a for a, b in zip(tops, tops[1:], strict=False)}
    assert pitches == {84}, f"DIRECTION §3.3 pitch is 84px: {pitches}"

    wrapped = hero(
        SceneRole.CONTENT,
        SHORT_HEADING,
        ["Check the real sender address and then the reply-to as well", *SHORT_BULLETS[1:]],
    )
    assert len(wrapped.bullets[0].lines) == 2
    # One line more, one line-height further down — never a collision.
    assert wrapped.bullets[1].rect.y - wrapped.bullets[0].rect.y == 84 + wrapped.bullets[
        0
    ].line_height


def test_the_first_bullet_does_not_move_when_a_heading_wraps():
    """A variable heading line count is what put the bullet stack at three different heights
    across four slides. The heading box is fixed and the type is bottom-aligned in it."""
    one_line = hero(SceneRole.CONTENT, SHORT_HEADING, SHORT_BULLETS)
    two_lines = hero(
        SceneRole.CONTENT, "Recognise the pressure tactics attackers use", SHORT_BULLETS
    )
    assert len(one_line.heading_lines) == 1
    assert len(two_lines.heading_lines) == 2

    assert two_lines.rule_rect == one_line.rule_rect
    assert two_lines.bullets[0].rect.y == one_line.bullets[0].rect.y
    # The one-line heading is the one that carries the reserved air, above its type.
    assert one_line.heading_offset_y == one_line.heading_line_height
    assert two_lines.heading_offset_y == 0


def test_the_accent_rule_is_one_fixed_width_everywhere_but_the_title_card():
    body = [
        hero(SceneRole.CONTENT, SHORT_HEADING, SHORT_BULLETS),
        hero(SceneRole.SUMMARY, "What to remember", SHORT_BULLETS),
        hero(SceneRole.CLOSING, "If in doubt, report", SHORT_BULLETS[:2]),
    ]
    assert {s.rule_rect.width for s in body} == {88}
    assert {s.rule_rect.height for s in body} == {4}
    title = tx.layout_slide(
        "How phishing works", [], VisualPlan(layout=SlideLayout.TITLE_CARD), HD
    )
    assert title.rule_rect is not None
    assert title.rule_rect.width == 120


# ================================================================== title card


def test_the_title_card_is_a_title_card_and_not_a_content_slide():
    title = tx.layout_slide(
        "How phishing works", BULLETS, VisualPlan(layout=SlideLayout.TITLE_CARD), HD
    )
    content = hero(SceneRole.CONTENT, SHORT_HEADING, SHORT_BULLETS)

    assert title.geometry.role is SceneRole.TITLE, "a title_card layout implies the role"
    assert title.bullets == [], "bullet_budget 0 — the opener earns its impact by having none"
    assert title.geometry.align == "center"
    assert title.heading_size > content.heading_size * 1.3
    assert title.kicker == "TRAINING MODULE"
    assert title.kicker_size == tx.type_size(tx.TYPE_STEP_KICKER, HD.width)
    assert title.kicker_tracking > 0, "uppercase caps need tracking"
    assert title.kicker_rect is not None
    assert title.kicker_rect.bottom <= title.heading_rect.y + title.heading_line_height


def test_the_title_block_is_optically_centred_not_top_anchored():
    """The rejected opener top-anchored its block and left the bottom 55% of the frame
    empty, which reads as a slide that failed to load."""
    title = tx.layout_slide(
        "How phishing works", [], VisualPlan(layout=SlideLayout.TITLE_CARD), HD
    )
    box = tx.Rect.union(title.text_rects)
    assert box is not None
    centre = box.y + box.height / 2
    assert centre == pytest.approx(HD.height * tx.TITLE_OPTICAL_CENTRE_RATIO, abs=40)
    assert centre < HD.height / 2, "optical centre sits above geometric centre"


def test_a_role_never_moves_the_image_region():
    """``ffmpeg_backend`` asks for the hero rectangle without knowing the role. If a role
    moved it, the image and the rounded hole cut for it would land in different places."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    regions = {
        tx.slide_geometry(plan, HD, role=role).image_region.as_tuple() for role in SceneRole
    }
    assert len(regions) == 1
    assert tx.image_region(plan, HD).as_tuple() == regions.pop()


def test_the_bullet_budget_is_enforced_wherever_the_slide_is_built():
    """Even if a script writes five points onto a closing scene."""
    for role in SceneRole:
        slide = hero(role, SHORT_HEADING, SHORT_BULLETS + ["A fifth point too many"])
        assert len(slide.bullets) == role.bullet_budget


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


def test_emphasis_is_off_by_default_so_every_bullet_is_set_identically():
    """The rejected build signalled emphasis three ways at once (size, weight, outline) and
    it read as a font-substitution bug. Colour and marker shape are both off the table, and
    a uniform list beats a hierarchy nobody notices — so the default is nothing at all."""
    assert tx.EMPHASIS_MODE == "off"
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=Theme())
    assert any(b.emphasis for b in slide.bullets), "test data must contain an emphasis flag"

    assert len({b.text_colour for b in slide.bullets}) == 1
    assert len({b.font for b in slide.bullets}) == 1
    assert len({b.size for b in slide.bullets}) == 1
    assert len({b.stroke_ratio for b in slide.bullets}) == 1
    assert len({b.marker_shape for b in slide.bullets}) == 1
    assert len({b.marker_diameter for b in slide.bullets}) == 1
    assert len({b.indent for b in slide.bullets}) == 1
    assert {b.faux_bold for b in slide.bullets} == {0.0}


def test_weight_emphasis_changes_the_face_and_nothing_else():
    """The one switchable emphasis. Measured at 1080p it is a 38% increase in ink mass —
    visible, but it also reads as a different font and lands the baseline 8px low, which is
    why it is opt-in. It must not touch a single other metric."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=Theme(), emphasis_mode="weight")
    emphasised = next(b for b in slide.bullets if b.emphasis)
    normal = next(b for b in slide.bullets if not b.emphasis)

    if tx.heavier_font(slide.font):
        assert emphasised.font != normal.font
        assert emphasised.faux_bold == 0.0
    else:
        assert emphasised.faux_bold > 0.0
    assert emphasised.text_colour == normal.text_colour
    assert emphasised.size == normal.size
    assert emphasised.stroke_ratio == normal.stroke_ratio
    assert emphasised.marker_shape == normal.marker_shape
    assert emphasised.marker_diameter == normal.marker_diameter
    assert emphasised.indent == normal.indent


def test_an_unknown_emphasis_mode_falls_back_to_off():
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), emphasis_mode="rainbow")
    assert len({b.font for b in slide.bullets}) == 1


def test_markers_are_graphic_so_they_keep_the_accent_colour():
    """`accent` is still allowed — on the rule and the markers, which are not text."""
    theme = Theme()
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    assert {b.marker_colour for b in slide.bullets} == {theme.accent}
    assert slide.rule_rect is not None
    # The text indent is shared so the left edge lines up regardless of emphasis.
    assert len({b.indent for b in slide.bullets}) == 1


@pytest.mark.parametrize("shape", list(tx.MARKER_SHAPES))
def test_one_marker_shape_per_video_whatever_the_theme_asks_for(shape):
    """`Theme.marker` is the only thing that decides the shape, and it decides it once."""
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=Theme(marker=shape))
    assert {b.marker_shape for b in slide.bullets} == {shape}
    # The gutter is a fixed 1.0 em, so the text edge does not move with the shape.
    assert {b.indent for b in slide.bullets} == {slide.bullets[0].size}


def test_an_unknown_marker_shape_falls_back_to_a_disc():
    theme = Theme.model_construct(**{**Theme().model_dump(), "marker": "sparkle"})
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    assert {b.marker_shape for b in slide.bullets} == {"disc"}


def test_uniform_text_is_switchable_back_to_accent_for_emphasis():
    """`uniform_text=False` must reproduce the pre-brand-rule behaviour: colour only."""
    theme = Theme(uniform_text=False)
    slide = layout(VisualPlan(layout=SlideLayout.HERO_RIGHT), theme=theme)
    emphasised = [b for b in slide.bullets if b.emphasis]
    normal = [b for b in slide.bullets if not b.emphasis]
    assert emphasised and normal

    assert all(b.text_colour == theme.accent for b in emphasised)
    assert all(b.text_colour == theme.text for b in normal)
    # ...and the geometry is left alone, markers included: colour does all the work.
    assert {b.font for b in slide.bullets} == {slide.font}
    assert {b.size for b in slide.bullets} == {normal[0].size}
    assert len({b.marker_shape for b in slide.bullets}) == 1
    assert len({b.marker_diameter for b in slide.bullets}) == 1


def test_centred_layouts_share_one_offset_so_the_stack_reads_as_a_block():
    # A centred *body* stack: the title card has no bullets at all (bullet_budget 0).
    slide = layout(full_bleed(TextPosition.CENTER))
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
    assert any(len(b.lines) > 1 for b in slide.bullets), "need a multi-line wrap to test"

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
def test_the_same_bullet_shares_a_png_whatever_its_emphasis_flag_says(tmp_path):
    """The strongest possible statement of uniformity: with emphasis off, two bullets that
    differ only in the flag are *the same file*. The profile still has to split them."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    plain = tx.build_scene_text(HEADING, [BulletPoint(text="Same")], plan, HD, tmp_path)
    flagged = tx.build_scene_text(
        HEADING, [BulletPoint(text="Same", emphasis=True)], plan, HD, tmp_path
    )
    draft = tx.build_scene_text(HEADING, [BulletPoint(text="Same")], plan, DRAFT, tmp_path)

    def bullet_png(scene):
        return next(x for x in scene.layers if x.kind == "bullet").png_path

    assert bullet_png(plain) == bullet_png(flagged)
    assert bullet_png(draft) != bullet_png(plain)


@needs_magick
def test_weight_emphasis_is_part_of_the_cache_key(tmp_path):
    """...but when it *is* switched on, two bullets differing only in weight must not
    collide on one PNG."""
    if tx.heavier_font(tx.find_font()) is None:
        pytest.skip("no heavier face installed")
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    plain = tx.build_scene_text(
        HEADING, [BulletPoint(text="Same")], plan, HD, tmp_path, emphasis_mode="weight"
    )
    heavy = tx.build_scene_text(
        HEADING, [BulletPoint(text="Same", emphasis=True)], plan, HD, tmp_path,
        emphasis_mode="weight",
    )
    assert (
        next(x for x in plain.layers if x.kind == "bullet").png_path
        != next(x for x in heavy.layers if x.kind == "bullet").png_path
    )


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
@pytest.mark.parametrize("shape", ["dash", "disc", "ring", "chevron"])
@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_every_marker_in_a_scene_is_pixel_identical(tmp_path, shape, theme_name):
    """The whole point of the change, measured on pixels.

    Crop the marker's box out of every bullet layer and hash it. One hash means the ink is
    identical byte for byte — same shape, same size, same position, same colour — including
    on the bullet carrying ``emphasis=True``, which is what used to get a different shape.
    """
    theme = (
        Theme(marker=shape)
        if theme_name == "dark"
        else Theme(marker=shape, bg="#F7F8FA", surface="#FFFFFF", text="#111827",
                   muted="#55607A", accent="#1D4ED8")
    )
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    slide = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=theme)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path, theme=theme, slide=slide)
    layers = [x for x in scene.sorted_layers() if x.kind == "bullet"]
    binary = tx.require_imagemagick()
    assert any(b.emphasis for b in slide.bullets)

    digests, accents = set(), set()
    for block, layer in zip(slide.bullets, layers, strict=True):
        radius = max(2, block.marker_diameter // 2)
        cx = block.offset_x + max(radius, block.indent // 2)
        cy = round(block.size * 0.52)
        pad = radius + 6  # the ink plus its halo, and nothing of the text
        box = f"{2 * pad}x{2 * pad}+{cx - pad}+{cy - pad}"
        raw = subprocess.run(  # noqa: S603
            [binary, str(layer.png_path), "-crop", box, "+repage", "-depth", "8", "RGBA:-"],
            capture_output=True, check=True,
        ).stdout
        digests.add(hashlib.sha256(raw).hexdigest())
        histogram = subprocess.run(  # noqa: S603
            [binary, str(layer.png_path), "-crop", box, "+repage",
             "-format", "%c", "histogram:info:"],
            capture_output=True, text=True, check=True,
        ).stdout.upper()
        accents.add(theme.accent.lstrip("#").upper() in histogram)

    assert len(digests) == 1, f"{len(digests)} different markers in one stack"
    assert accents == {True}, "a marker is missing its accent ink"


@needs_magick
def test_the_marker_gutter_is_the_only_place_accent_ink_appears_in_a_bullet(tmp_path):
    """`uniform_text` in pixels: the words are one colour and it is never the accent."""
    theme = Theme()
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    slide = tx.layout_slide(HEADING, BULLETS, plan, HD, theme=theme)
    scene = tx.build_scene_text(HEADING, BULLETS, plan, HD, tmp_path, theme=theme, slide=slide)
    binary = tx.require_imagemagick()
    accent = theme.accent.lstrip("#").upper()

    for block, layer in zip(
        slide.bullets, [x for x in scene.sorted_layers() if x.kind == "bullet"], strict=True
    ):
        text_only = (
            f"{block.rect.width - block.offset_x - block.indent}x{block.rect.height}"
            f"+{block.offset_x + block.indent}+0"
        )
        histogram = subprocess.run(  # noqa: S603
            [binary, str(layer.png_path), "-crop", text_only, "+repage",
             "-format", "%c", "histogram:info:"],
            capture_output=True, text=True, check=True,
        ).stdout.upper()
        assert accent not in histogram, "accent ink found in the text column"
        assert theme.text.lstrip("#").upper() in histogram, "text is not `theme.text`"


@needs_magick
def test_weight_emphasis_is_visible_but_costs_the_baseline(tmp_path):
    """Why emphasis defaults to off, measured rather than asserted.

    A heavier face *is* visible — 35%+ more ink for the same words at the same size. But
    ImageMagick lays a block out from the face's own ascent, so the heavier face's first
    baseline lands several pixels lower for an identical ``-annotate`` origin: the one
    emphasised line sits off the rhythm of the stack around it. Visible, and visible as a
    rendering fault rather than as hierarchy — which is exactly the complaint.
    """
    base = tx.find_font()
    heavier = tx.heavier_font(base)
    if heavier is None:
        pytest.skip("no heavier face installed")
    binary = tx.require_imagemagick()

    def render(font: str) -> tuple[float, int]:
        text_file = tx.write_text_file(tmp_path, "w.txt", ["Treat urgency as a warning"])
        out = tmp_path / f"{Path(font).stem}.png"
        subprocess.run(  # noqa: S603
            [binary, "-size", "1200x160", "xc:none", "-font", font, "-pointsize", "44",
             "-gravity", "northwest", "-fill", "white", "-annotate", "+10+10",
             tx.imagemagick_text_arg(text_file), f"PNG32:{out}"],
            check=True,
        )
        mass = float(subprocess.run(  # noqa: S603
            [binary, str(out), "-alpha", "extract", "-format", "%[fx:mean*w*h]", "info:"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return mass, ink_bbox(out)[3]

    heavy_mass, heavy_top = render(heavier)
    light_mass, light_top = render(base)
    assert heavy_mass > light_mass * 1.15, f"weight barely reads: {heavy_mass} vs {light_mass}"
    assert heavy_top > light_top, "expected the heavier face to sit lower in its canvas"


def test_available_fonts_are_all_real_files():
    assert all(Path(path).is_file() for path in tx.available_fonts())
