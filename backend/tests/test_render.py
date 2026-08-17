"""Tests for app.render.

The planner and every filtergraph *builder* are pure, so most of this file runs with
no ffmpeg at all. The handful of tests that shell out are marked ``integration`` and
skipped when no usable ffmpeg is on the box.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.core.models import (
    Motion,
    RenderProfile,
    Scene,
    TextPosition,
    Timeline,
    Transition,
    VisualPlan,
    Word,
)
from app.render import captions, text_overlay
from app.render import ffmpeg as ff
from app.render.ffmpeg_backend import FFmpegBackend, frames_for
from app.render.planner import (
    MAX_ZOOM_SPAN,
    MIN_ZOOM_SPAN,
    MOTION_ROTATION,
    RuleBasedPlanner,
)

HD = RenderProfile()
integration = pytest.mark.skipif(not ff.available(), reason="no usable ffmpeg")


def make_timeline(durations: list[float], motions: list[Motion] | None = None) -> Timeline:
    scenes: list[Scene] = []
    cursor = 0.0
    for index, duration in enumerate(durations):
        plan = VisualPlan(motion=motions[index]) if motions else None
        scenes.append(
            Scene(
                id=index + 1,
                narration=f"narration {index}",
                heading=f"Heading {index}",
                image_prompt="prompt",
                start=cursor,
                end=cursor + duration,
                plan=plan,
            )
        )
        cursor += duration
    return Timeline(
        job_id="test", topic="t", title="T", scenes=scenes, voice="v", profile=HD
    )


# ============================================================ planner: purity


def test_planner_is_pure_and_deterministic():
    source = make_timeline([4.0, 3.0, 5.0])
    planner = RuleBasedPlanner()

    first = planner.plan(source)
    second = planner.plan(source)

    assert all(scene.plan is None for scene in source.scenes), "input must not be mutated"
    assert first.model_dump() == second.model_dump(), "planning must be deterministic"
    assert all(scene.plan is not None for scene in first.scenes)


def test_planner_satisfies_protocol():
    from app.core.ports import VideoBackend, VisualPlanner

    assert isinstance(RuleBasedPlanner(), VisualPlanner)
    assert isinstance(FFmpegBackend(text_mode="scrim"), VideoBackend)


# ========================================================== planner: variety


def test_no_two_consecutive_scenes_share_a_motion():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 8))
    motions = [scene.plan.motion for scene in timeline.scenes]
    assert all(a != b for a, b in zip(motions, motions[1:], strict=False)), motions


def test_repeated_llm_motion_is_rotated_away():
    # The LLM picked ZOOM_IN for every scene; the planner must break the repetition.
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 5, [Motion.ZOOM_IN] * 5))
    motions = [scene.plan.motion for scene in timeline.scenes]

    assert motions[0] is Motion.ZOOM_IN, "the first request should be respected"
    assert all(a != b for a, b in zip(motions, motions[1:], strict=False)), motions
    assert all(m in MOTION_ROTATION for m in motions)


def test_llm_motion_is_respected_when_it_does_not_repeat():
    requested = [Motion.PAN_LEFT, Motion.ZOOM_OUT, Motion.STATIC, Motion.PAN_RIGHT]
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 4, requested))
    assert [scene.plan.motion for scene in timeline.scenes] == requested


def test_repeated_static_rotates_to_a_moving_shot():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 4.0], [Motion.STATIC] * 2))
    assert timeline.scenes[0].plan.motion is Motion.STATIC
    assert timeline.scenes[1].plan.motion is not Motion.STATIC


# ============================================================= planner: zooms


def test_zoom_amounts_stay_subtle():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 8))
    for scene in timeline.scenes:
        plan = scene.plan
        span = abs(plan.zoom_to - plan.zoom_from)
        if plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
            assert MIN_ZOOM_SPAN - 1e-9 <= span <= MAX_ZOOM_SPAN + 1e-9, plan
        else:
            assert span == pytest.approx(0.0), "pans hold a fixed zoom"
        assert 1.0 <= plan.zoom_from <= 1.0 + MAX_ZOOM_SPAN
        assert 1.0 <= plan.zoom_to <= 1.0 + MAX_ZOOM_SPAN


def test_pans_hold_enough_zoom_to_have_somewhere_to_travel():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 8))
    for scene in timeline.scenes:
        if scene.plan.motion in (Motion.PAN_LEFT, Motion.PAN_RIGHT):
            assert scene.plan.zoom_from > 1.0


# ======================================================= planner: transitions


def test_first_scene_fades_up_from_black_then_transitions_alternate():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 7))
    transitions = [scene.plan.transition_in for scene in timeline.scenes]
    assert transitions[0] is Transition.FADE
    assert transitions[1:] == [
        Transition.DISSOLVE,
        Transition.SLIDE_LEFT,
        Transition.WIPE_RIGHT,
        Transition.DISSOLVE,
        Transition.SLIDE_LEFT,
        Transition.WIPE_RIGHT,
    ]
    assert all(a != b for a, b in zip(transitions[1:], transitions[2:], strict=False))


def test_transition_duration_clamped_to_40_percent_of_shorter_neighbour():
    # 0.8s scene between two long ones: 0.5s would swallow most of it.
    timeline = RuleBasedPlanner().plan(make_timeline([5.0, 0.8, 5.0]))
    assert timeline.scenes[1].plan.transition_duration == pytest.approx(0.32)
    # The scene *after* the short one is clamped by the short one too.
    assert timeline.scenes[2].plan.transition_duration == pytest.approx(0.32)
    # Long neighbours keep the default.
    roomy = RuleBasedPlanner().plan(make_timeline([5.0, 5.0]))
    assert roomy.scenes[1].plan.transition_duration == 0.5


def test_transition_never_exceeds_40_percent_for_any_scene_length():
    durations = [0.4, 6.0, 1.0, 0.9, 3.0, 0.6]
    timeline = RuleBasedPlanner().plan(make_timeline(durations))
    for index, scene in enumerate(timeline.scenes):
        neighbours = [scene.duration] + ([durations[index - 1]] if index else [])
        assert scene.plan.transition_duration <= 0.4 * min(neighbours) + 1e-9


def test_unusably_short_transition_is_demoted_to_a_cut():
    # 0.2s scene -> 0.08s crossfade, which reads as a dropped frame.
    timeline = RuleBasedPlanner().plan(make_timeline([5.0, 0.2, 5.0]))
    assert timeline.scenes[1].plan.transition_in is Transition.CUT
    assert timeline.scenes[1].plan.transition_duration == 0.0


def test_text_position_alternates_so_slides_do_not_stack_text():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 6))
    positions = [scene.plan.text_position for scene in timeline.scenes]
    assert positions[0] is TextPosition.LOWER_THIRD
    assert all(a != b for a, b in zip(positions, positions[1:], strict=False))
    assert all(scene.plan.scrim_opacity == pytest.approx(0.45) for scene in timeline.scenes)


# ========================================================= duration arithmetic


def test_final_duration_subtracts_every_transition_overlap():
    """xfade consumes overlap, so the total shrinks by one transition per boundary."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 4.0, 4.0, 4.0]))
    overlaps = [scene.plan.transition_duration for scene in timeline.scenes[1:]]

    assert timeline.narration_duration == pytest.approx(16.0)
    assert overlaps == [0.5, 0.5, 0.5]
    assert timeline.final_duration() == pytest.approx(16.0 - 1.5)
    # The naive (wrong) answer differs by 1.5s -- ~0.5s of drift per transition.
    assert abs(timeline.final_duration() - timeline.narration_duration) == pytest.approx(1.5)


def test_final_duration_ignores_cuts():
    timeline = make_timeline([4.0, 4.0, 4.0])
    for scene in timeline.scenes:
        scene.plan = VisualPlan(transition_in=Transition.CUT, transition_duration=0.0)
    assert timeline.final_duration() == pytest.approx(12.0)


def test_video_chain_offsets_match_final_duration():
    """The xfade offsets the backend emits must add up to final_duration()."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 3.0, 5.0, 2.0]))
    backend = FFmpegBackend(text_mode="scrim")
    clip_durations = [scene.duration for scene in timeline.scenes]

    parts, starts, length = backend._video_chain(timeline, clip_durations)

    assert length == pytest.approx(timeline.final_duration())
    # Each scene starts one transition earlier than its narration timestamp implies.
    cumulative = 0.0
    for index, scene in enumerate(timeline.scenes):
        if index:
            cumulative += scene.plan.transition_duration
        assert starts[index] == pytest.approx(scene.start - cumulative)
    graph = ";".join(parts)
    for index in range(1, len(timeline.scenes)):
        assert f"offset={starts[index]:.6f}" in graph


def test_video_chain_handles_a_cut_without_consuming_time():
    timeline = RuleBasedPlanner().plan(make_timeline([5.0, 0.2, 5.0]))
    backend = FFmpegBackend(text_mode="scrim")
    durations = [scene.duration for scene in timeline.scenes]

    parts, starts, length = backend._video_chain(timeline, durations)
    graph = ";".join(parts)

    assert "concat=n=2:v=1:a=0" in graph
    assert starts[1] == pytest.approx(5.0), "a cut starts exactly where the last clip ends"
    assert length == pytest.approx(timeline.final_duration())


def test_frames_for_is_frame_exact():
    assert frames_for(4.68, 30) == 140
    assert frames_for(1 / 30, 30) == 1
    assert frames_for(0.001, 30) == 1


# ============================================================ zoompan builder


def _evaluate(expression: str, **variables: float) -> float:
    return float(eval(expression, {"__builtins__": {}}, variables))  # noqa: S307


def test_zoom_expression_hits_both_endpoints_with_smoothstep_easing():
    plan = VisualPlan(motion=Motion.ZOOM_IN, zoom_from=1.0, zoom_to=1.12)
    z, _, _ = FFmpegBackend._zoompan_expressions(plan, frames=150)

    assert _evaluate(z, on=0) == pytest.approx(1.0)
    assert _evaluate(z, on=149) == pytest.approx(1.12)
    # Smoothstep: derivative is ~0 at both ends, so the move eases in and out.
    early = _evaluate(z, on=1) - _evaluate(z, on=0)
    middle = _evaluate(z, on=75) - _evaluate(z, on=74)
    assert early < middle / 5


def test_linear_easing_is_actually_linear():
    plan = VisualPlan(motion=Motion.ZOOM_IN, zoom_from=1.0, zoom_to=1.10, easing="linear")
    z, _, _ = FFmpegBackend._zoompan_expressions(plan, frames=101)
    assert _evaluate(z, on=50) == pytest.approx(1.05)


def test_zoom_out_expression_descends():
    plan = VisualPlan(motion=Motion.ZOOM_OUT, zoom_from=1.15, zoom_to=1.0)
    z, _, _ = FFmpegBackend._zoompan_expressions(plan, frames=100)
    assert _evaluate(z, on=0) == pytest.approx(1.15)
    assert _evaluate(z, on=99) == pytest.approx(1.0)


def test_pan_expressions_sweep_the_full_available_width_in_both_directions():
    frames = 100
    width = 1920.0
    right, left = (
        FFmpegBackend._zoompan_expressions(
            VisualPlan(motion=motion, zoom_from=1.10, zoom_to=1.10), frames
        )
        for motion in (Motion.PAN_RIGHT, Motion.PAN_LEFT)
    )
    travel = width - width / 1.10

    assert _evaluate(right[1], on=0, iw=width, zoom=1.10) == pytest.approx(0.0)
    assert _evaluate(right[1], on=99, iw=width, zoom=1.10) == pytest.approx(travel)
    assert _evaluate(left[1], on=0, iw=width, zoom=1.10) == pytest.approx(travel)
    assert _evaluate(left[1], on=99, iw=width, zoom=1.10) == pytest.approx(0.0)


def test_pan_forces_headroom_when_the_plan_asks_for_zoom_1():
    plan = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.0, zoom_to=1.0)
    z, x, _ = FFmpegBackend._zoompan_expressions(plan, frames=60)
    assert _evaluate(z, on=0) > 1.0, "at zoom 1.0 a pan has nowhere to travel"


def test_single_frame_clip_does_not_divide_by_zero():
    plan = VisualPlan(motion=Motion.ZOOM_IN)
    z, x, y = FFmpegBackend._zoompan_expressions(plan, frames=1)
    assert _evaluate(z, on=0) == pytest.approx(plan.zoom_from)


def test_scene_graph_upscales_before_zoompan_to_kill_the_stepping():
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    backend = FFmpegBackend(text_mode="scrim")
    graph = backend._scene_graph(
        src_size=(1920, 1080),
        plan=VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.1, zoom_to=1.1),
        profile=profile,
        frames=90,
        layout=text_overlay.layout_heading("Hi", VisualPlan(), profile),
        heading="Hi",
        has_text_input=False,
    )
    assert "scale=7680:4320" in graph, "source must be pre-upscaled by upscale_factor"
    assert "s=1920x1080" in graph, "zoompan must emit the final size"
    assert graph.index("scale=7680:4320") < graph.index("zoompan")


def test_static_motion_skips_zoompan_entirely():
    profile = RenderProfile(width=640, height=360, upscale_factor=4)
    graph = FFmpegBackend(text_mode="scrim")._scene_graph(
        src_size=(640, 360),
        plan=VisualPlan(motion=Motion.STATIC),
        profile=profile,
        frames=30,
        layout=text_overlay.layout_heading("Hi", VisualPlan(), profile),
        heading="Hi",
        has_text_input=False,
    )
    assert "zoompan" not in graph
    assert "scale=640:360" in graph


# ================================================================ image fitting


def test_matching_aspect_uses_cover_and_crop_never_a_stretch():
    parts = FFmpegBackend._fit_chain((1920, 1080), (1920, 1080), (3840, 2160))
    chain = ";".join(parts)
    assert "force_original_aspect_ratio=increase" in chain
    assert "crop=3840:2160" in chain
    assert "gblur" not in chain


def test_portrait_source_in_landscape_frame_uses_blurred_fill():
    parts = FFmpegBackend._fit_chain((1080, 1920), (1920, 1080), (3840, 2160))
    chain = ";".join(parts)
    assert "gblur" in chain, "a 9:16 image centre-cropped to 16:9 loses most of the frame"
    assert "force_original_aspect_ratio=decrease" in chain, "foreground must be contained"
    assert "overlay=(W-w)/2:(H-h)/2" in chain
    # No filter may set both dimensions without an aspect-preserving flag.
    assert "scale=3840:2160:force_original_aspect_ratio=decrease" in chain


def test_mild_aspect_mismatch_still_crops():
    chain = ";".join(FFmpegBackend._fit_chain((1600, 1200), (1920, 1080), (1920, 1080)))
    assert "gblur" not in chain


# ============================================================== text: escaping


HOSTILE_HEADINGS = [
    "It's Simple: 40% Faster, Smoother",
    "Ratio 1:2, 3:4 -- 100% [done]",
    "back\\slash and 'quotes'",
    "semi;colon",
    "key=value pairs",
]


@pytest.mark.parametrize("raw", HOSTILE_HEADINGS)
def test_escaping_only_ever_adds_prefixes_and_is_reversible(raw):
    escaped = text_overlay.escape_drawtext(raw)
    assert escaped.replace("\\\\\\", "") == raw
    for char in set(raw) & set(text_overlay.FILTERGRAPH_SPECIALS):
        assert "\\\\\\" + char in escaped, f"{char!r} was not escaped in {escaped!r}"


def test_apostrophe_and_colon_escaping_is_exact():
    assert text_overlay.escape_drawtext("It's") == "It\\\\\\'s"
    assert text_overlay.escape_drawtext("a:b") == "a\\\\\\:b"
    assert text_overlay.escape_drawtext("a,b") == "a\\\\\\,b"


def test_percent_is_left_alone_and_handled_by_expansion_none():
    # % is not special to the filtergraph parser, only to drawtext's own expansion.
    assert text_overlay.escape_drawtext("40% off") == "40% off"


def test_drawtext_filter_disables_percent_expansion():
    layout = text_overlay.layout_heading("50% Off: Today's Deal", VisualPlan(), HD)
    chain = text_overlay.drawtext_filters(layout, font="/tmp/fake.ttf")
    assert "expansion=none" in chain, "otherwise drawtext eats % as a strftime directive"
    assert "50%" in chain
    assert chain.startswith("drawbox=")
    assert chain.count("drawtext=") == len(layout.lines)
    # A raw colon or comma here would split the filter chain, not just look wrong.
    for line in layout.lines:
        assert text_overlay.escape_drawtext(line) in chain


@integration
@pytest.mark.parametrize("raw", HOSTILE_HEADINGS)
def test_escaping_round_trips_through_ffmpeg_own_parser(raw):
    """Push the escaped string through a real filtergraph and read it back.

    ``drawtext`` needs libfreetype and is missing from some builds (including the
    current Homebrew bottle), so this uses ``metadata``, whose ``value`` option goes
    through exactly the same option-value parsing, and which can print what it got.
    """
    escaped = text_overlay.escape_drawtext(raw)
    stderr = ff.run(
        [
            ff.ffmpeg_bin(),
            "-hide_banner",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=0.04",
            "-vf",
            f"metadata=mode=add:key=h:value={escaped},metadata=mode=print:key=h",
            "-f",
            "null",
            "-",
        ]
    )
    printed = [line.split("] h=", 1)[1] for line in stderr.splitlines() if "] h=" in line]
    assert printed, f"filtergraph did not even parse for {raw!r} -> {escaped!r}"
    assert printed[0] == raw


# =============================================================== text: wrapping


def test_short_heading_stays_on_one_line():
    assert text_overlay.wrap_text("Short Heading", 40) == ["Short Heading"]


def test_long_heading_wraps_to_two_balanced_lines():
    text = "This Extremely Long Heading Demonstrates Automatic Two Line Wrapping"
    lines = text_overlay.wrap_text(text, 40)
    assert len(lines) == 2
    assert all(len(line) <= 40 for line in lines)
    assert " ".join(lines) == text
    shorter, longer = sorted(len(line) for line in lines)
    assert shorter / longer >= 0.4, "a lopsided split looks accidental"


def test_wrap_reports_failure_rather_than_overflowing():
    assert text_overlay.wrap_text("a" * 80, 20) is None


def test_layout_shrinks_the_font_for_a_very_long_heading():
    long_heading = (
        "This Extremely Long Heading Demonstrates Automatic Two Line Wrapping "
        "Without Ever Overflowing The Frame Edges"
    )
    short = text_overlay.layout_heading("Short", VisualPlan(), HD)
    long = text_overlay.layout_heading(long_heading, VisualPlan(), HD)

    assert len(short.lines) == 1
    assert len(long.lines) == 2
    assert long.font_size < short.font_size
    assert " ".join(long.lines) == long_heading


def test_layout_never_exceeds_two_lines_even_for_absurd_input():
    layout = text_overlay.layout_heading("word " * 200, VisualPlan(), HD)
    assert len(layout.lines) <= 2
    assert layout.lines[-1].endswith(text_overlay.ELLIPSIS)


# ================================================================= text: layout


@pytest.mark.parametrize(
    "position", [TextPosition.LOWER_THIRD, TextPosition.UPPER_THIRD, TextPosition.CENTER]
)
def test_text_stays_inside_the_safe_area(position):
    plan = VisualPlan(text_position=position)
    layout = text_overlay.layout_heading("A Reasonably Long Heading Here", plan, HD)
    margin = HD.height * (1.0 - text_overlay.SAFE_AREA)

    assert layout.block_top >= 0
    assert layout.block_bottom <= HD.height
    if position is not TextPosition.CENTER:
        assert layout.block_top >= margin - 1
        assert layout.block_bottom <= HD.height - margin + 1
    estimated_width = max(len(line) for line in layout.lines) * layout.font_size * 0.52
    assert estimated_width <= HD.width * text_overlay.SAFE_AREA


@pytest.mark.parametrize(
    "position", [TextPosition.LOWER_THIRD, TextPosition.UPPER_THIRD, TextPosition.CENTER]
)
def test_a_solid_scrim_band_sits_behind_every_line_of_text(position):
    """The whole point of the scrim: white text must never sit on a bare image."""
    layout = text_overlay.layout_heading("Heading Over A Bright Sky", VisualPlan(
        text_position=position), HD)
    solid = [band for band in layout.scrim_bands() if band[2] == "solid"]
    assert solid, "no solid scrim band"
    assert any(
        y <= layout.block_top and y + height >= layout.block_bottom for y, height, _ in solid
    ), f"solid band {solid} does not cover text {layout.block_top}-{layout.block_bottom}"
    for y, height, _ in layout.scrim_bands():
        assert y >= 0 and y + height <= layout.height


def test_scrim_filter_uses_drawbox_which_exists_in_every_ffmpeg_build():
    layout = text_overlay.layout_heading("Hi", VisualPlan(), HD)
    chain = text_overlay.scrim_filter(layout)
    assert chain.startswith("drawbox=")
    assert "color=black@0.45" in chain


# =================================================================== text: font


def test_the_chosen_font_file_actually_exists():
    font = text_overlay.find_font()
    assert Path(font).is_file(), f"hardcoded font path is a lie: {font}"


def test_font_fallback_skips_missing_candidates():
    chosen = text_overlay.find_font(
        ("/nope/missing-one.ttf", "/nope/missing-two.ttf", *text_overlay.FONT_CANDIDATES)
    )
    assert Path(chosen).is_file()


def test_no_font_at_all_is_an_explicit_error():
    with pytest.raises(text_overlay.FontNotFoundError):
        text_overlay.find_font(("/nope/a.ttf", "/nope/b.ttf"))


# ====================================================================== captions


def test_ass_timestamps_are_centiseconds():
    assert captions.format_ass_time(0.0) == "0:00:00.00"
    assert captions.format_ass_time(1.5) == "0:00:01.50"
    assert captions.format_ass_time(61.234) == "0:01:01.23"
    assert captions.format_ass_time(3661.0) == "1:01:01.00"


def test_karaoke_durations_are_centiseconds_and_cover_the_gaps():
    words = [
        Word(word="hello", start=0.0, end=0.4),
        Word(word="world", start=0.5, end=1.0),
    ]
    line = captions.karaoke_line(words)
    # First word dwells until the next one starts (0.5s = 50cs), not just its own end.
    assert line == r"{\k50}hello {\k50}world"


def test_ass_document_is_well_formed_and_carries_every_word():
    timeline = make_timeline([2.0])
    timeline.scenes[0].words = [
        Word(word="one", start=0.0, end=0.5),
        Word(word="two", start=0.5, end=1.0),
    ]
    document = captions.build_ass(timeline)
    assert "[Script Info]" in document
    assert f"PlayResX: {HD.width}" in document
    assert document.count("Dialogue:") == 1
    assert r"{\k" in document


def test_caption_shift_matches_the_xfade_overlap():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 4.0, 4.0]))
    shifts = captions.shift_for_transitions(timeline)
    assert shifts[1] == pytest.approx(0.0)
    assert shifts[2] == pytest.approx(-0.5)
    assert shifts[3] == pytest.approx(-1.0)


def test_ass_braces_are_escaped():
    assert captions.escape_ass_text("{not a tag}") == r"\{not a tag\}"


# ================================================================== ffmpeg wrap


def test_ffmpeg_error_includes_the_stderr_tail_and_the_command():
    stderr = "\n".join(f"line {n}" for n in range(200))
    error = ff.FFmpegError(["ffmpeg", "-i", "a b.mp4"], 234, stderr)
    message = str(error)
    assert "exited 234" in message
    assert "'a b.mp4'" in message, "the command must be copy-pasteable"
    assert "line 199" in message, "the useful part of ffmpeg stderr is at the end"
    assert "line 160" in message, f"expected the last {ff.STDERR_TAIL_LINES} lines"
    assert "line 159" not in message
    assert "line 100" not in message


def test_capability_probe_agrees_with_this_build():
    filters = ff._filters(ff.ffmpeg_bin())
    assert {"zoompan", "xfade", "overlay", "scale", "drawbox"} <= filters
    assert ff.has_encoder("libx264")


def test_text_mode_falls_back_when_drawtext_is_missing():
    mode = text_overlay.resolve_text_mode("auto")
    assert mode in ("drawtext", "png", "scrim")
    if ff.has_filter("drawtext"):
        assert mode == "drawtext"
    elif text_overlay.imagemagick_bin():
        assert mode == "png", "must not silently drop the heading"


# =================================================================== integration

FPS = 12
TINY = RenderProfile(width=320, height=180, fps=FPS, upscale_factor=2, crf=28)


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> dict[str, Path]:
    """Synthesise images and audio with ffmpeg so the tests need no fixtures on disk."""
    directory = tmp_path_factory.mktemp("render_assets")
    landscape = directory / "landscape.png"
    portrait = directory / "portrait.png"
    ff.ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=1", "-frames:v", "1", landscape])
    ff.ffmpeg(["-f", "lavfi", "-i", "smptebars=s=360x640:r=1", "-frames:v", "1", portrait])
    voices = []
    for index, (freq, seconds) in enumerate(((330, 1.0), (440, 0.6))):
        wav = directory / f"voice_{index}.wav"
        ff.ffmpeg(
            ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-c:a",
             "pcm_s16le", wav]
        )
        voices.append(wav)
    music = directory / "music.wav"
    ff.ffmpeg(["-f", "lavfi", "-i", "sine=frequency=110:duration=0.8", "-c:a", "pcm_s16le", music])
    return {
        "landscape": landscape,
        "portrait": portrait,
        "voices": voices,
        "music": music,
    }


@integration
def test_render_scene_is_frame_exact(assets, tmp_path):
    backend = FFmpegBackend()
    plan = VisualPlan(motion=Motion.ZOOM_IN, zoom_from=1.0, zoom_to=1.12)
    clip = backend.render_scene(
        assets["landscape"], plan, "It's Exact: 100% Frame Accurate", 1.0, tmp_path / "s.mp4", TINY
    )

    summary = ff.probe_summary(clip)
    assert summary["duration"] == pytest.approx(1.0, abs=1.5 / FPS)
    assert summary["nb_frames"] == FPS
    assert (summary["width"], summary["height"]) == (TINY.width, TINY.height)
    assert summary["audio_codec"] is None, "scene clips are video-only by design"


@integration
def test_render_scene_handles_a_portrait_source_without_distortion(assets, tmp_path):
    clip = FFmpegBackend().render_scene(
        assets["portrait"],
        VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.1, zoom_to=1.1),
        "Portrait Source",
        1.0,
        tmp_path / "p.mp4",
        TINY,
    )
    summary = ff.probe_summary(clip)
    assert (summary["width"], summary["height"]) == (TINY.width, TINY.height)


@integration
def test_assembled_duration_matches_final_duration(assets, tmp_path):
    """The regression that matters: xfade overlap must not desync the narration."""
    timeline = Timeline(
        job_id="itest",
        topic="t",
        title="T",
        voice="v",
        profile=TINY,
        music_path=str(assets["music"]),
        scenes=[
            Scene(
                id=1,
                narration="one",
                heading="First Slide: It's Fine",
                image_prompt="p",
                image_path=str(assets["landscape"]),
                audio_path=str(assets["voices"][0]),
                start=0.0,
                end=1.0,
            ),
            Scene(
                id=2,
                narration="two",
                heading="A Considerably Longer Second Heading That Has To Wrap Somewhere",
                image_prompt="p",
                image_path=str(assets["portrait"]),
                audio_path=str(assets["voices"][1]),
                start=1.0,
                end=2.0,
            ),
        ],
    )
    timeline = RuleBasedPlanner().plan(timeline)
    assert timeline.scenes[1].plan.transition_duration == pytest.approx(0.4)

    backend = FFmpegBackend()
    timeline = backend.render_all(timeline, tmp_path)
    out = backend.assemble(timeline, tmp_path / "final.mp4")

    expected = timeline.final_duration()
    summary = ff.probe_summary(out)
    assert expected == pytest.approx(1.6), "2.0s of scenes minus one 0.4s crossfade"
    assert summary["duration"] == pytest.approx(expected, abs=max(0.1, 4 / FPS))
    assert summary["audio_channels"] == 2
    assert summary["audio_codec"] == "aac"
    assert summary["video_codec"] == "h264"
    # And the naive answer would have been 2.0s -- 0.4s of narration drift.
    assert abs(summary["duration"] - timeline.narration_duration) > 0.3


@integration
def test_assemble_refuses_to_lie_about_a_duration_mismatch(assets, tmp_path):
    from app.render.ffmpeg_backend import DurationMismatchError

    backend = FFmpegBackend()
    scene = Scene(
        id=1,
        narration="x",
        heading="Solo",
        image_prompt="p",
        image_path=str(assets["landscape"]),
        start=0.0,
        end=1.0,
        plan=VisualPlan(motion=Motion.STATIC, transition_in=Transition.CUT,
                        transition_duration=0.0),
    )
    timeline = Timeline(
        job_id="j", topic="t", title="T", voice="v", profile=TINY, scenes=[scene]
    )
    timeline = backend.render_all(timeline, tmp_path)
    # Lie about the scene length after rendering: the check must catch it.
    timeline.scenes[0].end = 9.0
    with pytest.raises(DurationMismatchError, match="final_duration"):
        backend.assemble(timeline, tmp_path / "bad.mp4")


@integration
def test_upscaling_removes_the_zoompan_stepping(assets, tmp_path):
    """A slow pan without pre-upscaling repeats frames outright; with it, none do.

    zoompan truncates x/y to integers, so at output resolution a slow pan sits on the
    same pixel offset for several frames. Duplicate frames are that artefact, counted.
    """
    # Deliberately slow: 640/1.05 leaves ~30px of travel spread over 120 frames, i.e.
    # a quarter of a pixel per frame -- exactly where integer truncation shows up.
    plan = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.05, zoom_to=1.05, easing="linear")
    pan_fps, pan_seconds = 30, 4.0
    duplicates = {}
    for upscale in (1, 4):
        profile = RenderProfile(width=640, height=360, fps=pan_fps, upscale_factor=upscale, crf=20)
        clip = FFmpegBackend(text_mode="scrim").render_scene(
            assets["landscape"], plan, "", pan_seconds, tmp_path / f"pan{upscale}.mp4", profile
        )
        stderr = ff.run(
            [ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", clip, "-vf",
             "mpdecimate=hi=64*4:lo=64*2:frac=0.1", "-f", "null", "-"]
        )
        kept = max(
            int(line.split("frame=")[1].split()[0])
            for line in stderr.splitlines()
            if "frame=" in line and "fps=" in line
        )
        duplicates[upscale] = int(round(pan_fps * pan_seconds)) - kept

    assert duplicates[1] > 0, f"expected visible stepping without pre-upscaling: {duplicates}"
    assert duplicates[4] < duplicates[1] / 2, f"upscaling did not help: {duplicates}"


@integration
def test_missing_clip_is_reported_clearly(tmp_path):
    from app.render.ffmpeg_backend import RenderError

    timeline = RuleBasedPlanner().plan(make_timeline([1.0]))
    with pytest.raises(RenderError, match="no rendered clip"):
        FFmpegBackend().assemble(timeline, tmp_path / "x.mp4")


@integration
def test_ffmpeg_failure_surfaces_the_stderr_tail():
    with pytest.raises(ff.FFmpegError) as excinfo:
        ff.ffmpeg(["-i", "/definitely/not/a/file.png", "-f", "null", "-"])
    assert "No such file" in str(excinfo.value) or "not a file" in str(excinfo.value).lower()


def test_expected_duration_helper_matches_the_model():
    from app.render.ffmpeg_backend import expected_assembled_duration

    timeline = RuleBasedPlanner().plan(make_timeline([3.3, 1.7, 2.9, 4.1]))
    assert expected_assembled_duration(timeline) == pytest.approx(timeline.final_duration())
    assert not math.isnan(expected_assembled_duration(timeline))
