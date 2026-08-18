"""Tests for app.render.

The planner and every filtergraph *builder* are pure, so most of this file runs with
no ffmpeg at all. The handful of tests that shell out are marked ``integration`` and
skipped when no usable ffmpeg is on the box.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import pytest

from app.core.models import (
    BulletPoint,
    Motion,
    RenderProfile,
    Scene,
    SceneRole,
    SlideLayout,
    TextAnimation,
    TextPosition,
    Theme,
    Timeline,
    Transition,
    VisualPlan,
    Word,
)
from app.render import captions, text_overlay
from app.render import ffmpeg as ff
from app.render.contracts import SceneText, TextLayer
from app.render.ffmpeg_backend import (
    AUTO_LOGO,
    CANVAS_PIXEL_BUDGET,
    CLIP_SEAM_CROSSFADE,
    EASE_VELOCITY_FLOOR,
    STROBE_STEP_PIXELS,
    FFmpegBackend,
    clip_loop_count,
    clip_loop_span,
    clip_seam,
    detail_upscale,
    eased_progress,
    fallback_region,
    ffmpeg_colour,
    frames_for,
    layout_region,
    motion_canvas,
    motion_travel,
    peak_step,
    plan_zoom_ceiling,
    resolve_logo_source,
    slowest_step,
)
from app.render.planner import (
    BULLET_ANIMATION_ROTATION,
    HEADING_ANIMATION_ROTATION,
    MAX_ZOOM_SPAN,
    MIN_ANIM_DURATION,
    MIN_ZOOM_SPAN,
    MOTION_ROTATION,
    TRANSITION_ROTATION,
    RuleBasedPlanner,
)

HD = RenderProfile()
integration = pytest.mark.skipif(not ff.available(), reason="no usable ffmpeg")


def make_timeline(
    durations: list[float],
    motions: list[Motion] | None = None,
    bullets: list[list[BulletPoint]] | None = None,
) -> Timeline:
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
                bullets=bullets[index] if bullets else [],
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


# ================================================= planner: one video, one style
#
# These tests replace an older set that asserted the *opposite* — that no two consecutive
# scenes shared a motion, a layout, a transition or an entrance. That variety is what the
# viewer described as "all over the place"; `docs/DIRECTION.md` §0 is the argument, and the
# rule is now "repetition is the design". The old rotations survive behind `hold_*=False`,
# and the tests below pin both halves.


def test_every_scene_shares_one_camera_move():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 8))
    motions = {scene.plan.motion for scene in timeline.scenes}
    spans = {round(abs(s.plan.zoom_to - s.plan.zoom_from), 4) for s in timeline.scenes}

    assert motions == {Motion.ZOOM_IN}, motions
    assert spans == {0.06}, f"one zoom amount for the video: {spans}"
    assert {s.plan.easing for s in timeline.scenes} == {"linear"}


def test_the_scripts_preferred_move_is_honoured_once_for_the_whole_video():
    """The LLM still gets a vote — on the video, which is the level a house style is set at."""
    requested = [Motion.PAN_RIGHT] * 4 + [Motion.ZOOM_OUT]
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 5, requested))
    assert {s.plan.motion for s in timeline.scenes} == {Motion.PAN_RIGHT}


def test_a_static_request_never_becomes_a_video_of_dead_stills():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 3, [Motion.STATIC] * 3))
    assert {s.plan.motion for s in timeline.scenes} == {Motion.ZOOM_IN}


def test_the_old_per_scene_rotation_is_still_available():
    timeline = RuleBasedPlanner(hold_motion=False).plan(make_timeline([4.0] * 8))
    motions = [scene.plan.motion for scene in timeline.scenes]
    assert all(a != b for a, b in zip(motions, motions[1:], strict=False)), motions
    assert all(m in MOTION_ROTATION for m in motions)


def test_llm_motion_is_respected_when_it_does_not_repeat():
    requested = [Motion.PAN_LEFT, Motion.ZOOM_OUT, Motion.STATIC, Motion.PAN_RIGHT]
    timeline = RuleBasedPlanner(hold_motion=False).plan(make_timeline([4.0] * 4, requested))
    assert [scene.plan.motion for scene in timeline.scenes] == requested


def test_repeated_static_rotates_to_a_moving_shot():
    timeline = RuleBasedPlanner(hold_motion=False).plan(
        make_timeline([4.0, 4.0], [Motion.STATIC] * 2)
    )
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


def test_every_boundary_including_the_first_is_the_same_crossfade():
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 7))
    transitions = [scene.plan.transition_in for scene in timeline.scenes]
    assert transitions == [Transition.FADE] * 7
    # ...and the old rotation is still reachable, for a deck that is not burned-in text.
    rotated = RuleBasedPlanner(hold_transition=False).plan(make_timeline([4.0] * 7))
    assert all(s.plan.transition_in in TRANSITION_ROTATION for s in rotated.scenes)


def test_only_a_crossfade_is_ever_used_because_wipes_shred_burned_in_text():
    """``slideleft`` and ``wiperight`` are gone, and this is not a taste question.

    The text is burned into the scene clip, so a wipe cuts through both stacks at once —
    a measured frame of the rejected render has two headings and eight bullet fragments on
    screen ("ize / re Tactics", "e Artificial Panic"). ``dissolve`` stays out too: on this
    ffmpeg build it is a dither, not a blend.
    """
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 12))
    chosen = {scene.plan.transition_in for scene in timeline.scenes}
    assert chosen == {Transition.FADE}, chosen
    assert Transition.SLIDE_LEFT not in chosen
    assert Transition.WIPE_RIGHT not in chosen
    # The enum members all survive so old plans deserialise.
    for name in ("DISSOLVE", "SLIDE_LEFT", "WIPE_RIGHT"):
        assert hasattr(Transition, name)


def test_transition_duration_clamped_to_40_percent_of_shorter_neighbour():
    # 0.8s scene between two long ones: a full transition would swallow most of it.
    timeline = RuleBasedPlanner().plan(make_timeline([5.0, 0.8, 5.0]))
    assert timeline.scenes[1].plan.transition_duration == pytest.approx(0.32)
    # The scene *after* the short one is clamped by the short one too.
    assert timeline.scenes[2].plan.transition_duration == pytest.approx(0.32)
    # Long neighbours keep the default: 0.35s, since only the photograph cross-dissolves
    # now that both frames share one grid.
    roomy = RuleBasedPlanner().plan(make_timeline([5.0, 5.0]))
    assert roomy.scenes[1].plan.transition_duration == 0.35


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


def test_text_position_follows_the_slide_layout_by_default():
    """The training layout puts text in a panel beside the image, not over it."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 6))
    by_layout = {
        scene.plan.layout: scene.plan.text_position for scene in timeline.scenes
    }
    assert by_layout[SlideLayout.TITLE_CARD] is TextPosition.CENTER
    assert by_layout[SlideLayout.HERO_RIGHT] is TextPosition.LEFT_PANEL
    assert set(by_layout) == {SlideLayout.TITLE_CARD, SlideLayout.HERO_RIGHT}
    assert all(scene.plan.scrim_opacity == pytest.approx(0.45) for scene in timeline.scenes)


def test_alternating_text_position_is_still_available_for_photo_decks():
    timeline = RuleBasedPlanner(alternate_text_position=True).plan(make_timeline([4.0] * 6))
    positions = [scene.plan.text_position for scene in timeline.scenes]
    assert positions[0] is TextPosition.LOWER_THIRD
    assert all(a != b for a, b in zip(positions, positions[1:], strict=False))


def test_an_explicit_text_position_pins_every_scene():
    timeline = RuleBasedPlanner(text_position=TextPosition.LOWER_THIRD).plan(
        make_timeline([4.0] * 4)
    )
    assert all(s.plan.text_position is TextPosition.LOWER_THIRD for s in timeline.scenes)


# ============================================================= planner: layouts


def test_the_opener_is_a_title_card_and_the_body_never_changes_layout():
    """Two layouts in the whole video. The body holding still IS the design.

    Alternating hero sides moved the text block ~940px sideways between consecutive scenes,
    so the viewer re-hunted for the text on every slide.
    """
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 8))
    layouts = [scene.plan.layout for scene in timeline.scenes]
    assert layouts[0] is SlideLayout.TITLE_CARD, "open on type, not a photo"
    assert set(layouts[1:]) == {SlideLayout.HERO_RIGHT}, layouts


def test_the_hero_side_can_be_mirrored_for_the_whole_video_but_never_per_scene():
    timeline = RuleBasedPlanner(hero_side="left").plan(make_timeline([4.0] * 6))
    layouts = [scene.plan.layout for scene in timeline.scenes]
    assert set(layouts[1:]) == {SlideLayout.HERO_LEFT}, layouts


def test_the_retired_layouts_are_never_emitted():
    """``image_band`` and ``full_bleed`` keep their enum members so old timelines
    deserialise, but no video gets one: ``full_bleed`` measured 10.45:1 contrast against
    18.8:1 on the solid slides, i.e. the only layout that varied was the only one that put
    the heading on unknown pixels."""
    for count in range(1, 15):
        layouts = [
            scene.plan.layout
            for scene in RuleBasedPlanner().plan(make_timeline([4.0] * count)).scenes
        ]
        assert SlideLayout.FULL_BLEED not in layouts, layouts
        assert SlideLayout.IMAGE_BAND not in layouts, layouts
        assert len(set(layouts)) <= 2, layouts


def test_title_card_opener_can_be_turned_off():
    timeline = RuleBasedPlanner(title_card_opener=False).plan(make_timeline([4.0] * 3))
    assert timeline.scenes[0].plan.layout is not SlideLayout.TITLE_CARD


# ============================================================== planner: roles


def test_a_deck_gets_a_shape_even_when_nothing_upstream_assigned_roles():
    """Nothing fills ``Scene.role`` in yet, so every scene arrives as CONTENT. Without an
    inference a deck would be a queue of identical slabs, which is the other half of what
    "all over the place" meant. DIRECTION §1.1 fixes the sequence per scene count."""
    def roles_for(count: int) -> list[SceneRole]:
        planned = RuleBasedPlanner().plan(make_timeline([15.0] * count))
        return [scene.role for scene in planned.scenes]

    shapes = {count: roles_for(count) for count in (3, 4, 6, 7, 9)}
    assert shapes[4] == [SceneRole.TITLE, SceneRole.CONTENT, SceneRole.CONTENT, SceneRole.CLOSING]
    assert shapes[6] == [SceneRole.TITLE] + [SceneRole.CONTENT] * 4 + [SceneRole.CLOSING]
    # A recap only exists from seven scenes up: below that the closing already restates the
    # point, and a summary costs a teaching slide.
    assert SceneRole.SUMMARY not in shapes[6]
    assert shapes[7][-2:] == [SceneRole.SUMMARY, SceneRole.CLOSING]
    assert shapes[9].count(SceneRole.SUMMARY) == 1
    for roles in shapes.values():
        assert roles.count(SceneRole.TITLE) <= 1
        assert roles.count(SceneRole.CLOSING) <= 1


def test_a_timeline_that_already_has_roles_is_taken_at_its_word():
    timeline = make_timeline([15.0] * 4)
    timeline.scenes[2].role = SceneRole.CLOSING
    planned = RuleBasedPlanner().plan(timeline)
    assert [s.role for s in planned.scenes] == [
        SceneRole.CONTENT,
        SceneRole.CONTENT,
        SceneRole.CLOSING,
        SceneRole.CONTENT,
    ]
    assert planned.scenes[2].plan.layout is SlideLayout.HERO_RIGHT


def test_the_bullet_budget_is_applied_to_the_planned_copy():
    bullets = [[BulletPoint(text=f"point {i}", appear_at=1.2 + 1.6 * i) for i in range(5)]] * 4
    timeline = RuleBasedPlanner().plan(make_timeline([15.0] * 4, bullets=bullets))
    assert [len(s.bullets) for s in timeline.scenes] == [0, 4, 4, 2]
    assert all(len(s.bullets) == s.role.bullet_budget for s in timeline.scenes)
    # ...and the input is untouched, as ever.
    assert all(scene.role is SceneRole.CONTENT for scene in make_timeline([15.0] * 4).scenes)


# ========================================================== planner: animation


def test_one_entrance_for_every_element_in_the_video():
    """A fade-and-rise, everywhere. Four rotating entrances across nine slides is four
    house styles; the difference between elements is carried by *duration* and *travel*,
    which live on the layer (see ``text_overlay.BULLET_ANIM_DURATION``), not on the plan."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 9))
    assert {s.plan.heading_animation for s in timeline.scenes} == {TextAnimation.SLIDE_UP}
    assert {s.plan.bullet_animation for s in timeline.scenes} == {TextAnimation.SLIDE_UP}
    assert {s.plan.anim_duration for s in timeline.scenes} == {0.4}


def test_bullets_in_one_scene_always_share_an_entrance():
    """A stack arriving from four directions reads as chaos. There is exactly one
    ``bullet_animation`` per plan by construction; sequence is timing, not direction."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 9))
    for scene in timeline.scenes:
        assert isinstance(scene.plan.bullet_animation, TextAnimation)
    assert {s.plan.bullet_min_gap for s in timeline.scenes} == {1.6}


def test_the_old_animation_rotation_is_still_available():
    timeline = RuleBasedPlanner(hold_animation=False).plan(make_timeline([4.0] * 9))
    headings = [scene.plan.heading_animation for scene in timeline.scenes]
    assert all(a != b for a, b in zip(headings, headings[1:], strict=False)), headings
    assert all(a in HEADING_ANIMATION_ROTATION for a in headings)
    assert all(s.plan.bullet_animation in BULLET_ANIMATION_ROTATION for s in timeline.scenes)
    for scene in timeline.scenes:
        assert scene.plan.heading_animation is not scene.plan.bullet_animation, scene.plan


def test_typewriter_is_never_chosen_by_default():
    """The renderer can only approximate it as a wipe, so it stays opt-in."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0] * 12))
    chosen = {s.plan.heading_animation for s in timeline.scenes} | {
        s.plan.bullet_animation for s in timeline.scenes
    }
    assert TextAnimation.TYPEWRITER not in chosen


def test_anim_duration_never_exceeds_the_gap_to_the_next_bullet():
    """Overlapping entrances read as a smear rather than a sequence."""
    bullets = [
        [
            BulletPoint(text="a", appear_at=0.5),
            BulletPoint(text="b", appear_at=0.8),  # a 0.3s gap
            BulletPoint(text="c", appear_at=2.0),
        ]
    ]
    timeline = RuleBasedPlanner().plan(make_timeline([8.0], bullets=bullets))
    anim = timeline.scenes[0].plan.anim_duration
    assert anim <= 0.3 + 1e-9, anim
    assert anim < 0.45, "the default must have been clamped down"
    assert anim >= MIN_ANIM_DURATION


def test_anim_duration_finishes_well_before_the_scene_ends():
    bullets = [[BulletPoint(text="last", appear_at=1.7)]]
    timeline = RuleBasedPlanner().plan(make_timeline([2.0], bullets=bullets))
    plan = timeline.scenes[0].plan
    # 1.7s reveal in a 2.0s scene leaves no room, so the floor applies -- and the
    # floor is small enough that the entrance is over before the crossfade starts.
    assert plan.anim_duration == pytest.approx(MIN_ANIM_DURATION)
    assert 1.7 + plan.anim_duration < 2.0


def test_anim_duration_is_clamped_to_a_fraction_of_a_short_scene():
    timeline = RuleBasedPlanner().plan(make_timeline([1.2]))
    anim = timeline.scenes[0].plan.anim_duration
    assert anim <= 1.2 * 0.18 + 1e-9, anim
    assert anim >= MIN_ANIM_DURATION


def test_anim_duration_keeps_the_default_when_there_is_room():
    timeline = RuleBasedPlanner().plan(
        make_timeline([8.0], bullets=[[BulletPoint(text="x", appear_at=2.0)]])
    )
    assert timeline.scenes[0].plan.anim_duration == pytest.approx(0.40)


def test_anim_duration_never_goes_negative_for_absurd_input():
    bullets = [[BulletPoint(text="late", appear_at=99.0)]]
    timeline = RuleBasedPlanner().plan(make_timeline([0.1], bullets=bullets))
    assert timeline.scenes[0].plan.anim_duration == pytest.approx(MIN_ANIM_DURATION)


# ========================================================= duration arithmetic


def test_final_duration_subtracts_every_transition_overlap():
    """xfade consumes overlap, so the total shrinks by one transition per boundary."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 4.0, 4.0, 4.0]))
    overlaps = [scene.plan.transition_duration for scene in timeline.scenes[1:]]

    assert timeline.narration_duration == pytest.approx(16.0)
    assert overlaps == [0.35, 0.35, 0.35]
    assert timeline.final_duration() == pytest.approx(16.0 - 1.05)
    # The naive (wrong) answer differs by 1.5s -- ~0.5s of drift per transition.
    assert abs(timeline.final_duration() - timeline.narration_duration) == pytest.approx(1.05)


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


# ================================================================== watermark


LOGO_SVG = Path("/Users/argo/ab/prompt-to-video-v2/frontend/public/favicon.svg")
needs_magick = pytest.mark.skipif(
    text_overlay.imagemagick_bin() is None, reason="ImageMagick not installed"
)


def test_resolve_logo_source_auto_falls_back_to_the_apps_own_mark():
    resolved = resolve_logo_source(AUTO_LOGO)
    if LOGO_SVG.is_file():
        assert resolved == LOGO_SVG
    else:
        assert resolved is None


@pytest.mark.parametrize("disabled", [None, "", ".", "none", Path("")])
def test_an_empty_or_absent_logo_setting_disables_branding_silently(disabled):
    """`VIDEO_LOGO_PATH=` in .env arrives as Path('.'), which is a directory, not a mark."""
    assert resolve_logo_source(disabled) is None


def test_a_configured_logo_that_does_not_exist_is_skipped_not_raised(tmp_path):
    """Branding must never be able to fail a render."""
    assert resolve_logo_source(tmp_path / "missing.svg") is None
    real = tmp_path / "here.png"
    real.write_bytes(b"x")
    assert resolve_logo_source(real) == real


def test_a_backend_with_branding_off_asks_for_no_logo():
    assert FFmpegBackend(text_mode="scrim", logo_path=None).logo_source is None
    assert FFmpegBackend(text_mode="scrim", logo_path=None).logo_png(HD, Path("/tmp")) is None


def test_the_logo_offsets_are_even_because_yuv420p_subsamples_chroma():
    """An odd overlay offset lands the layer on a half-pixel of the chroma plane."""
    backend = FFmpegBackend(text_mode="scrim")
    for profile in (HD, RenderProfile(width=1280, height=717), RenderProfile.draft()):
        region = backend.logo_region(profile)
        assert region.x % 2 == 0 and region.y % 2 == 0
        assert region.x >= 0 and region.y >= 0
        assert region.y + region.height <= profile.height + region.height


def test_the_logo_is_the_last_stage_of_the_video_chain():
    """After the fades, not before them.

    Before the fades the mark would dim with the opening fade-up and the closing
    fade-out, and 'constant for the entire video' is the requirement.
    """
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 3.0, 5.0]))
    backend = FFmpegBackend(text_mode="scrim")
    durations = [scene.duration for scene in timeline.scenes]
    box = backend.logo_region(HD)

    parts, _, length = backend._video_chain(timeline, durations, logo_input=9, logo_box=box)
    graph = ";".join(parts)

    assert f"[faded][logo]overlay=x={box.x}:y={box.y}" in graph
    assert graph.endswith("[vout]")
    overlay_stage = parts[-1]
    assert "overlay=" in overlay_stage and overlay_stage.endswith("format=yuv420p[vout]")
    # The fades are upstream of the overlay, so they cannot touch it.
    faded = next(p for p in parts if p.endswith("[faded]"))
    assert "fade=t=in" in faded and "fade=t=out" in faded
    assert "fade=" not in overlay_stage
    # And the mark itself gets no fade, no trim, nothing time-varying.
    logo_prep = next(p for p in parts if p.startswith("[9:v]"))
    assert logo_prep == "[9:v]format=rgba,setsar=1[logo]"
    assert length == pytest.approx(timeline.final_duration())


def test_the_logo_overlay_does_not_change_the_chain_length():
    """`assemble` asserts against final_duration(); branding must be duration-neutral."""
    timeline = RuleBasedPlanner().plan(make_timeline([4.0, 3.0, 5.0, 2.0]))
    backend = FFmpegBackend(text_mode="scrim")
    durations = [scene.duration for scene in timeline.scenes]

    plain_parts, plain_starts, plain_len = backend._video_chain(timeline, durations)
    box = backend.logo_region(HD)
    logo_parts, logo_starts, logo_len = backend._video_chain(
        timeline, durations, logo_input=4, logo_box=box
    )

    assert logo_len == pytest.approx(plain_len)
    assert logo_starts == plain_starts
    # Same xfade stages, plus exactly two more filters (the prep and the overlay).
    assert len(logo_parts) == len(plain_parts) + 2
    for filt in ("xfade", "offset="):
        assert ";".join(plain_parts).count(filt) == ";".join(logo_parts).count(filt)


def test_the_logo_chain_is_valid_even_with_no_fades_and_no_captions():
    """A filterchain cannot be empty; with a single cut scene the tail would be."""
    timeline = make_timeline([3.0])
    timeline.scenes[0].plan = VisualPlan(
        transition_in=Transition.CUT, transition_duration=0.0
    )
    backend = FFmpegBackend(text_mode="scrim", final_fade_out=False)
    box = backend.logo_region(HD)

    parts, _, _ = backend._video_chain(timeline, [3.0], logo_input=1, logo_box=box)
    faded = next(p for p in parts if p.endswith("[faded]"))
    assert faded == "[c0]null[faded]", faded


def test_no_layout_the_planner_produces_collides_with_the_logo():
    """The corner is reserved by geometry, so this should hold without intervention."""
    timeline = RuleBasedPlanner().plan(
        make_timeline([6.0] * 5, bullets=[[BulletPoint(text="A point", appear_at=0.5)]] * 5)
    )
    backend = FFmpegBackend(text_mode="scrim")
    assert backend.logo_conflicts(timeline, backend.logo_region(HD)) == []


def test_a_collision_is_reported_rather_than_silently_overlapped():
    """Move the logo on top of the text and the check must say so, per scene.

    Reported, not resolved: nudging the mark per scene would make it move, which is
    exactly what a persistent brand mark must not do.
    """
    timeline = RuleBasedPlanner().plan(
        make_timeline([6.0, 6.0], bullets=[[BulletPoint(text="A point", appear_at=0.5)]] * 2)
    )
    backend = FFmpegBackend(text_mode="scrim")
    # A logo occupying most of the frame overlaps every slide's text by construction.
    huge = FFmpegBackend(text_mode="scrim").logo_region(HD)
    huge = type(huge)(0, 0, HD.width, HD.height)

    conflicts = backend.logo_conflicts(timeline, huge)
    assert len(conflicts) == len(timeline.scenes)
    for scene, message in zip(timeline.scenes, conflicts, strict=True):
        assert f"scene {scene.id}" in message
        assert scene.plan.layout.value in message
        assert "overlaps the logo" in message


@needs_magick
def test_the_logo_png_is_cached_in_the_job_directory(tmp_path):
    backend = FFmpegBackend(text_mode="scrim")
    if backend.logo_source is None:
        pytest.skip("no logo source available")
    png = backend.logo_png(HD, tmp_path)
    assert png is not None
    assert tmp_path in png.parents, "the rasterised mark must live under the job dir"
    assert backend.logo_png(HD, tmp_path) == png


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
    # Still clearly eased: the opening step is a small fraction of the mid-move step.
    early = _evaluate(z, on=1) - _evaluate(z, on=0)
    middle = _evaluate(z, on=75) - _evaluate(z, on=74)
    assert early < middle / 3


def test_the_eased_curve_never_actually_stops_moving():
    """The other half of the stepping fix, and the reason the threshold above is /3 not /5.

    A pure smoothstep's velocity is exactly zero at both endpoints, so the opening and
    closing frames of every move held position no matter how much canvas headroom zoompan
    was given -- no finite upscale can divide into a zero. Measured on
    hero_right/pan_right at 1080p over 604 frames, putting a floor under the velocity took
    the duplicate-frame ratio in the evaluator's 4s window from 11.67% to 2.50%, for free.
    """
    plan = VisualPlan(motion=Motion.ZOOM_IN, zoom_from=1.0, zoom_to=1.12)
    z, _, _ = FFmpegBackend._zoompan_expressions(plan, frames=600)

    early = _evaluate(z, on=1) - _evaluate(z, on=0)
    middle = _evaluate(z, on=300) - _evaluate(z, on=299)
    late = _evaluate(z, on=599) - _evaluate(z, on=598)

    assert early > 0, "a stationary opening frame is the defect"
    assert late > 0, "a stationary closing frame is the same defect"
    # The floor is a known fraction of the mean rate, not an accident.
    mean = (1.12 - 1.0) / 599
    assert early == pytest.approx(EASE_VELOCITY_FLOOR * mean, rel=0.05)
    assert late == pytest.approx(EASE_VELOCITY_FLOOR * mean, rel=0.05)
    assert middle > 4 * early, "and it is still an ease, not a linear ramp"


def test_eased_progress_endpoints_are_exact_so_the_move_lands_where_planned():
    for easing in ("ease_in_out", "linear"):
        expression = eased_progress("u", easing)
        assert _eval_expr(expression, u=0.0) == pytest.approx(0.0)
        assert _eval_expr(expression, u=1.0) == pytest.approx(1.0)
    # Monotonic: the move never doubles back on itself.
    expression = eased_progress("u", "ease_in_out")
    values = [_eval_expr(expression, u=i / 200) for i in range(201)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))


def test_a_move_fast_enough_to_strobe_is_reported(caplog):
    """The opposite failure to stepping. Reported, not silently slowed down."""
    region = layout_region(VisualPlan(layout=SlideLayout.FULL_BLEED), HD)
    # A big zoom span crammed into a 2s scene.
    manic = VisualPlan(layout=SlideLayout.FULL_BLEED, motion=Motion.PAN_RIGHT,
                       zoom_from=1.5, zoom_to=1.5)
    assert peak_step(manic, region, 60) > STROBE_STEP_PIXELS

    with caplog.at_level("WARNING"):
        FFmpegBackend(text_mode="scrim")._warn_if_strobing(manic, region, 60)
    assert any("strobing" in record.message for record in caplog.records)

    # And an ordinary hero pan is nowhere near it.
    calm = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                      zoom_from=1.12, zoom_to=1.12)
    hero = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    assert peak_step(calm, hero, 604) < 1.0


def test_travel_is_measured_in_output_pixels_of_the_region():
    """A fixed upscale ignored this, which is precisely why it was wrong for a panel."""
    hero = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    bleed = layout_region(VisualPlan(layout=SlideLayout.FULL_BLEED), HD)
    pan = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.12, zoom_to=1.12)

    assert motion_travel(pan, hero) == pytest.approx(hero.width * (1 - 1 / 1.12))
    assert motion_travel(pan, bleed) > motion_travel(pan, hero), (
        "the same plan travels further across the frame than inside a panel -- the whole "
        "reason a factor calibrated on 1920 is short at 856"
    )
    # A pan with no zoom still gets the MIN_PAN_ZOOM headroom rather than zero travel.
    flat = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.0, zoom_to=1.0)
    assert motion_travel(flat, hero) > 0

    # Slowest step scales the mean rate by the easing floor; linear has no floor.
    eased = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.12, zoom_to=1.12)
    straight = VisualPlan(motion=Motion.PAN_RIGHT, zoom_from=1.12, zoom_to=1.12,
                          easing="linear")
    assert slowest_step(eased, hero, 600) == pytest.approx(
        EASE_VELOCITY_FLOOR * slowest_step(straight, hero, 600)
    )


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


def _graph(backend: FFmpegBackend, plan: VisualPlan, profile: RenderProfile, **kwargs) -> str:
    """Build a scene graph, defaulting the boilerplate the tests do not care about."""
    return backend._scene_graph(
        src_size=kwargs.pop("src_size", (1920, 1080)),
        plan=plan,
        profile=profile,
        frames=kwargs.pop("frames", 90),
        text_layout=text_overlay.layout_heading(kwargs.get("heading", "Hi"), plan, profile),
        heading=kwargs.pop("heading", "Hi"),
        has_image_input=kwargs.pop(
            "has_image_input", layout_region(plan, profile) is not None
        ),
        **kwargs,
    )


def test_the_canvas_is_upscaled_before_zoompan_and_zoompan_emits_the_final_size():
    """The ordering that makes the headroom work at all: scale up, *then* zoompan down."""
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    plan = VisualPlan(
        layout=SlideLayout.FULL_BLEED, motion=Motion.PAN_RIGHT, zoom_from=1.1, zoom_to=1.1
    )
    graph = _graph(FFmpegBackend(text_mode="scrim"), plan, profile)

    assert "s=1920x1080" in graph, "zoompan must emit the final size"
    scales = [int(m) for m in re.findall(r"scale=(\d+):\d+", graph)]
    assert scales, graph
    assert max(scales) > profile.width, "the canvas must be wider than the output"
    assert graph.index("scale=") < graph.index("zoompan")


def test_the_upscale_is_derived_from_the_move_not_a_fixed_factor():
    """The regression this replaces.

    A fixed 4x was calibrated when zoompan filled the frame. Two moves over the same
    region with the same frame count but very different travel must now get very
    different canvases -- that is the whole point of deriving it.
    """
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    region = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), profile)
    crawl = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                       zoom_from=1.06, zoom_to=1.06)
    sweep = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                       zoom_from=1.6, zoom_to=1.6)

    slow = motion_canvas(crawl, region, 600, profile, src_size=(2752, 1536))
    fast = motion_canvas(sweep, region, 600, profile, src_size=(2752, 1536))

    assert slow.canvas[0] > fast.canvas[0], "a slower move needs more headroom"
    assert slow.canvas[0] > region.width * 4, "4x was measured as insufficient here"
    # And the frame count matters just as much as the distance.
    brief = motion_canvas(crawl, region, 60, profile, src_size=(2752, 1536))
    assert brief.canvas[0] < slow.canvas[0], "fewer frames means faster per frame"


def test_ken_burns_happens_inside_the_image_region_not_across_the_frame():
    """The whole correctness point of the designed-frame layout.

    zoompan must emit the *region's* size, and the upscale headroom is relative to the
    region too -- otherwise the pan travels across the frame and the panel edge moves.
    """
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN)
    region = layout_region(plan, profile)
    graph = _graph(FFmpegBackend(text_mode="scrim"), plan, profile, src_size=(2752, 1536))

    assert region is not None and region.width < profile.width
    zoompan = next(part for part in graph.split(";") if "zoompan" in part)
    assert f"s={region.width}x{region.height}" in zoompan, "zoompan must emit region size"
    assert "s=1920x1080" not in zoompan, "a hero panel is not the frame"
    assert f"overlay=x={region.x}:y={region.y}" in graph, "region must land at its offset"
    assert graph.startswith("color=c=0x0B1220:s=1920x1080"), "solid background first"

    sizing = motion_canvas(plan, region, 90, profile, src_size=(2752, 1536))
    assert f"{sizing.canvas[0]}:{sizing.canvas[1]}" in graph
    assert sizing.canvas[0] > region.width, "the canvas must exceed the region it feeds"


def test_a_pans_canvas_is_anamorphic_because_only_one_axis_travels():
    """The cost fix: precision on the travel axis, detail on the other.

    zoompan crops proportionally to its input and scales to the region, so a canvas that
    is stretched on one axis is exactly un-squeezed on the way out. A pan holds a fixed
    zoom and never moves vertically, so buying vertical precision is buying nothing --
    and buying it isotropically is what made the honest factor unaffordable.
    """
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    region = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), profile)
    pan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                     zoom_from=1.12, zoom_to=1.12)

    sizing = motion_canvas(pan, region, 604, profile, src_size=(2752, 1536))
    canvas_w, canvas_h = sizing.canvas
    assert canvas_w / region.width > canvas_h / region.height, (
        "a pan must not pay for cross-axis precision it cannot use"
    )
    assert sizing.stretched, "an anamorphic canvas is the point"
    assert canvas_h >= sizing.fit[1], "the stretch may never downscale the lanczos fit"

    graph = _graph(FFmpegBackend(text_mode="scrim"), pan, profile,
                   src_size=(2752, 1536), frames=604)
    # The expensive lanczos fit happens at the small isotropic size...
    assert f"scale={sizing.fit[0]}:{sizing.fit[1]}" in graph
    # ...and only a cheap bilinear stretch reaches the wide canvas.
    stretch = f"scale={canvas_w}:{canvas_h}:flags=bilinear"
    assert stretch in graph, graph
    assert graph.index(stretch) < graph.index("zoompan")


def test_a_zoom_keeps_cross_axis_precision_because_it_moves_both_axes():
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    region = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), profile)
    pan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                     zoom_from=1.12, zoom_to=1.12)
    zoom = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN,
                      zoom_from=1.0, zoom_to=1.12)

    panned = motion_canvas(pan, region, 604, profile, src_size=(2752, 1536))
    zoomed = motion_canvas(zoom, region, 604, profile, src_size=(2752, 1536))
    assert zoomed.canvas[1] > panned.canvas[1], (
        "a zoom changes the crop's height every frame; a pan does not"
    )


def test_zoompans_own_crop_is_never_an_upscale_of_the_fitted_still():
    """The subtle resolution bug: canvas must be at least region * zoom on both axes.

    zoompan crops ``canvas/zoom`` and scales that to the region. If the canvas only just
    matches the region, the crop is *smaller* than the region and the last resample is an
    upscale -- measurably softer than the still it came from. Measured: getting this wrong
    on full_bleed cost 37% of the panel's horizontal gradient energy.
    """
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    for layout in (SlideLayout.HERO_RIGHT, SlideLayout.IMAGE_BAND, SlideLayout.FULL_BLEED):
        region = layout_region(VisualPlan(layout=layout), profile)
        for motion, zf, zt in (
            (Motion.PAN_RIGHT, 1.08, 1.08), (Motion.PAN_LEFT, 1.0, 1.0),
            (Motion.ZOOM_IN, 1.0, 1.12), (Motion.ZOOM_OUT, 1.15, 1.0),
        ):
            plan = VisualPlan(layout=layout, motion=motion, zoom_from=zf, zoom_to=zt)
            sizing = motion_canvas(plan, region, 600, profile, src_size=(2752, 1536))
            zoom = plan_zoom_ceiling(plan)
            assert sizing.canvas[0] / zoom >= region.width - 2, (
                f"{layout}/{motion}: crop width {sizing.canvas[0] / zoom:.0f} upscales to "
                f"{region.width}"
            )
            assert sizing.canvas[1] / zoom >= region.height - 2, (
                f"{layout}/{motion}: crop height {sizing.canvas[1] / zoom:.0f} upscales to "
                f"{region.height}"
            )


def test_the_canvas_stays_inside_its_area_budget():
    """Cost is the canvas area resampled per frame, so that is what is budgeted."""
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    for layout in SlideLayout:
        region = layout_region(VisualPlan(layout=layout), profile)
        if region is None:
            continue
        for motion in (Motion.PAN_RIGHT, Motion.ZOOM_IN, Motion.ZOOM_OUT):
            plan = VisualPlan(layout=layout, motion=motion, zoom_from=1.0, zoom_to=1.12)
            sizing = motion_canvas(plan, region, 600, profile, src_size=(2752, 1536))
            area = sizing.canvas[0] * sizing.canvas[1]
            # The floors (no-upscale, keep the fit) can exceed the budget; nothing else may.
            floor = max(sizing.fit[0], region.width * 2) * max(sizing.fit[1], region.height * 2)
            assert area <= max(CANVAS_PIXEL_BUDGET * 1.05, floor), (
                f"{layout}/{motion}: {area / 1e6:.1f} Mpx"
            )


def test_the_detail_factor_never_invents_pixels_the_source_does_not_have():
    """The 'source pixels vs region output size' half of the derivation."""
    profile = RenderProfile(width=1920, height=1080, upscale_factor=4)
    hero = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), profile)
    bleed = layout_region(VisualPlan(layout=SlideLayout.FULL_BLEED), profile)

    # A 2752x1536 still is already bigger than an 856x816 panel, so a little headroom.
    assert detail_upscale((2752, 1536), hero) >= 2
    # The same still has to be *upscaled* to cover 1080p, so no detail headroom at all.
    assert detail_upscale((2752, 1536), bleed) == 1
    # A tiny source never gets lanczos-inflated either.
    assert detail_upscale((320, 180), bleed) == 1
    assert detail_upscale((0, 0), hero) == 1, "an unprobeable source must not divide by zero"


def test_the_area_budget_still_honours_the_profile():
    """`upscale_factor` is now a cost budget rather than the factor. It must still bite."""
    region = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    crawl = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.PAN_RIGHT,
                       zoom_from=1.06, zoom_to=1.06)
    generous = motion_canvas(
        crawl, region, 600, RenderProfile(upscale_factor=4), src_size=(2752, 1536)
    )
    thrifty = motion_canvas(
        crawl, region, 600, RenderProfile(upscale_factor=1), src_size=(2752, 1536)
    )
    assert thrifty.canvas[0] < generous.canvas[0], "a draft profile buys less smoothness"
    assert thrifty.canvas[0] >= region.width, "but never less than the region it feeds"


def test_a_static_shot_asks_for_no_canvas_at_all():
    region = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    sizing = motion_canvas(plan, region, 600, HD, src_size=(2752, 1536))
    assert sizing.canvas == (region.width, region.height)
    assert not sizing.stretched
    assert motion_travel(plan, region) == 0.0
    assert slowest_step(plan, region, 600) == 0.0
    assert plan_zoom_ceiling(plan) == 1.0


def test_static_motion_skips_zoompan_entirely():
    profile = RenderProfile(width=640, height=360, upscale_factor=4)
    plan = VisualPlan(layout=SlideLayout.FULL_BLEED, motion=Motion.STATIC)
    graph = _graph(FFmpegBackend(text_mode="scrim"), plan, profile, src_size=(640, 360))
    assert "zoompan" not in graph
    assert "scale=640:360" in graph


# ============================================================== slide geometry


def test_every_layout_but_the_title_card_has_an_image_region_inside_the_frame():
    for layout in SlideLayout:
        plan = VisualPlan(layout=layout)
        region = layout_region(plan, HD)
        if layout is SlideLayout.TITLE_CARD:
            assert region is None, "a title card is type on colour; it has no image"
            continue
        assert region is not None
        assert region.x >= 0 and region.y >= 0
        assert region.x + region.width <= HD.width
        assert region.y + region.height <= HD.height


def test_region_edges_are_even_because_yuv420p_subsamples_chroma():
    for layout in SlideLayout:
        region = layout_region(VisualPlan(layout=layout), HD)
        if region is None:
            continue
        assert region.x % 2 == 0 and region.y % 2 == 0, region
        assert region.width % 2 == 0 and region.height % 2 == 0, region


def test_hero_image_occupies_roughly_a_third_leaving_the_text_the_larger_column():
    # 720x900 (4:5) of a 1920x1080 frame, per the grid in `docs/DIRECTION.md` §6.2. The
    # share dropped from ~0.47 to 0.375 when the text column widened to the 904px that a
    # 78px heading needs for 22 characters; text_overlay.slide_geometry owns both numbers,
    # so the panel and the hole cut for it cannot disagree.
    for layout in (SlideLayout.HERO_LEFT, SlideLayout.HERO_RIGHT):
        region = layout_region(VisualPlan(layout=layout), HD)
        share = region.width / HD.width
        assert 0.35 <= share <= 0.45, f"{layout}: {share:.3f}"
        assert region.width / region.height == pytest.approx(0.8), "the hero region is 4:5"
        assert region.width < HD.width - region.width, "text needs the larger share"


def test_hero_left_and_hero_right_are_mirrors():
    left = layout_region(VisualPlan(layout=SlideLayout.HERO_LEFT), HD)
    right = layout_region(VisualPlan(layout=SlideLayout.HERO_RIGHT), HD)
    assert (left.width, left.height) == (right.width, right.height)
    assert left.x < right.x
    assert HD.width - (right.x + right.width) == pytest.approx(left.x, abs=2)


def test_full_bleed_fills_the_frame():
    region = layout_region(VisualPlan(layout=SlideLayout.FULL_BLEED), HD)
    assert (region.x, region.y) == (0, 0)
    assert (region.width, region.height) == (HD.width, HD.height)


def test_regions_scale_with_the_profile_rather_than_being_hardcoded():
    small = RenderProfile(width=960, height=540)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)
    big, tiny = layout_region(plan, HD), layout_region(plan, small)
    assert tiny.width / small.width == pytest.approx(big.width / HD.width, abs=0.02)


def test_fallback_region_matches_the_shape_of_the_real_one():
    """Used only when text_overlay cannot supply geometry; must not be absurd."""
    for layout in SlideLayout:
        fallback = fallback_region(layout, HD)
        real = layout_region(VisualPlan(layout=layout), HD)
        assert (fallback is None) == (real is None), layout
        if fallback is None:
            continue
        assert fallback.x + fallback.width <= HD.width
        assert fallback.y + fallback.height <= HD.height


def test_title_card_graph_needs_no_image_input_at_all():
    """render_all must not demand an image_path for a slide that has no image."""
    plan = VisualPlan(layout=SlideLayout.TITLE_CARD)
    graph = _graph(FFmpegBackend(text_mode="scrim"), plan, HD)
    assert "[0:v]" not in graph, "nothing may reference an image input"
    assert "zoompan" not in graph
    assert graph.startswith("color=c=0x0B1220")


def test_bounded_panels_cover_crop_while_full_bleed_may_blur_fill():
    """A blurred letterbox inside a panel is mush; the design wants a clean edge."""
    cropped = ";".join(
        FFmpegBackend._fit_chain((1080, 1920), (856, 816), (856, 816), blur_fill=False)
    )
    assert "gblur" not in cropped
    assert "force_original_aspect_ratio=increase" in cropped
    filled = ";".join(FFmpegBackend._fit_chain((1080, 1920), (1920, 1080), (1920, 1080)))
    assert "gblur" in filled


def test_fit_chain_labels_do_not_collide_with_the_slide_background():
    """Duplicate filtergraph labels are a parse error, not a warning."""
    chain = ";".join(FFmpegBackend._fit_chain((1080, 1920), (1920, 1080), (1920, 1080)))
    assert "[bg]" not in chain, "the solid slide background already owns [bg]"


def test_theme_colour_is_converted_to_the_form_ffmpeg_accepts():
    assert ffmpeg_colour("#0B1220") == "0x0B1220"
    assert ffmpeg_colour("0b1220") == "0x0B1220"
    with pytest.raises(Exception, match="RRGGBB"):
        ffmpeg_colour("rebeccapurple")


def test_the_background_is_the_themes_colour_not_a_hardcoded_one():
    backend = FFmpegBackend(text_mode="scrim", theme=Theme(bg="#123456"))
    graph = _graph(backend, VisualPlan(layout=SlideLayout.TITLE_CARD), HD)
    assert "color=c=0x123456:s=1920x1080:r=30[bg]" in graph


def test_hero_panels_get_rounded_corners_and_full_bleed_does_not():
    backend = FFmpegBackend(text_mode="scrim")
    hero = _graph(backend, VisualPlan(layout=SlideLayout.HERO_RIGHT), HD, src_size=(2752, 1536))
    assert "alphamerge" in hero, "rounded corners are an alpha mask"
    assert "loop=loop=-1:size=1" in hero, "the mask is static; evaluate geq once"
    assert "geq=lum=" in hero

    bleed = _graph(backend, VisualPlan(layout=SlideLayout.FULL_BLEED), HD)
    assert "alphamerge" not in bleed, "a full-bleed image IS the frame"


# ============================================================ text animation


def _eval_expr(expression: str, **variables: float) -> float:
    """Evaluate an ffmpeg expression as Python. The subset we emit is compatible."""
    scope = {"min": min, "max": max, "pow": pow}
    return float(eval(expression, {"__builtins__": {}}, scope | variables))  # noqa: S307


def fake_layer(**kwargs) -> TextLayer:
    """A TextLayer without touching text_overlay -- the contract is the only coupling."""
    defaults = dict(
        png_path=Path("/tmp/layer.png"),
        x=120,
        y=400,
        width=800,
        height=60,
        appear_at=1.0,
        animation=TextAnimation.FADE_IN,
        anim_duration=0.4,
        slide_distance=60,
        kind="bullet",
    )
    return TextLayer(**(defaults | kwargs))


def fake_scene_text(count: int = 4, *, first: float = 0.5, step: float = 1.0) -> SceneText:
    """A scrim, a heading, and ``count`` staggered bullets."""
    layers = [
        TextLayer(
            png_path=Path("/tmp/scrim.png"),
            x=0,
            y=0,
            width=1920,
            height=1080,
            appear_at=0.0,
            animation=TextAnimation.FADE_IN,
            anim_duration=0.3,
            kind="scrim",
        ),
        fake_layer(appear_at=0.2, animation=TextAnimation.SLIDE_UP, kind="heading", y=120),
    ]
    for index in range(count):
        layers.append(
            fake_layer(
                png_path=Path(f"/tmp/b{index}.png"),
                appear_at=first + index * step,
                y=400 + index * 90,
                animation=TextAnimation.SLIDE_LEFT,
            )
        )
    return SceneText(layers=layers)


def test_a_layer_is_fully_transparent_before_its_appear_at():
    """The classic bug: the PNG shows from frame 0 and every reveal is spoiled.

    ``fade=t=in`` with ``st>0`` holds alpha at zero for every earlier frame, and the
    ``enable`` gate skips the overlay entirely. Both are asserted because the fade is
    what makes it correct and the gate is what makes it cheap.
    """
    layer = fake_layer(appear_at=1.25, anim_duration=0.4)
    prep = FFmpegBackend._layer_prep(layer, fps=30)

    assert "format=rgba" in prep, "fade cannot touch an alpha channel that is not there"
    assert prep.index("format=rgba") < prep.index("fade="), "alpha must exist first"
    assert "fade=t=in:st=1.2500:d=0.4000:alpha=1" in prep
    assert FFmpegBackend._visibility_expr(layer, fps=30) == "gte(t,1.2500)"


def test_alpha_fade_is_the_only_thing_that_interpolates_opacity():
    """`enable` cannot interpolate, so it must never be the whole mechanism."""
    layer = fake_layer(animation=TextAnimation.FADE_IN, appear_at=2.0, anim_duration=0.5)
    prep = FFmpegBackend._layer_prep(layer, fps=30)
    assert "alpha=1" in prep, "without alpha=1 the fade goes to black, not transparent"
    assert ":d=0.5000" in prep


def test_disappear_at_gets_a_matching_fade_out_and_a_bounded_gate():
    layer = fake_layer(appear_at=1.0, disappear_at=4.0, anim_duration=0.4)
    prep = FFmpegBackend._layer_prep(layer, fps=30)
    assert "fade=t=out:st=4.0000:d=0.4000:alpha=1" in prep
    assert prep.index("fade=t=in") < prep.index("fade=t=out")
    gate = FFmpegBackend._visibility_expr(layer, fps=30)
    assert gate.startswith("between(t,1.0000,")
    assert _eval_expr(gate.replace("between(t,", "").split(",")[1].rstrip(")")) > 4.0


def test_a_layer_with_no_disappear_time_stays_to_the_end():
    prep = FFmpegBackend._layer_prep(fake_layer(disappear_at=None), fps=30)
    assert "fade=t=out" not in prep


@pytest.mark.parametrize("fps", [12, 24, 30, 60])
def test_none_and_typewriter_appear_within_a_single_frame(fps):
    """Instant, but never visible early: a one-frame ramp still gates frame 0."""
    for animation in (TextAnimation.NONE, TextAnimation.TYPEWRITER):
        layer = fake_layer(animation=animation, anim_duration=0.5)
        assert FFmpegBackend._fade_in(layer, fps=fps) == pytest.approx(1 / fps)


def test_pop_fades_faster_than_it_moves():
    layer = fake_layer(animation=TextAnimation.POP, anim_duration=0.5)
    assert FFmpegBackend._fade_in(layer, fps=30) < layer.anim_duration
    assert FFmpegBackend._fade_in(layer, fps=30) == pytest.approx(0.3)


def test_slide_up_travels_from_below_and_lands_exactly_on_target():
    layer = fake_layer(animation=TextAnimation.SLIDE_UP, appear_at=1.0, anim_duration=0.4, y=400)
    x_expr, y_expr = FFmpegBackend._anim_position(layer)

    assert x_expr == "120", "a vertical slide must not move horizontally"
    assert _eval_expr(y_expr, t=0.0) == pytest.approx(460), "starts slide_distance below"
    assert _eval_expr(y_expr, t=1.0) == pytest.approx(460)
    assert _eval_expr(y_expr, t=1.4) == pytest.approx(400), "lands on the final y"
    assert _eval_expr(y_expr, t=1.2) < 460, "and it actually moved in between"


def test_slide_left_travels_from_the_right_and_lands_exactly_on_target():
    layer = fake_layer(animation=TextAnimation.SLIDE_LEFT, appear_at=2.0, anim_duration=0.5, x=120)
    x_expr, y_expr = FFmpegBackend._anim_position(layer)

    assert y_expr == "400"
    assert _eval_expr(x_expr, t=0.0) == pytest.approx(180)
    assert _eval_expr(x_expr, t=2.5) == pytest.approx(120)
    assert 120 < _eval_expr(x_expr, t=2.25) < 180


@pytest.mark.parametrize(
    "animation", [TextAnimation.SLIDE_UP, TextAnimation.SLIDE_LEFT, TextAnimation.POP]
)
def test_position_is_clamped_to_the_final_value_forever_after(animation):
    """An unclamped ramp keeps travelling for the rest of the scene."""
    layer = fake_layer(animation=animation, appear_at=1.0, anim_duration=0.4, x=120, y=400)
    x_expr, y_expr = FFmpegBackend._anim_position(layer)
    for t in (1.4, 1.5, 3.0, 10.0, 600.0):
        assert _eval_expr(x_expr, t=t) == pytest.approx(120), t
        assert _eval_expr(y_expr, t=t) == pytest.approx(400), t


@pytest.mark.parametrize(
    "animation", [TextAnimation.SLIDE_UP, TextAnimation.SLIDE_LEFT, TextAnimation.POP]
)
def test_position_is_pinned_at_the_start_value_before_appear_at(animation):
    layer = fake_layer(animation=animation, appear_at=2.0, anim_duration=0.4)
    x_expr, y_expr = FFmpegBackend._anim_position(layer)
    for t in (0.0, 0.5, 1.99, 2.0):
        assert _eval_expr(x_expr, t=t) == pytest.approx(_eval_expr(x_expr, t=0.0)), t
        assert _eval_expr(y_expr, t=t) == pytest.approx(_eval_expr(y_expr, t=0.0)), t


def test_slides_are_eased_not_linear():
    """A linear slide starts and stops abruptly and reads as mechanical.

    Smoothstep has zero velocity at both ends, so the first and last steps of the
    move are a fraction of the mid-move step.
    """
    layer = fake_layer(animation=TextAnimation.SLIDE_UP, appear_at=0.0, anim_duration=1.0)
    _, y_expr = FFmpegBackend._anim_position(layer)

    first = abs(_eval_expr(y_expr, t=0.05) - _eval_expr(y_expr, t=0.0))
    middle = abs(_eval_expr(y_expr, t=0.55) - _eval_expr(y_expr, t=0.5))
    last = abs(_eval_expr(y_expr, t=1.0) - _eval_expr(y_expr, t=0.95))

    assert first < middle / 4, f"eases in: {first} vs {middle}"
    assert last < middle / 4, f"eases out: {last} vs {middle}"
    # And the midpoint of a smoothstep is exactly halfway.
    assert _eval_expr(y_expr, t=0.5) == pytest.approx(430)


def test_pop_overshoots_past_its_target_then_settles():
    """POP is approximated as a positional overshoot -- see _anim_position."""
    layer = fake_layer(animation=TextAnimation.POP, appear_at=0.0, anim_duration=1.0, y=400)
    _, y_expr = FFmpegBackend._anim_position(layer)

    assert _eval_expr(y_expr, t=0.0) == pytest.approx(420), "starts below the mark"
    overshoot = min(_eval_expr(y_expr, t=t / 100) for t in range(0, 101))
    assert overshoot < 400, f"never went past the target: {overshoot}"
    assert overshoot > 392, f"overshoot must stay subtle, not bounce: {overshoot}"
    assert _eval_expr(y_expr, t=1.0) == pytest.approx(400), "settles exactly on target"


def test_none_and_fade_in_do_not_move_the_layer():
    for animation in (TextAnimation.NONE, TextAnimation.FADE_IN, TextAnimation.TYPEWRITER):
        x_expr, y_expr = FFmpegBackend._anim_position(fake_layer(animation=animation))
        assert (x_expr, y_expr) == ("120", "400"), animation


def test_typewriter_is_a_left_to_right_alpha_wipe():
    """Documented approximation: without drawtext there is no glyph-level clock."""
    layer = fake_layer(animation=TextAnimation.TYPEWRITER, appear_at=1.0, anim_duration=0.8)
    prep = FFmpegBackend._layer_prep(layer, fps=30)

    assert "geq=" in prep and "alpha(X,Y)" in prep
    assert prep.index("format=rgba") < prep.index("geq=")
    reveal = FFmpegBackend._wipe_filter(layer)
    fraction = reveal.split("lt(X,W*")[1].split(")),")[0] + ")"
    assert _eval_expr(fraction, T=0.0) == pytest.approx(0.0), "nothing revealed early"
    assert _eval_expr(fraction, T=1.4) == pytest.approx(0.5), "half way through"
    assert _eval_expr(fraction, T=5.0) == pytest.approx(1.0), "fully revealed and clamped"


# ------------------------------------------------------------- chain assembly


def test_layers_are_overlaid_in_sorted_order_scrim_first():
    """The scrim must land *under* the type it exists to make legible."""
    scene_text = fake_scene_text(count=3)
    parts, final = FFmpegBackend(text_mode="scrim")._text_chain(
        scene_text, base="base", first_input=1, fps=30
    )
    graph = ";".join(parts)
    kinds = [layer.kind for layer in scene_text.sorted_layers()]

    assert kinds == ["scrim", "heading", "bullet", "bullet", "bullet"]
    assert final == "ov4", "the last overlay's label is what the caller composites"
    # Overlay stages are strictly chained: each consumes the previous one's output.
    assert "[base][tl0]overlay=" in graph
    for index in range(1, 5):
        assert f"[ov{index - 1}][tl{index}]overlay=" in graph


def test_every_filtergraph_label_is_unique_across_seven_layers():
    """A duplicate label is a parse error, and 1 scrim + 1 heading + 5 bullets is legal."""
    scene_text = fake_scene_text(count=5)
    assert len(scene_text.layers) == 7
    parts, _ = FFmpegBackend(text_mode="scrim")._text_chain(
        scene_text, base="base", first_input=1, fps=30
    )
    produced = [part.rsplit("[", 1)[1].rstrip("]") for part in parts]
    assert len(produced) == len(set(produced)), produced


def test_each_layer_reads_its_own_input_index():
    scene_text = fake_scene_text(count=4)
    parts, _ = FFmpegBackend(text_mode="scrim")._text_chain(
        scene_text, base="base", first_input=1, fps=30
    )
    graph = ";".join(parts)
    for index in range(len(scene_text.layers)):
        assert f"[{index + 1}:v]format=rgba" in graph


def test_text_layers_start_at_input_zero_when_the_slide_has_no_image():
    parts, _ = FFmpegBackend(text_mode="scrim")._text_chain(
        fake_scene_text(count=1), base="bg", first_input=0, fps=30
    )
    assert "[0:v]format=rgba" in ";".join(parts)


def test_animated_layers_ask_for_per_frame_evaluation_and_static_ones_do_not():
    """`eval=frame` is required for a time-varying expression and wasted otherwise."""
    backend = FFmpegBackend(text_mode="scrim")
    moving = SceneText(layers=[fake_layer(animation=TextAnimation.SLIDE_UP)])
    still = SceneText(layers=[fake_layer(animation=TextAnimation.FADE_IN)])

    assert "eval=frame" in ";".join(backend._text_chain(moving, base="b", first_input=1, fps=30)[0])
    assert "eval=init" in ";".join(backend._text_chain(still, base="b", first_input=1, fps=30)[0])


def test_scene_graph_composites_every_layer_and_ends_in_yuv420p():
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN)
    graph = _graph(
        FFmpegBackend(text_mode="scrim"),
        plan,
        HD,
        src_size=(2752, 1536),
        scene_text=fake_scene_text(count=4),
    )
    assert graph.count("overlay=") == 1 + 6, "one for the hero panel, one per text layer"
    assert graph.endswith("format=yuv420p[vout]")
    assert graph.count("[vout]") == 1


def test_too_many_text_layers_is_refused_rather_than_rendered(tmp_path):
    from app.render.ffmpeg_backend import MAX_TEXT_LAYERS, RenderError

    backend = FFmpegBackend(text_mode="scrim")
    crowded = SceneText(layers=[fake_layer() for _ in range(MAX_TEXT_LAYERS + 1)])
    with pytest.raises(RenderError, match="exceeds"):
        backend.render_scene(
            None, VisualPlan(layout=SlideLayout.TITLE_CARD), "Hi", 1.0,
            tmp_path / "nope.mp4", HD, scene_text=crowded,
        )


def test_scene_graph_construction_holds_no_shared_state():
    """Scenes render on four threads; two graphs built from the same inputs must match
    and must not influence each other."""
    backend = FFmpegBackend(text_mode="scrim")
    plans = [
        VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN),
        VisualPlan(layout=SlideLayout.HERO_LEFT, motion=Motion.PAN_LEFT),
    ]
    first = [_graph(backend, plan, HD, src_size=(2752, 1536)) for plan in plans]
    second = [_graph(backend, plan, HD, src_size=(2752, 1536)) for plan in reversed(plans)]
    assert first == list(reversed(second))


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
    assert shifts[2] == pytest.approx(-0.35)
    assert shifts[3] == pytest.approx(-0.7)


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
    assert timeline.scenes[1].plan.transition_duration == pytest.approx(0.35)

    backend = FFmpegBackend()
    timeline = backend.render_all(timeline, tmp_path)
    out = backend.assemble(timeline, tmp_path / "final.mp4")

    expected = timeline.final_duration()
    summary = ff.probe_summary(out)
    assert expected == pytest.approx(1.65), "2.0s of scenes minus one 0.35s crossfade"
    assert summary["duration"] == pytest.approx(expected, abs=max(0.1, 4 / FPS))
    assert summary["audio_channels"] == 2
    assert summary["audio_codec"] == "aac"
    assert summary["video_codec"] == "h264"
    # And the naive answer would have been 2.0s -- 0.4s of narration drift.
    assert abs(summary["duration"] - timeline.narration_duration) > 0.3


@integration
@needs_magick
def test_branding_is_duration_neutral_and_survives_every_transition(assets, tmp_path):
    """The two things the watermark must not do: shift timing, or pulse at a boundary.

    The transition here is ``slideleft`` on purpose, and it is *harsher than anything the
    planner now emits* — ``TRANSITION_ROTATION`` is ``(FADE,)``, so a slide never actually
    ships. It is kept because a logo burnt into each scene clip would literally slide off the
    left edge with the outgoing frame, which a crossfade would hide. (An earlier version of
    this docstring claimed a slide was what the rotation picked for the second boundary. That
    was true once and is not any more.)

    Constancy is asserted on the mark's **absolute** value, not on its lift over a control
    patch elsewhere in the frame. The differential version of this measurement was wrong, and
    wrong in a way that accused a working watermark: the control patch sits along the bottom
    edge three logo-widths to the right, so mid-``slideleft`` the *incoming picture* slides
    through it. Measured on this exact scenario, at the first boundary's midpoint the control
    patch jumped from ~31 to ~87 while the mark itself only moved 195 -> 183, so the
    differential collapsed by half and reported a pulse that was not there. Absolute spread
    over the same five samples is 17.6% (steady); under ``FADE``, where the control patch
    holds still, both readings agree at ~1-2%.

    The mark's absolute value is still allowed to vary a little, because ``logo_opacity`` is
    0.85 and the picture behind it genuinely changes at a boundary. Hence a tolerance rather
    than an equality -- but the tolerance is on the thing that actually means "the mark
    dimmed", so it is a real guard.
    """
    scenes = [
        Scene(
            id=index + 1,
            narration="n",
            heading=f"Slide {index + 1}",
            image_prompt="p",
            image_path=str(assets["landscape"] if index % 2 == 0 else assets["portrait"]),
            start=float(index),
            end=float(index + 1),
            plan=VisualPlan(
                layout=SlideLayout.HERO_RIGHT,
                motion=Motion.STATIC,
                transition_in=Transition.FADE if index == 0 else Transition.SLIDE_LEFT,
                transition_duration=0.0 if index == 0 else 0.25,
            ),
        )
        for index in range(3)
    ]
    timeline = Timeline(
        job_id="brand", topic="t", title="T", voice="v", profile=TINY, scenes=scenes
    )
    backend = FFmpegBackend()
    if backend.logo_source is None:
        pytest.skip("no logo source available")
    timeline = backend.render_all(timeline, tmp_path)

    branded = backend.assemble(timeline, tmp_path / "branded.mp4")
    plain = FFmpegBackend(logo_path=None).assemble(timeline, tmp_path / "plain.mp4")

    expected = timeline.final_duration()
    branded_d, plain_d = ff.probe_duration(branded), ff.probe_duration(plain)
    assert branded_d == pytest.approx(expected, abs=max(0.1, 4 / FPS))
    assert branded_d == pytest.approx(plain_d, abs=1 / FPS), "the overlay shifted the timing"

    logo = backend.logo_png(TINY, tmp_path)
    assert logo is not None
    region = backend.logo_region(TINY, logo)
    binary = text_overlay.require_imagemagick()
    mask = tmp_path / "mask.png"
    # Alpha is pre-multiplied by theme.logo_opacity, so "opaque" is that, not 1.0.
    threshold = int(backend.theme.logo_opacity * 90)
    subprocess.run(  # noqa: S603
        [binary, str(logo), "-alpha", "extract", "-threshold", f"{threshold}%", str(mask)],
        check=True,
    )

    durations = [ff.probe_duration(s.clip_path) for s in timeline.scenes]
    _, starts, chain = backend._video_chain(timeline, durations)
    stamps = [starts[0] + durations[0] / 2]
    for index in range(1, len(scenes)):
        boundary = starts[index]
        half = scenes[index].plan.transition_duration / 2
        stamps += [boundary + half, starts[index] + durations[index] / 2]

    share = float(subprocess.run(  # noqa: S603
        [binary, str(mask), "-format", "%[fx:mean]", "info:"],
        capture_output=True, text=True, check=True).stdout)

    def logo_core(video: Path, when: float) -> float:
        """Mean blue of the mark's own opaque core, 0-255. Absolute, not differential."""
        frame = tmp_path / f"f{video.stem}{when:.3f}.png"
        ff.ffmpeg(["-ss", f"{when:.4f}", "-i", video, "-frames:v", "1", "-update", "1", frame])
        crop = tmp_path / "crop.png"
        subprocess.run(  # noqa: S603
            [binary, str(frame), "-crop",
             f"{region.width}x{region.height}+{region.x}+{region.y}", "+repage", str(crop)],
            check=True,
        )
        masked = tmp_path / "masked.png"
        subprocess.run(  # noqa: S603
            [binary, str(crop), str(mask), "-compose", "Multiply", "-composite", str(masked)],
            check=True,
        )
        core_b = float(subprocess.run(  # noqa: S603
            [binary, str(masked), "-format", "%[fx:mean.b]", "info:"],
            capture_output=True, text=True, check=True).stdout)
        return 255 * core_b / max(share, 1e-6)

    cores = [logo_core(branded, when) for when in stamps]
    assert all(core > 120 for core in cores), f"the mark is missing somewhere: {cores}"
    # Constant, not merely present: the spread across the whole video stays small.
    assert max(cores) - min(cores) < 0.35 * max(cores), f"the mark pulses: {cores}"
    # And with branding off the same patch is just dark background, so the metric is
    # measuring the logo and not the picture behind it.
    bare = logo_core(plain, stamps[0])
    assert bare < 80, f"the unbranded control is not dark: {bare}"
    assert min(cores) > 1.5 * bare, f"branded {min(cores)} vs unbranded {bare}"


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


EVALUATOR_MPDECIMATE = "mpdecimate=hi=128:lo=64:frac=0.05"
"""Exactly what ``app.evaluate.metrics`` scores against, so the numbers are comparable."""


def _duplicate_ratio(clip: Path, mpdecimate: str = EVALUATOR_MPDECIMATE) -> float:
    """Fraction of frames in ``clip`` that mpdecimate calls a repeat of the previous one."""
    def count(vf: str | None) -> int:
        argv = [ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(clip), "-an"]
        if vf:
            argv += ["-vf", vf, "-fps_mode", "passthrough"]
        stderr = ff.run(argv + ["-f", "null", "-"])
        return max(
            int(line.split("frame=")[1].split()[0])
            for line in stderr.splitlines()
            if "frame=" in line and "fps=" in line
        )

    total = count(None)
    return max(0, total - count(mpdecimate)) / max(1, total)


@integration
def test_the_derived_upscale_removes_the_zoompan_stepping(assets, tmp_path):
    """A slow pan with no headroom repeats frames outright; with the derived canvas it
    does not. Measured with the evaluator's own mpdecimate settings.

    zoompan truncates x/y to integers in *canvas* pixels, so with canvas == region a
    slow pan sits on the same offset for several frames. Duplicate frames are that
    artefact, counted.
    """
    plan = VisualPlan(
        layout=SlideLayout.FULL_BLEED,
        motion=Motion.PAN_RIGHT,
        zoom_from=1.05,
        zoom_to=1.05,
        easing="linear",
    )
    pan_fps, pan_seconds = 30, 4.0
    profile = RenderProfile(width=640, height=360, fps=pan_fps, upscale_factor=4, crf=20)
    region = layout_region(plan, profile)

    derived = FFmpegBackend(text_mode="scrim").render_scene(
        assets["landscape"], plan, "", pan_seconds, tmp_path / "derived.mp4", profile
    )
    # The control: force the canvas back to the region, i.e. no sub-pixel headroom at all.
    import app.render.ffmpeg_backend as backend_module

    original = backend_module.motion_canvas
    region_only = backend_module.MotionCanvas(
        fit=(profile.width, profile.height),
        canvas=(profile.width, profile.height),
        detail=1,
    )
    backend_module.motion_canvas = lambda *a, **k: region_only
    try:
        none = FFmpegBackend(text_mode="scrim").render_scene(
            assets["landscape"], plan, "", pan_seconds, tmp_path / "flat.mp4", profile
        )
    finally:
        backend_module.motion_canvas = original

    without, with_headroom = _duplicate_ratio(none), _duplicate_ratio(derived)
    assert without > 0.2, f"expected gross stepping with no headroom, got {without:.3f}"
    assert with_headroom < without / 2, (
        f"derived canvas did not help: {with_headroom:.3f} vs {without:.3f}"
    )
    # And the derivation put the headroom on the travel axis, not both.
    sizing = motion_canvas(
        plan, region, int(pan_fps * pan_seconds), profile, src_size=(640, 360)
    )
    assert sizing.canvas[0] / region.width >= sizing.canvas[1] / region.height


def _pixel(frame: Path, x: int, y: int) -> tuple[int, int, int]:
    """One RGB pixel out of an image, straight from ffmpeg's raw output.

    Reading pixels beats reading ``signalstats`` metadata: the metadata filter prints
    at INFO level, which the wrapper's ``-loglevel error`` swallows.
    """
    raw = subprocess.run(  # noqa: S603
        [
            ff.ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error",
            "-i", str(frame), "-vf", f"crop=2:2:{x}:{y},format=rgb24",
            "-f", "rawvideo", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return raw[0], raw[1], raw[2]


@integration
def test_text_layers_really_appear_one_at_a_time_in_a_rendered_clip(assets, tmp_path):
    """The proof the unit tests cannot give: render it and count what is on screen.

    Three bars enter at 0.3s / 0.9s / 1.5s over a solid background. A passing
    expression test does not mean ffmpeg agreed, so this reads the actual pixels.
    """
    profile = RenderProfile(width=320, height=180, fps=30, upscale_factor=1, crf=20)
    layers = []
    for index in range(3):
        png = tmp_path / f"bar{index}.png"
        ff.ffmpeg(
            ["-f", "lavfi", "-i", "color=c=white:s=120x20,format=rgba", "-frames:v", "1", png]
        )
        layers.append(
            TextLayer(
                png_path=png,
                x=40,
                y=30 + index * 40,
                width=120,
                height=20,
                appear_at=0.3 + index * 0.6,
                animation=TextAnimation.SLIDE_LEFT,
                anim_duration=0.25,
                slide_distance=30,
                kind="bullet",
            )
        )

    clip = FFmpegBackend(text_mode="scrim").render_scene(
        None,
        VisualPlan(layout=SlideLayout.TITLE_CARD, motion=Motion.STATIC),
        "",
        2.2,
        tmp_path / "anim.mp4",
        profile,
        scene_text=SceneText(layers=layers),
    )

    def white_bars(t: float) -> int:
        """How many of the three bar rows are lit at time ``t``."""
        frame = tmp_path / f"f{t}.png"
        ff.ffmpeg(["-ss", str(t), "-i", clip, "-frames:v", "1", frame])
        return sum(
            _pixel(frame, 100, 38 + index * 40)[0] > 128 for index in range(3)
        )

    assert white_bars(0.1) == 0, "nothing may be visible before the first appear_at"
    assert white_bars(0.7) == 1
    assert white_bars(1.3) == 2
    assert white_bars(2.0) == 3, "every layer must be on screen by the end"


@integration
def test_a_rendered_slide_keeps_the_solid_background_outside_the_image_region(assets, tmp_path):
    """A hero panel must not bleed into the frame: the margin stays brand colour."""
    profile = RenderProfile(width=640, height=360, fps=24, upscale_factor=2, crf=20)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN)
    region = layout_region(plan, profile)
    clip = FFmpegBackend(text_mode="scrim", theme=Theme(bg="#101010")).render_scene(
        assets["landscape"], plan, "", 0.5, tmp_path / "hero.mp4", profile
    )

    frame = tmp_path / "hero.png"
    ff.ffmpeg(["-ss", "0.2", "-i", clip, "-frames:v", "1", frame])

    margin = _pixel(frame, max(0, region.x - 16), region.y + 8)
    inside = _pixel(frame, region.x + region.width // 2, region.y + region.height // 2)
    assert max(margin) < 40, f"margin should be the near-black theme colour, got {margin}"
    assert inside != margin, "the image region must actually contain the image"


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


def test_the_per_process_thread_cap_reaches_the_concurrent_scene_encode():
    """render_all divides the box between workers; the scene encode is the process
    that runs concurrently, so the cap has to land there and not only on assemble."""
    profile = RenderProfile(encoder_threads=3)
    assert FFmpegBackend._thread_args(profile) == ["-threads", "3"]
    assert FFmpegBackend._thread_args(RenderProfile()) == [], "0/None means let ffmpeg decide"


# ======================================================= master bus: duration

def _master_chain_of(backend: FFmpegBackend, total: float) -> str:
    parts, label = backend._master_chain("aout", total=total)
    assert label == "[amaster]"
    return ";".join(parts)


def test_the_master_bus_rebases_timestamps_before_it_clamps_the_length():
    """The 2.4-frame drift regression, pinned at the builder level.

    `loudnorm` re-bases timestamps off its own internal block clock instead of passing the
    input's through. `atrim` selects on timestamps, so once loudnorm has moved them the
    clamp no longer lines up with the samples and leaks. Measured on the reference render,
    loudnorm alone put +0.066992s (+2.01 frames at 30fps) past an `atrim=0:74.633008`, and
    a container is as long as its longest stream. `asetpts=N/SR/TB` regenerates the clock
    from the sample count, so the clamp means what it says.
    """
    chain = _master_chain_of(FFmpegBackend(text_mode="scrim"), 74.633008)
    if "loudnorm" not in chain:
        pytest.skip("this ffmpeg has no loudnorm, so there is nothing to rebase")

    assert "asetpts=N/SR/TB" in chain, "the clamp is meaningless without a sample-exact clock"
    assert chain.index("loudnorm") < chain.index("asetpts=N/SR/TB"), "rebase after loudnorm"
    assert chain.index("asetpts=N/SR/TB") < chain.index("atrim="), "rebase before the clamp"
    assert "atrim=0:74.633008" in chain
    assert "apad=whole_dur=74.633008" in chain
    assert chain.index("atrim=") < chain.index("apad="), "trim long, then pad short"


def test_the_logo_is_not_involved_in_the_master_bus_at_all():
    """Bisected: branding was the other suspect for the drift and it is duration-neutral."""
    branded = _master_chain_of(FFmpegBackend(text_mode="scrim"), 10.0)
    plain = _master_chain_of(FFmpegBackend(text_mode="scrim", logo_path=None), 10.0)
    assert branded == plain


# ============================================================= generated clips

VEO_CLIP = Path(
    "/private/tmp/claude-501/-Users-argo-ab-prompt-to-video-v2/"
    "17f5789b-d93a-4c4f-af36-254d779b6e1c/scratchpad/veo_clip.mp4"
)
needs_veo = pytest.mark.skipif(not VEO_CLIP.is_file(), reason="no Veo fixture clip on disk")


def test_a_clip_shorter_than_the_scene_loops_enough_times_to_cover_it():
    """n passes crossfaded at the seam yield n*clip - (n-1)*seam, exactly like xfade
    between scenes. Getting this wrong is how a scene ends on a frozen frame."""
    clip = 8.0
    for needed in (4.0, 8.0, 8.5, 14.0, 19.0, 20.0, 24.0, 60.0):
        loops = clip_loop_count(clip, needed)
        span = clip_loop_span(clip, loops)
        assert span >= needed - 1e-9, f"{needed}s needs more than {loops} passes ({span}s)"
        if loops > 1:
            assert clip_loop_span(clip, loops - 1) < needed, (
                f"{loops} passes is one more than {needed}s actually needs"
            )

    assert clip_loop_count(8.0, 8.0) == 1, "an exact fit must not loop at all"
    assert clip_loop_count(8.0, 4.0) == 1, "a clip longer than the scene never loops"
    assert clip_loop_count(0.0, 20.0) == 1, "an unprobeable clip must not loop forever"


def test_the_seam_crossfade_cannot_swallow_a_short_clip():
    assert clip_seam(8.0) == pytest.approx(CLIP_SEAM_CROSSFADE)
    assert clip_seam(1.0) == pytest.approx(0.25), "a 1s clip cannot give up half a second"
    assert clip_seam(0.0) == 0.0


def _clip_graph(backend: FFmpegBackend, plan: VisualPlan, profile: RenderProfile,
                *, frames: int, src_size: tuple[int, int] = (1280, 720),
                clip_duration: float = 8.0) -> str:
    return backend._scene_graph(
        src_size=src_size,
        plan=plan,
        profile=profile,
        frames=frames,
        text_layout=None,
        heading="",
        has_image_input=True,
        clip_duration=clip_duration,
        clip_fps=24.0,
    )


def test_a_clip_scene_never_gets_a_camera_move_on_top_of_the_footage():
    """A zoompan over moving footage reads as seasick. The clip supplies the movement."""
    for motion in Motion:
        plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=motion)
        graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=600)
        assert "zoompan" not in graph, f"{motion} must not add a move to a clip"


def test_a_clip_is_converted_to_the_timelines_frame_rate():
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=600)
    assert f"fps={HD.fps}" in graph
    # The cadence is normalised before anything is split, so every looped branch matches.
    assert graph.index("fps=") < graph.index("split=")


def test_a_clip_is_covered_and_centre_cropped_into_its_region_never_stretched():
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    region = layout_region(plan, HD)
    graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=600)

    assert "force_original_aspect_ratio=increase" in graph, "cover, so no letterbox"
    assert f"crop={region.width}:{region.height}" in graph, "then centre-crop the excess"
    assert f"overlay=x={region.x}:y={region.y}" in graph
    # No filter may set both dimensions of the footage without an aspect-preserving flag.
    for match in re.findall(r"scale=(\d+):(\d+)(:[^,;\[]*)?", graph):
        options = match[2] or ""
        assert "force_original_aspect_ratio" in options or "flags=bilinear" in options, match


def test_a_clip_scene_references_only_the_video_stream_of_its_input():
    """Defence in depth against the provider's AAC track: narration is authoritative."""
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=600)
    assert "[0:a]" not in graph
    assert "[0:v]" in graph


def test_a_clip_scene_crossfades_its_seams_rather_than_cutting_or_freezing():
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=600)  # 20s
    loops = clip_loop_count(8.0, 20.0)

    assert f"split={loops}" in graph
    assert graph.count("xfade=transition=fade") == loops - 1
    for index in range(1, loops):
        assert f"offset={index * (8.0 - CLIP_SEAM_CROSSFADE):.6f}" in graph
    # tpad only ever adds frames past the end; -frames:v does the cutting.
    assert "tpad=stop_mode=clone" in graph
    assert "setpts=PTS-STARTPTS" in graph


def test_a_clip_that_already_fills_the_scene_is_not_looped():
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    graph = _clip_graph(FFmpegBackend(text_mode="scrim"), plan, HD, frames=150)  # 5s
    assert "xfade" not in graph
    assert "split=" not in graph


@pytest.mark.parametrize(
    ("layout", "factor"),
    [
        # Measured. A 720p clip has to be *upscaled* for every layout at 1080p -- including
        # the hero panel, which is only 720 wide but 900 tall, so it is the panel's height
        # and not its width that runs out of pixels. Worth pinning: "1280x720 into an ~857px
        # region is plenty" is the intuition, and it is wrong.
        (SlideLayout.HERO_RIGHT, "1.25x"),
        (SlideLayout.IMAGE_BAND, "1.50x"),
        (SlideLayout.FULL_BLEED, "1.50x"),
    ],
)
def test_a_clip_that_cannot_fill_its_region_is_flagged_as_an_upscale(caplog, layout, factor):
    region = layout_region(VisualPlan(layout=layout), HD)
    with caplog.at_level("WARNING"):
        FFmpegBackend._warn_if_clip_is_upscaled((1280, 720), region, VisualPlan(layout=layout))
    warnings = [r.getMessage() for r in caplog.records if "upscale" in r.getMessage()]
    assert warnings, f"{layout.value} needs a {factor} upscale and must not pass silently"
    assert factor in warnings[0], warnings[0]


def test_a_clip_with_pixels_to_spare_is_not_flagged(caplog):
    """At draft resolution the same clip is a downscale everywhere, so nothing is said."""
    draft = RenderProfile.draft()
    for layout in (SlideLayout.HERO_RIGHT, SlideLayout.IMAGE_BAND, SlideLayout.FULL_BLEED):
        region = layout_region(VisualPlan(layout=layout), draft)
        caplog.clear()
        with caplog.at_level("WARNING"):
            FFmpegBackend._warn_if_clip_is_upscaled(
                (1280, 720), region, VisualPlan(layout=layout)
            )
        assert not [r for r in caplog.records if "upscale" in r.getMessage()], layout
    # And an unprobeable clip says nothing rather than dividing by zero.
    caplog.clear()
    with caplog.at_level("WARNING"):
        FFmpegBackend._warn_if_clip_is_upscaled((0, 0), region, VisualPlan())
    assert not caplog.records


def test_render_all_rejects_a_scene_with_neither_a_still_nor_a_clip(tmp_path):
    """The validation gate must name both options now that there are two."""
    from app.render.ffmpeg_backend import RenderError

    timeline = make_timeline([5.0])
    timeline.scenes[0].plan = VisualPlan(layout=SlideLayout.HERO_RIGHT)

    with pytest.raises(RenderError, match="image_path or video_path"):
        FFmpegBackend(text_mode="scrim").render_all(timeline, tmp_path)


@integration
@needs_veo
def test_render_all_accepts_a_scene_whose_visual_is_only_a_clip(tmp_path):
    """A clip scene has no image_path at all, and that must be allowed through."""
    profile = RenderProfile(width=320, height=180, fps=30, upscale_factor=2, crf=28)
    timeline = make_timeline([5.0])
    timeline.scenes[0].plan = VisualPlan(
        layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN,
        transition_in=Transition.CUT, transition_duration=0.0,
    )
    timeline.scenes[0].image_path = None
    timeline.scenes[0].video_path = str(VEO_CLIP)
    timeline.profile = profile

    rendered = FFmpegBackend(text_mode="scrim").render_all(timeline, tmp_path)
    clip = Path(rendered.scenes[0].clip_path)
    assert clip.is_file()
    assert ff.count_frames(clip) == frames_for(5.0, profile.fps)


@integration
@needs_veo
def test_the_veo_fixture_still_has_the_properties_this_code_assumes(tmp_path):
    """Hard constraints, asserted rather than remembered."""
    summary = ff.probe_summary(VEO_CLIP)
    assert summary["duration"] == pytest.approx(8.0, abs=0.01)
    assert (summary["width"], summary["height"]) == (1280, 720)
    assert summary["fps"] == pytest.approx(24.0, abs=0.01)
    assert summary["audio_codec"] == "aac", "the track this code exists to discard"


@integration
@needs_veo
@pytest.mark.parametrize("seconds", [6.0, 8.0, 14.0, 20.0])
def test_a_clip_scene_is_exactly_as_many_frames_as_a_still_scene(tmp_path, seconds):
    """Where a bug was most expected: the clip path must hit the same frame count.

    A still input is infinite (`-loop 1`), so `-frames:v` can always be satisfied. A clip
    is finite, and a graph that came up one frame short would silently write a shorter
    file, desyncing the narration from that scene onwards.
    """
    profile = RenderProfile(width=640, height=360, fps=30, upscale_factor=2, crf=24)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN)
    backend = FFmpegBackend(text_mode="scrim")

    clip = backend.render_scene(
        None, plan, "", seconds, tmp_path / f"clip{seconds}.mp4", profile,
        video_path=VEO_CLIP,
    )

    want = frames_for(seconds, profile.fps)
    assert ff.count_frames(clip) == want, "the clip path is off the frame grid"
    assert ff.probe_duration(clip) == pytest.approx(want / profile.fps, abs=1.0 / profile.fps)
    summary = ff.probe_summary(clip)
    assert (summary["width"], summary["height"]) == (profile.width, profile.height)
    assert summary["audio_codec"] is None, "the clip's AAC track reached the output"


@integration
@needs_veo
def test_a_clip_scene_and_a_still_scene_agree_frame_for_frame(assets, tmp_path):
    """Same timing contract, different visual source."""
    profile = RenderProfile(width=640, height=360, fps=30, upscale_factor=2, crf=24)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.ZOOM_IN)
    backend = FFmpegBackend(text_mode="scrim")

    still = backend.render_scene(
        assets["landscape"], plan, "", 17.0, tmp_path / "still.mp4", profile
    )
    moving = backend.render_scene(
        assets["landscape"], plan, "", 17.0, tmp_path / "moving.mp4", profile,
        video_path=VEO_CLIP,
    )
    assert ff.count_frames(still) == ff.count_frames(moving)
    assert ff.probe_duration(still) == pytest.approx(ff.probe_duration(moving), abs=1e-3)


@integration
@needs_veo
def test_a_looped_clip_never_freezes_and_never_repeats_a_frame_at_the_seam(tmp_path):
    """The shortfall-coverage choice, measured rather than asserted.

    Holding the final frame would park ~60% of a 20s scene on one image; slowing the clip
    down would hold every frame for three. Both show up as duplicate frames. A crossfaded
    loop keeps moving throughout, so the ratio stays at the encoder's noise floor.
    """
    profile = RenderProfile(width=640, height=360, fps=30, upscale_factor=2, crf=20)
    plan = VisualPlan(layout=SlideLayout.HERO_RIGHT, motion=Motion.STATIC)
    clip = FFmpegBackend(text_mode="scrim").render_scene(
        None, plan, "", 20.0, tmp_path / "looped.mp4", profile, video_path=VEO_CLIP
    )

    ratio = _duplicate_ratio(clip)
    # Measured floor for any clip scene: the 24 -> 30 fps resample alone contributes
    # 34/240 = 14.2% on this fixture, and the whole looped 20s scene measures ~11%.
    # A final-frame hold would put 12 of the 20 seconds at 100%, i.e. ~0.60 overall.
    assert ratio < 0.25, f"the scene freezes or judders somewhere: {ratio:.3f}"

    # And specifically: nothing is frozen *past the 8s mark*, which is exactly where a
    # final-frame hold would park for the remaining 12 seconds.
    def frame_at(when: float) -> bytes:
        png = tmp_path / f"probe{when:.2f}.png"
        ff.ffmpeg(["-ss", f"{when:.3f}", "-i", clip, "-frames:v", "1", "-update", "1", png])
        return png.read_bytes()

    for when in (9.0, 11.0, 13.0, 15.0, 17.0, 19.0):
        assert frame_at(when) != frame_at(when + 0.5), (
            f"the picture is identical at {when}s and {when + 0.5}s -- it has frozen"
        )
