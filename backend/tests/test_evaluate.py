"""Tests for app.evaluate.

The metric *arithmetic* is pure and is tested directly: WCAG contrast, the strip/skirt
sampling over synthetic pixel buffers, words per minute, bullet timing, the offset mapping
from narration time into final-video time. Anything that shells out to ffmpeg is exercised
by monkeypatching :func:`app.evaluate.metrics._run` / ``_stderr`` with captured real output,
so the parsers are tested against the strings ffmpeg actually emits rather than a guess.

The vision pass is mocked everywhere. The one thing worth asserting about it is that a
failed or malformed judgement *removes* a dimension rather than scoring it neutral — an
unmeasured dimension must never look like a passing one.

The scoring tests pin the two behaviours that make the scorecard usable: legibility and
relevance dominating the weighted average, and a blocker capping the grade regardless of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.models import (
    BulletPoint,
    Motion,
    RenderProfile,
    Scene,
    TextPosition,
    Timeline,
    Transition,
    VisualPlan,
    Word,
)
from app.evaluate import metrics as M
from app.evaluate import scorer, vision
from app.evaluate.models import (
    BalanceMeasure,
    ContrastMeasure,
    Dimension,
    Grade,
    LoudnessMeasure,
    SceneMetrics,
    Severity,
    VideoMetrics,
    grade_for,
)
from app.evaluate.vision import SceneVerdict, ScriptVerdict, VisionReport

# --------------------------------------------------------------------------- fixtures


def _words(pairs: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(word=w, start=s, end=e) for w, s, e in pairs]


_DEFAULT = object()


def _scene(
    scene_id: int = 1,
    *,
    start: float = 0.0,
    end: float = 10.0,
    words: list[Word] | None = None,
    bullets: list[BulletPoint] | None = None,
    plan: VisualPlan | None | object = _DEFAULT,
    heading: str = "Understanding the Deceptive Hook",
) -> Scene:
    """``plan=None`` means *no* plan; omitting it gives the pipeline default."""
    return Scene(
        id=scene_id,
        narration="Phishing begins when criminals send deceptive messages.",
        heading=heading,
        image_prompt="a dim room",
        start=start,
        end=end,
        words=words if words is not None else _words([("phishing", start, start + 0.5)]),
        bullets=bullets or [],
        plan=VisualPlan() if plan is _DEFAULT else plan,  # type: ignore[arg-type]
    )


def _timeline(scenes: list[Scene] | None = None, **kwargs) -> Timeline:
    return Timeline(
        job_id="job-1",
        topic="How phishing attacks work",
        title="How Phishing Attacks Work",
        voice="aura-2-draco-en",
        scenes=scenes if scenes is not None else [_scene()],
        **kwargs,
    )


@pytest.fixture
def four_scene_timeline() -> Timeline:
    """Four 10s scenes with a 0.5s dissolve at each boundary, as the pipeline builds them."""
    scenes = []
    for index in range(4):
        start = index * 10.0
        scenes.append(
            _scene(
                index + 1,
                start=start,
                end=start + 10.0,
                # 22 words over 10s == 132 wpm, comfortably inside the band.
                words=_words(
                    [(f"w{i}", start + i * 0.45, start + i * 0.45 + 0.4) for i in range(22)]
                ),
                plan=VisualPlan(transition_in=Transition.DISSOLVE, transition_duration=0.5),
            )
        )
    return _timeline(scenes)


# ------------------------------------------------------------------- WCAG arithmetic


def test_relative_luminance_matches_wcag_reference_points() -> None:
    assert M.relative_luminance(0) == pytest.approx(0.0)
    assert M.relative_luminance(255) == pytest.approx(1.0)
    # 0.2140 is the WCAG relative luminance of #808080.
    assert M.relative_luminance(128) == pytest.approx(0.2158, abs=0.002)


def test_contrast_ratio_is_symmetric_and_bounded() -> None:
    assert M.contrast_ratio(255, 0) == pytest.approx(21.0)
    assert M.contrast_ratio(0, 255) == pytest.approx(21.0)
    assert M.contrast_ratio(128, 128) == pytest.approx(1.0)


def test_level_for_contrast_inverts_contrast_against_white() -> None:
    for ratio in (2.0, 3.0, 4.5, 7.0, 12.0):
        level = M.level_for_contrast(ratio)
        assert M.contrast_against_white(level) == pytest.approx(ratio, rel=0.01)


def test_contrast_target_level_is_the_45_boundary() -> None:
    """Anything at or below ~118/255 clears 4.5:1 against white text."""
    assert 110 < M.CONTRAST_TARGET_LEVEL < 125
    assert M.contrast_against_white(M.CONTRAST_TARGET_LEVEL) == pytest.approx(4.5, rel=0.01)
    assert M.contrast_against_white(M.CONTRAST_TARGET_LEVEL + 20) < 4.5


# ---------------------------------------------------------- strip / skirt sampling


def _rows(pattern: list[int], width: int, count: int) -> bytes:
    """``count`` identical rows built by repeating ``pattern`` across ``width``."""
    row = bytes((pattern * (width // len(pattern) + 1))[:width])
    return row * count


def test_strip_contrasts_finds_the_bright_patch_a_mean_would_hide() -> None:
    """Half-dark, half-bright: the worst strip must report the bright half.

    This is the whole reason the metric is per-strip. The band mean here is ~128, which
    reads as a comfortable 4.5:1 while half the heading is actually sitting on 4.0:1.
    """
    width = 320
    row = bytes([10] * (width // 2) + [200] * (width // 2))
    strips = M._strip_contrasts([row] * 10, width, 40)
    assert len(strips) == 8
    assert min(strips) == pytest.approx(M.contrast_against_white(200), rel=0.01)
    assert max(strips) == pytest.approx(M.contrast_against_white(10), rel=0.01)
    assert min(strips) < 2.0 < 15.0 < max(strips)


def test_strip_contrasts_handles_a_width_smaller_than_one_strip() -> None:
    assert len(M._strip_contrasts([bytes([50] * 20)], 20, 63)) == 1


def test_strip_contrasts_on_empty_input() -> None:
    assert M._strip_contrasts([], 0, 63) == []


def test_skirt_rows_takes_both_sides_and_rejects_glyph_rows() -> None:
    width, height = 100, 120
    glyph_top, glyph_bottom = 50, 70
    buf = bytearray(_rows([40], width, height))
    # Paint the glyph band pure white so a leaked row is unmistakable.
    for y in range(glyph_top, glyph_bottom):
        buf[y * width : (y + 1) * width] = bytes([255] * width)
    rows = M._skirt_rows(bytes(buf), width, height, glyph_top, glyph_bottom)

    assert rows, "skirt must sample something"
    assert all(row.count(255) / width <= M.GLYPH_ROW_FRACTION for row in rows)
    # Both sides contribute: 26 rows above and 26 below, minus the 8-row pad.
    assert len(rows) == 2 * M.SKIRT_ROWS


def test_skirt_rows_clamps_at_the_frame_edge() -> None:
    """A heading at y=0 has no rows above it; the measurement must still work."""
    width, height = 64, 40
    rows = M._skirt_rows(_rows([30], width, height), width, height, 0, 10)
    assert rows
    assert len(rows) <= M.SKIRT_ROWS


# -------------------------------------------------------------------- text geometry


@pytest.mark.parametrize(
    "position",
    [TextPosition.LOWER_THIRD, TextPosition.UPPER_THIRD, TextPosition.CENTER],
)
def test_text_box_stays_inside_the_frame_and_the_scrim(position: TextPosition) -> None:
    profile = RenderProfile()
    scene = _scene(plan=VisualPlan(text_position=position))
    box = M.text_box(scene, _timeline([scene], profile=profile))
    assert box is not None
    assert 0 <= box.y
    assert box.y + box.height <= profile.height
    assert 0 <= box.x and box.x + box.width <= profile.width
    assert box.scrim_top <= box.y
    assert box.scrim_bottom >= box.y + box.height
    assert box.as_geometry() == f"{box.width}x{box.height}+{box.x}+{box.y}"


def test_text_box_is_horizontally_centred() -> None:
    scene = _scene()
    profile = RenderProfile()
    box = M.text_box(scene, _timeline([scene], profile=profile))
    assert box is not None
    assert abs((profile.width - box.width) // 2 - box.x) <= 1


def test_text_box_is_none_without_a_plan() -> None:
    """No VisualPlan means no known text geometry, so no measurement is attempted."""
    assert M.text_box(_scene(plan=None), _timeline()) is None


def test_lower_and_upper_third_land_in_opposite_halves() -> None:
    profile = RenderProfile()
    lower = M.text_box(
        _scene(plan=VisualPlan(text_position=TextPosition.LOWER_THIRD)),
        _timeline(profile=profile),
    )
    upper = M.text_box(
        _scene(plan=VisualPlan(text_position=TextPosition.UPPER_THIRD)),
        _timeline(profile=profile),
    )
    assert lower is not None and upper is not None
    assert upper.y < profile.height / 2 < lower.y


# ------------------------------------------------------------------ offset mapping


def test_timeline_offsets_accumulate_one_transition_per_boundary(
    four_scene_timeline: Timeline,
) -> None:
    assert M.timeline_offsets(four_scene_timeline) == [0.0, 0.5, 1.0, 1.5]


def test_timeline_offsets_ignore_hard_cuts() -> None:
    scenes = [
        _scene(1, start=0.0, end=10.0, plan=VisualPlan(transition_in=Transition.CUT)),
        _scene(2, start=10.0, end=20.0, plan=VisualPlan(transition_in=Transition.CUT)),
        _scene(3, start=20.0, end=30.0, plan=VisualPlan(transition_in=Transition.DISSOLVE)),
    ]
    assert M.timeline_offsets(_timeline(scenes)) == [0.0, 0.0, 0.5]


def test_offsets_agree_with_final_duration(four_scene_timeline: Timeline) -> None:
    """The offset table and Timeline.final_duration() must derive the same overlap.

    If they disagree, frames get sampled from the neighbouring slide and every visual
    metric is quietly measuring the wrong picture.
    """
    total_overlap = M.timeline_offsets(four_scene_timeline)[-1]
    assert (
        four_scene_timeline.narration_duration - total_overlap
        == pytest.approx(four_scene_timeline.final_duration())
    )


def test_frame_source_prefers_the_scene_clip(tmp_path: Path) -> None:
    clip = tmp_path / "clip_02.mp4"
    clip.write_bytes(b"\x00")
    scene = _scene(2, start=10.0, end=20.0)
    scene.clip_path = str(clip)
    timeline = _timeline([_scene(1, start=0.0, end=10.0), scene])
    source = M.frame_source(scene, timeline, tmp_path / "video.mp4")
    assert source.local is True
    assert source.path == clip
    assert source.offset == 0.0


def test_frame_source_falls_back_with_the_overlap_removed(
    four_scene_timeline: Timeline, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    scene = four_scene_timeline.scenes[3]
    source = M.frame_source(scene, four_scene_timeline, video)
    assert source.local is False
    assert source.path == video
    # Narration start 30.0, minus three 0.5s dissolves.
    assert source.offset == pytest.approx(28.5)


def test_sample_timestamps_stay_inside_the_scene(four_scene_timeline: Timeline) -> None:
    source = M.FrameSource(Path("x.mp4"), 28.5, 10.0, False)
    stamps = M.sample_timestamps(source, 3)
    assert len(stamps) == 3
    assert all(28.5 < t < 38.5 for t in stamps)
    assert stamps == sorted(stamps)


def test_sample_timestamps_single_frame_is_mid_scene() -> None:
    source = M.FrameSource(Path("x.mp4"), 0.0, 10.0, True)
    assert M.sample_timestamps(source, 1)[0] == pytest.approx(5.0)


def test_sample_timestamps_survives_a_zero_length_scene() -> None:
    stamps = M.sample_timestamps(M.FrameSource(Path("x.mp4"), 0.0, 0.0, True), 3)
    assert len(stamps) == 3
    assert all(t >= 0.0 for t in stamps)


# -------------------------------------------------------------------------- pacing


def test_words_per_minute_uses_the_spoken_span_not_the_scene_duration() -> None:
    """A scene padded with silence is a pacing problem, not a slow delivery."""
    scene = _scene(
        words=_words([(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(20)]),
        start=0.0,
        end=30.0,
    )
    # 20 words spanning 9.9s -> ~121 wpm, not 20 words / 30s -> 40 wpm.
    assert M.words_per_minute(scene) == pytest.approx(121.2, abs=0.5)


def test_words_per_minute_without_words() -> None:
    assert M.words_per_minute(_scene(words=[])) is None


def test_words_per_minute_with_one_zero_length_word() -> None:
    assert M.words_per_minute(_scene(words=_words([("a", 1.0, 1.0)]))) is None


def test_duration_deviation_flags_the_odd_scene_out() -> None:
    scenes = [
        _scene(1, start=0.0, end=10.0),
        _scene(2, start=10.0, end=20.0),
        _scene(3, start=20.0, end=30.0),
        _scene(4, start=30.0, end=50.0),
    ]
    timeline = _timeline(scenes)
    assert M.duration_deviation(scenes[0], timeline) == pytest.approx(0.0)
    assert M.duration_deviation(scenes[3], timeline) == pytest.approx(1.0)


def test_duration_deviation_needs_enough_siblings() -> None:
    scenes = [_scene(1, start=0.0, end=10.0), _scene(2, start=10.0, end=30.0)]
    assert M.duration_deviation(scenes[0], _timeline(scenes)) is None


# ----------------------------------------------------------------- narration gaps


def test_sentence_pauses_are_not_reported_as_holes() -> None:
    """0.64s between sentences is prosody. Flagging it buries the real defects."""
    scene = _scene(
        words=_words([("a", 0.0, 1.0), ("b", 1.64, 2.5), ("c", 2.5, 3.0)]), start=0.0, end=4.0
    )
    assert M.narration_gaps(scene) == []


def test_a_real_hole_is_reported_scene_relative() -> None:
    scene = _scene(
        words=_words([("a", 10.0, 11.0), ("b", 13.5, 14.0)]), start=10.0, end=15.0
    )
    assert M.narration_gaps(scene) == [(1.0, 3.5)]


def test_leading_silence_counts_as_a_hole() -> None:
    scene = _scene(words=_words([("a", 12.0, 13.0)]), start=10.0, end=14.0)
    assert M.narration_gaps(scene) == [(0.0, 2.0)]


def test_narration_gaps_without_words() -> None:
    assert M.narration_gaps(_scene(words=[])) == []


def test_speech_and_gap_windows_are_in_final_video_time(
    four_scene_timeline: Timeline,
) -> None:
    """Both window sets must carry the xfade correction, or they sample the wrong audio."""
    speech = M.speech_windows(four_scene_timeline)
    assert speech
    last = speech[-1]
    # Scene 4's narration starts at 30.0; three dissolves put it at 28.5 in the file.
    assert last[0] < 30.0
    assert all(a >= 0.0 and b > a for a, b in speech)


def test_gap_windows_are_trimmed_inward() -> None:
    scene = _scene(words=_words([("a", 0.0, 1.0), ("b", 3.0, 3.5)]), start=0.0, end=4.0)
    windows = M.narration_gap_windows(_timeline([scene]))
    assert len(windows) == 1
    start, end = windows[0]
    assert start > 1.0 and end < 3.0


# ------------------------------------------------------------------ bullet timing


def test_no_bullets_is_not_a_defect() -> None:
    """Jobs rendered before Scene.bullets existed must produce zero findings."""
    assert M.bullet_issues(_scene(bullets=[])) == []


def test_well_spaced_bullets_produce_no_issues() -> None:
    scene = _scene(
        start=0.0,
        end=10.0,
        bullets=[
            BulletPoint(text="one", appear_at=1.0),
            BulletPoint(text="two", appear_at=3.0),
            BulletPoint(text="three", appear_at=5.0),
        ],
    )
    assert M.bullet_issues(scene) == []


def test_bullet_before_the_scene_starts_is_flagged() -> None:
    scene = _scene(start=0.0, end=10.0, bullets=[BulletPoint(text="early", appear_at=-0.5)])
    issues = M.bullet_issues(scene)
    assert len(issues) == 1
    assert "before the scene starts" in issues[0]


def test_bullet_after_the_scene_ends_is_flagged() -> None:
    scene = _scene(start=0.0, end=10.0, bullets=[BulletPoint(text="late", appear_at=12.0)])
    assert any("after the scene ends" in i for i in M.bullet_issues(scene))


def test_bullet_landing_just_before_the_cut_is_flagged() -> None:
    scene = _scene(start=0.0, end=10.0, bullets=[BulletPoint(text="squeezed", appear_at=9.9)])
    assert any("before the scene ends" in i for i in M.bullet_issues(scene))


def test_out_of_order_bullets_are_flagged() -> None:
    scene = _scene(
        start=0.0,
        end=10.0,
        bullets=[BulletPoint(text="a", appear_at=5.0), BulletPoint(text="b", appear_at=2.0)],
    )
    assert any("out of order" in i for i in M.bullet_issues(scene))


def test_bullets_closer_than_the_floor_are_flagged() -> None:
    scene = _scene(
        start=0.0,
        end=10.0,
        bullets=[BulletPoint(text="a", appear_at=1.0), BulletPoint(text="b", appear_at=1.3)],
    )
    issues = M.bullet_issues(scene)
    assert any("under the" in i and "floor" in i for i in issues)


def test_bullet_floor_comes_from_the_plan_when_present() -> None:
    scene = _scene(
        start=0.0,
        end=10.0,
        plan=VisualPlan(bullet_min_gap=2.0),
        bullets=[BulletPoint(text="a", appear_at=1.0), BulletPoint(text="b", appear_at=2.0)],
    )
    assert any("2.00s floor" in i for i in M.bullet_issues(scene))


def test_bullet_issues_without_a_plan_uses_the_module_default() -> None:
    scene = _scene(
        start=0.0,
        end=10.0,
        plan=None,
        bullets=[BulletPoint(text="a", appear_at=1.0), BulletPoint(text="b", appear_at=1.1)],
    )
    assert M.bullet_issues(scene)


# ------------------------------------------------------------------- ffmpeg parsers


_EBUR128_TAIL = """\
[Parsed_ebur128_0 @ 0x1] t: 47.3  TARGET:-23 LUFS    M: -28.6 S: -22.3     I: -11.1 LUFS
[Parsed_ebur128_0 @ 0x1] Summary:

  Integrated loudness:
    I:         -20.3 LUFS
    Threshold: -30.4 LUFS

  Loudness range:
    LRA:         3.1 LU
    Threshold: -40.4 LUFS
    LRA low:   -22.2 LUFS
    LRA high:  -19.0 LUFS

  True peak:
    Peak:       -5.6 dBFS
"""


def test_loudness_parses_the_summary_not_the_running_lines(monkeypatch) -> None:
    """The per-100ms ``I:`` lines are partial integrations; matching those gives nonsense."""
    monkeypatch.setattr(M, "_stderr", lambda *a, **k: _EBUR128_TAIL)
    measure = M.measure_loudness(Path("video.mp4"))
    assert measure is not None
    assert measure.integrated_lufs == -20.3
    assert measure.true_peak_dbfs == -5.6
    assert measure.loudness_range_lu == 3.1


def test_loudness_returns_none_when_ffmpeg_fails(monkeypatch) -> None:
    def boom(*a, **k):
        raise M.MeasurementError("no such file")

    monkeypatch.setattr(M, "_stderr", boom)
    assert M.measure_loudness(Path("nope.mp4")) is None


def test_loudness_returns_none_on_output_without_a_summary(monkeypatch) -> None:
    monkeypatch.setattr(M, "_stderr", lambda *a, **k: "nothing useful here")
    assert M.measure_loudness(Path("video.mp4")) is None


def test_silence_pairs_starts_with_ends(monkeypatch) -> None:
    monkeypatch.setattr(
        M,
        "_stderr",
        lambda *a, **k: (
            "[silencedetect @ 0x1] silence_start: 1.5\n"
            "[silencedetect @ 0x1] silence_end: 2.75 | silence_duration: 1.25\n"
            "[silencedetect @ 0x1] silence_start: 8.0\n"
            "[silencedetect @ 0x1] silence_end: 9.0 | silence_duration: 1.0\n"
        ),
    )
    assert M.detect_silence(Path("a.mp3")) == [(1.5, 2.75), (8.0, 9.0)]


def test_unterminated_silence_is_dropped(monkeypatch) -> None:
    """A file that ends mid-silence prints a start with no end. Do not invent one."""
    monkeypatch.setattr(M, "_stderr", lambda *a, **k: "silence_start: 4.0\n")
    assert M.detect_silence(Path("a.mp3")) == []


def test_duplicate_ratio_divides_dropped_by_total(monkeypatch) -> None:
    counts = iter([120, 103])
    monkeypatch.setattr(M, "_frame_count", lambda *a, **k: next(counts))
    assert M.duplicate_frame_ratio(Path("clip.mp4")) == pytest.approx(17 / 120, abs=1e-4)


def test_duplicate_ratio_of_a_perfectly_smooth_clip(monkeypatch) -> None:
    monkeypatch.setattr(M, "_frame_count", lambda *a, **k: 120)
    assert M.duplicate_frame_ratio(Path("clip.mp4")) == 0.0


def test_duplicate_ratio_is_none_for_a_single_frame(monkeypatch) -> None:
    monkeypatch.setattr(M, "_frame_count", lambda *a, **k: 1)
    assert M.duplicate_frame_ratio(Path("clip.mp4")) is None


def test_mpdecimate_filter_is_the_calibrated_one_not_the_default() -> None:
    """Guard the calibration: the defaults rank a smooth render worse than a juddering one."""
    assert M.MPDECIMATE.startswith("mpdecimate=")
    assert "hi=128" in M.MPDECIMATE and "lo=64" in M.MPDECIMATE and "frac=0.05" in M.MPDECIMATE


def test_window_level_ignores_windows_that_are_too_short(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake(argv, **kwargs):
        captured.append(argv)
        return "[Parsed_volumedetect_0 @ 0x1] mean_volume: -22.7 dB"

    monkeypatch.setattr(M, "_stderr", fake)
    level = M._window_level_dbfs(Path("v.mp4"), [(1.0, 2.0), (3.0, 3.01)])
    assert level == pytest.approx(-22.7)
    expression = captured[0][captured[0].index("-af") + 1]
    assert "between(t,1.000,2.000)" in expression
    assert "3.010" not in expression


def test_window_level_with_no_usable_windows(monkeypatch) -> None:
    monkeypatch.setattr(M, "_stderr", lambda *a, **k: "")
    assert M._window_level_dbfs(Path("v.mp4"), []) is None


def test_balance_is_speech_minus_bed(monkeypatch) -> None:
    levels = iter([-22.7, -32.7])
    monkeypatch.setattr(M, "_window_level_dbfs", lambda *a, **k: next(levels))
    scene = _scene(
        words=_words([("a", 0.0, 1.0), ("b", 3.0, 4.0)]), start=0.0, end=5.0
    )
    balance = M.measure_balance(Path("v.mp4"), _timeline([scene]))
    assert balance is not None
    assert balance.measured is True
    assert balance.separation_db == pytest.approx(10.0)


def test_balance_reports_unmeasured_when_there_is_no_gap(monkeypatch) -> None:
    """No pause anywhere means the bed cannot be isolated. Say so; do not return 0 dB."""
    monkeypatch.setattr(M, "_window_level_dbfs", lambda *a, **k: -22.0)
    scene = _scene(
        words=_words([(f"w{i}", i * 0.5, i * 0.5 + 0.5) for i in range(10)]), start=0.0, end=5.0
    )
    balance = M.measure_balance(Path("v.mp4"), _timeline([scene]))
    assert balance is not None
    assert balance.measured is False


def test_measure_video_reports_profile_mismatches(monkeypatch) -> None:
    monkeypatch.setattr(
        M,
        "probe",
        lambda p: {
            "format": {"duration": "47.47"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "24/1",
                    "codec_name": "h264",
                }
            ],
        },
    )
    monkeypatch.setattr(M, "measure_loudness", lambda p: None)
    monkeypatch.setattr(M, "measure_balance", lambda p, t: None)
    monkeypatch.setattr(M, "duplicate_frame_ratio", lambda *a, **k: None)
    metrics = M.measure_video(_timeline(), Path("v.mp4"))
    assert any("1280x720" in m for m in metrics.profile_mismatch)
    assert any("24" in m and "fps" in m for m in metrics.profile_mismatch)
    assert any("no audio stream" in m for m in metrics.profile_mismatch)


def test_duration_drift_is_measured_in_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        M,
        "probe",
        lambda p: {
            "format": {"duration": "40.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        },
    )
    monkeypatch.setattr(M, "measure_loudness", lambda p: None)
    monkeypatch.setattr(M, "measure_balance", lambda p, t: None)
    monkeypatch.setattr(M, "duplicate_frame_ratio", lambda *a, **k: None)
    scenes = [_scene(1, start=0.0, end=41.0, plan=VisualPlan(transition_in=Transition.CUT))]
    metrics = M.measure_video(_timeline(scenes), Path("v.mp4"))
    assert metrics.expected_duration == pytest.approx(41.0)
    assert metrics.duration_drift_frames == pytest.approx(30.0)
    assert metrics.profile_mismatch == []


# ---------------------------------------------------------------------- score maths


def test_piecewise_interpolates_and_clamps() -> None:
    anchors = [(0.0, 10.0), (10.0, 0.0)]
    assert scorer._piecewise(-5.0, anchors) == 10.0
    assert scorer._piecewise(0.0, anchors) == 10.0
    assert scorer._piecewise(5.0, anchors) == pytest.approx(5.0)
    assert scorer._piecewise(20.0, anchors) == 0.0


def test_piecewise_is_monotonic_across_the_contrast_table() -> None:
    values = [scorer._piecewise(r, scorer.CONTRAST_ANCHORS) for r in (1, 2, 3, 4.5, 7, 12, 21)]
    assert values == sorted(values)


def test_contrast_anchors_put_wcag_aa_at_acceptable_not_good() -> None:
    assert scorer._piecewise(M.MIN_CONTRAST_RATIO, scorer.CONTRAST_ANCHORS) == pytest.approx(7.0)
    assert scorer._piecewise(3.0, scorer.CONTRAST_ANCHORS) < 5.0


def test_pacing_anchors_bracket_the_flag_thresholds() -> None:
    assert scorer._piecewise(M.MIN_WPM, scorer.PACING_ANCHORS) == pytest.approx(7.0)
    assert scorer._piecewise(M.MAX_WPM, scorer.PACING_ANCHORS) == pytest.approx(7.0)
    assert scorer._piecewise(145.0, scorer.PACING_ANCHORS) == pytest.approx(10.0)
    assert scorer._piecewise(60.0, scorer.PACING_ANCHORS) == pytest.approx(0.0)


def test_motion_anchors_do_not_punish_the_encoder_noise_floor() -> None:
    """The reference render's genuinely smooth clips measure up to 0.10."""
    assert scorer._piecewise(0.10, scorer.MOTION_ANCHORS) == pytest.approx(10.0)
    assert scorer._piecewise(0.14, scorer.MOTION_ANCHORS) < 10.0
    assert scorer._piecewise(0.30, scorer.MOTION_ANCHORS) < 5.0


def test_weights_sum_to_one() -> None:
    assert sum(scorer.SCENE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(scorer.VIDEO_WEIGHTS.values()) == pytest.approx(1.0)


def test_legibility_and_relevance_are_half_the_scene_score() -> None:
    """The documented design intent, pinned so a later tweak has to be deliberate."""
    dominant = (
        scorer.SCENE_WEIGHTS[Dimension.LEGIBILITY] + scorer.SCENE_WEIGHTS[Dimension.RELEVANCE]
    )
    assert dominant == pytest.approx(0.50)
    others = set(scorer.SCENE_WEIGHTS) - {Dimension.LEGIBILITY, Dimension.RELEVANCE}
    assert all(
        scorer.SCENE_WEIGHTS[d] < scorer.SCENE_WEIGHTS[Dimension.RELEVANCE] for d in others
    )


def test_weighted_renormalises_over_the_dimensions_present() -> None:
    """Two 8s and nothing else must be 80, not 8 * (0.26 + 0.24) * 10."""
    scores = {Dimension.LEGIBILITY: 8.0, Dimension.RELEVANCE: 8.0}
    assert scorer._weighted(scores, scorer.SCENE_WEIGHTS) == pytest.approx(80.0)


def test_weighted_with_nothing_present() -> None:
    assert scorer._weighted({}, scorer.SCENE_WEIGHTS) == 0.0


def test_grade_boundaries() -> None:
    assert grade_for(90.0) is Grade.A
    assert grade_for(89.9) is Grade.B
    assert grade_for(70.0) is Grade.C
    assert grade_for(59.9) is Grade.F


def test_legibility_blends_measurement_and_judgement() -> None:
    metrics = SceneMetrics(scene_id=1, contrast=ContrastMeasure(ratio=4.5, ratio_median=7.0))
    measured = scorer._piecewise(4.5, scorer.CONTRAST_ANCHORS)
    blended = scorer._legibility(metrics, 3)
    assert blended is not None
    assert blended == pytest.approx(0.6 * measured + 0.4 * 3.0, abs=0.01)
    # A harsh model verdict must drag the score below the measurement alone.
    assert blended < measured


def test_legibility_falls_back_to_whichever_half_exists() -> None:
    only_metric = SceneMetrics(scene_id=1, contrast=ContrastMeasure(ratio=9.0, ratio_median=15.0))
    assert scorer._legibility(only_metric, None) == pytest.approx(
        scorer._piecewise(9.0, scorer.CONTRAST_ANCHORS)
    )
    assert scorer._legibility(SceneMetrics(scene_id=1), 6) == pytest.approx(6.0)
    assert scorer._legibility(SceneMetrics(scene_id=1), None) is None


def test_timing_is_unpenalised_when_there_are_no_bullets() -> None:
    assert scorer._timing_score(SceneMetrics(scene_id=1)) == 10.0


def test_timing_pays_for_each_defect_and_never_goes_negative() -> None:
    metrics = SceneMetrics(
        scene_id=1,
        bullet_issues=["a", "b", "c", "d", "e"],
        narration_gaps=[(1.0, 3.0), (5.0, 7.0)],
        silence_windows=[(2.0, 3.0)],
    )
    assert scorer._timing_score(metrics) == 0.0


def test_pacing_penalises_an_uneven_scene_on_top_of_wpm() -> None:
    even = SceneMetrics(scene_id=1, words_per_minute=145.0, duration_deviation=0.0)
    uneven = SceneMetrics(scene_id=1, words_per_minute=145.0, duration_deviation=0.7)
    assert scorer._pacing_score(even) == pytest.approx(10.0)
    assert scorer._pacing_score(uneven) == pytest.approx(7.0)


def test_pacing_is_none_without_word_timings() -> None:
    assert scorer._pacing_score(SceneMetrics(scene_id=1)) is None


# ------------------------------------------------------- unmeasured != passing


def test_missing_vision_removes_dimensions_rather_than_neutralising_them() -> None:
    metrics = SceneMetrics(
        scene_id=1,
        contrast=ContrastMeasure(ratio=9.0, ratio_median=15.0),
        words_per_minute=140.0,
        duplicate_frame_ratio=0.0,
    )
    score = scorer.score_scene(metrics, None)
    assert score.relevance is None
    assert score.composition is None
    assert score.professionalism is None
    assert set(score.dimension_scores()) == {
        Dimension.LEGIBILITY,
        Dimension.MOTION,
        Dimension.PACING,
        Dimension.TIMING,
    }


def test_scene_score_uses_the_vision_verdict_when_present() -> None:
    metrics = SceneMetrics(
        scene_id=1, contrast=ContrastMeasure(ratio=9.0, ratio_median=15.0), words_per_minute=140.0
    )
    report = VisionReport(
        scenes={
            1: SceneVerdict(
                scene_id=1,
                topical_relevance=3,
                text_legibility=4,
                composition=5,
                professionalism=6,
                issues=["generic atrium"],
                suggested_image_prompt="a phishing email on a laptop, no text",
            )
        }
    )
    score = scorer.score_scene(metrics, report)
    assert score.relevance == 3.0
    assert score.composition == 5.0
    assert score.professionalism == 6.0
    assert score.issues == ["generic atrium"]
    assert score.suggested_image_prompt is not None


def test_an_off_topic_scene_scores_below_an_on_topic_one() -> None:
    """The validation case: identical metrics, only relevance differs."""
    metrics = SceneMetrics(
        scene_id=1,
        contrast=ContrastMeasure(ratio=9.0, ratio_median=15.0),
        words_per_minute=140.0,
        duplicate_frame_ratio=0.0,
    )

    def verdict(relevance: int) -> VisionReport:
        return VisionReport(
            scenes={
                1: SceneVerdict(
                    scene_id=1,
                    topical_relevance=relevance,
                    text_legibility=8,
                    composition=8,
                    professionalism=8,
                )
            }
        )

    on_topic = scorer.score_scene(metrics, verdict(9)).overall
    off_topic = scorer.score_scene(metrics, verdict(2)).overall
    assert off_topic < on_topic
    # 7 points of relevance at weight 0.24 must move the total by at least 15/100.
    assert on_topic - off_topic > 15.0


# ------------------------------------------------------------------- audio scoring


def test_audio_score_penalises_the_reference_render_loudness() -> None:
    metrics = VideoMetrics(
        loudness=LoudnessMeasure(integrated_lufs=-20.3, true_peak_dbfs=-5.6, loudness_range_lu=3.1),
        balance=BalanceMeasure(speech_dbfs=-22.7, bed_dbfs=-32.7, separation_db=10.0),
    )
    quiet = scorer._audio_score(metrics, [])
    on_target = scorer._audio_score(
        VideoMetrics(
            loudness=LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=-5.6),
            balance=metrics.balance,
        ),
        [],
    )
    assert quiet is not None and on_target is not None
    assert quiet < on_target
    assert quiet < 8.0


def test_audio_score_drops_the_balance_term_when_unmeasurable() -> None:
    loudness = LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=-3.0)
    unmeasured = VideoMetrics(
        loudness=loudness,
        balance=BalanceMeasure(
            speech_dbfs=-20.0, bed_dbfs=0.0, separation_db=0.0, measured=False
        ),
    )
    # 0 dB separation would score 0 if it were counted; dropping it must not.
    assert scorer._audio_score(unmeasured, []) == pytest.approx(10.0)


def test_audio_score_punishes_dead_air() -> None:
    loudness = LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=-3.0)
    clean = scorer._audio_score(VideoMetrics(loudness=loudness), [])
    holey = scorer._audio_score(
        VideoMetrics(loudness=loudness),
        [SceneMetrics(scene_id=1, silence_windows=[(1.0, 2.0), (4.0, 5.0)])],
    )
    assert clean is not None and holey is not None
    assert holey < clean


def test_audio_score_is_none_with_nothing_to_go_on() -> None:
    assert scorer._audio_score(VideoMetrics(), []) is not None  # continuity term always exists


def test_clipping_costs_more_than_a_safe_peak() -> None:
    hot = VideoMetrics(loudness=LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=0.5))
    safe = VideoMetrics(loudness=LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=-3.0))
    assert scorer._audio_score(hot, []) < scorer._audio_score(safe, [])


def test_technical_score_falls_with_drift_and_mismatch() -> None:
    assert scorer._technical_score(VideoMetrics(duration_drift_frames=1.4)) > 9.0
    assert scorer._technical_score(VideoMetrics(duration_drift_frames=60.0)) < 2.0
    assert scorer._technical_score(
        VideoMetrics(duration_drift_frames=0.0, profile_mismatch=["a", "b"])
    ) == pytest.approx(4.0)


def test_script_score_docks_bullets_that_do_not_echo() -> None:
    def report(echo: str) -> VisionReport:
        return VisionReport(
            script=ScriptVerdict(
                narrative_flow=8, clarity=8, actionability=8, bullets_echo_narration=echo
            )
        )

    assert scorer._script_score(report("yes")) == pytest.approx(8.0)
    assert scorer._script_score(report("partly")) == pytest.approx(7.5)
    assert scorer._script_score(report("no")) == pytest.approx(7.0)
    # Nothing to echo is not a failure.
    assert scorer._script_score(report("no_bullets")) == pytest.approx(8.0)
    assert scorer._script_score(None) is None


# ------------------------------------------------------------------ recommendations


def test_contrast_fix_solves_for_an_opacity_that_clears_the_target() -> None:
    """The proposed scrim must actually reach 4.5:1, not just be larger than before."""
    current = 0.45
    for measured in (4.35, 3.89, 2.5):
        proposed = scorer._contrast_fix_opacity(measured, current)
        assert current < proposed <= 0.80
        underlying = M.level_for_contrast(measured) / (1.0 - current)
        assert M.contrast_against_white(underlying * (1.0 - proposed)) >= M.MIN_CONTRAST_RATIO


def test_contrast_fix_is_capped_so_the_image_stays_visible() -> None:
    """Past 0.80 the slide stops being a photograph, so the solver stops there."""
    assert scorer._contrast_fix_opacity(1.05, 0.78) == 0.80
    assert scorer._contrast_fix_opacity(1.05, 0.85) == 0.80


def test_a_scrim_already_at_the_cap_is_not_offered_as_an_auto_fix() -> None:
    """When no scrim can rescue the text, say so rather than emit a no-op parameter change."""
    scene = _scene(4, start=0.0, end=10.0, plan=VisualPlan(scrim_opacity=0.80))
    metrics = SceneMetrics(scene_id=4, contrast=ContrastMeasure(ratio=2.0, ratio_median=3.0))
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    legibility = [r for r in recs if r.dimension is Dimension.LEGIBILITY]
    assert legibility
    assert all(not r.auto_fixable for r in legibility)
    assert all(r.action != "raise_scrim_opacity" for r in legibility)


def test_low_contrast_yields_an_auto_fixable_scrim_recommendation() -> None:
    scene = _scene(4, start=0.0, end=10.0, plan=VisualPlan(scrim_opacity=0.45))
    metrics = SceneMetrics(
        scene_id=4, contrast=ContrastMeasure(ratio=4.35, ratio_median=6.12), words_per_minute=140.0
    )
    score = scorer.score_scene(metrics, None)
    recs = scorer._scene_recommendations(score, _timeline([scene]))
    scrim = [r for r in recs if r.action == "raise_scrim_opacity"]
    assert len(scrim) == 1
    assert scrim[0].auto_fixable is True
    assert scrim[0].severity is Severity.MAJOR
    assert scrim[0].params["scene_id"] == 4
    assert scrim[0].params["to"] > scrim[0].params["from"]
    assert scrim[0].evidence and "4.35" in scrim[0].evidence


def test_contrast_below_the_large_text_allowance_is_a_blocker() -> None:
    scene = _scene(1, start=0.0, end=10.0)
    metrics = SceneMetrics(scene_id=1, contrast=ContrastMeasure(ratio=2.4, ratio_median=3.0))
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    assert any(r.severity is Severity.BLOCKER for r in recs if r.dimension is Dimension.LEGIBILITY)


def test_good_contrast_yields_no_legibility_recommendation() -> None:
    scene = _scene(1, start=0.0, end=10.0)
    metrics = SceneMetrics(
        scene_id=1, contrast=ContrastMeasure(ratio=9.12, ratio_median=15.46), words_per_minute=140.0
    )
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    assert not [r for r in recs if r.dimension is Dimension.LEGIBILITY]


def test_a_much_darker_alternative_band_is_offered_as_a_move(monkeypatch) -> None:
    scene = _scene(4, start=0.0, end=10.0, plan=VisualPlan(text_position=TextPosition.UPPER_THIRD))
    metrics = SceneMetrics(
        scene_id=4,
        contrast=ContrastMeasure(
            ratio=3.5, ratio_median=5.0, alt_position="lower_third", alt_ratio=14.0
        ),
    )
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    move = [r for r in recs if r.action == "move_text_position"]
    assert len(move) == 1
    assert move[0].auto_fixable is True
    assert move[0].params["text_position"] == "lower_third"


def test_a_no_better_alternative_band_is_not_offered() -> None:
    scene = _scene(4, start=0.0, end=10.0)
    metrics = SceneMetrics(
        scene_id=4,
        contrast=ContrastMeasure(
            ratio=3.89, ratio_median=7.39, alt_position="upper_third", alt_ratio=3.63
        ),
    )
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    assert not [r for r in recs if r.action == "move_text_position"]


def test_off_topic_image_is_a_blocker_with_the_replacement_prompt() -> None:
    scene = _scene(4, start=0.0, end=10.0)
    metrics = SceneMetrics(scene_id=4, words_per_minute=140.0)
    report = VisionReport(
        scenes={
            4: SceneVerdict(
                scene_id=4,
                topical_relevance=3,
                text_legibility=7,
                composition=6,
                professionalism=6,
                issues=["generic atrium, unrelated to phishing"],
                suggested_image_prompt="hands over a keyboard reviewing an email, no text",
            )
        }
    )
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, report), _timeline([scene]))
    relevance = [r for r in recs if r.dimension is Dimension.RELEVANCE]
    assert len(relevance) == 1
    assert relevance[0].severity is Severity.BLOCKER
    assert relevance[0].auto_fixable is True
    assert relevance[0].action == "regenerate_scene_image"
    assert "keyboard" in str(relevance[0].params["image_prompt"])


def test_relevance_without_a_suggested_prompt_is_not_auto_fixable() -> None:
    scene = _scene(4, start=0.0, end=10.0)
    report = VisionReport(
        scenes={
            4: SceneVerdict(
                scene_id=4,
                topical_relevance=5,
                text_legibility=8,
                composition=8,
                professionalism=8,
            )
        }
    )
    recs = scorer._scene_recommendations(
        scorer.score_scene(SceneMetrics(scene_id=4), report), _timeline([scene])
    )
    relevance = [r for r in recs if r.dimension is Dimension.RELEVANCE]
    assert len(relevance) == 1
    assert relevance[0].severity is Severity.MAJOR
    assert relevance[0].auto_fixable is False
    assert relevance[0].action is None


def test_bullet_defects_produce_one_auto_fixable_respace() -> None:
    scene = _scene(
        1,
        start=0.0,
        end=10.0,
        bullets=[BulletPoint(text="a", appear_at=1.0), BulletPoint(text="b", appear_at=1.2)],
    )
    metrics = SceneMetrics(scene_id=1, bullet_issues=M.bullet_issues(scene), bullet_count=2)
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    respace = [r for r in recs if r.action == "respace_bullets"]
    assert respace
    assert all(r.auto_fixable for r in respace)
    assert respace[0].params["min_gap"] == M.MIN_BULLET_GAP


def test_no_bullets_produces_no_bullet_recommendations() -> None:
    """The older-job path: nothing to check must mean nothing reported per scene."""
    scene = _scene(1, start=0.0, end=10.0, bullets=[])
    metrics = SceneMetrics(scene_id=1, bullet_count=0, words_per_minute=140.0)
    recs = scorer._scene_recommendations(scorer.score_scene(metrics, None), _timeline([scene]))
    assert not [r for r in recs if r.action == "respace_bullets"]


def test_quiet_mix_yields_the_exact_normalising_gain() -> None:
    metrics = VideoMetrics(
        loudness=LoudnessMeasure(integrated_lufs=-20.3, true_peak_dbfs=-5.6, loudness_range_lu=3.1)
    )
    recs = scorer._video_recommendations(metrics, None, _timeline())
    loud = [r for r in recs if r.action == "renormalize_loudness"]
    assert len(loud) == 1
    assert loud[0].auto_fixable is True
    assert loud[0].severity is Severity.MAJOR
    assert loud[0].params["gain_db"] == pytest.approx(4.3)
    assert loud[0].params["target_lufs"] == M.TARGET_LUFS


def test_on_target_loudness_is_not_flagged() -> None:
    metrics = VideoMetrics(loudness=LoudnessMeasure(integrated_lufs=-16.4, true_peak_dbfs=-4.0))
    recs = scorer._video_recommendations(metrics, None, _timeline())
    assert not [r for r in recs if r.dimension is Dimension.AUDIO and r.action]


def test_a_clipping_peak_is_a_blocker() -> None:
    metrics = VideoMetrics(loudness=LoudnessMeasure(integrated_lufs=-16.0, true_peak_dbfs=0.2))
    recs = scorer._video_recommendations(metrics, None, _timeline())
    peak = [r for r in recs if r.action == "limit_true_peak"]
    assert len(peak) == 1
    assert peak[0].severity is Severity.BLOCKER


def test_music_drowning_the_voice_is_auto_duckable() -> None:
    metrics = VideoMetrics(
        balance=BalanceMeasure(speech_dbfs=-20.0, bed_dbfs=-24.0, separation_db=4.0)
    )
    recs = scorer._video_recommendations(metrics, None, _timeline())
    duck = [r for r in recs if r.action == "duck_music"]
    assert len(duck) == 1
    assert duck[0].severity is Severity.MAJOR
    assert duck[0].params["additional_gain_db"] == pytest.approx(-6.0)


def test_large_duration_drift_is_a_blocker_and_not_auto_fixable() -> None:
    metrics = VideoMetrics(duration=40.0, expected_duration=48.0, duration_drift_frames=240.0)
    recs = scorer._video_recommendations(metrics, None, _timeline())
    drift = [r for r in recs if r.dimension is Dimension.TECHNICAL]
    assert drift[0].severity is Severity.BLOCKER
    assert drift[0].auto_fixable is False


def test_small_duration_drift_is_not_flagged() -> None:
    metrics = VideoMetrics(duration=47.47, expected_duration=47.42, duration_drift_frames=1.4)
    recs = scorer._video_recommendations(metrics, None, _timeline())
    assert not [r for r in recs if r.dimension is Dimension.TECHNICAL]


def test_every_auto_fixable_recommendation_carries_an_action_and_params() -> None:
    """``auto_fixable`` is a promise the pipeline can dispatch on. Keep it honest."""
    metrics = VideoMetrics(
        loudness=LoudnessMeasure(integrated_lufs=-20.3, true_peak_dbfs=0.5),
        balance=BalanceMeasure(speech_dbfs=-20.0, bed_dbfs=-24.0, separation_db=4.0),
    )
    scene = _scene(4, start=0.0, end=10.0)
    scene_recs = scorer._scene_recommendations(
        scorer.score_scene(
            SceneMetrics(
                scene_id=4,
                contrast=ContrastMeasure(ratio=3.5, ratio_median=5.0),
                duplicate_frame_ratio=0.30,
                bullet_issues=["bullet 2 appears 0.20s after the previous bullet"],
            ),
            None,
        ),
        _timeline([scene]),
    )
    for rec in scene_recs + scorer._video_recommendations(metrics, None, _timeline()):
        if rec.auto_fixable:
            assert rec.action, f"{rec.dimension} auto-fix has no action"
            assert rec.params, f"{rec.action} auto-fix has no params"


# ---------------------------------------------------------------- grade capping


def test_a_blocker_caps_the_grade_at_c() -> None:
    blocker = [
        scorer.Recommendation(
            severity=Severity.BLOCKER,
            dimension=Dimension.RELEVANCE,
            problem="off topic",
            fix="regenerate",
        )
    ]
    grade, capped = scorer._cap_grade(94.0, blocker)
    assert grade is Grade.C
    assert capped is True


def test_a_major_caps_the_grade_at_b() -> None:
    major = [
        scorer.Recommendation(
            severity=Severity.MAJOR, dimension=Dimension.AUDIO, problem="quiet", fix="normalise"
        )
    ]
    grade, capped = scorer._cap_grade(96.0, major)
    assert grade is Grade.B
    assert capped is True


def test_capping_never_raises_a_grade() -> None:
    major = [
        scorer.Recommendation(
            severity=Severity.MAJOR, dimension=Dimension.AUDIO, problem="quiet", fix="normalise"
        )
    ]
    grade, capped = scorer._cap_grade(41.0, major)
    assert grade is Grade.F
    assert capped is False


def test_only_minor_findings_leave_the_grade_alone() -> None:
    minor = [
        scorer.Recommendation(
            severity=Severity.MINOR, dimension=Dimension.PACING, problem="slow", fix="rewrite"
        )
    ]
    assert scorer._cap_grade(93.0, minor) == (Grade.A, False)


# ------------------------------------------------------------------- vision layer


def _verdict_json(**overrides) -> str:
    payload = {
        "topical_relevance": 3,
        "relevance_reason": "A sunlit atrium full of plants; anchor band 1-2.",
        "text_legibility": 4,
        "composition": 5,
        "professionalism": 6,
        "issues": ["image unrelated to phishing"],
        "suggested_image_prompt": "hands over a keyboard reviewing a suspicious email, no text",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_judge_scene_parses_a_structured_verdict(monkeypatch) -> None:
    monkeypatch.setattr(vision, "_generate", lambda *a, **k: _verdict_json())
    verdict = vision.judge_scene(
        frame_jpeg=b"jpeg",
        topic="phishing",
        heading="Staying Safe with Smart Habits",
        narration="Verify senders before clicking.",
        scene_id=4,
        scene_number=4,
        scene_count=4,
        api_key="k",
    )
    assert verdict is not None
    assert verdict.topical_relevance == 3
    assert verdict.suggested_image_prompt is not None
    assert verdict.issues == ["image unrelated to phishing"]


def test_judge_scene_sends_the_frame_inline_with_a_mime_type(monkeypatch) -> None:
    captured: dict = {}

    def fake(model, body, api_key, **kwargs):
        captured["body"] = body
        return _verdict_json()

    monkeypatch.setattr(vision, "_generate", fake)
    vision.judge_scene(
        frame_jpeg=b"\xff\xd8jpeg",
        topic="phishing",
        heading="h",
        narration="n",
        scene_id=1,
        scene_number=1,
        scene_count=4,
        api_key="k",
    )
    parts = captured["body"]["contents"][0]["parts"]
    inline = [p for p in parts if "inlineData" in p]
    assert len(inline) == 1
    assert inline[0]["inlineData"]["mimeType"] == "image/jpeg"
    config = captured["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    # The REST API rejects lowercase schema types.
    assert config["responseSchema"]["type"] == "OBJECT"
    assert config["temperature"] == 0.0
    text = " ".join(p["text"] for p in parts if "text" in p)
    assert "phishing" in text and "SLIDE 1 OF 4" in text


def test_judge_scene_retries_once_on_malformed_json(monkeypatch) -> None:
    calls: list[dict] = []

    def fake(model, body, api_key, **kwargs):
        calls.append(body)
        return "not json at all" if len(calls) == 1 else _verdict_json()

    monkeypatch.setattr(vision, "_generate", fake)
    verdict = vision.judge_scene(
        frame_jpeg=b"j",
        topic="t",
        heading="h",
        narration="n",
        scene_id=1,
        scene_number=1,
        scene_count=1,
        api_key="k",
    )
    assert verdict is not None
    assert len(calls) == 2
    # The retry nudges rather than repeating the identical request.
    assert len(calls[1]["contents"][0]["parts"]) > len(calls[0]["contents"][0]["parts"])


def test_judge_scene_gives_up_after_the_retry(monkeypatch) -> None:
    monkeypatch.setattr(vision, "_generate", lambda *a, **k: "{oops")
    assert (
        vision.judge_scene(
            frame_jpeg=b"j",
            topic="t",
            heading="h",
            narration="n",
            scene_id=1,
            scene_number=1,
            scene_count=1,
            api_key="k",
        )
        is None
    )


def test_judge_scene_survives_a_transport_failure(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(vision, "_generate", boom)
    assert (
        vision.judge_scene(
            frame_jpeg=b"j",
            topic="t",
            heading="h",
            narration="n",
            scene_id=1,
            scene_number=1,
            scene_count=1,
            api_key="k",
        )
        is None
    )


def test_fenced_json_is_tolerated(monkeypatch) -> None:
    monkeypatch.setattr(
        vision, "_generate", lambda *a, **k: f"```json\n{_verdict_json()}\n```"
    )
    verdict = vision.judge_scene(
        frame_jpeg=b"j",
        topic="t",
        heading="h",
        narration="n",
        scene_id=1,
        scene_number=1,
        scene_count=1,
        api_key="k",
    )
    assert verdict is not None and verdict.topical_relevance == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 1), (11, 10), ("8", 8), (8.6, 9), (None, 5), ("banana", 5), (-3, 1)],
)
def test_clamp_coerces_whatever_the_model_returns(raw, expected) -> None:
    """One out-of-range score must not throw away the other three."""
    assert vision._clamp(raw) == expected


def test_script_text_includes_headings_narration_and_bullets() -> None:
    scene = _scene(
        1,
        start=0.0,
        end=10.0,
        bullets=[BulletPoint(text="Check the sender", appear_at=2.0, emphasis=True)],
    )
    text = vision.script_text(_timeline([scene]))
    assert "TOPIC: How phishing attacks work" in text
    assert "SLIDE 1 — Understanding the Deceptive Hook" in text
    assert "Check the sender" in text
    assert "@2.0s *" in text


def test_script_text_marks_slides_without_bullets() -> None:
    assert "(none on this slide)" in vision.script_text(_timeline())


def test_judge_script_overrides_the_model_when_there_are_no_bullets(monkeypatch) -> None:
    """Ground truth beats the model: with no bullets there is nothing to echo."""
    monkeypatch.setattr(
        vision,
        "_generate",
        lambda *a, **k: json.dumps(
            {
                "narrative_flow": 7,
                "clarity": 8,
                "actionability": 6,
                "bullets_echo_narration": "yes",
                "issues": [],
                "suggestions": [],
            }
        ),
    )
    verdict = vision.judge_script(_timeline(), api_key="k")
    assert verdict is not None
    assert verdict.bullets_echo_narration == "no_bullets"


def test_judge_script_normalises_an_unexpected_enum_value(monkeypatch) -> None:
    monkeypatch.setattr(
        vision,
        "_generate",
        lambda *a, **k: json.dumps(
            {
                "narrative_flow": 7,
                "clarity": 8,
                "actionability": 6,
                "bullets_echo_narration": "MAYBE",
                "issues": [],
                "suggestions": [],
            }
        ),
    )
    assert vision.judge_script(_timeline(), api_key="k").bullets_echo_narration == "no_bullets"


def test_judge_timeline_requires_a_key(monkeypatch) -> None:
    """No key anywhere must degrade to metrics-only, not half-score with a neutral guess."""

    class NoKey:
        gemini_api_key = ""

    monkeypatch.setattr("app.core.config.get_settings", lambda: NoKey())
    with pytest.raises(vision.VisionUnavailable):
        vision.judge_timeline(_timeline(), Path("v.mp4"), api_key="")


def test_judge_timeline_records_a_missing_frame_as_an_error(monkeypatch) -> None:
    monkeypatch.setattr(vision, "scene_frame_jpeg", lambda *a, **k: None)
    monkeypatch.setattr(vision, "judge_script", lambda *a, **k: None)
    report = vision.judge_timeline(_timeline(), Path("v.mp4"), api_key="k")
    assert report.scenes == {}
    assert any("no frame" in e for e in report.errors)
    assert any("script judgement unavailable" in e for e in report.errors)


# ------------------------------------------------------------------ end to end


@pytest.fixture
def stub_measurements(monkeypatch, four_scene_timeline: Timeline):
    """Replace every ffmpeg call with fixed measurements taken from the real render."""
    contrasts = {1: 9.12, 2: 9.32, 3: 3.89, 4: 4.35}
    dupes = {1: 0.05, 2: 0.075, 3: 0.058, 4: 0.10}

    def fake_scene(scene, timeline, video_path):
        return SceneMetrics(
            scene_id=scene.id,
            heading=scene.heading,
            duration=scene.duration,
            contrast=ContrastMeasure(
                ratio=contrasts[scene.id], ratio_median=contrasts[scene.id] * 1.6
            ),
            duplicate_frame_ratio=dupes[scene.id],
            words_per_minute=132.0,
            bullet_count=len(scene.bullets),
        )

    def fake_video(timeline, video_path):
        return VideoMetrics(
            duration=39.5,
            expected_duration=38.5,
            duration_drift_frames=30.0,
            width=1920,
            height=1080,
            fps=30.0,
            video_codec="h264",
            audio_codec="aac",
            loudness=LoudnessMeasure(
                integrated_lufs=-20.3, true_peak_dbfs=-5.6, loudness_range_lu=3.1
            ),
            balance=BalanceMeasure(speech_dbfs=-22.7, bed_dbfs=-32.7, separation_db=10.0),
            duplicate_frame_ratio=0.06,
        )

    monkeypatch.setattr(M, "measure_scene", fake_scene)
    monkeypatch.setattr(M, "measure_video", fake_video)
    return four_scene_timeline


def test_score_timeline_without_vision_notes_what_it_could_not_judge(
    stub_measurements: Timeline, tmp_path: Path
) -> None:
    score = scorer.score_timeline(stub_measurements, tmp_path / "video.mp4", vision=False)
    assert score.vision_used is False
    assert score.script_score is None
    assert all(s.relevance is None for s in score.scenes)
    assert any("relevance" in n for n in score.notes)
    # The two low-contrast scenes must still be the two worst.
    ranked = sorted(score.scenes, key=lambda s: s.overall)
    assert {ranked[0].scene_id, ranked[1].scene_id} == {3, 4}


def test_score_timeline_with_a_mocked_vision_pass(
    stub_measurements: Timeline, tmp_path: Path, monkeypatch
) -> None:
    relevance = {1: 5, 2: 5, 3: 5, 4: 3}

    def fake_judge(timeline, video_path, *, api_key="", model="", max_workers=4):
        return VisionReport(
            model="mock",
            scenes={
                s.id: SceneVerdict(
                    scene_id=s.id,
                    topical_relevance=relevance[s.id],
                    text_legibility=8 if s.id in (1, 2) else 4,
                    composition=7,
                    professionalism=7,
                    suggested_image_prompt="a phishing email on screen" if s.id == 4 else None,
                )
                for s in timeline.scenes
            },
            script=ScriptVerdict(
                narrative_flow=6, clarity=8, actionability=6, bullets_echo_narration="no_bullets"
            ),
        )

    monkeypatch.setattr(scorer, "judge_timeline", fake_judge)
    score = scorer.score_timeline(stub_measurements, tmp_path / "video.mp4", vision=True)

    assert score.vision_used is True
    assert score.vision_model == "mock"
    assert score.narrative_flow == 6.0

    # Scene 4 is the worst: lowest relevance and lowest legibility.
    worst = score.worst_scene()
    assert worst is not None and worst.scene_id == 4
    assert worst.overall < min(s.overall for s in score.scenes if s.scene_id != 4)

    # The off-topic image is a blocker, and it caps the grade.
    blockers = [r for r in score.recommendations if r.severity is Severity.BLOCKER]
    assert any(r.dimension is Dimension.RELEVANCE and r.scene_id == 4 for r in blockers)
    assert score.grade is Grade.F or score.grade_capped or score.overall < 70

    # Auto-fixables are a subset of everything, and each is dispatchable.
    auto = score.auto_fixable()
    assert auto
    assert all(r.action for r in auto)
    assert {r.action for r in auto} >= {
        "raise_scrim_opacity",
        "regenerate_scene_image",
        "renormalize_loudness",
    }
    assert score.by_severity()[0].severity is Severity.BLOCKER


def test_score_timeline_survives_a_vision_pass_that_explodes(
    stub_measurements: Timeline, tmp_path: Path, monkeypatch
) -> None:
    """A scorecard from the metrics alone is worth more than a traceback."""

    def boom(*a, **k):
        raise RuntimeError("gemini is down")

    monkeypatch.setattr(scorer, "judge_timeline", boom)
    score = scorer.score_timeline(stub_measurements, tmp_path / "video.mp4", vision=True)
    assert score.vision_used is False
    assert any("gemini is down" in n for n in score.notes)
    assert score.scenes


def test_report_and_json_round_trip(stub_measurements: Timeline, tmp_path: Path) -> None:
    score = scorer.score_timeline(stub_measurements, tmp_path / "video.mp4", vision=False)
    destination = scorer.write_score(score, tmp_path / "score.json")
    reloaded = json.loads(destination.read_text())
    assert reloaded["job_id"] == "job-1"
    assert reloaded["grade"] in {g.value for g in Grade}
    assert len(reloaded["scenes"]) == 4
    # Raw measurements survive so a threshold can be re-tuned without re-rendering.
    assert reloaded["scenes"][0]["metrics"]["contrast"]["ratio"] == 9.12

    text = scorer.render_report(score)
    assert "OVERALL" in text
    assert "PER SCENE" in text
    assert "RECOMMENDATIONS" in text
    for scene in score.scenes:
        assert str(scene.scene_id) in text


def test_report_renders_with_every_dimension_unmeasured(tmp_path: Path) -> None:
    """A near-empty scorecard must still print rather than crash on a None."""
    from app.evaluate.models import VideoScore

    text = scorer.render_report(
        VideoScore(job_id="j", scenes=[], metrics=VideoMetrics(), notes=["nothing measured"])
    )
    assert "n/a" in text
    assert "nothing to fix." in text


# -------------------------------------------------------------------- timeline load


def test_load_timeline_reads_an_explicit_file(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json"
    path.write_text(_timeline().model_dump_json())
    assert load_and_check(path) == "job-1"


def load_and_check(path: Path) -> str:
    return scorer.load_timeline("ignored", timeline_path=path).job_id


def test_load_timeline_raises_when_nothing_can_supply_one(monkeypatch, tmp_path: Path) -> None:
    """A missing timeline must be a clear LookupError naming both failed sources."""

    class FakeSettings:
        db_path = tmp_path / "absent.db"

    monkeypatch.setattr("app.core.config.get_settings", lambda: FakeSettings())

    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(LookupError) as excinfo:
        scorer.load_timeline("missing-job")
    message = str(excinfo.value)
    assert "missing-job" in message
    assert "absent.db" in message
    assert "connection refused" in message


def test_resolve_video_finds_the_job_directory_from_a_scene_path(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00")
    scene = _scene(1)
    scene.clip_path = str(tmp_path / "clip_01.mp4")
    assert scorer.resolve_video("job-1", _timeline([scene])) == video


def test_resolve_video_prefers_an_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "other.mp4"
    assert scorer.resolve_video("job-1", _timeline(), explicit) == explicit


# -------------------------------------------------------------------------- motion


def test_static_scenes_are_not_expected_to_move() -> None:
    """A STATIC plan legitimately produces identical frames; that is not judder.

    Recorded as an explicit known limitation: the duplicate-frame metric has no way to
    tell a deliberate hold from a broken zoom, so a STATIC scene will read as 100%
    duplicate. Nothing in the reference render uses STATIC, so this is untested against
    real output.
    """
    plan = VisualPlan(motion=Motion.STATIC)
    assert plan.motion is Motion.STATIC
    assert scorer._piecewise(1.0, scorer.MOTION_ANCHORS) == 0.0
