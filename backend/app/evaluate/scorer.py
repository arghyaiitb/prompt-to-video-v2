"""Fold measurements and judgements into a scorecard, and say what to change.

WEIGHTS AND WHY
===============

Scene dimensions (they sum to 1.00, and are renormalised over whichever dimensions were
actually assessed):

    legibility       0.26   If the heading cannot be read, the slide has not delivered its
                            one piece of on-screen information. Training video text is
                            *read*, not merely watched, and nothing else on the slide can
                            compensate. This is the single heaviest weight.
    relevance        0.24   An image that does not match the narration is worse than a
                            plain one: it actively misdirects attention and reads as
                            careless. Nearly tied with legibility.
    composition      0.12   Real, but recoverable by moving the text rather than
                            regenerating the asset — so it costs less than the two above.
    pacing           0.10   Delivery outside ~110-170 wpm measurably hurts retention.
    timing           0.10   Bullets landing before the scene, after it, or on top of each
                            other are visible defects, and they are cheap to fix.
    professionalism  0.10   Tone and brand fit. Subjective, so deliberately light.
    motion           0.08   Judder is noticeable and annoying, but it never stops a viewer
                            understanding the slide. Lowest weight of the seven.

Legibility + relevance = 0.50 by design. A beautiful video whose text you cannot read is a
failure, and so is a legible one illustrated with the wrong pictures; between them they can
sink a scene on their own, which is the intended behaviour.

Video components:

    scene average    0.60   The scenes *are* the video.
    audio            0.15   Loudness, peak, narration/music balance, dead air. Weighted
                            equal to the script because bad audio makes a video unwatchable
                            just as fast as a bad script makes it pointless.
    script           0.15   Flow, clarity, actionability.
    technical        0.10   Duration drift against the Timeline, container conformance.

GRADE CAPPING
=============
The weighted average is an average, and averages hide outliers: four excellent slides and
one off-topic slide is a video you cannot publish, but it averages to a B. So a BLOCKER
finding caps the grade at C and a MAJOR at B, regardless of the number. ``overall`` itself is
left honest and uncapped, because it is the quantity a re-run compares against to prove a
fix landed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.core.models import Timeline
from app.evaluate import metrics as M
from app.evaluate.models import (
    Dimension,
    Grade,
    Recommendation,
    SceneMetrics,
    SceneScore,
    Severity,
    VideoMetrics,
    VideoScore,
    grade_for,
)
from app.evaluate.vision import VisionReport, judge_timeline

logger = logging.getLogger(__name__)

EVALUATOR_VERSION = "1"

SCENE_WEIGHTS: dict[Dimension, float] = {
    Dimension.LEGIBILITY: 0.26,
    Dimension.RELEVANCE: 0.24,
    Dimension.COMPOSITION: 0.12,
    Dimension.PACING: 0.10,
    Dimension.TIMING: 0.10,
    Dimension.PROFESSIONALISM: 0.10,
    Dimension.MOTION: 0.08,
}

VIDEO_WEIGHTS: dict[str, float] = {
    "scenes": 0.60,
    "audio": 0.15,
    "script": 0.15,
    "technical": 0.10,
}

METRIC_LEGIBILITY_SHARE = 0.6
"""How much of the legibility score comes from the WCAG measurement rather than the model.

The measurement is objective and reproducible, so it leads. The model still gets 0.4
because contrast arithmetic is blind to things that genuinely stop you reading — a busy
high-frequency background, or a heading crossing a face — and those are real defects.
"""

GRADE_CEILING: dict[Severity, Grade] = {Severity.BLOCKER: Grade.C, Severity.MAJOR: Grade.B}
_GRADE_ORDER = [Grade.A, Grade.B, Grade.C, Grade.D, Grade.F]


# ------------------------------------------------------------------- score mapping


def _piecewise(value: float, anchors: list[tuple[float, float]]) -> float:
    """Linear interpolation between ``(measurement, score)`` anchors, clamped at the ends.

    Anchors instead of a formula because the mapping is a judgement about *thresholds*
    (4.5:1, 170 wpm, -16 LUFS) and a table makes those visible and arguable. Every table
    below sits next to the reasoning for its breakpoints.
    """
    points = sorted(anchors)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= value <= x1:
            span = x1 - x0
            return y0 if span <= 0 else y0 + (value - x0) * (y1 - y0) / span
    return points[-1][1]


#: 4.5:1 (WCAG AA) is deliberately mapped to 7.0 — "acceptable", not "good". 3:1, the
#: large-text allowance, only earns 4.0 because these frames also get viewed downscaled.
CONTRAST_ANCHORS = [(1.0, 0.0), (2.0, 2.0), (3.0, 4.0), (4.5, 7.0), (7.0, 9.0), (12.0, 10.0)]

#: Flat 10 up to the noise floor, because the reference render's genuinely smooth clips
#: measure 0.05-0.10 and penalising that would be scoring the encoder, not the motion.
#: 0.14 is what the same zoom measures without pre-upscaling — visibly stepping — so the
#: usable range is narrow and the decline past the floor is steep on purpose.
MOTION_ANCHORS = [
    (0.0, 10.0), (M.DUPLICATE_NOISE_FLOOR, 10.0), (0.14, 8.0),
    (0.20, 6.0), (0.30, 4.0), (0.50, 1.5), (1.0, 0.0),
]  # fmt: skip

#: 135-155 wpm is the comfortable band for narrated explanation. 110 and 170 are the flag
#: thresholds and both land on 7.0 — right at "acceptable".
PACING_ANCHORS = [
    (60.0, 0.0), (85.0, 3.0), (110.0, 7.0), (120.0, 8.5), (135.0, 10.0),
    (155.0, 10.0), (165.0, 8.5), (170.0, 7.0), (195.0, 3.0), (230.0, 0.0),
]  # fmt: skip

#: Absolute LU from the -16 LUFS target. 1 LU is inaudible; 4 LU is the difference between
#: "fine" and "why is this so quiet". The reference render is 4.3 LU off.
LOUDNESS_ANCHORS = [
    (0.0, 10.0), (1.0, 9.5), (2.0, 8.0), (3.0, 6.5),
    (4.0, 5.0), (6.0, 3.0), (8.0, 1.0), (12.0, 0.0),
]  # fmt: skip

#: dB of narration above the bed. Below 6 the music eats consonants; 14+ is clean.
BALANCE_ANCHORS = [(0.0, 0.0), (3.0, 2.0), (6.0, 5.0), (10.0, 8.0), (14.0, 10.0)]

#: Drift in frames against ``Timeline.final_duration()``. 2 frames is the flag threshold.
DRIFT_ANCHORS = [(0.0, 10.0), (2.0, 9.0), (5.0, 7.0), (10.0, 4.0), (30.0, 1.5), (90.0, 0.0)]

#: True peak. -1 dBFS is the ceiling; at 0 the file is clipping outright.
PEAK_ANCHORS = [(-6.0, 10.0), (-1.0, 10.0), (-0.3, 6.0), (0.0, 2.0), (1.0, 0.0)]


def _clamp10(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 2)


def _weighted(scores: dict[Dimension, float], weights: dict[Dimension, float]) -> float:
    """Weighted mean over the dimensions present, renormalised. 0-100."""
    usable = {k: v for k, v in scores.items() if v is not None and k in weights}
    total = sum(weights[k] for k in usable)
    if total <= 0:
        return 0.0
    return round(sum(usable[k] * weights[k] for k in usable) / total * 10.0, 1)


# -------------------------------------------------------------------- timeline load


def load_timeline(
    job_id: str, *, timeline_path: Path | None = None, api_base: str = "http://127.0.0.1:8000"
) -> Timeline:
    """Find a job's Timeline: explicit file, then the SQLite snapshot, then the API.

    The SQLite read goes through stdlib ``sqlite3`` in read-only URI mode rather than the
    app's own session layer. The evaluator has to be able to score a job while the rest of
    the backend is being edited, and it must never be able to write to the job table.
    """
    if timeline_path is not None:
        return Timeline.model_validate_json(Path(timeline_path).read_text())

    db_error = ""
    try:
        from app.core.config import get_settings

        db = get_settings().db_path
        if db.exists():
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT timeline_json FROM job WHERE id = ?", (job_id,)
                ).fetchone()
            if row and row[0]:
                return Timeline.model_validate_json(row[0])
            db_error = f"job {job_id} has no timeline_json in {db}"
        else:
            db_error = f"no database at {db}"
    except (sqlite3.Error, OSError, ValueError) as exc:
        db_error = f"sqlite read failed: {exc}"

    try:
        import httpx

        response = httpx.get(f"{api_base}/api/jobs/{job_id}/timeline", timeout=15.0)
        response.raise_for_status()
        return Timeline.model_validate(response.json())
    except Exception as exc:  # noqa: BLE001 - report both failures together
        raise LookupError(
            f"could not load a timeline for {job_id}: {db_error}; API fallback: {exc}"
        ) from exc


def resolve_video(job_id: str, timeline: Timeline, explicit: Path | None = None) -> Path:
    """Locate the rendered file. Scene clip paths pin the job directory for free."""
    if explicit is not None:
        return Path(explicit)
    for scene in timeline.scenes:
        for candidate in (scene.clip_path, scene.image_path, scene.audio_path):
            if candidate:
                guess = Path(candidate).parent / "video.mp4"
                if guess.exists():
                    return guess
    from app.core.config import get_settings

    return get_settings().video_output_dir / job_id / "video.mp4"


# ----------------------------------------------------------------------- scene score


def _legibility(metrics: SceneMetrics, vision_score: int | None) -> float | None:
    measured = (
        _piecewise(metrics.contrast.ratio, CONTRAST_ANCHORS) if metrics.contrast else None
    )
    if measured is None and vision_score is None:
        return None
    if measured is None:
        return _clamp10(float(vision_score or 0))
    if vision_score is None:
        return _clamp10(measured)
    share = METRIC_LEGIBILITY_SHARE
    return _clamp10(share * measured + (1 - share) * float(vision_score))


def _timing_score(metrics: SceneMetrics) -> float:
    """Bullet sanity and narration continuity. Starts at 10 and pays for each defect.

    A scene with no bullets is not penalised — older jobs predate ``Scene.bullets`` and an
    absent feature is not a broken one. It is reported as a note instead.
    """
    score = 10.0
    score -= min(6.0, 2.0 * len(metrics.bullet_issues))
    score -= 1.5 * len(metrics.narration_gaps)
    score -= 1.0 * len(metrics.silence_windows)
    return _clamp10(score)


def _pacing_score(metrics: SceneMetrics) -> float | None:
    if metrics.words_per_minute is None:
        return None
    score = _piecewise(metrics.words_per_minute, PACING_ANCHORS)
    deviation = abs(metrics.duration_deviation or 0.0)
    if deviation >= 0.60:
        score -= 3.0
    elif deviation >= 0.35:
        score -= 1.5
    return _clamp10(score)


def score_scene(metrics: SceneMetrics, report: VisionReport | None) -> SceneScore:
    verdict = report.scenes.get(metrics.scene_id) if report else None
    scores: dict[Dimension, float] = {}

    legibility = _legibility(metrics, verdict.text_legibility if verdict else None)
    if legibility is not None:
        scores[Dimension.LEGIBILITY] = legibility
    if verdict:
        scores[Dimension.RELEVANCE] = _clamp10(float(verdict.topical_relevance))
        scores[Dimension.COMPOSITION] = _clamp10(float(verdict.composition))
        scores[Dimension.PROFESSIONALISM] = _clamp10(float(verdict.professionalism))
    if metrics.duplicate_frame_ratio is not None:
        scores[Dimension.MOTION] = _clamp10(
            _piecewise(metrics.duplicate_frame_ratio, MOTION_ANCHORS)
        )
    pacing = _pacing_score(metrics)
    if pacing is not None:
        scores[Dimension.PACING] = pacing
    scores[Dimension.TIMING] = _timing_score(metrics)

    overall = _weighted(scores, SCENE_WEIGHTS)
    return SceneScore(
        scene_id=metrics.scene_id,
        heading=metrics.heading,
        duration=metrics.duration,
        legibility=scores.get(Dimension.LEGIBILITY),
        relevance=scores.get(Dimension.RELEVANCE),
        composition=scores.get(Dimension.COMPOSITION),
        professionalism=scores.get(Dimension.PROFESSIONALISM),
        motion=scores.get(Dimension.MOTION),
        pacing=scores.get(Dimension.PACING),
        timing=scores.get(Dimension.TIMING),
        overall=overall,
        grade=grade_for(overall),
        issues=list(verdict.issues) if verdict else [],
        suggested_image_prompt=verdict.suggested_image_prompt if verdict else None,
        metrics=metrics,
    )


# ----------------------------------------------------------------- video components


def _audio_score(metrics: VideoMetrics, scene_metrics: list[SceneMetrics]) -> float | None:
    parts: list[tuple[float, float]] = []
    if metrics.loudness:
        deviation = abs(metrics.loudness.integrated_lufs - M.TARGET_LUFS)
        parts.append((0.45, _piecewise(deviation, LOUDNESS_ANCHORS)))
        parts.append((0.15, _piecewise(metrics.loudness.true_peak_dbfs, PEAK_ANCHORS)))
    if metrics.balance and metrics.balance.measured:
        parts.append((0.25, _piecewise(metrics.balance.separation_db, BALANCE_ANCHORS)))
    holes = sum(len(s.silence_windows) for s in scene_metrics)
    parts.append((0.15, _clamp10(10.0 - 2.0 * holes)))
    total = sum(w for w, _ in parts)
    if total <= 0:
        return None
    return _clamp10(sum(w * s for w, s in parts) / total)


def _script_score(report: VisionReport | None) -> float | None:
    if report is None or report.script is None:
        return None
    verdict = report.script
    base = (verdict.narrative_flow + verdict.clarity + verdict.actionability) / 3.0
    penalty = {"no": 1.0, "partly": 0.5}.get(verdict.bullets_echo_narration, 0.0)
    return _clamp10(base - penalty)


def _technical_score(metrics: VideoMetrics) -> float:
    score = _piecewise(metrics.duration_drift_frames, DRIFT_ANCHORS)
    score -= 3.0 * len(metrics.profile_mismatch)
    return _clamp10(score)


# ----------------------------------------------------------------- recommendations


def _contrast_fix_opacity(ratio: float, current: float) -> float:
    """Scrim opacity that would bring the worst strip up to 4.5:1.

    A black scrim at opacity *a* scales an encoded luma by ``1 - a`` (both ``drawbox`` and
    the PNG ``overlay`` path blend on encoded values), so the underlying image level is
    recoverable from the measurement and the opacity can be solved rather than guessed.
    Capped at 0.80: past that the slide stops being a photograph.
    """
    worst_level = M.level_for_contrast(ratio)
    if worst_level <= 0:
        return current
    underlying = worst_level / max(0.05, 1.0 - current)
    needed = 1.0 - (M.CONTRAST_TARGET_LEVEL / underlying)
    # Round up to the next 0.05 so the fix clears the threshold instead of grazing it.
    stepped = min(0.80, max(current + 0.05, (int(needed / 0.05) + 1) * 0.05))
    return round(stepped, 2)


def _scene_recommendations(score: SceneScore, timeline: Timeline) -> list[Recommendation]:
    out: list[Recommendation] = []
    metrics = score.metrics
    if metrics is None:
        return out
    scene = next((s for s in timeline.scenes if s.id == score.scene_id), None)
    scene_id = score.scene_id

    # ---- legibility
    contrast = metrics.contrast
    if contrast is not None and contrast.ratio < M.MIN_CONTRAST_RATIO:
        current = scene.plan.scrim_opacity if scene and scene.plan else 0.45
        proposed = _contrast_fix_opacity(contrast.ratio, current)
        severity = Severity.BLOCKER if contrast.ratio < 3.0 else Severity.MAJOR
        evidence = (
            f"worst-strip contrast {contrast.ratio}:1 (typical {contrast.ratio_median}:1) "
            f"over {contrast.frames_sampled} frames at {contrast.region}"
        )
        problem = (
            f"White heading measures {contrast.ratio}:1 against its background at the "
            f"worst point — below the {M.MIN_CONTRAST_RATIO}:1 target."
        )
        if proposed > current:
            out.append(
                Recommendation(
                    severity=severity,
                    dimension=Dimension.LEGIBILITY,
                    scene_id=scene_id,
                    problem=problem,
                    fix=(
                        f"Raise scrim_opacity from {current} to {proposed} for this scene "
                        f"(brings the worst strip to about {M.MIN_CONTRAST_RATIO}:1)."
                    ),
                    auto_fixable=True,
                    action="raise_scrim_opacity",
                    params={"scene_id": scene_id, "from": current, "to": proposed},
                    evidence=evidence,
                )
            )
        else:
            # No scrim can rescue this without obliterating the image. Emitting a
            # from-0.80-to-0.80 "auto-fix" would burn a pipeline pass and change nothing.
            out.append(
                Recommendation(
                    severity=severity,
                    dimension=Dimension.LEGIBILITY,
                    scene_id=scene_id,
                    problem=problem,
                    fix=(
                        f"scrim_opacity is already at the {current} ceiling and still is not "
                        "enough. Regenerate this image with a darker area where the text sits, "
                        "or move the heading."
                    ),
                    auto_fixable=False,
                    evidence=evidence,
                )
            )
        if (
            contrast.alt_ratio is not None
            and contrast.alt_position
            and contrast.alt_ratio >= max(M.MIN_CONTRAST_RATIO, contrast.ratio * 1.5)
        ):
            out.append(
                Recommendation(
                    severity=Severity.MINOR,
                    dimension=Dimension.LEGIBILITY,
                    scene_id=scene_id,
                    problem=(
                        f"The {contrast.alt_position} band of this image is much darker than the "
                        f"band the heading is actually in."
                    ),
                    fix=(
                        f"Set text_position to {contrast.alt_position}: contrast there measures "
                        f"{contrast.alt_ratio}:1 versus {contrast.ratio}:1 where it is now. "
                        f"Cheaper than a heavier scrim and it keeps the image visible."
                    ),
                    auto_fixable=True,
                    action="move_text_position",
                    params={"scene_id": scene_id, "text_position": contrast.alt_position},
                    evidence=f"alt band worst strip {contrast.alt_ratio}:1",
                )
            )

    if score.legibility is not None and score.legibility < 5.0 and contrast is None:
        out.append(
            Recommendation(
                severity=Severity.MAJOR,
                dimension=Dimension.LEGIBILITY,
                scene_id=scene_id,
                problem="The heading was judged hard to read and no contrast measurement exists.",
                fix="Check the scrim actually rendered; the scene has no VisualPlan to measure.",
                auto_fixable=False,
            )
        )

    # ---- relevance
    if score.relevance is not None and score.relevance < 7.0:
        severity = Severity.BLOCKER if score.relevance <= 4.0 else Severity.MAJOR
        prompt = score.suggested_image_prompt
        out.append(
            Recommendation(
                severity=severity,
                dimension=Dimension.RELEVANCE,
                scene_id=scene_id,
                problem=(
                    f"The image does not depict this scene's subject "
                    f"(topical relevance {score.relevance:.0f}/10 for "
                    f"{score.heading!r})."
                ),
                fix=(
                    "Regenerate the image with the suggested prompt."
                    if prompt
                    else "Regenerate the image with a prompt naming the narration's actual subject."
                ),
                auto_fixable=bool(prompt),
                action="regenerate_scene_image" if prompt else None,
                params={"scene_id": scene_id, "image_prompt": prompt} if prompt else {},
                evidence="; ".join(score.issues[:3]) or None,
            )
        )

    # ---- motion
    ratio = metrics.duplicate_frame_ratio
    if ratio is not None and ratio >= M.DUPLICATE_NOISE_FLOOR:
        upscale = timeline.profile.upscale_factor
        out.append(
            Recommendation(
                severity=Severity.MAJOR if ratio >= 0.20 else Severity.MINOR,
                dimension=Dimension.MOTION,
                scene_id=scene_id,
                problem=(
                    f"{ratio:.1%} of frames repeat the previous one — the camera move steps "
                    f"(noise floor for this measurement is {M.DUPLICATE_NOISE_FLOOR:.0%})."
                ),
                fix=(
                    f"Raise RenderProfile.upscale_factor from {upscale} to {max(4, upscale * 2)}: "
                    "zoompan truncates x/y to integers, so a slow move holds position for "
                    "several frames unless the source is pre-upscaled."
                ),
                auto_fixable=True,
                action="raise_upscale_factor",
                params={"from": upscale, "to": max(4, upscale * 2)},
                evidence=f"{M.MPDECIMATE} over {M.DUPLICATE_SAMPLE_SECONDS}s",
            )
        )

    # ---- pacing
    wpm = metrics.words_per_minute
    if wpm is not None and not (M.MIN_WPM <= wpm <= M.MAX_WPM):
        slow = wpm < M.MIN_WPM
        out.append(
            Recommendation(
                severity=Severity.MINOR,
                dimension=Dimension.PACING,
                scene_id=scene_id,
                problem=(
                    f"Narration runs at {wpm:.0f} wpm, "
                    f"{'below' if slow else 'above'} the {M.MIN_WPM:.0f}-{M.MAX_WPM:.0f} band."
                ),
                fix=(
                    (
                        "Add a clause or two so the delivery has something to carry"
                        if slow
                        else "Cut a clause so the delivery can breathe"
                    )
                    + ", or pick a voice whose natural pace is "
                    + ("faster" if slow else "slower")
                    + ". Changing tempo mechanically would desync the word timings the "
                    "whole render is built on."
                ),
                auto_fixable=False,
                evidence=f"{len(scene.words) if scene else 0} words over the spoken span",
            )
        )
    deviation = metrics.duration_deviation
    if deviation is not None and abs(deviation) >= 0.35:
        out.append(
            Recommendation(
                severity=Severity.MINOR,
                dimension=Dimension.PACING,
                scene_id=scene_id,
                problem=(
                    f"This scene is {abs(deviation):.0%} "
                    f"{'longer' if deviation > 0 else 'shorter'} than the median scene."
                ),
                fix="Rebalance the narration across scenes so the slides hold roughly equal time.",
                auto_fixable=False,
            )
        )

    # ---- timing
    for issue in metrics.bullet_issues:
        blocker = "after the scene ends" in issue or "out of order" in issue
        out.append(
            Recommendation(
                severity=Severity.BLOCKER if blocker else Severity.MINOR,
                dimension=Dimension.TIMING,
                scene_id=scene_id,
                problem=issue,
                fix=(
                    f"Re-space the scene's bullet appear_at values to the "
                    f"{M.MIN_BULLET_GAP}s floor inside the scene's own duration."
                ),
                auto_fixable=True,
                action="respace_bullets",
                params={"scene_id": scene_id, "min_gap": M.MIN_BULLET_GAP},
            )
        )
    for start, end in metrics.narration_gaps:
        out.append(
            Recommendation(
                severity=Severity.MAJOR if end - start >= 1.0 else Severity.MINOR,
                dimension=Dimension.TIMING,
                scene_id=scene_id,
                problem=f"{end - start:.2f}s hole in the narration at {start:.2f}s into the scene.",
                fix="Check the TTS output for a dropped clause; re-synthesize this scene.",
                auto_fixable=False,
                evidence=f"gap {start:.2f}-{end:.2f}s (scene-relative)",
            )
        )
    for start, end in metrics.silence_windows:
        out.append(
            Recommendation(
                severity=Severity.MINOR,
                dimension=Dimension.TIMING,
                scene_id=scene_id,
                problem=f"Silence in the narration asset from {start:.2f}s to {end:.2f}s.",
                fix="Re-synthesize this scene's narration; the voice dropped out mid-scene.",
                auto_fixable=False,
            )
        )

    # ---- composition / professionalism, from the vision pass only
    if score.composition is not None and score.composition <= 5.0:
        out.append(
            Recommendation(
                severity=Severity.MINOR,
                dimension=Dimension.COMPOSITION,
                scene_id=scene_id,
                problem=f"Composition scored {score.composition:.0f}/10 — no clean area for text.",
                fix=(
                    "Add an explicit empty-space instruction to the image prompt "
                    "(\"generous uncluttered lower third\") and regenerate."
                ),
                auto_fixable=False,
                evidence="; ".join(score.issues[:2]) or None,
            )
        )
    if score.professionalism is not None and score.professionalism <= 5.0:
        out.append(
            Recommendation(
                severity=Severity.MINOR,
                dimension=Dimension.PROFESSIONALISM,
                scene_id=scene_id,
                problem=f"Reads as stock filler rather than training material "
                f"({score.professionalism:.0f}/10).",
                fix="Tighten the image prompt toward the subject matter and a consistent look.",
                auto_fixable=False,
            )
        )
    return out


def _video_recommendations(
    metrics: VideoMetrics, report: VisionReport | None, timeline: Timeline
) -> list[Recommendation]:
    out: list[Recommendation] = []

    loudness = metrics.loudness
    if loudness:
        deviation = loudness.integrated_lufs - M.TARGET_LUFS
        if abs(deviation) >= 1.5:
            out.append(
                Recommendation(
                    severity=Severity.MAJOR if abs(deviation) >= 3.0 else Severity.MINOR,
                    dimension=Dimension.AUDIO,
                    scene_id=None,
                    problem=(
                        f"Integrated loudness is {loudness.integrated_lufs:.1f} LUFS, "
                        f"{abs(deviation):.1f} LU {'below' if deviation < 0 else 'above'} the "
                        f"{M.TARGET_LUFS:.0f} LUFS web target."
                    ),
                    fix=(
                        f"Normalise the final mix: apply {-deviation:+.1f} dB, or add "
                        f"loudnorm=I={M.TARGET_LUFS:.0f}:TP={M.MAX_TRUE_PEAK_DBFS:.0f}:LRA=11 "
                        "to the assembly step."
                    ),
                    auto_fixable=True,
                    action="renormalize_loudness",
                    params={
                        "target_lufs": M.TARGET_LUFS,
                        "measured_lufs": loudness.integrated_lufs,
                        "gain_db": round(-deviation, 1),
                        "true_peak_ceiling_dbfs": M.MAX_TRUE_PEAK_DBFS,
                    },
                    evidence=(
                        f"ebur128 I={loudness.integrated_lufs} LUFS, "
                        f"LRA={loudness.loudness_range_lu} LU"
                    ),
                )
            )
        if loudness.true_peak_dbfs > M.MAX_TRUE_PEAK_DBFS:
            out.append(
                Recommendation(
                    severity=Severity.BLOCKER if loudness.true_peak_dbfs >= 0.0 else Severity.MAJOR,
                    dimension=Dimension.AUDIO,
                    scene_id=None,
                    problem=(
                        f"True peak is {loudness.true_peak_dbfs:.1f} dBFS, above the "
                        f"{M.MAX_TRUE_PEAK_DBFS:.0f} dBFS ceiling."
                    ),
                    fix=f"Add alimiter=limit={M.MAX_TRUE_PEAK_DBFS:.0f}dB to the mix.",
                    auto_fixable=True,
                    action="limit_true_peak",
                    params={"ceiling_dbfs": M.MAX_TRUE_PEAK_DBFS},
                    evidence=f"ebur128 true peak {loudness.true_peak_dbfs} dBFS",
                )
            )

    balance = metrics.balance
    if balance and balance.measured and balance.separation_db < M.MIN_SPEECH_BED_SEPARATION_DB:
        shortfall = M.MIN_SPEECH_BED_SEPARATION_DB - balance.separation_db
        out.append(
            Recommendation(
                severity=Severity.MAJOR if balance.separation_db < 6.0 else Severity.MINOR,
                dimension=Dimension.AUDIO,
                scene_id=None,
                problem=(
                    f"Narration sits only {balance.separation_db:.1f} dB above the music bed "
                    f"(speech {balance.speech_dbfs:.1f} dBFS, bed {balance.bed_dbfs:.1f} dBFS)."
                ),
                fix=(
                    f"Duck the music a further {shortfall:.1f} dB during speech "
                    f"(VIDEO_MUSIC_DUCK_DB), or use sidechaincompress against the narration."
                ),
                auto_fixable=True,
                action="duck_music",
                params={"additional_gain_db": -round(shortfall, 1)},
                evidence=f"measured over {balance.windows_sampled} narration gaps",
            )
        )
    elif balance and not balance.measured:
        out.append(
            Recommendation(
                severity=Severity.NIT,
                dimension=Dimension.AUDIO,
                scene_id=None,
                problem="Narration/music balance could not be measured: no long enough gap.",
                fix="No action; the metric needs a >=0.35s pause somewhere in the narration.",
                auto_fixable=False,
            )
        )

    if metrics.duration_drift_frames > M.MAX_DURATION_DRIFT_FRAMES:
        out.append(
            Recommendation(
                severity=Severity.BLOCKER if metrics.duration_drift_frames > 30 else Severity.MAJOR,
                dimension=Dimension.TECHNICAL,
                scene_id=None,
                problem=(
                    f"File is {metrics.duration:.2f}s but the Timeline predicts "
                    f"{metrics.expected_duration:.2f}s — "
                    f"{metrics.duration_drift_frames:.1f} frames "
                    "of drift."
                ),
                fix=(
                    "Check the xfade offsets in the assembly step: each boundary consumes one "
                    "transition_duration, and losing one desyncs narration cumulatively."
                ),
                auto_fixable=False,
                evidence=(
                    f"ffprobe {metrics.duration}s vs "
                    f"final_duration() {metrics.expected_duration}s"
                ),
            )
        )
    for mismatch in metrics.profile_mismatch:
        out.append(
            Recommendation(
                severity=Severity.MAJOR,
                dimension=Dimension.TECHNICAL,
                scene_id=None,
                problem=f"Output disagrees with its RenderProfile: {mismatch}.",
                fix="Re-render, or correct the profile stored on the Timeline.",
                auto_fixable=False,
            )
        )

    if report and report.script:
        verdict = report.script
        if verdict.actionability <= 6:
            out.append(
                Recommendation(
                    severity=Severity.MINOR,
                    dimension=Dimension.SCRIPT,
                    scene_id=None,
                    problem=f"Actionability {verdict.actionability}/10 — the viewer is informed "
                    "but not told what to do.",
                    fix=(verdict.suggestions[0] if verdict.suggestions else
                         "Add a closing slide with two or three concrete actions."),
                    auto_fixable=False,
                )
            )
        if verdict.narrative_flow <= 6:
            out.append(
                Recommendation(
                    severity=Severity.MINOR,
                    dimension=Dimension.SCRIPT,
                    scene_id=None,
                    problem=f"Narrative flow {verdict.narrative_flow}/10.",
                    fix=(verdict.issues[0] if verdict.issues else "Reorder or bridge the slides."),
                    auto_fixable=False,
                )
            )
        if verdict.bullets_echo_narration == "no_bullets":
            out.append(
                Recommendation(
                    severity=Severity.NIT,
                    dimension=Dimension.SCRIPT,
                    scene_id=None,
                    problem="No slide carries bullets, so bullet timing could not be assessed.",
                    fix=(
                        "Expected for jobs rendered before Scene.bullets existed. New jobs should "
                        "carry 3-5 points per slide echoing the narration."
                    ),
                    auto_fixable=False,
                )
            )
        elif verdict.bullets_echo_narration in {"no", "partly"}:
            out.append(
                Recommendation(
                    severity=Severity.MINOR,
                    dimension=Dimension.SCRIPT,
                    scene_id=None,
                    problem=f"Bullets only {verdict.bullets_echo_narration} echo the narration.",
                    fix="Regenerate bullets as verbatim phrases lifted from the narration.",
                    auto_fixable=False,
                )
            )

    if not any(s.bullets for s in timeline.scenes) and not (report and report.script):
        out.append(
            Recommendation(
                severity=Severity.NIT,
                dimension=Dimension.TIMING,
                scene_id=None,
                problem="No scene carries bullets; bullet timing was not assessed.",
                fix="Expected for jobs predating Scene.bullets.",
                auto_fixable=False,
            )
        )
    return out


# --------------------------------------------------------------------------- top level


def _cap_grade(overall: float, recommendations: list[Recommendation]) -> tuple[Grade, bool]:
    """Weighted grade, held down by the worst finding. See the module docstring."""
    grade = grade_for(overall)
    ceiling = grade
    for severity, limit in GRADE_CEILING.items():
        if any(r.severity == severity for r in recommendations):
            if _GRADE_ORDER.index(limit) > _GRADE_ORDER.index(ceiling):
                ceiling = limit
    return ceiling, ceiling != grade


def score_timeline(
    timeline: Timeline,
    video_path: Path,
    *,
    vision: bool = True,
    vision_model: str | None = None,
    api_key: str = "",
) -> VideoScore:
    """Measure, judge, and fold into one :class:`VideoScore`."""
    from app.evaluate.vision import DEFAULT_MODEL, VisionUnavailable

    notes: list[str] = []
    report: VisionReport | None = None
    if vision:
        try:
            report = judge_timeline(
                timeline, video_path, api_key=api_key, model=vision_model or DEFAULT_MODEL
            )
            notes.extend(report.errors)
        except VisionUnavailable as exc:
            notes.append(f"vision pass skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - a scorecard is still worth producing
            logger.warning("vision pass failed", exc_info=True)
            notes.append(f"vision pass failed: {exc}")
    else:
        notes.append("vision pass disabled (--no-vision): relevance, composition, "
                     "professionalism and script were not assessed")

    measure = measure_all(timeline, video_path)
    video_metrics = measure.video
    scene_scores = [score_scene(m, report) for m in measure.scenes]

    recommendations: list[Recommendation] = []
    for score in scene_scores:
        recommendations.extend(_scene_recommendations(score, timeline))
    recommendations.extend(_video_recommendations(video_metrics, report, timeline))

    scene_average = (
        round(sum(s.overall for s in scene_scores) / len(scene_scores), 1) if scene_scores else 0.0
    )
    audio = _audio_score(video_metrics, measure.scenes)
    script = _script_score(report)
    technical = _technical_score(video_metrics)

    components = {"scenes": scene_average / 10.0, "audio": audio, "script": script,
                  "technical": technical}
    usable = {k: v for k, v in components.items() if v is not None}
    total_weight = sum(VIDEO_WEIGHTS[k] for k in usable)
    overall = (
        round(sum(usable[k] * VIDEO_WEIGHTS[k] for k in usable) / total_weight * 10.0, 1)
        if total_weight
        else 0.0
    )
    grade, capped = _cap_grade(overall, recommendations)

    if report is None and vision:
        notes.append("scored on deterministic metrics only; relevance was not assessed")

    return VideoScore(
        job_id=timeline.job_id,
        topic=timeline.topic,
        title=timeline.title,
        video_path=str(video_path),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        evaluator_version=EVALUATOR_VERSION,
        vision_used=report is not None,
        vision_model=report.model if report else None,
        overall=overall,
        grade=grade,
        grade_capped=capped,
        scene_average=scene_average,
        audio_score=audio,
        script_score=script,
        technical_score=technical,
        narrative_flow=float(report.script.narrative_flow) if report and report.script else None,
        clarity=float(report.script.clarity) if report and report.script else None,
        actionability=float(report.script.actionability) if report and report.script else None,
        bullets_echo_narration=(
            report.script.bullets_echo_narration == "yes" if report and report.script else None
        ),
        scenes=scene_scores,
        recommendations=recommendations,
        metrics=video_metrics,
        notes=notes,
    )


class _Measurements:
    """Scene and video measurements taken together, so the video is probed once."""

    def __init__(self, scenes: list[SceneMetrics], video: VideoMetrics) -> None:
        self.scenes = scenes
        self.video = video


def measure_all(timeline: Timeline, video_path: Path) -> _Measurements:
    return _Measurements(
        [M.measure_scene(s, timeline, video_path) for s in timeline.scenes],
        M.measure_video(timeline, video_path),
    )


def score_job(
    job_id: str,
    *,
    video_path: Path | None = None,
    timeline_path: Path | None = None,
    vision: bool = True,
    vision_model: str | None = None,
    api_key: str = "",
) -> VideoScore:
    """Load a job's Timeline and rendered file, then score them."""
    timeline = load_timeline(job_id, timeline_path=timeline_path)
    resolved = resolve_video(job_id, timeline, video_path)
    if not resolved.exists():
        raise FileNotFoundError(f"no rendered video at {resolved}")
    return score_timeline(
        timeline, resolved, vision=vision, vision_model=vision_model, api_key=api_key
    )


# ------------------------------------------------------------------------- reporting


def write_score(score: VideoScore, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(score.model_dump(mode="json"), indent=2) + "\n")
    return path


_BAR_WIDTH = 20
_SEVERITY_TAG = {
    Severity.BLOCKER: "BLOCKER",
    Severity.MAJOR: "MAJOR  ",
    Severity.MINOR: "minor  ",
    Severity.NIT: "note   ",
}


def _bar(value: float | None, out_of: float = 10.0) -> str:
    if value is None:
        return "·" * _BAR_WIDTH
    filled = int(round(_BAR_WIDTH * max(0.0, min(1.0, value / out_of))))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _cell(value: float | None) -> str:
    return " n/a" if value is None else f"{value:4.1f}"


def render_report(score: VideoScore, *, width: int = 92) -> str:
    """The terminal scorecard. Ordered worst-first, because that is the work queue."""
    rule = "─" * width
    lines: list[str] = [
        rule,
        f"  {score.title or score.topic}",
        f"  job {score.job_id}   {score.video_path}",
        rule,
        f"  OVERALL {score.overall:.1f}/100   GRADE {score.grade.value}"
        + ("  (capped by a blocking finding)" if score.grade_capped else ""),
        "",
        f"    scenes     {_cell(score.scene_average / 10)}  {_bar(score.scene_average / 10)}"
        f"   weight {VIDEO_WEIGHTS['scenes']:.0%}",
        f"    audio      {_cell(score.audio_score)}  {_bar(score.audio_score)}"
        f"   weight {VIDEO_WEIGHTS['audio']:.0%}",
        f"    script     {_cell(score.script_score)}  {_bar(score.script_score)}"
        f"   weight {VIDEO_WEIGHTS['script']:.0%}",
        f"    technical  {_cell(score.technical_score)}  {_bar(score.technical_score)}"
        f"   weight {VIDEO_WEIGHTS['technical']:.0%}",
        rule,
        "  PER SCENE" + (" " * 12) + "legib  relev  compo  profe  motio  pacin  timin  = /100",
    ]
    for scene in score.scenes:
        label = f"{scene.scene_id}. {scene.heading}"[:26].ljust(26)
        lines.append(
            f"  {label}"
            f"{_cell(scene.legibility)}   {_cell(scene.relevance)}   {_cell(scene.composition)}   "
            f"{_cell(scene.professionalism)}   {_cell(scene.motion)}   {_cell(scene.pacing)}   "
            f"{_cell(scene.timing)}   {scene.overall:5.1f} {scene.grade.value}"
        )

    lines += [rule, "  MEASUREMENTS"]
    for scene in score.scenes:
        metrics = scene.metrics
        if metrics is None:
            continue
        contrast = metrics.contrast
        bits = [
            f"contrast {contrast.ratio}:1 (typ {contrast.ratio_median}:1)"
            if contrast
            else "contrast n/a",
            f"dup {metrics.duplicate_frame_ratio:.1%}"
            if metrics.duplicate_frame_ratio is not None
            else "dup n/a",
            f"{metrics.words_per_minute:.0f} wpm" if metrics.words_per_minute else "wpm n/a",
            f"{metrics.duration:.2f}s",
            f"{metrics.bullet_count} bullets",
        ]
        if contrast and contrast.alt_ratio is not None:
            bits.append(f"{contrast.alt_position} would be {contrast.alt_ratio}:1")
        lines.append(f"    scene {metrics.scene_id}: " + ", ".join(bits))
    if score.metrics:
        video = score.metrics
        loudness = video.loudness
        lines.append(
            f"    video: {video.duration:.2f}s (expected {video.expected_duration:.2f}s, "
            f"{video.duration_drift_frames:.1f} frames drift), {video.width}x{video.height} "
            f"@{video.fps:g}fps, {video.video_codec}+{video.audio_codec}"
        )
        if loudness:
            lines.append(
                f"    audio: {loudness.integrated_lufs:.1f} LUFS "
                f"(target {M.TARGET_LUFS:.0f}), true peak {loudness.true_peak_dbfs:.1f} dBFS, "
                f"LRA {loudness.loudness_range_lu:.1f} LU"
            )
        if video.balance:
            balance = video.balance
            lines.append(
                f"    balance: speech {balance.speech_dbfs:.1f} dBFS vs bed "
                f"{balance.bed_dbfs:.1f} dBFS = {balance.separation_db:.1f} dB separation"
                + ("" if balance.measured else " (not measurable)")
            )

    ordered = score.by_severity()
    lines += [rule, f"  RECOMMENDATIONS ({len(ordered)}, "
              f"{len(score.auto_fixable())} auto-fixable)"]
    if not ordered:
        lines.append("    nothing to fix.")
    for rec in ordered:
        scope = f"scene {rec.scene_id}" if rec.scene_id else "video"
        tag = "AUTO" if rec.auto_fixable else "    "
        lines.append(
            f"  [{_SEVERITY_TAG[rec.severity]}] [{tag}] {scope} / {rec.dimension.value}"
        )
        lines.append(f"      problem: {rec.problem}")
        lines.append(f"      fix:     {rec.fix}")
        if rec.action:
            lines.append(f"      action:  {rec.action} {json.dumps(rec.params)}")
        if rec.evidence:
            lines.append(f"      why:     {rec.evidence}")

    if score.notes:
        lines += [rule, "  NOTES"]
        lines += [f"    - {note}" for note in score.notes]
    lines.append(rule)
    return "\n".join(lines)
