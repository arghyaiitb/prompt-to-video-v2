"""Automated quality evaluation for a rendered video.

Three layers, deliberately separated so the cheap ones can run without the expensive one:

``metrics``
    Deterministic measurement via ffmpeg/ffprobe. No network, no model, no opinion.
    Every number here is reproducible and can be asserted on in CI.

``vision``
    Gemini judgement on a representative frame per scene plus one pass over the script.
    Catches the things arithmetic cannot: "this image has nothing to do with phishing".

``scorer``
    Folds both into :class:`~app.evaluate.models.VideoScore` — per-dimension 0-10, a
    weighted 0-100 overall, a letter grade, and a list of concrete
    :class:`~app.evaluate.models.Recommendation` objects, each flagged with whether the
    pipeline could apply it mechanically.

The point is the loop: score, fix the top recommendation, re-score, prove the number
moved. A rating that cannot be acted on is a vanity metric.
"""

from __future__ import annotations

from app.evaluate.models import (
    Dimension,
    Grade,
    Recommendation,
    SceneMetrics,
    SceneScore,
    Severity,
    VideoMetrics,
    VideoScore,
)
from app.evaluate.scorer import score_job, score_timeline

__all__ = [
    "Dimension",
    "Grade",
    "Recommendation",
    "SceneMetrics",
    "SceneScore",
    "Severity",
    "VideoMetrics",
    "VideoScore",
    "score_job",
    "score_timeline",
]
