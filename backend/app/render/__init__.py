"""Rendering: deterministic visual planning plus an ffmpeg execution backend."""

from app.render.ffmpeg_backend import FFmpegBackend
from app.render.planner import RuleBasedPlanner

__all__ = ["FFmpegBackend", "RuleBasedPlanner"]
