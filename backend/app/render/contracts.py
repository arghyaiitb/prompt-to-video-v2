"""Seam between text rasterisation and filtergraph construction.

`text_overlay` decides *what the text looks like and where it sits* and rasterises it.
`ffmpeg_backend` decides *how it enters the frame* by turning each asset into an
overlay chain. Neither needs to know the other's internals — this module is the
only shared vocabulary.

Kept separate from `core/models.py` because these are render-time artifacts
(files on disk), not part of the persisted Timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.models import TextAnimation


@dataclass(frozen=True)
class TextLayer:
    """One rasterised text element, positioned and timed, ready to overlay.

    Position is the layer's FINAL resting place in output-frame pixels. Entry
    animations move toward it; they never change where it ends up.
    """

    png_path: Path
    x: int
    y: int
    width: int
    height: int

    appear_at: float = 0.0
    """Seconds from scene start. Layers before this are fully transparent."""

    disappear_at: float | None = None
    """Optional exit time. None = stays until the clip ends."""

    animation: TextAnimation = TextAnimation.FADE_IN
    anim_duration: float = 0.45

    slide_distance: int = 60
    """Pixels travelled for SLIDE_* animations. Ignored otherwise."""

    kind: str = "bullet"
    """`scrim` | `heading` | `bullet` — for logging and ordering only."""


@dataclass
class SceneText:
    """Everything to composite over one scene, in draw order (back to front)."""

    layers: list[TextLayer] = field(default_factory=list)

    def sorted_layers(self) -> list[TextLayer]:
        """Scrim first, then heading, then bullets by appear time.

        Order matters: the scrim must land under the text it makes legible.
        """
        rank = {"scrim": 0, "heading": 1, "bullet": 2}
        return sorted(
            self.layers, key=lambda layer: (rank.get(layer.kind, 3), layer.appear_at)
        )

    @property
    def inputs(self) -> list[Path]:
        return [layer.png_path for layer in self.sorted_layers()]
