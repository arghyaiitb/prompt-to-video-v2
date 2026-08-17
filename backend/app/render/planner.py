"""Deterministic visual planning: every creative decision, no side effects.

Pure by construction — no network, no ffmpeg, no filesystem. The renderer is a dumb
executor of whatever this module decides, which is what makes both testable.
"""

from __future__ import annotations

from app.core.models import Motion, Scene, TextPosition, Timeline, Transition, VisualPlan

MOTION_ROTATION: tuple[Motion, ...] = (
    Motion.ZOOM_IN,
    Motion.PAN_RIGHT,
    Motion.ZOOM_OUT,
    Motion.PAN_LEFT,
)
"""Rotation used to break up repeats. STATIC is deliberately excluded as a *target*:
two dead slides in a row is exactly the boredom this rule exists to prevent."""

TRANSITION_ROTATION: tuple[Transition, ...] = (
    Transition.DISSOLVE,
    Transition.SLIDE_LEFT,
    Transition.WIPE_RIGHT,
)

ZOOM_SPANS: tuple[float, ...] = (0.10, 0.12, 0.15, 0.08)
"""Subtle by design. Anything past ~15% reads as a cheap slideshow."""

MIN_ZOOM_SPAN = 0.08
MAX_ZOOM_SPAN = 0.15

DEFAULT_TRANSITION_DURATION = 0.5
TRANSITION_MAX_SCENE_FRACTION = 0.40
"""A 0.5s dissolve across a 0.8s scene is a glitch, not a transition."""

MIN_TRANSITION_DURATION = 0.10
"""Below this a crossfade reads as a dropped frame; we demote to a hard CUT."""


class RuleBasedPlanner:
    """Fills in :class:`VisualPlan` for every scene. Satisfies ``VisualPlanner``."""

    def __init__(
        self,
        *,
        transition_duration: float = DEFAULT_TRANSITION_DURATION,
        scrim_opacity: float = 0.45,
        max_scene_fraction: float = TRANSITION_MAX_SCENE_FRACTION,
        min_transition_duration: float = MIN_TRANSITION_DURATION,
        easing: str = "ease_in_out",
    ) -> None:
        self.transition_duration = transition_duration
        self.scrim_opacity = scrim_opacity
        self.max_scene_fraction = max_scene_fraction
        self.min_transition_duration = min_transition_duration
        self.easing = easing

    # ---------------------------------------------------------------- public

    def plan(self, timeline: Timeline) -> Timeline:
        """Return a copy of ``timeline`` with every ``Scene.plan`` populated.

        The input is never mutated: planning is a pure function of the timeline,
        which keeps re-planning idempotent and unit tests trivial.
        """
        out = timeline.model_copy(deep=True)
        previous_motion: Motion | None = None

        for index, scene in enumerate(out.scenes):
            requested = scene.plan.motion if scene.plan is not None else self._default_motion(index)
            motion = self._enforce_variety(requested, previous_motion)
            zoom_from, zoom_to = self._zoom_range(motion, index)
            transition = self._transition_for(index)
            duration = self._transition_duration(out.scenes, index)

            if transition is not Transition.CUT and duration < self.min_transition_duration:
                # Keep transition_in and transition_duration self-consistent so
                # Timeline.final_duration() and the renderer agree on the maths.
                transition, duration = Transition.CUT, 0.0

            scene.plan = VisualPlan(
                motion=motion,
                zoom_from=zoom_from,
                zoom_to=zoom_to,
                easing="linear" if motion is Motion.STATIC else self.easing,  # type: ignore[arg-type]
                transition_in=transition,
                transition_duration=duration,
                text_position=self._text_position(index),
                scrim_opacity=self.scrim_opacity,
            )
            previous_motion = motion

        return out

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _default_motion(index: int) -> Motion:
        return MOTION_ROTATION[index % len(MOTION_ROTATION)]

    @staticmethod
    def _enforce_variety(requested: Motion, previous: Motion | None) -> Motion:
        """Never the same move twice in a row — rotate to the next sensible one."""
        if previous is None or requested is not previous:
            return requested
        if requested in MOTION_ROTATION:
            pos = MOTION_ROTATION.index(requested)
            return MOTION_ROTATION[(pos + 1) % len(MOTION_ROTATION)]
        return MOTION_ROTATION[0]  # e.g. STATIC twice in a row

    @staticmethod
    def _zoom_range(motion: Motion, index: int) -> tuple[float, float]:
        span = ZOOM_SPANS[index % len(ZOOM_SPANS)]
        span = min(MAX_ZOOM_SPAN, max(MIN_ZOOM_SPAN, span))
        if motion is Motion.ZOOM_IN:
            return 1.0, round(1.0 + span, 4)
        if motion is Motion.ZOOM_OUT:
            return round(1.0 + span, 4), 1.0
        if motion is Motion.STATIC:
            return 1.0, 1.0
        # Pans need headroom to travel across, so they sit at a fixed slight zoom.
        held = round(1.0 + span, 4)
        return held, held

    @staticmethod
    def _transition_for(index: int) -> Transition:
        if index == 0:
            return Transition.FADE  # fade up from black
        return TRANSITION_ROTATION[(index - 1) % len(TRANSITION_ROTATION)]

    def _transition_duration(self, scenes: list[Scene], index: int) -> float:
        """Clamp to a fraction of the *shorter* adjacent scene."""
        neighbours = [scenes[index].duration]
        if index > 0:
            neighbours.append(scenes[index - 1].duration)
        shortest = min(neighbours)
        limit = max(0.0, shortest * self.max_scene_fraction)
        return round(min(self.transition_duration, limit), 3)

    @staticmethod
    def _text_position(index: int) -> TextPosition:
        return TextPosition.LOWER_THIRD if index % 2 == 0 else TextPosition.UPPER_THIRD
