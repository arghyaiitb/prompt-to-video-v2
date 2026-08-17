"""Deterministic visual planning: every creative decision, no side effects.

Pure by construction — no network, no ffmpeg, no filesystem. The renderer is a dumb
executor of whatever this module decides, which is what makes both testable.

Four rules here exist because of something measured on real output rather than
something assumed:

*No dissolve.* ``xfade=transition=dissolve`` on this ffmpeg build is a noise/dither
dissolve — mid-transition frames are visibly grainy rather than a smooth blend. The
enum member stays (old timelines deserialise, and a future build may fix it) but the
planner uses ``fade``, which really is a smooth crossfade.

*One decision per video, not per scene.* This module used to alternate the layout, the
camera move, the transition and both text entrances on every scene, on the theory that
variety keeps a viewer awake. Watch the result: it reads as four unrelated templates
spliced together, and the first thing anyone said about it was that the video was "all
over the place". A real deck holds ONE body layout, ONE camera move, ONE transition and
ONE pair of entrances; what varies between scenes is the *content*. So every creative
choice below is resolved once, for the whole timeline, from the timeline. The
``hold_*`` flags turn each back into the old per-scene rotation if that is ever wanted,
and the rotations they'd use are still exported for tests.

*Structure comes from roles, not from position.* :class:`~app.core.models.SceneRole`
says what a scene is for; the type scale and the bullet budget follow from that. It does
**not** buy a scene its own layout: the title card is the only frame that differs, and a
summary or a closing earns its difference from having fewer bullets and different words.
The numbers come from ``docs/DIRECTION.md``, which is normative for this module.

*Variety inside a scene is timing.* The bullets in one scene share an entrance —
a stack arriving from four directions reads as chaos — and stagger instead.
"""

from __future__ import annotations

from collections import Counter

from app.core.models import (
    Motion,
    Scene,
    SceneRole,
    SlideLayout,
    TextAnimation,
    TextPosition,
    Timeline,
    Transition,
    VisualPlan,
)

MOTION_ROTATION: tuple[Motion, ...] = (
    Motion.ZOOM_IN,
    Motion.PAN_RIGHT,
    Motion.ZOOM_OUT,
    Motion.PAN_LEFT,
)
"""Only used when ``hold_motion=False``. STATIC is excluded as a *target* either way.

The default holds one move for the whole video — see :data:`HELD_MOTION`. Rotating the
camera direction between adjacent scenes is visible, and a Ken Burns move exists to keep the
frame from feeling dead, not to be noticed.
"""

TRANSITION_ROTATION: tuple[Transition, ...] = (Transition.FADE,)
"""One transition. ``SLIDE_LEFT`` and ``WIPE_RIGHT`` are **removed**, not deprioritised.

They are not a style choice, they are a defect when the text is burned into the scene clip
(DIRECTION §4.3, defects 7 and 8): at ``b85 t=57s`` the wipe edge cuts through both text
stacks and puts two headings and eight bullet fragments on screen at once; at ``t=38s`` the
slide catches the incoming heading cropped mid-word. ``DISSOLVE`` is absent too — see the
module docstring — leaving ``FADE``, the smooth crossfade this build actually has.
"""

LAYOUT_ROTATION: tuple[SlideLayout, ...] = (SlideLayout.HERO_RIGHT,)
"""**Two layouts in the whole video**: ``title_card`` for the opener, ``hero_right`` for
everything else. DIRECTION §6.1.

``hero_left``, ``image_band`` and ``full_bleed`` are retired — the enum members stay so old
timelines deserialise, but the planner never emits them:

* alternating hero sides moves the text block ~940px horizontally between consecutive
  scenes (``b85 t=21s`` vs ``t=45s``), so the viewer re-hunts for the text every scene;
* ``full_bleed`` puts the heading on unknown pixels — measured contrast 10.45 against 18.8
  on the solid slides, and the scorer's own alternative position measured 4.86. The one
  layout that varied was also the only one that risked legibility.

An earlier draft of this change gave the summary an ``image_band`` and the closer a
``full_bleed`` "so they read as a conclusion". That is the same variety instinct in a
smaller costume: a summary earns its difference from having fewer bullets and different
words, not from a different frame.
"""

HERO_LAYOUTS: tuple[SlideLayout, ...] = (SlideLayout.HERO_RIGHT, SlideLayout.HERO_LEFT)
"""Both hero sides, so ``hero_side="left"`` can still mirror a whole video if a brand needs
it. One video only ever uses one of them."""

SUMMARY_MIN_SCENES = 7
"""A recap only exists from seven scenes up. DIRECTION §1.1.

Below that the closing already restates the key point, so a summary is the redundancy
principle violated *and* it costs a teaching slide: a 6-scene deck has only 4 of them.
"""

HEADING_ANIMATION_ROTATION: tuple[TextAnimation, ...] = (
    TextAnimation.SLIDE_UP,
    TextAnimation.FADE_IN,
    TextAnimation.POP,
    TextAnimation.SLIDE_LEFT,
)
"""Only used when ``hold_animation=False``. The default is :data:`HEADING_ANIMATION`."""

BULLET_ANIMATION_ROTATION: tuple[TextAnimation, ...] = (
    TextAnimation.SLIDE_LEFT,
    TextAnimation.FADE_IN,
    TextAnimation.SLIDE_UP,
)
"""``TYPEWRITER`` is available but not in the rotation: without ``drawtext`` the
renderer can only approximate it as a wipe, so it is opt-in rather than a default."""

HEADING_ANIMATION = TextAnimation.SLIDE_UP
"""The heading entrance, for every scene in the video. A fade plus a 12px rise."""

BULLET_ANIMATION = TextAnimation.SLIDE_UP
"""The bullet entrance, for every bullet in the video. A fade plus an 8px rise.

The same *kind* of move as the heading, deliberately. ``SLIDE_LEFT`` is retired: on a
left-aligned list a horizontal entrance sweeps the text through the marker gutter
(DIRECTION §4.1). Nothing is lost by sharing the move — the heading is already on screen
before the first bullet reveals, so the two never travel together, and the bullets are
separated by 1.6s of stagger rather than by direction.
"""

HELD_TRANSITION = Transition.FADE
"""The transition at every scene boundary. One crossfade is a style; three is a demo reel."""

HELD_MOTION = Motion.ZOOM_IN
"""The camera move on every still, when the script has no opinion.

A slow push on every image is a house style. Alternating push/pull/pan/pan, which is what
this planner used to do, is four house styles — and alternating *direction* between adjacent
scenes is visible, which is the one thing a Ken Burns move must not be.
"""

ZOOM_SPANS: tuple[float, ...] = (0.10, 0.12, 0.15, 0.08)
"""Only used when ``hold_motion=False``."""

HELD_ZOOM_SPAN = 0.06
"""One zoom amount for the video. DIRECTION §4.4.

6% over 15s is 0.4%/s: never static, never noticed. It is also the span the ``upscale_factor``
arithmetic assumes — ``zoompan`` truncates to integer pixels, so a bigger span over a small
upscale is what produced a measured 0.37 duplicate-frame ratio.
"""

MIN_ZOOM_SPAN = 0.06
MAX_ZOOM_SPAN = 0.15

DEFAULT_TRANSITION_DURATION = 0.35
"""DIRECTION §4.3. Down from 0.5s: now that both frames share one grid, the only thing
actually cross-dissolving is the hero photograph, and 0.35s covers that while costing
0.15s less dead air at every boundary."""

TRANSITION_MAX_SCENE_FRACTION = 0.40
"""A 0.5s dissolve across a 0.8s scene is a glitch, not a transition."""

MIN_TRANSITION_DURATION = 0.20
"""Below this a crossfade reads as a dropped frame; we demote to a hard CUT. DIRECTION §4.3."""

DEFAULT_ANIM_DURATION = 0.40
"""The heading's entrance. DIRECTION §4.1. A bullet's is capped shorter by the layer
builder — see ``text_overlay.BULLET_ANIM_DURATION``."""

MIN_ANIM_DURATION = 0.12
"""Shorter than this and the easing has no frames to happen in — it reads as a cut."""

ANIM_MAX_SCENE_FRACTION = 0.18
"""An entrance that occupies a fifth of the slide is the slide."""

DEFAULT_BULLET_MIN_GAP = 1.6
"""Seconds between bullet reveals. DIRECTION §4.2.

0.6s lets two bullets land inside one spoken clause, so the eye is still on bullet N when
N+1 arrives. 1.6s is about one short clause at 135 wpm.
"""

ANIM_TAIL_MARGIN = 2.60
"""Seconds the last reveal must be finished and motionless before the scene cuts away.

DIRECTION §4.2, from the legibility dwell rule of 1 second per 13 characters: a 34-character
bullet needs 2.6s of stillness to be read. At the old 0.35s the last point of every scene was
legally on screen and practically unread — the highest-impact number in this module.
"""

ANIM_MAX_GAP_FRACTION = 0.6
"""Fraction of the smallest reveal gap an animation may occupy.

At 0.8 the previous bullet is still settling when the next one starts.
"""


class RuleBasedPlanner:
    """Fills in :class:`VisualPlan` for every scene. Satisfies ``VisualPlanner``."""

    def __init__(
        self,
        *,
        transition_duration: float = DEFAULT_TRANSITION_DURATION,
        scrim_opacity: float = 0.45,
        max_scene_fraction: float = TRANSITION_MAX_SCENE_FRACTION,
        min_transition_duration: float = MIN_TRANSITION_DURATION,
        easing: str = "linear",
        anim_duration: float = DEFAULT_ANIM_DURATION,
        bullet_min_gap: float = DEFAULT_BULLET_MIN_GAP,
        text_position: TextPosition | None = None,
        alternate_text_position: bool = False,
        title_card_opener: bool = True,
        hero_side: str | None = None,
        hold_layout: bool = True,
        hold_motion: bool = True,
        hold_animation: bool = True,
        hold_transition: bool = True,
        infer_roles: bool = True,
        enforce_bullet_budget: bool = True,
    ) -> None:
        self.transition_duration = transition_duration
        self.scrim_opacity = scrim_opacity
        self.max_scene_fraction = max_scene_fraction
        self.min_transition_duration = min_transition_duration
        self.easing = easing
        """``linear``, per DIRECTION §4.4. ``ease_in_out`` stalls a Ken Burns move at both
        ends, and a stalled move over an integer-truncating ``zoompan`` is where duplicate
        frames come from."""
        self.anim_duration = anim_duration
        self.bullet_min_gap = bullet_min_gap
        self.text_position = text_position
        """``None`` derives the position from the slide layout (the default).

        Pass an explicit ``TextPosition`` to pin every scene.
        """
        self.alternate_text_position = alternate_text_position
        """The pre-layout rhythm: upper/lower third, flipping every scene.

        Still the right answer for a full-bleed photo deck, where there is no solid
        column to put a text panel in. Wins over ``text_position``.
        """
        self.title_card_opener = title_card_opener
        self.hero_side = hero_side
        """``left`` | ``right``, or ``None`` to derive one per video from its title.

        Either way it is one side for the whole video. Which side barely matters; flipping
        it every scene is what made the deck feel like a slideshow of templates.
        """
        self.hold_layout = hold_layout
        self.hold_motion = hold_motion
        self.hold_animation = hold_animation
        self.hold_transition = hold_transition
        """Hold one choice for the whole video (the default) rather than rotating per
        scene. Set any of these ``False`` to get the old per-scene rotation back."""
        self.infer_roles = infer_roles
        """Assign :class:`~app.core.models.SceneRole` when the timeline has not.

        Nothing upstream fills ``Scene.role`` in yet, so every scene arrives as
        ``CONTENT`` and a deck would have no opener, recap or closer. When *any* scene
        carries a non-default role the timeline is taken at its word and this does nothing.
        """
        self.enforce_bullet_budget = enforce_bullet_budget
        """Trim each scene's bullets to ``role.bullet_budget`` on the planned copy."""

    # ---------------------------------------------------------------- public

    def plan(self, timeline: Timeline) -> Timeline:
        """Return a copy of ``timeline`` with every ``Scene.plan`` populated.

        The input is never mutated: planning is a pure function of the timeline,
        which keeps re-planning idempotent and unit tests trivial.

        ``Scene.role`` on the *output* is authoritative for the renderer: it is either what
        the timeline already said or what :meth:`_roles` inferred, and the layout, the type
        scale and the bullet budget all follow from it.
        """
        out = timeline.model_copy(deep=True)
        roles = self._roles(out.scenes)
        layouts = self._layouts(roles, out.title)
        held_motion = self._held_motion(out.scenes)
        heading_anim, bullet_anim = self._animations()
        previous_motion: Motion | None = None
        previous_heading: TextAnimation | None = None
        previous_bullet: TextAnimation | None = None

        for index, scene in enumerate(out.scenes):
            scene.role = roles[index]
            if self.enforce_bullet_budget:
                scene.bullets = scene.bullets[: scene.role.bullet_budget]

            if self.hold_motion:
                motion = held_motion
            else:
                requested = (
                    scene.plan.motion if scene.plan is not None else self._default_motion(index)
                )
                motion = self._enforce_variety(requested, previous_motion)
            zoom_from, zoom_to = self._zoom_range(motion, index)
            transition = self._transition_for(index)
            duration = self._transition_duration(out.scenes, index)

            if transition is not Transition.CUT and duration < self.min_transition_duration:
                # Keep transition_in and transition_duration self-consistent so
                # Timeline.final_duration() and the renderer agree on the maths.
                transition, duration = Transition.CUT, 0.0

            if not self.hold_animation:
                heading_anim = self._rotate_away(
                    HEADING_ANIMATION_ROTATION[index % len(HEADING_ANIMATION_ROTATION)],
                    HEADING_ANIMATION_ROTATION,
                    {previous_heading},
                )
                bullet_anim = self._rotate_away(
                    BULLET_ANIMATION_ROTATION[index % len(BULLET_ANIMATION_ROTATION)],
                    BULLET_ANIMATION_ROTATION,
                    # Also unlike this scene's heading: if the title and every bullet
                    # arrive identically the whole slide slides as one slab.
                    {previous_bullet, heading_anim},
                )

            layout = layouts[index]
            scene.plan = VisualPlan(
                layout=layout,
                motion=motion,
                zoom_from=zoom_from,
                zoom_to=zoom_to,
                easing="linear" if motion is Motion.STATIC else self.easing,  # type: ignore[arg-type]
                transition_in=transition,
                transition_duration=duration,
                text_position=self._text_position(index, layout),
                scrim_opacity=self.scrim_opacity,
                heading_animation=heading_anim,
                bullet_animation=bullet_anim,
                anim_duration=self._anim_duration(scene),
                bullet_min_gap=self.bullet_min_gap,
            )
            previous_motion = motion
            previous_heading = heading_anim
            previous_bullet = bullet_anim

        return out

    # --------------------------------------------------------------- helpers

    def _roles(self, scenes: list[Scene]) -> list[SceneRole]:
        """One :class:`~app.core.models.SceneRole` per scene.

        The timeline wins whenever it has an opinion — any scene with a non-default role
        means the roles were assigned deliberately upstream and this must not second-guess
        them. Otherwise a deck's shape is inferred from its length: announce, teach, recap,
        tell them what to do, dropping the parts a short deck has no room for.
        """
        count = len(scenes)
        if not self.infer_roles or any(scene.role is not SceneRole.CONTENT for scene in scenes):
            return [scene.role for scene in scenes]

        roles = [SceneRole.CONTENT] * count
        if count >= 2 and self.title_card_opener:
            roles[0] = SceneRole.TITLE
        if count >= 3:
            roles[-1] = SceneRole.CLOSING
        if count >= SUMMARY_MIN_SCENES:
            roles[-2] = SceneRole.SUMMARY
        return roles

    def _resolve_hero(self) -> SlideLayout:
        """The one body layout for this video. ``hero_right`` unless a caller mirrors it.

        An earlier draft derived the side from a digest of the video's title, so different
        topics would not look like carbon copies. DIRECTION §6.1 is right to reject that:
        looking like the same deck *is* the goal, ``hero_right`` is the side the grid in §6.2
        is specified for, and a per-video coin flip is a variability the reviewer cannot
        predict. It stays configurable for a brand that reads right-to-left.
        """
        return SlideLayout.HERO_LEFT if self.hero_side == "left" else SlideLayout.HERO_RIGHT

    def hero_layout(self) -> SlideLayout:
        """The body layout every non-title scene will get. Exposed for proofing."""
        return self._resolve_hero()

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
    def _rotate_away[T](
        requested: T, rotation: tuple[T, ...], forbidden: set[T | None]
    ) -> T:
        """The first choice at or after ``requested`` in ``rotation`` that is allowed.

        Falls back to ``requested`` when everything is forbidden, which beats
        returning something absurd just to satisfy a rule.
        """
        if requested not in forbidden:
            return requested
        start = rotation.index(requested) if requested in rotation else 0
        for step in range(1, len(rotation) + 1):
            candidate = rotation[(start + step) % len(rotation)]
            if candidate not in forbidden:
                return candidate
        return requested

    def _layouts(self, roles: list[SceneRole], title: str = "") -> list[SlideLayout]:
        """One layout per scene, decided by role rather than by position.

        The whole point is that the *body* does not change: every ``CONTENT`` scene gets the
        same hero side, so the frame the viewer learns in scene two is still the frame in
        scene six and only the words move. The three exceptions each happen at most once —
        the title card, the recap band, the full-bleed closer — which is what makes them
        read as structure instead of as variety.
        """
        del title  # the body layout no longer varies per video; see `hero_side`
        hero = self._resolve_hero()
        if not self.hold_layout:  # legacy per-scene rotation
            return [
                SlideLayout.TITLE_CARD
                if role is SceneRole.TITLE
                else LAYOUT_ROTATION[index % len(LAYOUT_ROTATION)]
                for index, role in enumerate(roles)
            ]
        return [
            SlideLayout.TITLE_CARD if role is SceneRole.TITLE else hero for role in roles
        ]

    def _held_motion(self, scenes: list[Scene]) -> Motion:
        """One camera move for the video: whatever the script asked for most often.

        The LLM's per-scene ``motion`` is still listened to, just at the level a house style
        is actually decided at. ``STATIC`` never wins — a whole deck of dead stills is not a
        style, it is a missing feature — and ties break toward the earliest scene's choice,
        so the answer stays deterministic.
        """
        requested = [s.plan.motion for s in scenes if s.plan is not None]
        votes = Counter(m for m in requested if m is not Motion.STATIC)
        if not votes:
            return HELD_MOTION
        best = max(votes.values())
        for motion in requested:  # first-mentioned winner, for determinism
            if votes.get(motion) == best:
                return motion
        return HELD_MOTION

    def _animations(self) -> tuple[TextAnimation, TextAnimation]:
        """The heading and bullet entrances for the whole video."""
        return HEADING_ANIMATION, BULLET_ANIMATION

    def _zoom_range(self, motion: Motion, index: int) -> tuple[float, float]:
        # One zoom amount for the video: a push that travels 8% on one slide and 15% on the
        # next is the same inconsistency as two marker shapes, just in the time axis.
        span = HELD_ZOOM_SPAN if self.hold_motion else ZOOM_SPANS[index % len(ZOOM_SPANS)]
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

    def _transition_for(self, index: int) -> Transition:
        if index == 0:
            return Transition.FADE  # fade up from black
        if self.hold_transition:
            return HELD_TRANSITION
        return TRANSITION_ROTATION[(index - 1) % len(TRANSITION_ROTATION)]

    def _transition_duration(self, scenes: list[Scene], index: int) -> float:
        """Clamp to a fraction of the *shorter* adjacent scene."""
        neighbours = [scenes[index].duration]
        if index > 0:
            neighbours.append(scenes[index - 1].duration)
        shortest = min(neighbours)
        limit = max(0.0, shortest * self.max_scene_fraction)
        return round(min(self.transition_duration, limit), 3)

    def _anim_duration(self, scene: Scene) -> float:
        """Longest entrance this scene can afford.

        Three independent ceilings, all of which have to hold:

        1. a fraction of the scene, so the entrance is not the slide;
        2. the gap to the next reveal, so bullets do not arrive on top of each other;
        3. enough room after the last reveal to finish before the scene cuts away.
        """
        reveals = sorted({0.0, *(bullet.appear_at for bullet in scene.bullets)})
        duration = max(0.0, scene.duration)

        limits = [self.anim_duration, duration * ANIM_MAX_SCENE_FRACTION]

        gaps = [b - a for a, b in zip(reveals, reveals[1:], strict=False) if b > a]
        if gaps:
            limits.append(min(gaps) * ANIM_MAX_GAP_FRACTION)

        limits.append(duration - ANIM_TAIL_MARGIN - reveals[-1])

        return round(max(MIN_ANIM_DURATION, min(limits)), 3)

    def _text_position(self, index: int, layout: SlideLayout) -> TextPosition:
        if self.alternate_text_position:
            return TextPosition.LOWER_THIRD if index % 2 == 0 else TextPosition.UPPER_THIRD
        if self.text_position is not None:
            return self.text_position
        return LAYOUT_TEXT_POSITION[layout]


LAYOUT_TEXT_POSITION: dict[SlideLayout, TextPosition] = {
    # LEFT_PANEL is the training layout: heading top-left, bullets stacked beneath.
    # HERO_LEFT wants the mirror of it; TextPosition has no RIGHT_PANEL member, so
    # text_overlay mirrors using plan.layout, which it already receives.
    SlideLayout.HERO_RIGHT: TextPosition.LEFT_PANEL,
    SlideLayout.HERO_LEFT: TextPosition.LEFT_PANEL,
    SlideLayout.TITLE_CARD: TextPosition.CENTER,
    SlideLayout.IMAGE_BAND: TextPosition.UPPER_THIRD,
    SlideLayout.FULL_BLEED: TextPosition.LOWER_THIRD,
}
"""Text goes wherever the image is not. Solid background, so no scrim is needed
except over full-bleed photography."""
