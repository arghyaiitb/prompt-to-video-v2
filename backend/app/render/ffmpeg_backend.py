"""ffmpeg implementation of ``VideoBackend``.

Architecture: **one intermediate clip per scene**, then a single assemble pass. A
monolithic filter_complex for a 12-scene video is unreadable, impossible to debug
from an error message, and forces a full re-render when one image changes. Per-scene
clips are near-lossless (x264 crf 12) so chaining them costs nothing visible.

Four things in here are load-bearing and easy to get wrong:

*The frame is designed, not photographed.* Every slide starts as a solid
``Theme.bg`` colour and the image is placed **inside a region** (see
:func:`layout_region`). ``SlideLayout.FULL_BLEED`` is the exception, not the rule.

*zoompan jitter.* ``zoompan`` truncates its ``x``/``y`` expressions to integers, so a
slow pan advances 0px for several frames and then jumps 1px — a visible stutter. We
pre-scale the source by ``profile.upscale_factor`` before zoompan and let zoompan
downscale to the *region's* size, which turns that 1px step into 1/N of an output
pixel. Ken Burns therefore happens inside the image panel, not across the frame.

*Text animation without drawtext.* This build has no ``drawtext`` (no libfreetype),
so text arrives as pre-rasterised RGBA PNGs via :mod:`app.render.contracts`. Each
layer is animated by ``fade=...:alpha=1`` (invisible before ``appear_at`` — verified,
see tests) plus a time-varying ``overlay`` x/y expression with smoothstep easing.

*xfade timing.* ``xfade`` **consumes** its overlap: two 5s clips with a 0.5s crossfade
produce 9.5s, not 10s. Every offset is therefore cumulative over already-shortened
output, the narration has to be shifted by the same amount, and the final duration is
``sum(durations) - sum(transitions)``. :meth:`assemble` checks its own output against
``Timeline.final_duration()`` and refuses to lie about it.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from app.core.config import REPO_ROOT, get_settings
from app.core.models import (
    BulletPoint,
    Motion,
    RenderProfile,
    Scene,
    SlideLayout,
    TextAnimation,
    Theme,
    Timeline,
    Transition,
    VisualPlan,
)
from app.render import captions as captions_mod
from app.render import ffmpeg as ff
from app.render import text_overlay as tx
from app.render.contracts import SceneText, TextLayer

logger = logging.getLogger(__name__)

INTERMEDIATE_CRF = 12
"""Near-lossless: concatenation must not compound generational loss."""

ASPECT_TOLERANCE = 0.25
"""Beyond 25% aspect mismatch, centre-cropping throws away too much of the image, so
we switch to a blurred fill instead. Stretching is never an option."""

MIN_PAN_ZOOM = 1.06
"""A pan needs headroom to travel across; at zoom 1.0 there is nowhere to go."""

AUDIO_RATE = 48_000

LOUDNESS_TARGET_LUFS = -16.0
"""Integrated loudness every render is normalised to.

The web convention, and what the evaluator scores against: an un-normalised assemble lands
near -20.3 LUFS, which reads as quiet next to anything else in a browser tab and cost every
video the same "+4.3 dB" note. Broadcast would want -23; this is not broadcast.
"""

LOUDNESS_TRUE_PEAK_DB = -1.0
"""True-peak ceiling, in dBFS. One dB of headroom survives lossy re-encoding downstream."""

LOUDNESS_RANGE = 11.0
"""Target loudness range. Narration plus a ducked bed is not dynamic material; a wider
target lets ``loudnorm`` leave quiet passages quiet and undoes the point of normalising."""

CLICK_FADE = 0.02
MUSIC_FADE_IN = 1.5
MUSIC_FADE_OUT = 2.0
FINAL_FADE_OUT = 0.6

# ------------------------------------------------------------- slide geometry

MARGIN_FRACTION = 0.045
"""Frame margin around image panels, as a fraction of frame *width*."""

HERO_WIDTH_FRACTION = 0.465
"""Image panel width for ``hero_left``/``hero_right``. Text gets the rest."""

BAND_HEIGHT_FRACTION = 0.45
"""Image band height for ``image_band``, as a fraction of frame height."""

POP_OVERSHOOT_C1 = 1.70158
"""Standard "back out" easing constant: overshoots ~10% then settles."""

POP_FADE_FRACTION = 0.6
"""A pop reads as snappy only if the opacity ramp beats the movement."""

MAX_TEXT_LAYERS = 8
"""1 background/scrim + 1 heading + up to 5 bullets, with one spare.

Each layer is an ffmpeg input and an overlay stage; past this the filtergraph stops
being reviewable and the render cost stops being worth it.
"""

_HEX_COLOUR = re.compile(r"^#?([0-9A-Fa-f]{6})$")

AUTO_LOGO = "auto"
"""Sentinel for ``FFmpegBackend(logo_path=...)``: resolve from settings.

``None`` disables branding outright; a path uses that file. A three-way knob rather than
two because "the caller said nothing" and "the caller said no logo" are different answers.
"""

DEFAULT_LOGO_RELATIVE = Path("frontend/public/favicon.svg")
"""Fallback mark, relative to the repo root: the app's own favicon."""

_DISABLED_PATHS = frozenset({"", ".", "none", "off"})
"""``VIDEO_LOGO_PATH=`` in ``.env`` arrives as ``Path('.')``, which is a directory, not a
mark. Treat the empty-ish spellings as an explicit "no branding"."""


class RenderError(RuntimeError):
    pass


class DurationMismatchError(RenderError):
    """The assembled output does not match ``Timeline.final_duration()``."""


@dataclass(frozen=True)
class Region:
    """Where the image lives inside the frame, in output pixels."""

    x: int
    y: int
    width: int
    height: int


def _even(value: float, *, minimum: int = 0) -> int:
    """Round down to an even number — yuv420p subsamples chroma 2x2.

    ``minimum=2`` for dimensions (a zero-width filter is a parse error); the default 0
    is for offsets, where 0 is both legal and extremely common.
    """
    whole = int(round(value))
    return max(minimum, whole - whole % 2)


def layout_region(
    plan: VisualPlan, profile: RenderProfile, *, theme: Theme | None = None
) -> Region | None:
    """The image's box, or ``None`` when the slide has no image (a title card).

    ``text_overlay.slide_geometry`` is the single source of truth here: it decides the
    text column *and* the image panel from the same numbers, so asking it rather than
    recomputing is what stops type and picture overlapping. :func:`fallback_region`
    covers a ``text_overlay`` that predates that function.

    Every edge is forced even — an odd overlay offset shifts the subsampled chroma
    plane by half a pixel and visibly softens the panel edge.
    """
    geometry = getattr(tx, "slide_geometry", None)
    if geometry is None:
        return fallback_region(plan.layout, profile)
    region = geometry(plan, profile, theme=theme or Theme()).image_region
    if region is None:
        return None
    return Region(
        _even(region.x),
        _even(region.y),
        _even(region.width, minimum=2),
        _even(region.height, minimum=2),
    )


def fallback_region(layout: SlideLayout, profile: RenderProfile) -> Region | None:
    """Proportional region maths, used when ``text_overlay`` cannot supply one.

    Same design intent as :func:`layout_region`, expressed in fractions of the frame
    so draft and final renders are the same slide at different sizes.
    """
    width, height = profile.width, profile.height
    margin = _even(width * MARGIN_FRACTION, minimum=2)

    if layout is SlideLayout.TITLE_CARD:
        return None
    if layout is SlideLayout.FULL_BLEED:
        return Region(0, 0, _even(width, minimum=2), _even(height, minimum=2))
    if layout is SlideLayout.IMAGE_BAND:
        band_w = _even(width - 2 * margin, minimum=2)
        band_h = _even(height * BAND_HEIGHT_FRACTION, minimum=2)
        return Region(margin, _even(height - margin - band_h), band_w, band_h)

    panel_w = _even(width * HERO_WIDTH_FRACTION, minimum=2)
    panel_h = _even(height - 2 * margin, minimum=2)
    x = margin if layout is SlideLayout.HERO_LEFT else _even(width - margin - panel_w)
    return Region(x, margin, panel_w, panel_h)


def corner_radius(
    plan: VisualPlan, region: Region, profile: RenderProfile, theme: Theme | None = None
) -> int:
    """Rounding for the image panel, in output pixels. 0 means square corners.

    Deferred to ``slide_geometry`` for the same reason as the region: a rounded panel
    and the card outline behind it have to agree. A full-bleed image *is* the frame,
    so it is never rounded.
    """
    theme = theme or Theme()
    if plan.layout is SlideLayout.FULL_BLEED:
        return 0
    geometry = getattr(tx, "slide_geometry", None)
    radius = (
        geometry(plan, profile, theme=theme).image_radius
        if geometry is not None
        else int(round(theme.image_radius * profile.width / 1920))
    )
    return max(0, min(radius, min(region.width, region.height) // 2))


def resolve_logo_source(configured: Path | str | None = AUTO_LOGO) -> Path | None:
    """The brand mark to composite, or ``None`` for no branding.

    Three inputs, three answers, and none of them can fail a render:

    * :data:`AUTO_LOGO` — take ``settings.video_logo_path``; if that is unset, fall back to
      ``<repo>/frontend/public/favicon.svg`` when it exists, else no branding.
    * ``None`` or an empty-ish path — no branding, silently. That is what
      ``VIDEO_LOGO_PATH=`` in ``.env`` means.
    * a real path — use it, or warn and skip if it is missing.

    A missing or unreadable logo is never an error. Branding is the last 1% of the frame;
    losing it must not cost the other 99%.
    """
    if configured is None:
        return None
    if configured == AUTO_LOGO:
        configured = get_settings().video_logo_path
        if configured is None:
            fallback = REPO_ROOT / DEFAULT_LOGO_RELATIVE
            return fallback if fallback.is_file() else None
    if str(configured).strip().lower() in _DISABLED_PATHS:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        logger.warning("configured brand logo does not exist, skipping branding: %s", path)
        return None
    return path


def ffmpeg_colour(value: str) -> str:
    """``#0B1220`` -> ``0x0B1220``. ``#`` is a comment character in filter scripts."""
    match = _HEX_COLOUR.match(value.strip())
    if not match:
        raise RenderError(f"theme colour must be #RRGGBB, got {value!r}")
    return f"0x{match.group(1).upper()}"


class FFmpegBackend:
    """Satisfies ``VideoBackend``. All creative choices arrive via ``VisualPlan``."""

    def __init__(
        self,
        *,
        text_mode: str = "auto",
        music_duck_db: int | None = None,
        final_fade_out: bool = True,
        burn_captions: bool = False,
        strict_duration: bool = True,
        theme: Theme | None = None,
        logo_path: Path | str | None = AUTO_LOGO,
    ) -> None:
        self.text_mode = tx.resolve_text_mode(text_mode)
        self.music_duck_db = (
            music_duck_db if music_duck_db is not None else get_settings().video_music_duck_db
        )
        self.final_fade_out = final_fade_out
        self.burn_captions = burn_captions
        self.strict_duration = strict_duration
        self.theme = theme or Theme()
        self.logo_source = resolve_logo_source(logo_path)
        """Brand mark, or ``None``. Resolved once so a render cannot change its mind."""
        self._font = tx.find_font() if self.text_mode in ("drawtext", "png") else None

    # ------------------------------------------------------------ scene clip

    def render_scene(
        self,
        image_path: Path | str | None,
        plan: VisualPlan,
        heading: str,
        duration: float,
        out_path: Path,
        profile: RenderProfile,
        *,
        bullets: list[BulletPoint] | None = None,
        scene_text: SceneText | None = None,
    ) -> Path:
        """Render one slide — solid background, image panel, animated text — to a clip.

        The clip is exactly ``round(duration * fps)`` frames so it matches its
        narration segment. Verified with ffprobe before returning.

        ``scene_text`` may be supplied by the caller (tests, or a pipeline that
        rasterises text elsewhere); otherwise it is built here from ``heading`` and
        ``bullets``. Everything written lands under ``out_path``'s stem, which is
        unique per scene — that is what makes concurrent scene renders safe.
        """
        out_path = Path(out_path)
        if duration <= 0:
            raise RenderError(f"scene duration must be positive, got {duration}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        region = layout_region(plan, profile, theme=self.theme)
        image_path = Path(image_path) if image_path else None
        if region is not None:
            if image_path is None:
                raise RenderError(f"layout {plan.layout.value} needs an image, none was given")
            if not image_path.is_file():
                raise RenderError(f"scene image not found: {image_path}")

        frames = max(1, int(round(duration * profile.fps)))
        src_size = ff.probe_image_size(image_path) if region is not None else (0, 0)

        inputs: list[str | Path] = []
        if region is not None:
            assert image_path is not None
            inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", image_path]

        text_layout = tx.layout_heading(heading, plan, profile)
        if scene_text is None and self.text_mode == "png":
            scene_text = self._build_scene_text(
                heading, bullets or [], plan, profile, out_path, image_path=image_path
            )

        text_png: Path | None = None
        if scene_text is not None:
            if len(scene_text.layers) > MAX_TEXT_LAYERS:
                raise RenderError(
                    f"{len(scene_text.layers)} text layers exceeds the {MAX_TEXT_LAYERS} "
                    "the filtergraph is designed for"
                )
            for png in scene_text.inputs:
                inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", png]
        elif self.text_mode == "png" and heading.strip():
            # Legacy single-PNG heading: still supported so a text_overlay without
            # build_scene_text keeps working.
            text_png = out_path.with_suffix(".text.png")
            tx.render_text_png(
                heading, plan, profile, text_png, font=self._font, layout=text_layout
            )
            inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", text_png]

        graph = self._scene_graph(
            src_size=src_size,
            plan=plan,
            profile=profile,
            frames=frames,
            text_layout=text_layout,
            heading=heading,
            has_text_input=text_png is not None,
            scene_text=scene_text,
            has_image_input=region is not None,
        )

        ff.ffmpeg(
            [
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[vout]",
                "-frames:v",
                str(frames),
                "-r",
                str(profile.fps),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                str(INTERMEDIATE_CRF),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                *self._thread_args(profile),
                out_path,
            ]
        )

        actual = ff.probe_duration(out_path)
        expected = frames / profile.fps
        if abs(actual - expected) > 1.5 / profile.fps:
            raise DurationMismatchError(
                f"{out_path.name}: wanted {expected:.4f}s ({frames} frames), got {actual:.4f}s"
            )
        logger.debug(
            "scene clip %s: %d frames, %.4fs (target %.4fs), motion=%s",
            out_path.name,
            frames,
            actual,
            duration,
            plan.motion.value,
        )
        return out_path

    def render_all(self, timeline: Timeline, clip_dir: Path) -> Timeline:
        """Render every scene, assigning ``clip_path``.

        Frame counts are rounded *cumulatively* (``round(end*fps) - round(start*fps)``)
        rather than per scene. Per-scene rounding leaks up to half a frame each time
        and the error accumulates; cumulative rounding keeps the clip boundaries on the
        narration's own frame grid.

        Scenes render concurrently — each clip is an independent ffmpeg process
        writing its own file, so there is no shared state to guard. Results are
        reassembled in scene order regardless of completion order.
        """
        out = timeline.model_copy(deep=True)
        fps = out.profile.fps
        clip_dir = Path(clip_dir)
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Validate everything up front: failing fast beats discovering a bad scene
        # after paying for three renders.
        for scene in out.scenes:
            if scene.plan is None:
                raise RenderError(f"scene {scene.id} has no VisualPlan; run the planner first")
            # A title card is solid colour plus type: it has no image region, so
            # demanding an image would fail a slide that is by design image-free.
            needs_image = layout_region(scene.plan, out.profile, theme=self.theme) is not None
            if needs_image and not scene.image_path:
                raise RenderError(
                    f"scene {scene.id} has no image_path but layout is {scene.plan.layout.value}"
                )

        workers, threads = out.profile.resolve_concurrency(len(out.scenes))
        profile = out.profile.model_copy(update={"encoder_threads": threads})

        jobs: list[tuple[int, Scene, Path, float]] = []
        for idx, scene in enumerate(out.scenes):
            frames = max(1, int(round(scene.end * fps)) - int(round(scene.start * fps)))
            jobs.append((idx, scene, clip_dir / f"scene_{scene.id:03d}.mp4", frames / fps))

        if workers == 1:
            for _, scene, clip, dur in jobs:
                self._render_one(scene, clip, dur, profile)
            return out

        logger.info(
            "rendering %d scenes with %d workers x %s threads",
            len(jobs),
            workers,
            threads or "auto",
        )
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
            futures = {
                pool.submit(self._render_one, scene, clip, dur, profile): (idx, scene)
                for idx, scene, clip, dur in jobs
            }
            errors: list[str] = []
            for fut in as_completed(futures):
                _, scene = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
                    errors.append(f"scene {scene.id}: {exc}")
        if errors:
            raise RenderError(
                f"{len(errors)} of {len(jobs)} scenes failed to render:\n" + "\n".join(errors)
            )
        logger.info("rendered %d scenes in %.1fs", len(jobs), time.monotonic() - started)
        return out

    def _render_one(
        self, scene: Scene, clip: Path, duration: float, profile: RenderProfile
    ) -> None:
        """Render one scene and record its clip. Runs on a worker thread."""
        assert scene.plan is not None  # validated by render_all
        self.render_scene(
            scene.image_path,
            scene.plan,
            scene.heading,
            duration,
            clip,
            profile,
            bullets=scene.bullets,
        )
        scene.clip_path = str(clip)

    def _build_scene_text(
        self,
        heading: str,
        bullets: list[BulletPoint],
        plan: VisualPlan,
        profile: RenderProfile,
        out_path: Path,
        *,
        image_path: Path | None = None,
    ) -> SceneText | None:
        """Rasterise this scene's text via the :mod:`app.render.contracts` seam.

        ``text_overlay.build_scene_text`` is the contract; if this build of
        ``text_overlay`` predates it we fall back to the legacy single-PNG heading
        rather than dropping the text on the floor.

        The work directory is derived from ``out_path``, which is unique per scene —
        no fixed temp names, so concurrent scene renders cannot collide.
        """
        builder = getattr(tx, "build_scene_text", None)
        if builder is None:
            logger.debug("text_overlay has no build_scene_text; using the legacy heading PNG")
            return None
        if not heading.strip() and not bullets:
            return None
        workdir = out_path.parent / f"{out_path.stem}.text"
        workdir.mkdir(parents=True, exist_ok=True)
        scene_text = builder(
            heading,
            bullets,
            plan,
            profile,
            workdir,
            image_path=image_path,
            theme=self.theme,
            font=self._font,
        )
        if scene_text is None or not scene_text.layers:
            return None
        return scene_text

    # -------------------------------------------------------- scene subgraph

    def _scene_graph(
        self,
        *,
        src_size: tuple[int, int],
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        text_layout: tx.TextLayout,
        heading: str,
        has_text_input: bool = False,
        scene_text: SceneText | None = None,
        has_image_input: bool = True,
    ) -> str:
        """The whole per-scene filtergraph, ending in ``[vout]``.

        Built back to front: solid background, image panel (with its own Ken Burns),
        then one overlay stage per animated text layer.
        """
        parts = self._background_chain(plan, profile)
        base = "bg"

        region = layout_region(plan, profile, theme=self.theme)
        if region is not None and has_image_input:
            parts += self._image_chain(
                src_size=src_size, plan=plan, profile=profile, frames=frames, region=region
            )
            parts.append(f"[{base}][hero]overlay=x={region.x}:y={region.y}:format=auto[base]")
            base = "base"

        if scene_text is not None:
            first_input = 1 if (region is not None and has_image_input) else 0
            layer_parts, base = self._text_chain(
                scene_text, base=base, first_input=first_input, fps=profile.fps
            )
            parts += layer_parts
            parts.append(f"[{base}]format=yuv420p[vout]")
            return ";".join(parts)

        if not heading.strip():
            parts.append(f"[{base}]format=yuv420p[vout]")
        elif has_text_input:
            index = 1 if (region is not None and has_image_input) else 0
            parts.append(f"[{base}][{index}:v]overlay=0:0:format=auto,format=yuv420p[vout]")
        elif self.text_mode == "drawtext":
            parts.append(f"[{base}]{tx.drawtext_filters(text_layout, font=self._font)}"
                         ",format=yuv420p[vout]")
        else:
            parts.append(f"[{base}]{tx.scrim_filter(text_layout)},format=yuv420p[vout]")

        return ";".join(parts)

    def _background_chain(self, plan: VisualPlan, profile: RenderProfile) -> list[str]:
        """The solid brand-colour canvas every slide is built on.

        A ``color`` *source* inside filter_complex rather than a CLI input: it keeps
        the input indices stable (image is 0, text layers follow) and needs no file.
        A full-bleed image covers this entirely, but generating it anyway costs one
        allocation and keeps every layout on the same code path.
        """
        colour = ffmpeg_colour(self.theme.bg)
        return [f"color=c={colour}:s={profile.width}x{profile.height}:r={profile.fps}[bg]"]

    def _image_chain(
        self,
        *,
        src_size: tuple[int, int],
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        region: Region,
    ) -> list[str]:
        """Fit, animate and round the image *inside its region*, ending in ``[hero]``.

        The upscale-before-zoompan trick is relative to the **region**, not the frame:
        a 46%-width panel needs 46% of the pixels, so this is cheaper than the
        full-frame version it replaces while removing the same integer stepping.
        """
        static = plan.motion is Motion.STATIC
        upscale = 1 if static else max(1, profile.upscale_factor)
        work = (region.width * upscale, region.height * upscale)

        parts = self._fit_chain(
            src_size,
            (region.width, region.height),
            work,
            blur_fill=plan.layout is SlideLayout.FULL_BLEED,
        )
        if static:
            parts.append("[fit]null[moved]")
        else:
            z, x, y = self._zoompan_expressions(plan, frames)
            parts.append(
                f"[fit]zoompan=z='{z}':x='{x}':y='{y}'"
                f":d={frames}:fps={profile.fps}:s={region.width}x{region.height}[moved]"
            )

        radius = corner_radius(plan, region, profile, self.theme)
        if radius <= 0:
            parts.append("[moved]null[hero]")
            return parts

        # The mask is constant, so evaluate geq for exactly one frame and let `loop`
        # repeat it forever. Per-frame geq over a 900x900 panel doubles the CPU of the
        # whole clip; this way it costs one frame's worth, once.
        parts.append(
            f"color=c=white:s={region.width}x{region.height}"
            f":r={profile.fps}:d={1.0 / profile.fps:.6f}[maskraw]"
        )
        parts.append(
            f"[maskraw]format=gray,geq=lum='{self._round_rect_expr(radius)}'"
            ",loop=loop=-1:size=1:start=0[mask]"
        )
        parts.append("[moved]format=rgba[heroa];[heroa][mask]alphamerge[hero]")
        return parts

    @staticmethod
    def _round_rect_expr(radius: int) -> str:
        """Luma mask for a rounded rectangle: 255 inside, 0 in the clipped corners.

        Distance is measured only from the corner arc centres, so the straight edges
        stay hard and only the corners curve.
        """
        dx = f"max(0,max({radius}-X,X-(W-1-{radius})))"
        dy = f"max(0,max({radius}-Y,Y-(H-1-{radius})))"
        return f"255*lte(pow({dx},2)+pow({dy},2),{radius}*{radius})"

    @staticmethod
    def _fit_chain(
        src: tuple[int, int],
        out: tuple[int, int],
        work: tuple[int, int],
        *,
        blur_fill: bool = True,
    ) -> list[str]:
        """Fit the still to the canvas without ever distorting it.

        Close aspect ratios: scale to *cover* and centre-crop the excess. Far apart
        (a portrait image in a landscape frame): blurred fill — a cover-cropped,
        blurred copy behind a fully contained foreground.

        ``blur_fill=False`` forces cover-crop regardless of mismatch. That is the right
        answer for a *bounded* image panel: the panel already sits on the brand
        background, so letterboxing it inside a blurred copy of itself just adds mush
        where the design wants a clean edge. Cropping a wide photo to a tall panel is
        the intent, not a compromise.
        """
        src_w, src_h = src
        out_w, out_h = out
        work_w, work_h = work
        src_ar = src_w / max(1, src_h)
        out_ar = out_w / max(1, out_h)
        mismatch = abs(src_ar - out_ar) / out_ar

        if not blur_fill or mismatch <= ASPECT_TOLERANCE:
            return [
                f"[0:v]scale={work_w}:{work_h}:force_original_aspect_ratio=increase"
                f":flags=lanczos,crop={work_w}:{work_h},setsar=1[fit]"
            ]

        sigma = max(6, round(out_w / 45))
        # Labels are prefixed `fit` because the slide's own solid background already
        # owns [bg] -- duplicate labels are a filtergraph parse error, not a warning.
        return [
            "[0:v]split=2[fitbgsrc][fitfgsrc]",
            # Blur at output size (cheap) and only then scale up to the work canvas.
            f"[fitbgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
            f":flags=lanczos,crop={out_w}:{out_h},gblur=sigma={sigma}:steps=2,"
            f"scale={work_w}:{work_h}[fitbg]",
            f"[fitfgsrc]scale={work_w}:{work_h}:force_original_aspect_ratio=decrease"
            f":flags=lanczos[fitfg]",
            "[fitbg][fitfg]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[fit]",
        ]

    @staticmethod
    def _zoompan_expressions(plan: VisualPlan, frames: int) -> tuple[str, str, str]:
        """Build the ``z``/``x``/``y`` expressions.

        Easing lives *in the expression* — a smoothstep on ``on/(frames-1)`` — because
        zoompan has no easing of its own and a linear ramp starts and stops abruptly.
        Every expression is emitted inside single quotes by the caller so commas and
        colons in future expressions cannot break the filtergraph.
        """
        last = max(1, frames - 1)
        t = f"(on/{last})" if frames > 1 else "(0)"
        progress = f"({t}*{t}*(3-2*{t}))" if plan.easing == "ease_in_out" else t

        centre_x, centre_y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"

        if plan.motion in (Motion.ZOOM_IN, Motion.ZOOM_OUT):
            delta = round(plan.zoom_to - plan.zoom_from, 6)
            z = f"({plan.zoom_from}+({delta})*{progress})"
            return z, centre_x, centre_y

        held = max(plan.zoom_from, MIN_PAN_ZOOM)
        if plan.motion is Motion.PAN_RIGHT:
            return f"({held})", f"((iw-iw/zoom)*{progress})", centre_y
        if plan.motion is Motion.PAN_LEFT:
            return f"({held})", f"((iw-iw/zoom)*(1-{progress}))", centre_y
        return "(1)", "(0)", "(0)"

    # -------------------------------------------------------- text animation

    def _text_chain(
        self, scene_text: SceneText, *, base: str, first_input: int, fps: int
    ) -> tuple[list[str], str]:
        """One prep filter + one overlay stage per layer, back to front.

        Returns the filter list and the label of the composited result.
        ``sorted_layers()`` decides z-order: the scrim/background panel first so it
        lands *under* the type it exists to make legible.
        """
        parts: list[str] = []
        previous = base
        for index, layer in enumerate(scene_text.sorted_layers()):
            source = f"{first_input + index}:v"
            prepped = f"tl{index}"
            parts.append(f"[{source}]{self._layer_prep(layer, fps=fps)}[{prepped}]")

            x_expr, y_expr = self._anim_position(layer)
            animated = "(" in x_expr or "(" in y_expr
            options = [
                f"x='{x_expr}'",
                f"y='{y_expr}'",
                "format=auto",
                f"eval={'frame' if animated else 'init'}",
                f"enable='{self._visibility_expr(layer, fps=fps)}'",
            ]
            label = f"ov{index}"
            parts.append(f"[{previous}][{prepped}]overlay={':'.join(options)}[{label}]")
            previous = label
        return parts, previous

    @classmethod
    def _layer_prep(cls, layer: TextLayer, *, fps: int) -> str:
        """Filters applied to the layer's own PNG stream before it is overlaid.

        ``fade=...:alpha=1`` is the load-bearing part: with ``t=in`` and ``st>0`` the
        alpha channel is held at **zero** for every frame before ``st``, so the layer
        genuinely does not exist until ``appear_at`` — the classic bug here is the PNG
        being visible from frame 0. ``format=rgba`` first, because fade can only touch
        an alpha channel that is actually there.
        """
        frame = 1.0 / max(1, fps)
        chain = ["format=rgba"]

        if layer.animation is TextAnimation.TYPEWRITER and ff.has_filter("geq"):
            chain.append(cls._wipe_filter(layer))

        chain.append(
            f"fade=t=in:st={layer.appear_at:.4f}:d={cls._fade_in(layer, fps=fps):.4f}:alpha=1"
        )
        if layer.disappear_at is not None:
            out_d = max(frame, min(layer.anim_duration, 0.5))
            chain.append(f"fade=t=out:st={layer.disappear_at:.4f}:d={out_d:.4f}:alpha=1")
        return ",".join(chain)

    @staticmethod
    def _fade_in(layer: TextLayer, *, fps: int) -> float:
        """How long the opacity ramp lasts.

        NONE and TYPEWRITER get a single frame — long enough to guarantee the
        pre-``appear_at`` transparency, short enough to read as instant (NONE) or to
        leave the reveal to the wipe (TYPEWRITER). POP fades faster than it moves,
        which is what makes it feel like a snap rather than a drift.
        """
        frame = 1.0 / max(1, fps)
        if layer.animation in (TextAnimation.NONE, TextAnimation.TYPEWRITER):
            return frame
        span = layer.anim_duration
        if layer.animation is TextAnimation.POP:
            span *= POP_FADE_FRACTION
        return max(frame, span)

    @staticmethod
    def _visibility_expr(layer: TextLayer, *, fps: int) -> str:
        """``enable`` gate for the overlay stage.

        Purely an optimisation and a belt-and-braces guard: ``enable`` cannot
        interpolate, so the fade above is what actually animates. Skipping the
        overlay entirely while a layer is invisible matters when a slide has seven
        of them.
        """
        if layer.disappear_at is None:
            return f"gte(t,{layer.appear_at:.4f})"
        tail = layer.disappear_at + max(1.0 / max(1, fps), min(layer.anim_duration, 0.5))
        return f"between(t,{layer.appear_at:.4f},{tail:.4f})"

    @classmethod
    def _anim_position(cls, layer: TextLayer) -> tuple[str, str]:
        """``overlay`` x/y for the layer — a constant, or an expression in ``t``.

        Every expression is clamped so it lands *exactly* on the final position once
        the animation is over and stays there; the layer's resting place is never a
        function of how it got there.
        """
        x, y = str(layer.x), str(layer.y)
        progress = cls._progress_expr(layer)

        if layer.animation is TextAnimation.SLIDE_UP:
            eased = cls._smoothstep(progress)
            return x, f"({layer.y}+({layer.slide_distance})*(1-{eased}))"
        if layer.animation is TextAnimation.SLIDE_LEFT:
            eased = cls._smoothstep(progress)
            return f"({layer.x}+({layer.slide_distance})*(1-{eased}))", y
        if layer.animation is TextAnimation.POP:
            # No scale animation is available (see module notes), so the overshoot is
            # spatial: a short rise that goes slightly past the mark and settles back.
            offset = max(6, round(layer.slide_distance / 3))
            eased = cls._back_out(progress)
            return x, f"({layer.y}+({offset})*(1-{eased}))"
        return x, y

    @staticmethod
    def _progress_expr(layer: TextLayer) -> str:
        """0 before ``appear_at``, 1 after the animation, linear in between.

        The clamp is what pins the layer to its final position afterwards -- an
        unclamped ramp keeps travelling for the rest of the scene.
        """
        duration = max(1e-3, layer.anim_duration)
        return f"min(1,max(0,(t-{layer.appear_at:.4f})/{duration:.4f}))"

    @staticmethod
    def _smoothstep(progress: str) -> str:
        """``3p^2-2p^3``: zero velocity at both ends. A linear slide looks mechanical."""
        return f"({progress}*{progress}*(3-2*{progress}))"

    @staticmethod
    def _back_out(progress: str) -> str:
        """Overshoots ~10% past 1 around p=0.75 and returns to exactly 1 at p=1."""
        c1 = POP_OVERSHOOT_C1
        c3 = c1 + 1
        shifted = f"({progress}-1)"
        return f"(1+{c3:.5f}*pow({shifted},3)+{c1:.5f}*pow({shifted},2))"

    @staticmethod
    def _wipe_filter(layer: TextLayer) -> str:
        """Left-to-right reveal of the layer's alpha channel.

        This is a **wipe, not a true typewriter**: without ``drawtext`` there is no
        glyph-level clock to step through, so the rasterised line is uncovered by a
        moving vertical edge. It reads as typing for short lines and as a wipe for
        long ones. ``geq`` is evaluated per pixel per frame, which is why only the
        alpha plane is computed and why the layer PNGs are line-sized, not full-frame.
        """
        duration = max(1e-3, layer.anim_duration)
        reveal = f"W*min(1,max(0,(T-{layer.appear_at:.4f})/{duration:.4f}))"
        return (
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
            f":a='if(lt(X,{reveal}),alpha(X,Y),0)'"
        )

    # -------------------------------------------------------------- branding
    #
    # The watermark is composited **once, over the finished chain** — never per scene.
    # That is the whole design point. A logo burnt into each scene clip is an input to
    # every `xfade`, so at each boundary the outgoing copy fades out while the incoming
    # one fades in; because the two are pixel-identical and `xfade` is a linear blend of
    # *frames*, not of layers, the result still dips wherever the crossfade curve does not
    # sum to one. On screen that is a logo that pulses at every cut. Applying it after the
    # last xfade means there is exactly one copy and nothing to blend it against.

    def logo_png(self, profile: RenderProfile, workdir: Path) -> Path | None:
        """The rasterised, opacity-baked watermark for this profile, or ``None``.

        Cached in ``workdir`` (the job directory) and keyed on the source file, the target
        height and the opacity, so a re-render or a second assemble reuses it.
        """
        if self.logo_source is None:
            return None
        return tx.rasterise_logo(
            self.logo_source,
            tx.logo_height(profile, self.theme),
            self.theme.logo_opacity,
            Path(workdir),
        )

    def logo_region(self, profile: RenderProfile, png: Path | None = None) -> Region:
        """The watermark's box in output pixels.

        The *offsets* are forced even for the same reason the image panel's are: an odd
        overlay offset lands the layer on a half-pixel of the subsampled chroma plane and
        softens its edges. The *size* is left exactly as rasterised — unlike the image
        panel this is not a scale target, it is a measurement of a file that already
        exists, and rounding it down under-reports the box the collision check needs.
        """
        size: tuple[int, int] | None = None
        if png is not None:
            try:
                size = ff.probe_image_size(png)
            except (ff.FFmpegError, OSError, KeyError, ValueError):
                size = None
        rect = tx.logo_rect(profile, self.theme, size=size)
        return Region(_even(rect.x), _even(rect.y), max(1, rect.width), max(1, rect.height))

    def logo_conflicts(self, timeline: Timeline, region: Region) -> list[str]:
        """Scenes whose *ink* would overlap the watermark. One string per collision.

        Reported rather than resolved. Nudging the logo per scene would make it move, which
        is exactly what a persistent brand mark must not do, and shrinking the text column
        for one slide would break the deck's rhythm. So the answer is to say so loudly and
        let the geometry be corrected upstream.

        Compared against :meth:`~app.render.text_overlay.SlidePlan.ink_rects`, not the
        layer canvases: a centred bullet's canvas spans the whole column while its words sit
        in the middle, and comparing canvases would cry wolf on every full-bleed slide.
        """
        box = tx.Rect(region.x, region.y, region.width, region.height)
        conflicts: list[str] = []
        for scene in timeline.scenes:
            if scene.plan is None:
                continue
            try:
                slide = tx.layout_slide(
                    scene.heading,
                    scene.bullets,
                    scene.plan,
                    timeline.profile,
                    theme=self.theme,
                    font=self._font,
                )
            except (tx.FontNotFoundError, OSError) as exc:
                logger.debug("could not lay out scene %s for the logo check (%s)", scene.id, exc)
                continue
            hits = [rect for rect in slide.ink_rects() if rect.intersects(box)]
            if hits:
                conflicts.append(
                    f"scene {scene.id} ({scene.plan.layout.value}/"
                    f"{scene.plan.text_position.value}): text at "
                    f"{[r.as_tuple() for r in hits]} overlaps the logo at {box.as_tuple()}"
                )
        return conflicts

    # --------------------------------------------------------------- assemble

    def assemble(self, timeline: Timeline, out_path: Path) -> Path:
        """Chain the scene clips with xfade, lock narration to them, duck the music."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scenes = timeline.scenes
        if not scenes:
            raise RenderError("timeline has no scenes")

        clips: list[Path] = []
        for scene in scenes:
            if not scene.clip_path or not Path(scene.clip_path).is_file():
                raise RenderError(f"scene {scene.id} has no rendered clip; call render_scene first")
            clips.append(Path(scene.clip_path))

        profile = timeline.profile
        # Real clip durations, not nominal ones: xfade offsets must line up with the
        # frames that actually exist, otherwise the crossfades creep.
        clip_durations = [ff.probe_duration(c) for c in clips]

        inputs: list[str | Path] = []
        for clip in clips:
            inputs += ["-i", clip]

        audio_index: dict[int, int] = {}
        next_index = len(clips)
        for position, scene in enumerate(scenes):
            if scene.audio_path and Path(scene.audio_path).is_file():
                inputs += ["-i", scene.audio_path]
                audio_index[position] = next_index
                next_index += 1
            elif scene.audio_path:
                logger.warning("scene %s narration missing: %s", scene.id, scene.audio_path)

        music_index: int | None = None
        if timeline.music_path and Path(timeline.music_path).is_file():
            inputs += ["-stream_loop", "-1", "-i", timeline.music_path]
            music_index = next_index
            next_index += 1
        elif timeline.music_path:
            logger.warning("music track missing: %s", timeline.music_path)

        # The watermark is a *single-frame* input on purpose: `overlay`'s default
        # `eof_action=repeat` holds that one frame for the whole video, so the logo is
        # present on every frame without introducing an unbounded `-loop 1` stream into a
        # pass that has no `-frames:v` to stop it.
        logo = self.logo_png(profile, out_path.parent)
        logo_input: int | None = None
        logo_box: Region | None = None
        if logo is not None:
            inputs += ["-i", logo]
            logo_input = next_index
            next_index += 1
            logo_box = self.logo_region(profile, logo)
            for conflict in self.logo_conflicts(timeline, logo_box):
                logger.warning("brand logo collides with slide text: %s", conflict)
            logger.info(
                "branding %s with %s at %dx%d+%d+%d, opacity %.2f",
                out_path.name,
                logo.name,
                logo_box.width,
                logo_box.height,
                logo_box.x,
                logo_box.y,
                self.theme.logo_opacity,
            )

        video_parts, starts, video_length = self._video_chain(
            timeline, clip_durations, logo_input=logo_input, logo_box=logo_box
        )
        graph = list(video_parts)

        audio_parts, audio_label = self._audio_chain(
            timeline,
            starts=starts,
            clip_durations=clip_durations,
            audio_index=audio_index,
            music_index=music_index,
            total=video_length,
        )
        graph += audio_parts

        maps: list[str | Path] = ["-map", "[vout]"]
        if audio_label:
            maps += ["-map", audio_label]

        expected = timeline.final_duration()
        logger.info(
            "assemble: %d clips, sum(clips)=%.4fs, overlap=%.4fs, "
            "chain length=%.4fs, Timeline.final_duration()=%.4fs",
            len(clips),
            sum(clip_durations),
            sum(clip_durations) - video_length,
            video_length,
            expected,
        )

        ff.ffmpeg(
            [
                *inputs,
                "-filter_complex",
                ";".join(graph),
                *maps,
                "-r",
                str(profile.fps),
                *self._video_encode_args(profile),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(AUDIO_RATE),
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )

        actual = ff.probe_duration(out_path)
        tolerance = max(0.1, (len(scenes) + 2) / profile.fps)
        drift = actual - expected
        message = (
            f"assembled {out_path.name}: actual={actual:.4f}s "
            f"final_duration()={expected:.4f}s drift={drift:+.4f}s "
            f"(tolerance {tolerance:.4f}s)"
        )
        if abs(drift) > tolerance:
            if self.strict_duration:
                raise DurationMismatchError(message)
            logger.warning(message)
        else:
            logger.info(message)
        return out_path

    def _video_chain(
        self,
        timeline: Timeline,
        clip_durations: list[float],
        *,
        logo_input: int | None = None,
        logo_box: Region | None = None,
    ) -> tuple[list[str], list[float], float]:
        """xfade/concat chain. Returns (filters, per-scene start times, total length).

        ``starts[i]`` is where scene *i* begins **in the assembled video**, which is
        also exactly the xfade offset used to bring it in. The narration is delayed by
        the same value, which is what keeps voice and picture together.

        The watermark, when there is one, is overlaid as the very last stage — after the
        fades, not before them. Before them it would dip with the opening fade-up and the
        closing fade-out, and "constant" is the requirement. It adds no filter that has a
        duration, so the chain length this returns is unchanged by it.
        """
        scenes = timeline.scenes
        profile = timeline.profile
        parts: list[str] = []
        for index in range(len(scenes)):
            parts.append(
                f"[{index}:v]fps={profile.fps},scale={profile.width}:{profile.height},"
                f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS[c{index}]"
            )

        acc = "c0"
        length = clip_durations[0]
        starts = [0.0]
        for index in range(1, len(scenes)):
            plan = scenes[index].plan
            transition = plan.transition_in if plan else Transition.DISSOLVE
            duration = plan.transition_duration if plan else 0.0
            label = f"x{index}"
            starts.append(length if transition is Transition.CUT else length - duration)
            if transition is Transition.CUT or duration <= 0:
                parts.append(f"[{acc}][c{index}]concat=n=2:v=1:a=0[{label}]")
                length += clip_durations[index]
            else:
                offset = length - duration
                parts.append(
                    f"[{acc}][c{index}]xfade=transition={transition.value}"
                    f":duration={duration:.6f}:offset={offset:.6f}[{label}]"
                )
                length += clip_durations[index] - duration
            acc = label

        # Scene 0's transition_in is a fade up from black, not a crossfade between
        # clips, so it costs no duration -- final_duration() deliberately ignores it.
        tail: list[str] = []
        first_plan = scenes[0].plan
        if first_plan and first_plan.transition_in in (Transition.FADE, Transition.DISSOLVE):
            fade_in = min(max(first_plan.transition_duration, 0.3), max(0.05, length / 4))
            tail.append(f"fade=t=in:st=0:d={fade_in:.3f}")
        if self.final_fade_out and length > 2 * FINAL_FADE_OUT:
            tail.append(f"fade=t=out:st={length - FINAL_FADE_OUT:.3f}:d={FINAL_FADE_OUT}")
        if self.burn_captions:
            tail.append(self._caption_filter(timeline))

        if logo_input is None or logo_box is None:
            tail.append("format=yuv420p")
            parts.append(f"[{acc}]" + ",".join(tail) + "[vout]")
            return parts, starts, length

        if not tail:
            # A filterchain cannot be empty, and with no fades and no captions it would be.
            tail.append("null")
        parts.append(f"[{acc}]" + ",".join(tail) + "[faded]")
        # Scaled and pre-multiplied by ImageMagick at rasterisation time, so there is
        # nothing left to do here but place it. `eof_action=repeat` is the default and is
        # what makes one PNG frame cover the whole timeline.
        parts.append(f"[{logo_input}:v]format=rgba,setsar=1[logo]")
        parts.append(
            f"[faded][logo]overlay=x={logo_box.x}:y={logo_box.y}"
            ":format=auto:eval=init:eof_action=repeat,format=yuv420p[vout]"
        )
        return parts, starts, length

    def _caption_filter(self, timeline: Timeline) -> str:
        if not ff.has_filter("subtitles"):
            raise RenderError(
                f"burn_captions=True but {ff.ffmpeg_bin()} has no 'subtitles' filter "
                "(built without libass)"
            )
        ass_path = Path(get_settings().job_dir(timeline.job_id)) / "captions.ass"
        captions_mod.write_ass(timeline, ass_path)
        return captions_mod.burn_filter(ass_path)

    def _audio_chain(
        self,
        timeline: Timeline,
        *,
        starts: list[float],
        clip_durations: list[float],
        audio_index: dict[int, int],
        music_index: int | None,
        total: float,
    ) -> tuple[list[str], str | None]:
        """Narration locked to the *video* clock, plus a ducked music bed.

        Each narration segment is trimmed/silence-padded to its scene's clip length and
        delayed to that scene's position in the assembled video. Segments therefore
        overlap by exactly the crossfade duration, which is what we want: the voice
        stays glued to the picture instead of drifting later with every transition.
        """
        parts: list[str] = []
        labels: list[str] = []
        common = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"

        for position, input_index in sorted(audio_index.items()):
            segment = clip_durations[position]
            delay_ms = int(round(starts[position] * 1000))
            chain = [
                common,
                f"atrim=0:{segment:.6f}",
                f"apad=whole_dur={segment:.6f}",
            ]
            if segment > 4 * CLICK_FADE:
                chain.append(f"afade=t=out:st={segment - CLICK_FADE:.6f}:d={CLICK_FADE}")
            if delay_ms > 0:
                chain.append(f"adelay=delays={delay_ms}:all=1")
            label = f"a{position}"
            parts.append(f"[{input_index}:a]" + ",".join(chain) + f"[{label}]")
            labels.append(label)

        def master(parts: list[str], label: str | None) -> tuple[list[str], str | None]:
            """Append the master bus, so every path is normalised the same way."""
            if label is None:
                return parts, None
            chain, out = self._master_chain(label, total=total)
            return parts + chain, out

        narration: str | None = None
        if len(labels) == 1:
            parts.append(f"[{labels[0]}]atrim=0:{total:.6f},apad=whole_dur={total:.6f}[narr]")
            narration = "narr"
        elif labels:
            joined = "".join(f"[{label}]" for label in labels)
            parts.append(
                f"{joined}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
                f"atrim=0:{total:.6f},apad=whole_dur={total:.6f}[narr]"
            )
            narration = "narr"

        if music_index is None:
            return master(parts, narration)

        fade_in = min(MUSIC_FADE_IN, max(0.1, total / 4))
        fade_out = min(MUSIC_FADE_OUT, max(0.1, total / 4))
        music_chain = [
            common,
            f"atrim=0:{total:.6f}",
            f"apad=whole_dur={total:.6f}",
            f"volume={self.music_duck_db}dB",
            f"afade=t=in:st=0:d={fade_in:.3f}",
            f"afade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out:.3f}",
        ]
        parts.append(f"[{music_index}:a]" + ",".join(music_chain) + "[music]")

        if narration is None:
            return master(parts, "music")

        voice, bed = narration, "music"
        if ff.has_filter("sidechaincompress"):
            # Static -18 dB alone still fights the voice on louder passages; the
            # sidechain dips the bed only while narration is actually present.
            parts.append(f"[{narration}]asplit=2[narr_out][narr_key]")
            parts.append(
                "[music][narr_key]sidechaincompress=threshold=0.03:ratio=8"
                ":attack=20:release=400:makeup=1:detection=rms:level_sc=1[music_ducked]"
            )
            voice, bed = "narr_out", "music_ducked"
        else:
            logger.warning("no sidechaincompress filter; music uses static ducking only")

        # Summing (normalize=0) preserves the narration level but can clip on peaks; the
        # limiter that catches that now lives on the master bus, after loudnorm, so it is
        # the last thing to touch the signal rather than the second-to-last.
        parts.append(f"[{voice}][{bed}]amix=inputs=2:normalize=0:duration=first[aout]")
        return master(parts, "aout")

    def _master_chain(self, label: str, *, total: float) -> tuple[list[str], str]:
        """Normalise the finished mix to :data:`LOUDNESS_TARGET_LUFS` and cap its peaks.

        Single pass. ``loudnorm`` can also be run twice — measure, then apply the measured
        offsets — which is exact; one pass is approximate but lands within a fraction of a
        dB and does not require decoding the whole programme twice for a value nothing
        downstream reads. Approximate is the right trade here.

        The ``atrim``/``apad`` pair is not decoration. ``loudnorm`` runs an internal
        lookahead buffer and can return a stream a few samples longer or shorter than it
        was handed; a container is as long as its longest stream, and :meth:`assemble`
        asserts that length against ``Timeline.final_duration()``. Clamping here is what
        keeps a *loudness* change from becoming a *duration* change.
        """
        chain: list[str] = []
        if ff.has_filter("loudnorm"):
            chain.append(
                f"loudnorm=I={LOUDNESS_TARGET_LUFS:g}"
                f":TP={LOUDNESS_TRUE_PEAK_DB:g}"
                f":LRA={LOUDNESS_RANGE:g}"
            )
        else:
            logger.warning(
                "no loudnorm filter in %s; shipping un-normalised audio", ff.ffmpeg_bin()
            )
        if ff.has_filter("alimiter"):
            # Matched to the loudnorm ceiling rather than left at a nominal 0.97, so the
            # two agree about what "true peak" means instead of one undoing the other.
            limit = 10 ** (LOUDNESS_TRUE_PEAK_DB / 20)
            chain.append(f"alimiter=limit={limit:.4f}:level=disabled")
        chain += [f"atrim=0:{total:.6f}", f"apad=whole_dur={total:.6f}"]
        return [f"[{label}]" + ",".join(chain) + "[amaster]"], "[amaster]"

    # ----------------------------------------------------------------- encode

    @staticmethod
    def _thread_args(profile: RenderProfile) -> list[str]:
        """Per-process thread cap. Empty list means "let ffmpeg decide".

        ``render_all`` divides the machine between its workers and records the share in
        ``profile.encoder_threads``; the scene encode is the process that actually runs
        concurrently, so this is where the cap has to land. Without it four x264
        instances each claim every core and spend their time contending.
        """
        if not profile.encoder_threads:
            return []
        return ["-threads", str(profile.encoder_threads)]

    def _video_encode_args(self, profile: RenderProfile) -> list[str]:
        codec = profile.video_codec
        if not ff.has_encoder(codec):
            logger.warning("encoder %s unavailable in %s; using libx264", codec, ff.ffmpeg_bin())
            codec = "libx264"

        if codec.endswith("videotoolbox"):
            # VideoToolbox ignores crf; map it onto its 1-100 constant-quality scale.
            quality = max(20, min(100, int(round(100 - profile.crf * 1.8))))
            return [
                "-c:v",
                codec,
                "-q:v",
                str(quality),
                "-allow_sw",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        args = ["-c:v", codec, "-crf", str(profile.crf), "-pix_fmt", "yuv420p"]
        if codec in ("libx264", "libx265"):
            args += ["-preset", "medium", "-profile:v", "high" if codec == "libx264" else "main"]
        return args + self._thread_args(profile)


def frames_for(duration: float, fps: int) -> int:
    """Frame count for a duration — exported so callers can reason about exactness."""
    return max(1, int(round(duration * fps)))


def expected_assembled_duration(timeline: Timeline) -> float:
    """Mirror of ``Timeline.final_duration()`` computed from ``math.fsum`` for tests."""
    overlap = math.fsum(
        scene.plan.transition_duration
        for scene in timeline.scenes[1:]
        if scene.plan and scene.plan.transition_in is not Transition.CUT
    )
    return timeline.narration_duration - overlap
