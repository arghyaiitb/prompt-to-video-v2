"""ffmpeg implementation of ``VideoBackend``.

Architecture: **one intermediate clip per scene**, then a single assemble pass. A
monolithic filter_complex for a 12-scene video is unreadable, impossible to debug
from an error message, and forces a full re-render when one image changes. Per-scene
clips are near-lossless (x264 crf 12) so chaining them costs nothing visible.

Two things in here are load-bearing and easy to get wrong:

*zoompan jitter.* ``zoompan`` truncates its ``x``/``y`` expressions to integers, so a
slow pan advances 0px for several frames and then jumps 1px — a visible stutter. We
pre-scale the source by ``profile.upscale_factor`` before zoompan and let zoompan
downscale to the final size, which turns that 1px step into 1/N of an output pixel.

*xfade timing.* ``xfade`` **consumes** its overlap: two 5s clips with a 0.5s crossfade
produce 9.5s, not 10s. Every offset is therefore cumulative over already-shortened
output, the narration has to be shifted by the same amount, and the final duration is
``sum(durations) - sum(transitions)``. :meth:`assemble` checks its own output against
``Timeline.final_duration()`` and refuses to lie about it.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.core.config import get_settings
from app.core.models import Motion, RenderProfile, Scene, Timeline, Transition, VisualPlan
from app.render import captions as captions_mod
from app.render import ffmpeg as ff
from app.render import text_overlay as tx

logger = logging.getLogger(__name__)

INTERMEDIATE_CRF = 12
"""Near-lossless: concatenation must not compound generational loss."""

ASPECT_TOLERANCE = 0.25
"""Beyond 25% aspect mismatch, centre-cropping throws away too much of the image, so
we switch to a blurred fill instead. Stretching is never an option."""

MIN_PAN_ZOOM = 1.06
"""A pan needs headroom to travel across; at zoom 1.0 there is nowhere to go."""

AUDIO_RATE = 48_000
CLICK_FADE = 0.02
MUSIC_FADE_IN = 1.5
MUSIC_FADE_OUT = 2.0
FINAL_FADE_OUT = 0.6


class RenderError(RuntimeError):
    pass


class DurationMismatchError(RenderError):
    """The assembled output does not match ``Timeline.final_duration()``."""


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
    ) -> None:
        self.text_mode = tx.resolve_text_mode(text_mode)
        self.music_duck_db = (
            music_duck_db if music_duck_db is not None else get_settings().video_music_duck_db
        )
        self.final_fade_out = final_fade_out
        self.burn_captions = burn_captions
        self.strict_duration = strict_duration
        self._font = tx.find_font() if self.text_mode in ("drawtext", "png") else None

    # ------------------------------------------------------------ scene clip

    def render_scene(
        self,
        image_path: Path,
        plan: VisualPlan,
        heading: str,
        duration: float,
        out_path: Path,
        profile: RenderProfile,
    ) -> Path:
        """Render one slide — fit, Ken Burns, heading — to its own clip.

        The clip is exactly ``round(duration * fps)`` frames so it matches its
        narration segment. Verified with ffprobe before returning.
        """
        image_path, out_path = Path(image_path), Path(out_path)
        if not image_path.is_file():
            raise RenderError(f"scene image not found: {image_path}")
        if duration <= 0:
            raise RenderError(f"scene duration must be positive, got {duration}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        frames = max(1, int(round(duration * profile.fps)))
        src_w, src_h = ff.probe_image_size(image_path)

        layout = tx.layout_heading(heading, plan, profile)
        inputs: list[str | Path] = ["-loop", "1", "-framerate", str(profile.fps), "-i", image_path]

        text_png: Path | None = None
        if self.text_mode == "png" and heading.strip():
            text_png = out_path.with_suffix(".text.png")
            tx.render_text_png(heading, plan, profile, text_png, font=self._font, layout=layout)
            inputs += ["-loop", "1", "-framerate", str(profile.fps), "-i", text_png]

        graph = self._scene_graph(
            src_size=(src_w, src_h),
            plan=plan,
            profile=profile,
            frames=frames,
            layout=layout,
            heading=heading,
            has_text_input=text_png is not None,
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
            if not scene.image_path:
                raise RenderError(f"scene {scene.id} has no image_path")

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

    def _render_one(self, scene: Scene, clip: Path, duration: float, profile: RenderProfile) -> None:
        """Render one scene and record its clip. Runs on a worker thread."""
        assert scene.plan is not None  # validated by render_all
        self.render_scene(
            Path(scene.image_path or ""),
            scene.plan,
            scene.heading,
            duration,
            clip,
            profile,
        )
        scene.clip_path = str(clip)

    # -------------------------------------------------------- scene subgraph

    def _scene_graph(
        self,
        *,
        src_size: tuple[int, int],
        plan: VisualPlan,
        profile: RenderProfile,
        frames: int,
        layout: tx.TextLayout,
        heading: str,
        has_text_input: bool,
    ) -> str:
        width, height = profile.width, profile.height
        static = plan.motion is Motion.STATIC
        upscale = 1 if static else max(1, profile.upscale_factor)
        work_w, work_h = width * upscale, height * upscale

        parts = self._fit_chain(src_size, (width, height), (work_w, work_h))

        if static:
            parts.append("[fit]null[moved]")
        else:
            z, x, y = self._zoompan_expressions(plan, frames)
            parts.append(
                f"[fit]zoompan=z='{z}':x='{x}':y='{y}'"
                f":d={frames}:fps={profile.fps}:s={width}x{height}[moved]"
            )

        if not heading.strip():
            parts.append("[moved]format=yuv420p[vout]")
        elif has_text_input:
            # Text layer is a still RGBA PNG; overlay it straight onto the moving frame.
            parts.append("[moved][1:v]overlay=0:0:format=auto,format=yuv420p[vout]")
        elif self.text_mode == "drawtext":
            drawn = tx.drawtext_filters(layout, font=self._font)
            parts.append(f"[moved]{drawn},format=yuv420p[vout]")
        else:
            parts.append(f"[moved]{tx.scrim_filter(layout)},format=yuv420p[vout]")

        return ";".join(parts)

    @staticmethod
    def _fit_chain(
        src: tuple[int, int], out: tuple[int, int], work: tuple[int, int]
    ) -> list[str]:
        """Fit the still to the canvas without ever distorting it.

        Close aspect ratios: scale to *cover* and centre-crop the excess. Far apart
        (a portrait image in a landscape frame): blurred fill — a cover-cropped,
        blurred copy behind a fully contained foreground.
        """
        src_w, src_h = src
        out_w, out_h = out
        work_w, work_h = work
        src_ar = src_w / max(1, src_h)
        out_ar = out_w / max(1, out_h)
        mismatch = abs(src_ar - out_ar) / out_ar

        if mismatch <= ASPECT_TOLERANCE:
            return [
                f"[0:v]scale={work_w}:{work_h}:force_original_aspect_ratio=increase"
                f":flags=lanczos,crop={work_w}:{work_h},setsar=1[fit]"
            ]

        sigma = max(6, round(out_w / 45))
        return [
            "[0:v]split=2[bgsrc][fgsrc]",
            # Blur at output size (cheap) and only then scale up to the work canvas.
            f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
            f":flags=lanczos,crop={out_w}:{out_h},gblur=sigma={sigma}:steps=2,"
            f"scale={work_w}:{work_h}[bg]",
            f"[fgsrc]scale={work_w}:{work_h}:force_original_aspect_ratio=decrease"
            f":flags=lanczos[fg]",
            "[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[fit]",
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

        video_parts, starts, video_length = self._video_chain(timeline, clip_durations)
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
        self, timeline: Timeline, clip_durations: list[float]
    ) -> tuple[list[str], list[float], float]:
        """xfade/concat chain. Returns (filters, per-scene start times, total length).

        ``starts[i]`` is where scene *i* begins **in the assembled video**, which is
        also exactly the xfade offset used to bring it in. The narration is delayed by
        the same value, which is what keeps voice and picture together.
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
        tail.append("format=yuv420p")
        parts.append(f"[{acc}]" + ",".join(tail) + "[vout]")
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
            return parts, (f"[{narration}]" if narration else None)

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
            return parts, "[music]"

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

        mix = f"[{voice}][{bed}]amix=inputs=2:normalize=0:duration=first"
        if ff.has_filter("alimiter"):
            # Summing (normalize=0) preserves narration level but can clip on peaks.
            mix += ",alimiter=limit=0.97:level=disabled"
        parts.append(f"{mix}[aout]")
        return parts, "[aout]"

    # ----------------------------------------------------------------- encode

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
        if profile.encoder_threads:
            # Cap per-process threads when scenes render concurrently; otherwise each
            # encoder claims every core and the processes fight each other.
            args += ["-threads", str(profile.encoder_threads)]
        return args


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
