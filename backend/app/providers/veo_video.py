"""Generated MOTION footage via Veo 3.1, as an alternative visual source to a still.

VERIFIED against the live key on 2026-08-17. The three models on this key
(`veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`,
`veo-3.1-lite-generate-preview`) expose ONLY `predictLongRunning` — there is no
`generateContent` path, so none of `_gemini.generate_content` applies. The shape is
submit / poll / download:

    POST /v1beta/models/{model}:predictLongRunning?key=...
    {"instances":[{"prompt":"..."}],"parameters":{"aspectRatio":"16:9"}}
    -> {"name": "models/veo-3.1-fast-generate-preview/operations/2wuadveixogd"}

    GET /v1beta/{operation_name}?key=...        # repeat until done
    -> {"done": true, "response": {"generateVideoResponse":
          {"generatedSamples": [{"video": {"uri": ".../files/9awr...:download?alt=media"}}]}}}

    GET {uri}  with header `x-goog-api-key: ...`, following redirects -> the mp4 bytes

MEASURED CLIP PROPERTIES — these are the constraints callers live with:

    duration   8.000s FIXED, whatever duration is asked for
    frame      1280x720 (720p, NOT 1080p), 24 fps, h264
    audio      an AAC 48 kHz stereo track we always discard: narration is authoritative
    size       ~1.8 MB per clip

COST NOTE — Veo is by far the most expensive call in this pipeline. Every scene that
uses motion is one 8-second generation, billed whether or not the scene is 8 seconds
long, and a 60-90 second wall-clock wait on the `fast` tier. For comparison an image is
one sub-10-second call and music is one call for the whole video. So:

  * `clip_budget()` tells a caller what a render will cost BEFORE any call is made;
  * `VeoVideoProvider(max_clips=...)` is a hard per-instance cap so a bug in a loop
    cannot quietly bill fifty clips (default `DEFAULT_MAX_CLIPS`);
  * `veo_enabled()` reads the `video_enable_veo` settings flag and defaults to FALSE
    when the flag does not exist, so motion is opt-in at the wiring layer. The provider
    itself does not consult it — constructing this class IS the decision to spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.providers._gemini import RETRY_STATUS
from app.providers._media import FFPROBE, MediaError, run_ffmpeg

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_MODEL = "veo-3.1-fast-generate-preview"
"""Cost default. The full tier is not measurably better for b-roll behind narration."""

CLIP_SECONDS = 8.0
"""Measured, fixed. Not a maximum and not a request — this is what every call returns."""

CLIP_WIDTH = 1280
CLIP_HEIGHT = 720
CLIP_FPS = 24.0

DEFAULT_ASPECT_RATIO = "16:9"

POLL_INTERVAL_SECONDS = 10.0
"""First poll gap. Nothing finishes faster than ~45s, so a tighter loop only burns quota."""

MAX_WAIT_SECONDS = 300.0
"""Hard ceiling. `fast` took 60-90s when measured; 300s allows for a bad queue day."""

EMPTY_RESULT_ATTEMPTS = 3
"""Resubmissions when an operation completes with an error or no samples.

Same class of failure as Lyria's empty candidate: a 200-shaped success that carries no
media. Each retry rephrases slightly, because a refusal can be deterministic per wording.
"""

DEFAULT_MAX_CLIPS = 12
"""Per-instance spend cap. Raise deliberately, with a number in hand."""

REQUEST_TIMEOUT_SECONDS = 120.0
HTTP_ATTEMPTS = 3

# Veo renders legible text badly and any burnt-in words fight the overlay the renderer
# draws on top, so text is suppressed the same way it is for stills.
NEGATIVE_SUFFIX = (
    "No text, no letters, no words, no numbers, no captions, no subtitles, no signage, "
    "no watermarks, no logos anywhere in the frame."
)

# Steered toward slow continuous camera movement: this footage sits behind narration, and
# a cut or a whip pan mid-scene reads as an editing mistake rather than as b-roll.
STYLE_SUFFIX = (
    "Cinematic live-action b-roll, one single continuous shot with no cuts and no scene "
    "changes, slow smooth deliberate camera movement, natural lighting, shallow depth of "
    "field, calm and unhurried, no people speaking to camera."
)

DEFAULT_PROMPT = "Slow cinematic push-in across a calm modern workspace, soft natural light"

_PLACEHOLDER_PALETTE: tuple[tuple[str, str], ...] = (
    ("0x1F3A5F", "0x0B1626"),
    ("0x3F1F5F", "0x140B26"),
    ("0x1F5F4A", "0x0B2620"),
    ("0x5F3A1F", "0x261408"),
    ("0x4A1F3F", "0x200B1A"),
    ("0x1F4A5F", "0x0B1F26"),
)


class VideoClipError(RuntimeError):
    """Generation failed, or succeeded without usable video."""


class VeoTimeoutError(VideoClipError):
    """The operation did not finish inside the budget. Carries the id for a hand retry."""

    def __init__(self, message: str, operation: str) -> None:
        super().__init__(message)
        self.operation = operation


class ClipBudgetError(VideoClipError):
    """The per-instance clip cap was reached. A spend guard, not an API failure."""


@dataclass(frozen=True)
class ClipProbe:
    """What ffprobe says is really in a file. Never inferred from the request."""

    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool


@dataclass(frozen=True)
class ClipInfo:
    """The truth about one delivered clip, for the caller that has to cover a scene.

    ``requested_duration`` is what was asked for; ``duration`` is what arrived. They
    routinely differ — see `VeoVideoProvider.generate`.
    """

    path: Path
    model: str
    operation: str
    requested_duration: float
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def shortfall(self) -> float:
        """Seconds of the requested scene this clip does NOT cover (0.0 if it covers it)."""
        return max(0.0, self.requested_duration - self.duration)

    @property
    def covers_request(self) -> bool:
        return self.shortfall <= 0.001


@dataclass(frozen=True)
class ClipBudget:
    """What a render will cost in Veo calls, computed before spending anything."""

    clips: int
    billed_seconds: float
    requested_seconds: float
    uncovered_seconds: float

    def summary(self) -> str:
        return (
            f"{self.clips} Veo clip(s) = {self.billed_seconds:.0f}s billed at "
            f"{CLIP_SECONDS:.0f}s each for {self.requested_seconds:.1f}s of scene; "
            f"{self.uncovered_seconds:.1f}s not covered by clip footage"
        )


def veo_enabled(settings: Any | None = None) -> bool:
    """Whether motion footage is switched on for this deployment.

    Reads `settings.video_enable_veo`. Defaults to **False** when the setting does not
    exist, so nothing starts spending Veo credits merely because this module was
    imported — adding the flag is what turns it on.
    """
    settings = settings if settings is not None else get_settings()
    return bool(getattr(settings, "video_enable_veo", False))


def clip_budget(target_durations: Sequence[float], *, clips_per_scene: int = 1) -> ClipBudget:
    """Cost of covering `target_durations` with generated clips, one call per clip.

    ``clips_per_scene`` is for a caller that intends to stitch several clips to fill a
    long scene; the default of 1 matches this provider, which generates exactly one clip
    per `generate` call and reports the shortfall rather than chaining more calls itself.
    """
    if clips_per_scene < 1:
        raise ValueError("clips_per_scene must be at least 1")
    scenes = [float(d) for d in target_durations if float(d) > 0]
    clips = len(scenes) * clips_per_scene
    covered_per_scene = CLIP_SECONDS * clips_per_scene
    return ClipBudget(
        clips=clips,
        billed_seconds=clips * CLIP_SECONDS,
        requested_seconds=sum(scenes),
        uncovered_seconds=sum(max(0.0, d - covered_per_scene) for d in scenes),
    )


def clips_needed(target_duration: float) -> int:
    """How many 8s clips it would take to cover a scene. Informational — see clip_budget."""
    if target_duration <= 0:
        return 0
    return max(1, math.ceil(target_duration / CLIP_SECONDS))


class VeoVideoProvider:
    """Satisfies `VideoClipProvider`. Generates one moving clip per call.

    ``target_duration`` IS A REQUEST, NOT A PROMISE. Veo returns 8.000 seconds no matter
    what is asked for, so a 14-second scene gets an 8-second clip and there is nothing
    this provider can do about it. It therefore never pretends otherwise:

      * `generate` returns the path, as the Protocol requires;
      * the real measured duration, frame size and fps land on `last_clip_info`
        (a `ClipInfo`, with `.shortfall` and `.covers_request`) — read it after every
        call and decide how to cover the rest (hold the last frame, loop, or slow down);
      * a shortfall is logged as a WARNING naming both durations.

    A caller that ignores `last_clip_info` and lays an 8s clip under 14s of narration
    will desync everything downstream. That is the one thing to understand here.

    Two other guarantees:

      * the AAC track Veo attaches is stripped (`-an`) — narration is authoritative;
      * the video stream is passed through untouched at its native 24 fps. Resampling to
        the render profile's frame rate is the renderer's business, not this provider's.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        max_wait: float = MAX_WAIT_SECONDS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_clips: int | None = DEFAULT_MAX_CLIPS,
    ) -> None:
        settings = get_settings()
        # `video_default_video_model` may not exist yet; the fast tier is the cost default.
        self.model = model or getattr(settings, "video_default_video_model", None) or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.aspect_ratio = aspect_ratio
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.timeout = timeout
        self.max_clips = max_clips
        self.clips_generated = 0
        self.last_clip_info: ClipInfo | None = None

    # ------------------------------------------------------------------ port surface

    def generate(self, prompt: str, target_duration: float, out_path: Path) -> Path:
        """Generate one clip for `prompt`; see the class docstring on `target_duration`."""
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if not self.api_key:
            raise VideoClipError("gemini_api_key is empty — set GEMINI_API_KEY in .env")
        if self.max_clips is not None and self.clips_generated >= self.max_clips:
            raise ClipBudgetError(
                f"{self.model}: refusing to generate clip {self.clips_generated + 1}; this "
                f"provider is capped at max_clips={self.max_clips}. Veo bills "
                f"{CLIP_SECONDS:.0f}s per call — check the caller is not looping, then raise "
                f"the cap explicitly."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        operation = ""
        last_error: Exception | None = None
        for attempt in range(1, EMPTY_RESULT_ATTEMPTS + 1):
            operation = self._submit(compose_prompt(prompt, attempt=attempt))
            try:
                uri = self._await_video_uri(operation)
                break
            except VeoTimeoutError:
                # A timeout is not an empty result: the operation may still be running and
                # resubmitting would double the bill. Surface it with the id instead.
                raise
            except VideoClipError as exc:
                last_error = exc
                logger.warning(
                    "%s operation %s produced no video (attempt %d/%d): %s",
                    self.model,
                    operation,
                    attempt,
                    EMPTY_RESULT_ATTEMPTS,
                    exc,
                )
                if attempt < EMPTY_RESULT_ATTEMPTS:
                    time.sleep(1.5 * attempt)
        else:
            raise VideoClipError(
                f"{self.model} returned no video after {EMPTY_RESULT_ATTEMPTS} attempts "
                f"(last operation {operation})"
            ) from last_error

        data = self._download(uri)
        with tempfile.TemporaryDirectory(prefix="veo-") as tmp:
            raw = Path(tmp) / "raw.mp4"
            raw.write_bytes(data)
            strip_audio(raw, out_path)

        self.clips_generated += 1
        self.last_clip_info = _describe(
            out_path,
            model=self.model,
            operation=operation,
            requested_duration=target_duration,
        )
        _warn_on_shortfall(self.last_clip_info)
        return out_path

    # ------------------------------------------------------------------ hand retry

    def fetch_completed(self, operation: str, out_path: Path, *, requested_duration: float) -> Path:
        """Finish a `VeoTimeoutError` by hand: poll `operation` once more and download it.

        The operation id from the exception is all that is needed; the generation is
        already paid for, so this never resubmits.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._poll(operation)
        if not payload.get("done"):
            raise VeoTimeoutError(f"operation {operation} is still running", operation)
        uri = video_uri_from(payload)
        with tempfile.TemporaryDirectory(prefix="veo-") as tmp:
            raw = Path(tmp) / "raw.mp4"
            raw.write_bytes(self._download(uri))
            strip_audio(raw, out_path)
        self.last_clip_info = _describe(
            out_path,
            model=self.model,
            operation=operation,
            requested_duration=requested_duration,
        )
        _warn_on_shortfall(self.last_clip_info)
        return out_path

    # ------------------------------------------------------------------ transport

    def _submit(self, prompt: str) -> str:
        """POST predictLongRunning, returning the operation name."""
        body = {
            "instances": [{"prompt": prompt}],
            "parameters": {"aspectRatio": self.aspect_ratio},
        }
        url = f"{BASE_URL}/models/{self.model}:predictLongRunning"
        payload = _request(
            "POST",
            url,
            params={"key": self.api_key},
            json_body=body,
            timeout=self.timeout,
            label=self.model,
        )
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise VideoClipError(f"{self.model}: submit returned no operation name: {payload!r}")
        logger.info("%s submitted operation %s", self.model, name)
        return name

    def _poll(self, operation: str) -> dict[str, Any]:
        return _request(
            "GET",
            f"{BASE_URL}/{operation}",
            params={"key": self.api_key},
            timeout=self.timeout,
            label=operation,
        )

    def _await_video_uri(self, operation: str) -> str:
        """Poll until done, then extract the sample uri. Raises on timeout or empty result."""
        deadline = time.monotonic() + self.max_wait
        interval = self.poll_interval
        while True:
            payload = self._poll(operation)
            if payload.get("done"):
                return video_uri_from(payload)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VeoTimeoutError(
                    f"{self.model}: operation {operation} did not finish within "
                    f"{self.max_wait:.0f}s. It is probably still running and already billed — "
                    f"retry by hand with fetch_completed({operation!r}, out_path, "
                    f"requested_duration=...) rather than resubmitting.",
                    operation,
                )
            time.sleep(min(interval, remaining))
            # Ease off slightly: nothing is gained by hammering a job that takes ~90s.
            interval = min(interval * 1.25, 30.0)

    def _download(self, uri: str) -> bytes:
        """GET the file uri. Auth is a header here, not a query param, and it redirects."""
        last_error = ""
        for attempt in range(1, HTTP_ATTEMPTS + 1):
            try:
                response = httpx.get(
                    uri,
                    headers={"x-goog-api-key": self.api_key},
                    follow_redirects=True,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == HTTP_ATTEMPTS:
                    raise VideoClipError(f"download {uri}: {last_error}") from exc
                time.sleep(2.0 * attempt)
                continue
            if response.status_code == 200:
                data = response.content
                if not data:
                    raise VideoClipError(f"download {uri} returned an empty body")
                return data
            last_error = f"HTTP {response.status_code}: {response.text[:400]}"
            if response.status_code in RETRY_STATUS and attempt < HTTP_ATTEMPTS:
                time.sleep(2.0 * attempt)
                continue
            raise VideoClipError(f"download {uri}: {last_error}")
        raise VideoClipError(f"download {uri}: exhausted attempts — {last_error}")


class PlaceholderVideoProvider:
    """Satisfies `VideoClipProvider` with a locally generated drifting gradient.

    Exists so the render and worker stages can be exercised end to end for free — Veo is
    the most expensive call in the pipeline and nothing about wiring motion into a scene
    needs real footage to test.

    It deliberately mimics Veo's awkward part: by default it emits a FIXED
    `CLIP_SECONDS` clip at 1280x720/24fps with no audio, so a caller that assumes it got
    the duration it asked for breaks here, cheaply, instead of in production. Pass
    `fixed_duration=None` to honour `target_duration` instead.
    """

    def __init__(
        self,
        *,
        fixed_duration: float | None = CLIP_SECONDS,
        width: int = CLIP_WIDTH,
        height: int = CLIP_HEIGHT,
        fps: float = CLIP_FPS,
    ) -> None:
        self.fixed_duration = fixed_duration
        self.width = width
        self.height = height
        self.fps = fps
        self.last_clip_info: ClipInfo | None = None

    def generate(self, prompt: str, target_duration: float, out_path: Path) -> Path:
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        duration = self.fixed_duration if self.fixed_duration is not None else target_duration
        duration = max(0.5, float(duration))
        digest = hashlib.sha256((prompt or "").encode("utf-8")).digest()
        c0, c1 = _PLACEHOLDER_PALETTE[digest[0] % len(_PLACEHOLDER_PALETTE)]
        seed = int.from_bytes(digest[4:8], "big")

        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                # speed drives the drift, which is the whole point: a still placeholder
                # would not exercise the renderer's motion path at all.
                f"gradients=s={self.width}x{self.height}:c0={c0}:c1={c1}"
                f":x0=0:y0=0:x1={self.width}:y1={self.height}"
                f":nb_colors=2:d={duration:.3f}:seed={seed}:speed=0.004:r={self.fps:g}",
                "-t",
                f"{duration:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )
        self.last_clip_info = _describe(
            out_path,
            model="placeholder",
            operation="local",
            requested_duration=target_duration,
        )
        _warn_on_shortfall(self.last_clip_info)
        return out_path


# --------------------------------------------------------------------------- helpers


def compose_prompt(prompt: str, *, attempt: int = 1) -> str:
    """Add motion/style and no-text guards. Later attempts get progressively plainer.

    An operation that finishes with an error or no samples can be reproducible for a
    given wording, so a retry on the identical string can fail identically — the same
    failure mode Lyria shows with empty candidates.
    """
    text = (prompt or "").strip()
    if attempt >= 3:
        return f"{DEFAULT_PROMPT}. {NEGATIVE_SUFFIX}"
    if not text:
        text = DEFAULT_PROMPT
    parts = [text if text.endswith((".", "!", "?")) else f"{text}."]
    lowered = text.lower()
    if attempt == 1 and "b-roll" not in lowered and "continuous shot" not in lowered:
        parts.append(STYLE_SUFFIX)
    if "no text" not in lowered:
        parts.append(NEGATIVE_SUFFIX)
    return " ".join(parts)


def video_uri_from(payload: dict[str, Any]) -> str:
    """Pull the sample uri out of a completed operation, or explain what came back instead.

    A completed operation is not a successful one: `done: true` with an `error` object is
    the normal way a blocked or failed generation is reported, and so is `done: true`
    with an empty `generatedSamples`.
    """
    error = payload.get("error")
    if error:
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise VideoClipError(
            f"operation {payload.get('name', '?')} failed (code={code}): {message}"
        )
    response = payload.get("response") or {}
    samples = (response.get("generateVideoResponse") or {}).get("generatedSamples") or []
    if not samples:
        # Empty samples usually means the prompt tripped a safety filter; the reason,
        # when there is one, sits next to the samples list.
        detail = (response.get("generateVideoResponse") or {}).get("raiMediaFilteredReasons")
        raise VideoClipError(
            f"operation {payload.get('name', '?')} finished with no generatedSamples "
            f"(filtered reasons={detail!r})"
        )
    uri = ((samples[0] or {}).get("video") or {}).get("uri")
    if not isinstance(uri, str) or not uri:
        raise VideoClipError(f"generatedSamples[0] carried no video uri: {samples[0]!r}")
    return uri


def strip_audio(src: Path, out_path: Path) -> Path:
    """Copy the video stream into `out_path` with the audio track dropped.

    Veo attaches AAC stereo; narration is authoritative, so it never reaches the mux.
    The video stream is stream-copied — no re-encode, no quality loss, and crucially no
    frame-rate change: 24 fps is reported, not silently resampled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src), "-map", "0:v:0", "-an"]
    # +faststart is an mp4-family muxer option; passing it elsewhere is not just useless,
    # it is a different muxer's namespace.
    if out_path.suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+faststart"]
    try:
        run_ffmpeg([*args, "-c:v", "copy", str(out_path)])
    except MediaError:
        # A container that cannot hold h264 as-is (or an unexpected codec from a future
        # model) — re-encode with the muxer's own default encoder rather than fail. Still
        # no audio, still the source frame rate.
        run_ffmpeg([*args, str(out_path)])
    return out_path


def probe_clip(path: Path | str) -> ClipProbe:
    """Measure a clip with ffprobe. Nothing here is inferred from the request."""
    proc = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe emitted non-json
        raise MediaError(f"could not parse ffprobe output for {path}") from exc

    streams = parsed.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = _as_float((parsed.get("format") or {}).get("duration"))
    if duration <= 0 and video is not None:
        duration = _as_float(video.get("duration"))

    if video is None:
        return ClipProbe(duration, 0, 0, 0.0, has_audio, False)
    return ClipProbe(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_as_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=has_audio,
        has_video=True,
    )


def _describe(
    path: Path, *, model: str, operation: str, requested_duration: float
) -> ClipInfo:
    probe = probe_clip(path)
    if not probe.has_video:
        raise VideoClipError(f"{path} has no video stream")
    if probe.has_audio:  # pragma: no cover - strip_audio would have to have failed
        raise VideoClipError(f"{path} still carries an audio track after -an")
    return ClipInfo(
        path=path,
        model=model,
        operation=operation,
        requested_duration=requested_duration,
        duration=probe.duration,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        has_audio=probe.has_audio,
    )


def _warn_on_shortfall(info: ClipInfo) -> None:
    if info.covers_request:
        logger.info(
            "clip %s: %.3fs %dx%d @%.3g fps (covers the %.3fs requested)",
            info.path.name,
            info.duration,
            info.width,
            info.height,
            info.fps,
            info.requested_duration,
        )
        return
    logger.warning(
        "clip %s is %.3fs but %.3fs was requested — %.3fs of the scene is NOT covered by "
        "footage. The renderer must hold, loop, or slow this clip; laying it down as-is "
        "will desync narration. See last_clip_info.",
        info.path.name,
        info.duration,
        info.requested_duration,
        info.shortfall,
    )


def _request(
    method: str,
    url: str,
    *,
    params: dict[str, str],
    json_body: dict[str, Any] | None = None,
    timeout: float,
    label: str,
) -> dict[str, Any]:
    """One JSON call with the same linear-backoff policy as `_gemini.generate_content`."""
    last_error = ""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            if method == "POST":
                response = httpx.post(
                    url,
                    params=params,
                    json=json_body,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
            else:
                response = httpx.get(url, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            last_error = f"transport error: {exc}"
            if attempt == HTTP_ATTEMPTS:
                raise VideoClipError(f"{label}: {last_error}") from exc
            time.sleep(2.0 * attempt)
            continue

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                raise VideoClipError(f"{label}: response was not JSON") from exc
            if not isinstance(payload, dict):
                raise VideoClipError(f"{label}: expected a JSON object, got {type(payload)}")
            return payload

        last_error = f"HTTP {response.status_code}: {response.text[:800]}"
        if response.status_code in RETRY_STATUS and attempt < HTTP_ATTEMPTS:
            time.sleep(2.0 * attempt)
            continue
        raise VideoClipError(f"{label}: {last_error}")

    raise VideoClipError(f"{label}: exhausted {HTTP_ATTEMPTS} attempts — {last_error}")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_fps(rate: Any) -> float:
    """ffprobe reports fps as the string "24/1"; "0/0" means it could not tell."""
    if not isinstance(rate, str) or "/" not in rate:
        return _as_float(rate)
    numerator, _, denominator = rate.partition("/")
    den = _as_float(denominator)
    return _as_float(numerator) / den if den else 0.0
