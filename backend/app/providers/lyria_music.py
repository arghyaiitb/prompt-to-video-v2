"""Background music via Lyria.

VERIFIED against the live key: POST
`/v1beta/models/lyria-3-clip-preview:generateContent` with a plain text part returns
`mimeType: audio/mpeg`, 44.1 kHz stereo mp3 at ~193 kbps.

Two shape details that bite:

  * the candidate carries TWO parts, `[{"text": ...}, {"inlineData": ...}]`, with the
    commentary text FIRST — so `parts[0]` is not the audio. We scan for inlineData.
  * the clip length is NOT a fixed constant. Separate calls returned 29.57s and 30.77s,
    so the loop/trim maths reads the real length with ffprobe on every call instead of
    assuming ~30s. Hardcoding the clip length silently mis-sizes every music bed.

`MusicProvider.generate` promises an arbitrary `target_duration`, so the clip is never
returned as-is:

  * target longer than the clip  -> the clip is looped, each seam joined with an
    `acrossfade` so there is no click where the tail meets the head;
  * target shorter than the clip -> trimmed, with a fade-out so it does not stop dead.

Either way the output is padded and trimmed to the exact target and then measured with
ffprobe, because a music bed that runs short leaves silence under the last scene and one
that runs long gets abruptly cut by the video mux.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from app.core.config import get_settings
from app.providers._gemini import GeminiError, generate_content, inline_data_from
from app.providers._media import MediaError, audio_duration, run_ffmpeg

logger = logging.getLogger(__name__)

EMPTY_PART_ATTEMPTS = 3
"""Retries when the model returns a 200 with no audio part."""

# Seam length for the loop crossfade. Long enough to hide the join, short enough that a
# 30s clip does not lose an audible chunk of its structure per repeat.
CROSSFADE_SECONDS = 2.5

FADE_IN_SECONDS = 1.5
FADE_OUT_SECONDS = 2.5

DURATION_TOLERANCE = 0.1

# Narration sits on top, so the bed is steered away from anything that competes for
# attention: no vocals, no melodic hooks, no big dynamic swings.
STYLE_SUFFIX = (
    "Purely instrumental background bed for a narrated explainer video. No vocals, no "
    "singing, no voices, no lyrics, no spoken word. Soft and unobtrusive, gentle sustained "
    "textures, steady quiet dynamics with no crescendos, drops, or sudden accents. Minimal "
    "sparse arrangement that sits well underneath a speaking voice, mixed low and warm, "
    "consistent volume throughout, no prominent lead melody."
)

DEFAULT_MOOD = "calm ambient background music, soft piano"

_MIME_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
}


class MusicError(RuntimeError):
    """Generation succeeded but the audio could not be fitted to the target duration."""


class LyriaMusicProvider:
    """Satisfies `MusicProvider`."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        crossfade: float = CROSSFADE_SECONDS,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.video_default_music_model
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.timeout = timeout
        self.crossfade = crossfade

    def generate(self, mood: str, target_duration: float, out_path: Path) -> Path:
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # A 200 can still return a candidate with no parts (observed:
        # finishReason='OTHER'). That is a generation-side hiccup, not a client error,
        # so it deserves the same retry treatment as a 503 — which generate_content's
        # status-code retry does not cover. Vary the prompt slightly each attempt so a
        # deterministic refusal on one phrasing doesn't repeat forever.
        last_error: Exception | None = None
        for attempt in range(1, EMPTY_PART_ATTEMPTS + 1):
            body = {"contents": [{"parts": [{"text": _compose_prompt(mood, attempt=attempt)}]}]}
            try:
                response = generate_content(self.model, body, self.api_key, timeout=self.timeout)
                mime, data = inline_data_from(response)
                break
            except GeminiError as exc:
                last_error = exc
                if "no parts" not in str(exc) and "no candidates" not in str(exc):
                    raise
                logger.warning(
                    "%s returned an empty candidate (attempt %d/%d): %s",
                    self.model,
                    attempt,
                    EMPTY_PART_ATTEMPTS,
                    exc,
                )
                time.sleep(1.5 * attempt)
        else:
            raise MusicError(
                f"{self.model} returned no audio after {EMPTY_PART_ATTEMPTS} attempts"
            ) from last_error

        suffix = _MIME_EXTENSIONS.get(mime.lower(), ".mp3")
        with tempfile.TemporaryDirectory(prefix="lyria-") as tmp:
            clip = Path(tmp) / f"clip{suffix}"
            clip.write_bytes(data)
            clip_duration = audio_duration(clip)
            if clip_duration <= 0:
                raise MusicError(f"generated clip had no duration (mime={mime!r})")
            _fit_duration(clip, clip_duration, target_duration, out_path, self.crossfade)

        actual = audio_duration(out_path)
        if abs(actual - target_duration) > DURATION_TOLERANCE:
            raise MusicError(
                f"fitted music is {actual:.3f}s but {target_duration:.3f}s was requested "
                f"(source clip {clip_duration:.3f}s)"
            )
        return out_path


# --------------------------------------------------------------------------- helpers


def _compose_prompt(mood: str, *, attempt: int = 1) -> str:
    """Build the prompt. Later attempts are progressively plainer.

    An empty candidate is sometimes reproducible for a given phrasing, so retrying the
    identical string can fail identically. Falling back toward a generic request gives
    the retry a real chance of differing.
    """
    mood = (mood or "").strip().rstrip(".") or DEFAULT_MOOD
    if attempt >= 3:
        return DEFAULT_MOOD
    if attempt == 2:
        return f"{DEFAULT_MOOD}. {STYLE_SUFFIX}"
    return f"{mood}. {STYLE_SUFFIX}"


def loop_count(clip_duration: float, target_duration: float, crossfade: float) -> int:
    """How many copies of the clip are needed to cover the target.

    Each `acrossfade` overlaps two copies, so N copies yield
    `N * clip - (N - 1) * crossfade` seconds, not `N * clip`. Ignoring the overlap is
    what leaves a music bed several seconds short of the narration.
    """
    if target_duration <= clip_duration:
        return 1
    usable = clip_duration - crossfade
    if usable <= 0:  # pragma: no cover - only for a clip shorter than the crossfade
        raise MusicError(
            f"clip ({clip_duration:.2f}s) is not longer than the crossfade ({crossfade:.2f}s)"
        )
    copies = 1
    covered = clip_duration
    while covered < target_duration:
        copies += 1
        covered += usable
    return copies


def _fit_duration(
    clip: Path, clip_duration: float, target: float, out_path: Path, crossfade: float
) -> None:
    """Render exactly `target` seconds of audio from `clip` into `out_path`."""
    crossfade = max(0.1, min(crossfade, clip_duration / 3))
    copies = loop_count(clip_duration, target, crossfade)

    fade_out = min(FADE_OUT_SECONDS, target / 4)
    fade_in = min(FADE_IN_SECONDS, target / 4)

    filters: list[str] = []
    if copies == 1:
        stage = "[0:a]"
    else:
        # Chain pairwise crossfades: (((0 x 1) x 2) x 3) ...
        previous = "[0:a]"
        for index in range(1, copies):
            label = f"[x{index}]"
            filters.append(
                f"{previous}[{index}:a]acrossfade=d={crossfade:.3f}:c1=tri:c2=tri{label}"
            )
            previous = label
        stage = previous

    # apad before atrim guarantees the exact length even if the crossfade math lands a
    # few milliseconds short; atrim then caps it. Together they make the output exact.
    filters.append(
        f"{stage}apad,atrim=0:{target:.3f},asetpts=N/SR/TB,"
        f"afade=t=in:st=0:d={fade_in:.3f},"
        f"afade=t=out:st={max(0.0, target - fade_out):.3f}:d={fade_out:.3f}[out]"
    )

    args: list[str] = []
    for _ in range(copies):
        args += ["-i", str(clip)]
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        *_codec_args(out_path),
        str(out_path),
    ]
    try:
        run_ffmpeg(args)
    except MediaError as exc:
        raise MusicError(str(exc)) from exc


def _codec_args(out_path: Path) -> list[str]:
    """Pick an encoder from the requested extension; the filter graph forces a re-encode."""
    suffix = out_path.suffix.lower()
    if suffix == ".wav":
        return ["-c:a", "pcm_s16le"]
    if suffix in (".m4a", ".aac", ".mp4"):
        return ["-c:a", "aac", "-b:a", "192k"]
    if suffix == ".flac":
        return ["-c:a", "flac"]
    return ["-c:a", "libmp3lame", "-b:a", "192k"]
