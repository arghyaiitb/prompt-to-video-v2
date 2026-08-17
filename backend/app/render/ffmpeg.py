"""Thin subprocess wrapper around ffmpeg / ffprobe.

Two rules drive everything here:

1. Argument *lists*, never shell strings. Filtergraphs contain ``'`` ``:`` ``,``
   ``[`` ``]`` and ``%`` — every one of which a shell will happily mangle.
2. ffmpeg's stderr is the only useful diagnostic it produces, and the interesting
   part is always at the *end*. On failure we raise with the tail attached, and we
   log the full command at DEBUG so any failure can be reproduced by hand.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

STDERR_TAIL_LINES = 40
"""ffmpeg errors are unreadable without a chunk of trailing context."""

_FALLBACK_FFMPEG = "/opt/homebrew/bin/ffmpeg"


class FFmpegError(RuntimeError):
    """Non-zero exit from ffmpeg/ffprobe, with the command and stderr tail."""

    def __init__(self, argv: list[str], returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        super().__init__(
            f"{Path(argv[0]).name} exited {returncode}\n"
            f"command: {shlex.join(self.argv)}\n"
            f"--- stderr (last {STDERR_TAIL_LINES} lines) ---\n{tail}"
        )


def ffmpeg_bin() -> str:
    """Resolve the ffmpeg binary. ``FFMPEG_BIN`` wins so callers can point at a
    build with different filters compiled in (see module docs in ffmpeg_backend)."""
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or _FALLBACK_FFMPEG


def ffprobe_bin() -> str:
    if env := os.environ.get("FFPROBE_BIN"):
        return env
    # Keep ffprobe next to whatever ffmpeg we resolved; mixing builds probes lies.
    sibling = Path(ffmpeg_bin()).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


def run(argv: list[str | Path], *, timeout: float | None = 1800.0) -> str:
    """Run a command, return stderr. Raises :class:`FFmpegError` on non-zero exit."""
    args = [str(a) for a in argv]
    logger.debug("exec: %s", shlex.join(args))
    proc = subprocess.run(  # noqa: S603 - argv list, no shell
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise FFmpegError(args, proc.returncode, proc.stderr or proc.stdout)
    return proc.stderr


def ffmpeg(argv: list[str | Path], *, timeout: float | None = 1800.0) -> str:
    """Run ffmpeg with the boilerplate flags every call wants."""
    return run(
        [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *argv],
        timeout=timeout,
    )


def probe(path: str | Path) -> dict:
    """Full ffprobe JSON for ``path``."""
    out = subprocess.run(  # noqa: S603
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise FFmpegError([ffprobe_bin(), str(path)], out.returncode, out.stderr)
    return json.loads(out.stdout or "{}")


def _stream(data: dict, kind: str) -> dict | None:
    for s in data.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return None


def probe_duration(path: str | Path) -> float:
    """Container duration in seconds (falls back to the video stream's own)."""
    data = probe(path)
    fmt = data.get("format", {})
    if (d := fmt.get("duration")) not in (None, "N/A"):
        return float(d)
    for kind in ("video", "audio"):
        st = _stream(data, kind) or {}
        if (d := st.get("duration")) not in (None, "N/A"):
            return float(d)
    raise FFmpegError([ffprobe_bin(), str(path)], 0, "no duration in ffprobe output")


def probe_image_size(path: str | Path) -> tuple[int, int]:
    st = _stream(probe(path), "video")
    if not st:
        raise FFmpegError([ffprobe_bin(), str(path)], 0, "no video stream found")
    return int(st["width"]), int(st["height"])


def count_frames(path: str | Path) -> int:
    """Video frames actually in the file, by decoding it.

    ``probe_summary()['nb_frames']`` is container metadata and some muxers simply do not
    write it. This decodes, so it is slower and it is the truth — which is what a claim
    about frame-exactness needs to rest on.
    """
    out = subprocess.run(  # noqa: S603
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise FFmpegError([ffprobe_bin(), str(path)], out.returncode, out.stderr)
    value = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not value.isdigit():
        raise FFmpegError([ffprobe_bin(), str(path)], 0, f"no frame count in {out.stdout!r}")
    return int(value)


def probe_summary(path: str | Path) -> dict:
    """The handful of numbers worth reporting after a render."""
    data = probe(path)
    v = _stream(data, "video") or {}
    a = _stream(data, "audio") or {}
    fps = 0.0
    if rate := v.get("r_frame_rate"):
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "width": int(v.get("width", 0) or 0),
        "height": int(v.get("height", 0) or 0),
        "fps": round(fps, 3),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "audio_channels": int(a.get("channels", 0) or 0),
        "audio_sample_rate": int(a.get("sample_rate", 0) or 0),
        "nb_frames": int(v.get("nb_frames", 0) or 0),
    }


@lru_cache(maxsize=8)
def _filters(binary: str) -> frozenset[str]:
    out = subprocess.run(  # noqa: S603
        [binary, "-hide_banner", "-filters"], capture_output=True, text=True, check=False
    )
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        # " T. drawtext  V->V  Draw text ..."  -> flags, name, io, description
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


@lru_cache(maxsize=8)
def _encoders(binary: str) -> frozenset[str]:
    out = subprocess.run(  # noqa: S603
        [binary, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    return name in _filters(ffmpeg_bin())


def has_encoder(name: str) -> bool:
    return name in _encoders(ffmpeg_bin())


def available() -> bool:
    """True when the resolved ffmpeg binary can actually be executed."""
    try:
        run([ffmpeg_bin(), "-hide_banner", "-version"], timeout=20)
    except (OSError, FFmpegError, subprocess.SubprocessError):
        return False
    return True
