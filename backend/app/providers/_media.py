"""ffmpeg/ffprobe shell-outs shared by the media providers.

Kept private to the providers package: nothing outside needs to know that duration
comes from ffprobe rather than from a vendor's response body.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class MediaError(RuntimeError):
    """ffmpeg/ffprobe failed. Carries stderr because that is the only useful part."""


def run_ffmpeg(args: list[str], *, timeout: float = 300.0) -> None:
    """Run ffmpeg with -nostdin -y. Raises MediaError with stderr on failure."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")


def _ffprobe(args: list[str], *, timeout: float = 60.0) -> str:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", *args], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    return proc.stdout.strip()


def audio_duration(path: Path | str) -> float:
    """Real decoded duration in seconds.

    The container header is authoritative for wav; for mp3 the stream duration can
    disagree with the format duration, so we prefer the format value and fall back.
    """
    out = _ffprobe(
        ["-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    )
    try:
        return float(out.splitlines()[0])
    except (ValueError, IndexError):
        pass
    out = _ffprobe(
        [
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    try:
        return float(out.splitlines()[0])
    except (ValueError, IndexError) as exc:  # pragma: no cover - corrupt file
        raise MediaError(f"could not read audio duration from {path}") from exc


def image_dimensions(path: Path | str) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    out = _ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    try:
        w, h = out.splitlines()[0].split(",")[:2]
        return int(w), int(h)
    except (ValueError, IndexError) as exc:
        raise MediaError(f"could not read image dimensions from {path}") from exc


def transcode(src: Path, dst: Path) -> Path:
    """Container/codec change only — never rescales. Used when a vendor hands back
    a jpeg for a path we promised as png."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(src), "-frames:v", "1", str(dst)])
    return dst
