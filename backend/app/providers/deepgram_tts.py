"""Narration via Deepgram Aura.

VERIFIED against the live key: POST /v1/speak with `Authorization: Token <key>` and
`{"text": "..."}` returns the wav container as the raw response body — there is no JSON
envelope and no polling step, so the bytes go straight to disk.

Duration is read back with ffprobe rather than estimated from word count. Every scene
boundary in the Timeline is derived from real audio length; a guessed duration desyncs
captions and transitions by a growing offset across the video.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.providers._media import audio_duration, run_ffmpeg

SPEAK_URL = "https://api.deepgram.com/v1/speak"

# Deepgram rejects oversized single requests; scene narration is far below this, but a
# caller handing over a whole script should get audio, not a 400.
MAX_CHARS = 1800

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


class SynthesisError(RuntimeError):
    """Deepgram refused or returned something that was not audio."""


class DeepgramSynthesizer:
    """Satisfies `SpeechSynthesizer`.

    linear16 / 24 kHz wav is requested deliberately: an uncompressed container gives the
    aligner a clean signal to time against and lets ffmpeg concatenate scenes without a
    decode-reencode generation loss.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_voice: str | None = None,
        sample_rate: int = 24000,
        timeout: float = 120.0,
        attempts: int = 3,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.deepgram_api_key
        self.default_voice = default_voice or settings.video_default_tts_voice
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.attempts = attempts

    def synthesize(self, text: str, voice: str, out_path: Path) -> Path:
        text = (text or "").strip()
        if not text:
            raise ValueError("cannot synthesize empty text")
        if not self.api_key:
            raise SynthesisError("deepgram_api_key is empty — set DEEPGRAM_API_KEY in .env")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        voice = voice or self.default_voice

        chunks = _chunk(text, MAX_CHARS)
        if len(chunks) == 1:
            out_path.write_bytes(self._speak(chunks[0], voice))
            return out_path

        with tempfile.TemporaryDirectory(prefix="dg-tts-") as tmp:
            tmp_dir = Path(tmp)
            pieces: list[Path] = []
            for index, chunk in enumerate(chunks):
                piece = tmp_dir / f"part{index:03d}.wav"
                piece.write_bytes(self._speak(chunk, voice))
                pieces.append(piece)
            _concat_wavs(pieces, out_path)
        return out_path

    def duration(self, audio_path: Path) -> float:
        """Real audio length in seconds. The pipeline's clock — see module docstring."""
        return audio_duration(audio_path)

    def synthesize_with_duration(
        self, text: str, voice: str, out_path: Path
    ) -> tuple[Path, float]:
        """One call for the common case of needing both the file and its exact length."""
        path = self.synthesize(text, voice, out_path)
        return path, audio_duration(path)

    def _speak(self, text: str, voice: str) -> bytes:
        params = {
            "model": voice,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "container": "wav",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = ""
        for attempt in range(1, self.attempts + 1):
            try:
                response = httpx.post(
                    SPEAK_URL,
                    params=params,
                    headers=headers,
                    json={"text": text},
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == self.attempts:
                    raise SynthesisError(last_error) from exc
                continue

            if response.status_code == 200:
                data = response.content
                if not data.startswith(b"RIFF"):
                    raise SynthesisError(
                        f"expected a wav body, got {response.headers.get('content-type')!r}: "
                        f"{data[:200]!r}"
                    )
                return data

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code in RETRY_STATUS and attempt < self.attempts:
                continue
            raise SynthesisError(last_error)

        raise SynthesisError(f"exhausted {self.attempts} attempts — {last_error}")


# --------------------------------------------------------------------------- helpers


def probe_duration(audio_path: Path | str) -> float:
    """Module-level duration helper for callers that have a path but no synthesizer."""
    return audio_duration(audio_path)


def _chunk(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries, packing as many sentences per request as fit."""
    if len(text) <= limit:
        return [text]

    sentences: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _concat_wavs(pieces: list[Path], out_path: Path) -> None:
    """Sample-accurate join. All pieces share a format, so the concat demuxer is exact."""
    listing = out_path.parent / f".{out_path.stem}.concat.txt"
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in pieces), encoding="utf-8"
    )
    try:
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out_path)]
        )
    finally:
        listing.unlink(missing_ok=True)
