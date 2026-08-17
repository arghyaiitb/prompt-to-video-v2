"""Narration via Deepgram Aura.

VERIFIED against the live key: POST /v1/speak with `Authorization: Token <key>` and
`{"text": "..."}` returns the wav container as the raw response body — there is no JSON
envelope and no polling step, so the bytes go straight to disk.

Duration is read back with ffprobe rather than estimated from word count. Every scene
boundary in the Timeline is derived from real audio length; a guessed duration desyncs
captions and transitions by a growing offset across the video.

SSML IS NOT SUPPORTED — SETTLED, DO NOT RE-LITIGATE
---------------------------------------------------
There is no flag, parameter, header or content type that turns SSML on. Deepgram says so,
and the API says so:

  * "SSML is not on our roadmap at this time. We're seeing that the industry is moving
    away from SSML, and toward naturally expressive TTS... We're planning to release a new
    version of our TTS, Aura-2... but it will not have SSML support."
    — jkroll-deepgram, https://github.com/orgs/deepgram/discussions/1031 (2024-12-23)
  * The request schema has no SSML field. Sending `{"ssml": ...}` returns HTTP 400
    `PAYLOAD_ERROR`: "Please specify exactly one of `text` or `url` in the JSON body."
  * `Content-Type: application/ssml+xml` returns HTTP 415: "`Content-Type` must be either
    `text/plain` or `application/json`."
  * `?ssml=true`, `?enable_ssml=true`, `?input_type=ssml`, `?text_type=ssml` and an
    `X-Deepgram-SSML: true` header are all silently IGNORED — unknown params do not error,
    which is exactly what makes this easy to get wrong.
  * https://developers.deepgram.com/reference/text-to-speech-api/speak documents `text` as
    the only body field; the feature overview and prompting guides never mention SSML.

Markup therefore reaches the voice model as literal characters, and it corrupts the
narration in three distinct ways — all measured on this key, round-tripped through
`/v1/listen?model=nova-3`:

    aura-2  tags are SPOKEN. `<speak>X<break time="1s"/>Y</speak>` transcribes as
            "Speak. X. Break time equals once. Y." — words INSERTED.
    aura-1  tags are not spoken but mangle the adjacent word ("carefully" -> "CarefulLab")
            and the break is NOT honoured (3.46s -> 3.61s) — words CORRUPTED.
    no <speak> wrapper
            everything after the first tag can be DROPPED: "Verify the domain
            carefully<break/>before you approve the payment request." transcribed as
            "Verify the domain carefully. Break time equals ones." — words LOST.

Any of those breaks the pipeline invariant that `DeepgramAligner` and
`bullet_timing` depend on: the spoken words must equal the plain reference text, or bullet
anchoring loses its verbatim n-gram. Hence `supports_ssml = False` AND the unconditional
`strip_markup` in `synthesize` — the flag routes well-behaved callers, the strip protects
us from the rest.

Deepgram's documented substitute for `<break>` is an ellipsis, and it is NOT a usable
substitute for timed pauses: synthesis is non-deterministic, and identical input measured
2.28-3.04s (plain) and 2.88-4.04s (with `......`) across repeat calls. A run-to-run spread
near 1s swamps the effect being asked for, so there is nothing to calibrate against. Pace
narration by writing punctuation, or use the `speed` query parameter, which does work.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.providers._media import audio_duration, run_ffmpeg

logger = logging.getLogger(__name__)

SPEAK_URL = "https://api.deepgram.com/v1/speak"

# Deepgram rejects oversized single requests; scene narration is far below this, but a
# caller handing over a whole script should get audio, not a 400.
MAX_CHARS = 1800

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# Only well-formed tag-like constructs, so prose containing a bare comparison ("latency <
# 200ms") is left completely alone. A tag name must start with a letter, which is what
# separates `<break time="1s"/>` from `<` used as arithmetic.
_MARKUP_TAG = re.compile(r"</?[A-Za-z][\w:.\-]*(?:\s[^<>]*?)?/?>")
_XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# SSML bodies arrive XML-escaped; once the tags are gone the escapes must be undone or the
# voice says "amp" out loud. Applied ONLY when markup was actually present, so plain text
# passes through byte-for-byte.
_ENTITIES = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
    ("&#39;", "'"),
    ("&#x27;", "'"),
    ("&nbsp;", " "),
    ("&amp;", "&"),  # last: undoing this first would re-create the others
)


class SynthesisError(RuntimeError):
    """Deepgram refused or returned something that was not audio."""


class DeepgramSynthesizer:
    """Satisfies `SpeechSynthesizer`.

    linear16 / 24 kHz wav is requested deliberately: an uncompressed container gives the
    aligner a clean signal to time against and lets ffmpeg concatenate scenes without a
    decode-reencode generation loss.
    """

    supports_ssml: bool = False
    """Aura parses no markup at all — see the module docstring for the doc citation and the
    three measured corruption modes. Callers route on this; `synthesize` strips regardless.
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

        # Last line of defence. A caller that ignored `supports_ssml` would otherwise ship
        # a video whose narrator reads tag names aloud, and the audio would pass every
        # automated check we have — it is the right length and it is valid wav.
        text, had_markup = strip_markup(text)
        if had_markup:
            logger.warning(
                "stripped markup before Deepgram synthesis — Aura vocalises SSML; "
                "check supports_ssml before sending markup (text now: %.80r)",
                text,
            )
        if not text:
            raise ValueError("text contained only markup — nothing left to synthesize")

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


def strip_markup(text: str) -> tuple[str, bool]:
    """Reduce SSML to the words inside it. Returns `(plain_text, had_markup)`.

    Deliberately WORD-PRESERVING, because the aligner is handed the plain reference text
    and `bullet_timing` anchors each bullet to a verbatim n-gram of it: whatever is spoken
    must contain the same words in the same order. So every tag is replaced by a SPACE
    rather than deleted — dropping `<break/>` outright would weld "carefully<break/>before"
    into the single non-word "carefullybefore" — and `<sub alias>` keeps its written form
    rather than the alias, since the alias is not in the reference text.

    Plain text is returned unchanged, entities included: the escape handling only runs when
    markup was actually found, so this is safe to call unconditionally on every scene.
    """
    original = text or ""
    without_comments = _XML_COMMENT.sub(" ", original)
    stripped = _MARKUP_TAG.sub(" ", without_comments)

    had_markup = stripped != original
    if not had_markup:
        return original.strip(), False

    for entity, char in _ENTITIES:
        stripped = stripped.replace(entity, char)
    # Tag removal leaves the gaps it replaced; collapse them so the voice model sees
    # ordinary prose spacing, and tidy space stranded before punctuation.
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    return stripped.strip(), True


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
