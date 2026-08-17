"""Word-level timing via Deepgram STT, mapped back onto the original script.

VERIFIED against the live key: POST /v1/listen with `Content-Type: audio/wav` and the
raw wav as the body returns
`results.channels[0].alternatives[0].words[*] = {word, start, end, confidence,
punctuated_word}` plus `metadata.duration`.

THE CORRECTNESS RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------------------
A round trip through speech-to-text is lossy in exactly the way captions cannot afford:
punctuation disappears, casing is invented, "twenty five" comes back as "25", and the odd
word is misheard outright. So the two signals are given different jobs:

    the reference script is the source of truth for WHAT IS DISPLAYED
    the transcript is the source of truth for WHEN

`align` therefore never returns transcript words. It pairs the two sequences with
`difflib.SequenceMatcher` over normalised forms, keeps the reference token as the display
text, and borrows only the transcript's start/end. Reference words the transcript did not
match get timings interpolated across the surrounding anchors, weighted by token length,
so a misheard word still lands in the right place instead of vanishing from the captions.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.models import Word

LISTEN_URL = "https://api.deepgram.com/v1/listen"

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

# Confidence stamped on words whose timing was interpolated rather than measured, when
# the gap contained no transcript words to average. Lets callers spot soft timings.
INTERPOLATED_CONFIDENCE = 0.0

_NON_WORD = re.compile(r"[^a-z0-9]+")


class AlignmentError(RuntimeError):
    """Deepgram refused, or returned a body without a transcript."""


class DeepgramAligner:
    """Satisfies `Aligner`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        attempts: int = 3,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.deepgram_api_key
        self.model = model or settings.video_default_aligner_model
        self.timeout = timeout
        self.attempts = attempts

    def align(self, audio_path: Path, reference_text: str) -> list[Word]:
        reference_tokens = tokenize(reference_text)
        if not reference_tokens:
            return []

        stt_words, total_duration = self.transcribe(Path(audio_path))
        return align_tokens(reference_tokens, stt_words, total_duration)

    def transcribe(self, audio_path: Path) -> tuple[list[Word], float]:
        """Raw transcript words and `metadata.duration`. Exposed for diagnostics."""
        if not self.api_key:
            raise AlignmentError("deepgram_api_key is empty — set DEEPGRAM_API_KEY in .env")
        audio = Path(audio_path).read_bytes()

        params = {"model": self.model, "smart_format": "true", "punctuate": "true"}
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"}

        last_error = ""
        for attempt in range(1, self.attempts + 1):
            try:
                response = httpx.post(
                    LISTEN_URL,
                    params=params,
                    headers=headers,
                    content=audio,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == self.attempts:
                    raise AlignmentError(last_error) from exc
                continue

            if response.status_code == 200:
                return _parse_listen(response.json())

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code in RETRY_STATUS and attempt < self.attempts:
                continue
            raise AlignmentError(last_error)

        raise AlignmentError(f"exhausted {self.attempts} attempts — {last_error}")


# --------------------------------------------------------------------------- pure core


def normalize(token: str) -> str:
    """Comparison key: lowercase, alphanumerics only.

    Dropping apostrophes and hyphens makes "don't"/"dont" and "world-class"/"worldclass"
    compare equal, which is the common shape of transcript drift.
    """
    return _NON_WORD.sub("", token.lower())


def tokenize(reference_text: str) -> list[str]:
    """Whitespace tokens of the script, minus anything with no letters or digits.

    Tokens keep their original punctuation and casing — that is the whole point; they
    become the on-screen text.
    """
    return [tok for tok in (reference_text or "").split() if normalize(tok)]


def _parse_listen(payload: dict) -> tuple[list[Word], float]:
    channels = (payload.get("results") or {}).get("channels") or []
    if not channels:
        raise AlignmentError("listen response had no channels")
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        raise AlignmentError("listen response had no alternatives")

    raw_words = alternatives[0].get("words") or []
    words = [
        Word(
            word=str(w.get("word", "")),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
            confidence=float(w.get("confidence", 1.0)),
            punctuated_word=w.get("punctuated_word"),
        )
        for w in raw_words
    ]
    duration = float((payload.get("metadata") or {}).get("duration") or 0.0)
    if duration <= 0.0 and words:
        duration = words[-1].end
    return words, duration


def align_tokens(
    reference_tokens: list[str], stt_words: list[Word], total_duration: float
) -> list[Word]:
    """Carry transcript timings onto reference tokens. Pure — no network, no ffmpeg.

    Returns one `Word` per reference token, in order, with non-decreasing timings.
    `word` holds the normalised form and `punctuated_word` the original token, so
    `Word.display` yields the script's own punctuation.
    """
    if not reference_tokens:
        return []

    span_end = total_duration if total_duration > 0 else (stt_words[-1].end if stt_words else 0.0)

    if not stt_words:
        # Nothing was heard. Spread the script across the clip so captions still advance.
        return _interpolated_run(
            reference_tokens, 0.0, span_end, confidence=INTERPOLATED_CONFIDENCE
        )

    reference_norm = [normalize(t) for t in reference_tokens]
    stt_norm = [normalize(w.word) for w in stt_words]

    # autojunk would classify common words as noise on scripts over 200 tokens and
    # silently degrade the pairing; it must stay off for alignment work.
    matcher = difflib.SequenceMatcher(a=reference_norm, b=stt_norm, autojunk=False)

    anchors: list[tuple[float, float, float] | None] = [None] * len(reference_tokens)
    gaps: list[tuple[int, int, int, int]] = []  # ref_lo, ref_hi, stt_lo, stt_hi

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                source = stt_words[j1 + offset]
                anchors[i1 + offset] = (source.start, source.end, source.confidence)
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            # Same count, different words: the transcript misheard but the clock is fine.
            for offset in range(i2 - i1):
                source = stt_words[j1 + offset]
                anchors[i1 + offset] = (source.start, source.end, source.confidence)
        elif tag in ("replace", "delete"):
            gaps.append((i1, i2, j1, j2))
        # "insert" means the transcript invented words; reference has nothing to time.

    words: list[Word] = [
        _make_word(reference_tokens[i], *anchors[i])  # type: ignore[misc]
        if anchors[i] is not None
        else _make_word(reference_tokens[i], 0.0, 0.0, INTERPOLATED_CONFIDENCE)
        for i in range(len(reference_tokens))
    ]

    for ref_lo, ref_hi, stt_lo, stt_hi in gaps:
        lower, upper = _gap_bounds(anchors, stt_words, ref_lo, ref_hi, stt_lo, stt_hi, span_end)
        confidence = _gap_confidence(stt_words, stt_lo, stt_hi)
        filled = _interpolated_run(
            reference_tokens[ref_lo:ref_hi], lower, upper, confidence=confidence
        )
        words[ref_lo:ref_hi] = filled

    return _enforce_monotonic(words, span_end)


def _make_word(token: str, start: float, end: float, confidence: float) -> Word:
    return Word(
        word=normalize(token),
        start=round(start, 3),
        end=round(end, 3),
        confidence=round(confidence, 4),
        punctuated_word=token,
    )


def _gap_bounds(
    anchors: list[tuple[float, float, float] | None],
    stt_words: list[Word],
    ref_lo: int,
    ref_hi: int,
    stt_lo: int,
    stt_hi: int,
    span_end: float,
) -> tuple[float, float]:
    """Time window the unmatched reference tokens must fit inside.

    Prefers the transcript words that fell in the same gap — they are measured audio.
    Falls back to the surrounding matched anchors, then to the clip boundaries.
    """
    if stt_hi > stt_lo:
        return stt_words[stt_lo].start, stt_words[stt_hi - 1].end

    previous = next((anchors[i] for i in range(ref_lo - 1, -1, -1) if anchors[i]), None)
    following = next(
        (anchors[i] for i in range(ref_hi, len(anchors)) if anchors[i]), None
    )
    lower = previous[1] if previous else 0.0
    upper = following[0] if following else span_end
    return lower, max(upper, lower)


def _gap_confidence(stt_words: list[Word], stt_lo: int, stt_hi: int) -> float:
    if stt_hi <= stt_lo:
        return INTERPOLATED_CONFIDENCE
    window = stt_words[stt_lo:stt_hi]
    return sum(w.confidence for w in window) / len(window)


def _interpolated_run(
    tokens: list[str], start: float, end: float, *, confidence: float
) -> list[Word]:
    """Distribute [start, end] across tokens proportionally to their length.

    Character count is a crude but effective proxy for how long a word takes to say —
    better than an even split, which drifts on runs mixing "a" with "infrastructure".
    """
    if not tokens:
        return []
    weights = [max(1, len(normalize(t))) for t in tokens]
    total = sum(weights)
    span = max(0.0, end - start)

    result: list[Word] = []
    cursor = start
    for token, weight in zip(tokens, weights, strict=True):
        slice_len = span * (weight / total)
        result.append(_make_word(token, cursor, cursor + slice_len, confidence))
        cursor += slice_len
    return result


def _enforce_monotonic(words: list[Word], span_end: float) -> list[Word]:
    """Clamp so captions never move backwards.

    Overlapping or inverted timings survive the pairing step when the transcript itself
    reports them; a caption renderer fed a negative duration draws nothing.
    """
    cursor = 0.0
    fixed: list[Word] = []
    for word in words:
        start = max(word.start, cursor)
        end = max(word.end, start)
        if span_end > 0:
            start = min(start, span_end)
            end = min(max(end, start), span_end)
        fixed.append(
            word.model_copy(update={"start": round(start, 3), "end": round(end, 3)})
        )
        cursor = end
    return fixed
