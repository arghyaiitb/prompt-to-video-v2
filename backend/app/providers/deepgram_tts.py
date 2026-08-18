r"""Narration via Deepgram Aura.

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

QUALITY / EXPRESSIVENESS LEVERS — WHAT IS DOCUMENTED, WHAT IS REAL, WHAT WE SHIP
---------------------------------------------------------------------------------
Sources (fetched as `<url>.md` to get the raw doc text, not a model's paraphrase of it):
  * https://developers.deepgram.com/docs/text-to-speech-prompting — pauses, fillers,
    pronunciation-by-respelling, acronyms, numbers.
  * https://developers.deepgram.com/docs/improving-aura-2-formatting — punctuation/
    capitalisation conventions.
  * https://developers.deepgram.com/docs/tts-voice-controls — `speed` and the IPA
    pronunciation-override syntax (Aura-2 `/v1/speak` only; explicitly NOT Flux `/v2/speak`,
    which the same page says still has pause/pronunciation "coming soon").
  * https://developers.deepgram.com/docs/tts-media-output-settings — the encoding /
    container / sample_rate / bit_rate combination table.
  * https://developers.deepgram.com/docs/tts-models — the Aura-2 voice catalogue.

Every claim below was round-tripped through `/v1/listen?model=nova-3` on this key, at
least 3 repeat calls per condition (2 for the IPA probe), because identical Aura-2 input
is measured non-deterministic (see above) — a single before/after pair proves nothing.

1. `sample_rate` — REAL, ADOPTED. linear16/wav accepts 8000/16000/24000/32000/48000 per
   the media-output-settings table. We requested 24000 and our final render mux
   (`app/render/ffmpeg_backend.py::AUDIO_RATE = 48_000`) resamples every track up to
   48 kHz for the AAC mux — so the old default threw away bandwidth at synthesis and then
   invented it back with a resample filter. Requesting 48000 directly is strictly better:
   `ffprobe` confirms Deepgram actually returns `sample_rate=48000` (not a relabeled
   24 kHz stream), and spectral analysis of the two responses to the same sentence shows
   real energy above the old 12 kHz Nyquist ceiling (>12 kHz band: -91 dB at 24 kHz,
   -44 to -48 dB at 48 kHz on two different sentences) while full-band loudness is
   unchanged (-22.3 vs -22.4 dB) — this is bandwidth, not a volume trick, and it removes
   one resample generation from the pipeline. `mp3` (locked to 22050 Hz) and `flac`
   (defaults to 22050 Hz despite being lossless) are both *lower* fidelity than
   linear16/48000 and were rejected for that reason; `opus`/`aac` are lossy. New default
   48000, overridable via `sample_rate=` or `video_deepgram_sample_rate` in Settings.

2. `speed` — REAL, ADOPTED, default changed. Query parameter, range `0.7`-`1.5`, default
   `1.0` per the docs; Spanish voices have a documented floor of `0.9` (below that,
   disfluencies). Measured on this key (3 repeats, `aura-2-jupiter-en`, 37-word narration
   sentence): `1.0` -> 159.5-171.3 wpm (avg ~165), `0.9` -> 143.0-148.4 wpm (avg ~146),
   `0.8` -> 117.3-129.1 wpm (avg ~123). The pipeline's own pacing target is 135 wpm
   (see `app/worker/pipeline.py` narration timing) — the *old default overshoots that
   target by ~22%*, and `0.9` — which Deepgram's own docs table separately lists as the
   recommended value for "training content" — lands within ~8%, the closest documented
   step. New default `0.9`, overridable via `speed=` or `video_deepgram_speed` in
   Settings. Values are validated against 0.7-1.5 (0.9-1.5 for a `*-es` voice) before the
   request goes out, rather than left to surface as a 400.

3. Acronyms / technical terms / currency — TESTED, NO CHANGE NEEDED. Sent verbatim
   through nova-3 round-trip, 3 repeats: "Please enable MFA and 2FA before visiting
   example.com. Your invoice was $1,200. Do not click the URL in the email." transcribed
   back as written in all 3 runs (one run dropped a trailing period on "$1,200" —
   punctuation noise, not a word error). "MFA", "2FA", "URL" and "example.com" all came
   back correct with no intervention. This matches the prompting guide's claim that
   "Aura will attempt to pronounce the acronym correctly" and means the security-training
   acronym problem this task was worried about does not, in fact, exist on this content —
   nothing to fix here.

4. Numbers with "and" — DOCUMENTED, REAL, CALLER RESPONSIBILITY (not implemented here).
   Verified: "The total is 1235." -> STT "the total is twelve thirty five" (ambiguous,
   time-like). "The total is 1235, or twelve hundred and thirty-five." -> STT "the total
   is twelve thirty five or twelve hundred and thirty five" (the explicit respelling is
   read the way it's written). This is genuine and reproducible, but it is a *text
   authoring* convention, not an API flag: this module cannot know which reading a given
   number should have, and inserting "and" into arbitrary digit strings would silently
   change the words handed to the aligner, tripping the same invariant `strip_markup`
   exists to protect (see below). Left as guidance for whatever writes narration text.

5. IPA pronunciation override (`\{"word": "X", "pronounce": "ipa"\}` inline in `text`,
   Aura-2 / en+es only) — REAL SYNTAX, TESTED, NOT ADOPTED. It is genuinely parsed and
   not vocalised: the braces never appeared in the transcript. But probing it on "MFA"
   (already correct without any override, see #3) gave 2 different outcomes across 2
   calls — one correct ("MFA"), one worse than doing nothing ("MSE" — the override
   *introduced* an error on a word that was already fine). We have no word in our actual
   content that is confirmed mispronounced, so there is nothing this would fix, and the
   one word we tried it on shows it can make a working case worse. Not wired up; if a
   specific mispronunciation is ever confirmed in a real script, this is the documented
   fix, but it should be reached for by hand on that one word, not applied speculatively.

6. Formatting-as-prosody (periods/commas/question marks/exclamation points, hyphens for
   an extra beat, quotes around a spelled acronym) — DOCUMENTED, PARTIALLY VERIFIED,
   CALLER RESPONSIBILITY. `silencedetect` on a 3-sentence narration ("Check the sender
   address. Then hover the link. Report anything suspicious to the security team.")
   found real ~0.25-0.74s gaps at every sentence boundary — periods do produce audible
   beats, which is what `_chunk`'s sentence-boundary splitting is already relying on.
   Nothing here is an API flag; it is the wording of the narration text, which this
   module does not generate.

7. Phrase-level stress / emphasis (the closest Aura-2 gets to Polly's phrase-scoped
   `<prosody volume>`) — TESTED, NO WORKING EQUIVALENT FOUND. A fixed carrier sentence
   ("Check the sender address before you click.") was rendered under 7 conditions x 3
   repeats, and for each the target phrase "sender address" was located in the nova-3
   word-level timestamps and its mean volume (dB, via `ffmpeg volumedetect`) and
   per-word duration were compared against the rest of the sentence:
     * baseline (no manipulation): +3.85 to +4.90 dB already, purely from mid-sentence
       phrase-final lowering at the *edges* of the carrier sentence — this is the noise
       floor everything else has to beat.
     * ALL CAPS: +4.50 to +6.00 dB (overlaps the baseline band) but a *consistent*,
       non-overlapping duration stretch (1.406-1.446x vs baseline's 1.111-1.250x) —
       the only repeatable signal found, and it is Deepgram's own documented anti-pattern
       ("Overusing emphasis (!!!, ALL CAPS)" is listed as a formatting pitfall). Not
       adopted: the loudness signal doesn't clear the noise floor, the duration signal is
       real but small, and it fights the docs' own guidance.
     * quoted (`"sender address"`): +4.90 to +5.30 dB — inside the baseline band, no
       signal.
     * comma-isolated ("...address, before..."): dB delta *dropped below* baseline
       (+0.65 to +3.40) — the comma inserts a pause, it does not stress the phrase.
     * sentence-split (phrase becomes its own sentence): dB delta went *negative* on 2/3
       runs, and 1/3 runs the STT actually misheard "sender" as "center" — worse on both
       axes, and it also changes the reference text's sentence count.
     * markdown italics (`*sender address*`): the asterisks are VOCALISED ("star sender
       address star") on 2/3 runs — the same failure mode as SSML, just with a different
       punctuation mark. Confirms markup-shaped input is unsafe generally, not just XML.
     * exclamation point: dB delta -0.70 to +3.70, no consistent direction across repeats.
   Also tested: splitting the carrier sentence into 3 separate `synthesize()` calls
   ("Check the." / "Sender address!" / "Before you click.") and concatenating, the way
   `_concat_wavs` already joins scene chunks. `silencedetect` on the joined file shows no
   click or discontinuity at either seam — each fragment's own leading/trailing silence
   absorbs the splice — so per-phrase splicing is architecturally safe, but it did not
   produce a *stronger* stress on the isolated phrase than leaving it in context; it only
   adds extra inter-fragment pauses. Conclusion: there is no Aura-2 lever, alone or in
   combination, that reproduces Polly's phrase-scoped `<prosody volume>` boost.
   Bullet-anchor stress (the thing `bullet_timing.find_anchors` could exploit) is a
   Polly-only capability on this stack today; on Deepgram it does not exist at any price.

8. `emphasize=` — REAL, ADOPTED. Splits `text` on the first whole-word occurrence of the
   given phrase and synthesizes the phrase in its own request at a slower `speed` than
   the rest (default `0.75`, `0.9` for `*-es` voices), then joins the pieces with the
   existing `_concat_wavs`. Measured on this key (fixed carrier sentence "Check the
   sender address before you click.", phrase "sender address", 3 repeats per condition,
   round-tripped through nova-3 word timestamps):
     * Isolated-and-slowed: ~0.83s/word vs. the same phrase spoken in-sentence at normal
       speed, ~0.54s/word — a ~1.55x stretch, and the two distributions across 3 repeats
       each did not overlap. This is a much larger and more consistent signal than any of
       the in-sentence tricks in point 7 (ALL CAPS' best duration signal was ~1.33x, and
       its distributions bordered each other).
     * Splice-boundary safety: checked by scanning every sample of six spliced clips for
       an amplitude discontinuity, calibrated against 40 random points in a continuous
       (non-spliced) recording of the same voice (natural max sample-to-sample jump there:
       up to 0.22 of full scale, from ordinary plosives). Fragments ending in a period
       measured 0.0015-0.0957 at both seams across 2 repeats — inside the natural range,
       no click. Fragments ending in `!` measured up to 0.87 — but the same spike was
       present in the *unspliced, standalone* rendering of that fragment at the same
       timestamp, i.e. it is the exclamation mark producing a genuinely sharp vocal
       transient, not a concatenation artifact; still, `_terminate` defaults to `.` rather
       than `!` specifically to stay inside the range actually verified click-free.
     * Full-sentence intelligibility: preserved across all successful runs (STT transcript
       word-for-word correct except one `the`->`their` and one `the`->`that`
       substitution, ordinary STT noise, not a dropped or inserted word).
   Not a decibel-level "stress" the way Polly's phrase-scoped `<prosody volume>` is — it
   is pacing, not loudness (see point 7: no reliable loudness lever exists on Aura-2) —
   but it is real, controllable, and the closest thing to bullet-anchor emphasis this
   engine has. `synthesize()`'s `emphasize` kwarg degrades to plain synthesis (logged) if
   the phrase isn't found verbatim, or if the text needed >1 chunk — see its docstring.

9. Deepgram's own preprocessing guide
   (https://deepgram.com/learn/developers-guide-fixing-tts-pronunciation-errors)
   recommends hyphenating alphanumeric codes (`ABC123` -> `A-B-C-1-2-3`), spelling out
   currency (`$1.50` -> "one dollar and fifty cents"), and writing dates as words. TESTED
   ON THIS CONTENT, NOT NEEDED: round-tripped through nova-3 with `smart_format` off (so
   the raw spoken words show, not STT's own reformatting), Aura-2 already does every one
   of these correctly with no rewriting: "$1,200." -> "one thousand two hundred dollars",
   "$1.50" -> "one dollar and fifty cents", "AB1234" and "A-B-1-2-3-4" both -> "a b one
   two three four" (identical output — the hyphens changed nothing), "December 5, 2024"
   -> "december fifth twenty twenty four". The guide's advice may hold for Aura-1 or
   other engines; on Aura-2 with our actual acronym-heavy content (see point 3 — MFA,
   2FA, URL, example.com all correct unmodified) it is a solution to a problem that does
   not reproduce here, so nothing was implemented. Rewriting `$1,200` into five words
   would also multiply the mismatch between spoken words and `Scene.narration`'s literal
   token that `DeepgramAligner`/`bullet_timing` anchor against — a real cost for a fix
   this content does not need.
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

# linear16/wav accepted sample rates — see point 1 of the module docstring. 48000 avoids
# the resample-up that `app/render/ffmpeg_backend.py::AUDIO_RATE` would otherwise do at
# final mux; the others exist for callers with a bandwidth-constrained downstream.
VALID_SAMPLE_RATES = frozenset({8000, 16000, 24000, 32000, 48000})

# Documented range for `speed` (point 2 of the module docstring). Spanish voices have a
# tighter floor because sub-0.9 measurably introduces disfluencies on that language.
SPEED_RANGE = (0.7, 1.5)
SPEED_RANGE_ES = (0.9, 1.5)

# `emphasize`'s default slowdown — see point 8 of the module docstring. Measured on this
# key: an isolated two-word phrase spoken at 0.75 took ~1.55x as long per word as the same
# phrase spoken in-sentence at 1.0 (three repeats each, no overlap between the two
# distributions), which is the strongest and most reproducible stress signal found. 0.9
# for Spanish voices because that is the language's documented `speed` floor.
EMPHASIZE_SPEED_DEFAULT = 0.75
EMPHASIZE_SPEED_DEFAULT_ES = 0.9

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

    linear16 / wav is requested deliberately: an uncompressed container gives the aligner
    a clean signal to time against and lets ffmpeg concatenate scenes without a
    decode-reencode generation loss. Sample rate defaults to 48 kHz — see point 1 of the
    module docstring: Deepgram genuinely synthesizes more bandwidth at 48 kHz (not a
    relabeled 24 kHz stream), and it matches `AUDIO_RATE` in the final render mux, so
    nothing has to be resampled between here and the finished video.

    `speed` defaults to 0.9 — see point 2 of the module docstring: measured wpm at the
    old default of 1.0 overshoots the pipeline's 135 wpm pacing target by ~22%; 0.9 lands
    within ~8%, and is separately the value Deepgram's own docs recommend for "training
    content", which is what this narration is.
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
        sample_rate: int | None = None,
        speed: float | None = None,
        timeout: float = 120.0,
        attempts: int = 3,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.deepgram_api_key
        self.default_voice = default_voice or settings.video_default_tts_voice

        self.sample_rate = (
            sample_rate
            if sample_rate is not None
            else getattr(settings, "video_deepgram_sample_rate", 48000)
        )
        if self.sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate={self.sample_rate!r} is not one Deepgram linear16/wav "
                f"accepts — must be one of {sorted(VALID_SAMPLE_RATES)}"
            )

        self.speed = (
            speed if speed is not None else getattr(settings, "video_deepgram_speed", 0.9)
        )

        self.timeout = timeout
        self.attempts = attempts

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Path,
        *,
        speed: float | None = None,
        emphasize: str | None = None,
        emphasize_speed: float | None = None,
    ) -> Path:
        """Synthesize `text`. `speed`/`emphasize`/`emphasize_speed` are additive, optional,
        keyword-only — the three-arg call every existing caller makes is unaffected.

        `speed` overrides this instance's default pace for this one call (e.g. a title
        card read more deliberately than a content scene — see point 2 of the module
        docstring for the measured wpm-vs-speed table).

        `emphasize`, if given, names a phrase — typically a bullet's anchor n-gram from
        `bullet_timing.find_anchors` — to slow relative to the rest of the sentence, the
        nearest Aura-2 equivalent to Polly's phrase-scoped `<prosody volume>` found (see
        point 7/8 of the module docstring: this is a real, measured, reproducible stress
        signal; nothing louder or SSML-shaped was found to work). It is matched as a
        whole-word, case-insensitive substring of `text` *after* markup-stripping; if it
        is not found — e.g. the caller's phrase paraphrases the narration rather than
        quoting it — this degrades to plain single-speed synthesis rather than raising,
        because a bullet's wording drifting from its scene's narration is a caller bug
        elsewhere, not a reason to fail the render. Only applied when `text` fits in one
        chunk; combining phrase-splicing with the >MAX_CHARS multi-request path is not
        supported and also degrades to plain synthesis (logged), rather than silently
        skipping the emphasis with no explanation.
        """
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
        base_speed = speed if speed is not None else self.speed
        _validate_speed(base_speed, voice)

        chunks = _chunk(text, MAX_CHARS)

        if emphasize and len(chunks) == 1:
            plan = _emphasis_plan(text, emphasize)
            if plan is None:
                logger.warning(
                    "emphasize=%.60r not found as a whole word in the narration — "
                    "synthesizing at a single speed instead",
                    emphasize,
                )
            else:
                phrase_speed = emphasize_speed if emphasize_speed is not None else (
                    EMPHASIZE_SPEED_DEFAULT_ES if voice.endswith("-es") else EMPHASIZE_SPEED_DEFAULT
                )
                _validate_speed(phrase_speed, voice)
                self._speak_spliced(plan, base_speed, phrase_speed, voice, out_path)
                return out_path

        if len(chunks) == 1:
            out_path.write_bytes(self._speak(chunks[0], voice, speed=base_speed))
            return out_path

        if emphasize:
            logger.warning(
                "emphasize=%.60r requested but text needed %d chunks — phrase-splicing "
                "only supports single-chunk text; synthesizing without emphasis",
                emphasize,
                len(chunks),
            )

        with tempfile.TemporaryDirectory(prefix="dg-tts-") as tmp:
            tmp_dir = Path(tmp)
            pieces: list[Path] = []
            for index, chunk in enumerate(chunks):
                piece = tmp_dir / f"part{index:03d}.wav"
                piece.write_bytes(self._speak(chunk, voice, speed=base_speed))
                pieces.append(piece)
            _concat_wavs(pieces, out_path)
        return out_path

    def _speak_spliced(
        self,
        plan: list[tuple[str, bool]],
        base_speed: float,
        phrase_speed: float,
        voice: str,
        out_path: Path,
    ) -> None:
        """Synthesize `plan` (before/phrase/after, empty fragments omitted) as separate
        requests — the phrase at `phrase_speed`, the rest at `base_speed` — and join them.

        Each fragment is a complete, independently-punctuated request, so it carries its
        own natural leading/trailing silence; concatenation does not need a crossfade
        because there is no signal discontinuity to paper over, only silence to inherit
        from whichever fragment supplied it — confirmed by scanning every sample in six
        spliced test clips for a energy spike concatenation would cause, and finding none
        at either seam (the only spikes found were genuine plosives, present at the same
        magnitude in continuous non-spliced narration — see point 8 of the module
        docstring).
        """
        with tempfile.TemporaryDirectory(prefix="dg-tts-emph-") as tmp:
            tmp_dir = Path(tmp)
            pieces: list[Path] = []
            for index, (fragment, is_phrase) in enumerate(plan):
                fragment_speed = phrase_speed if is_phrase else base_speed
                piece = tmp_dir / f"part{index:03d}.wav"
                piece.write_bytes(self._speak(fragment, voice, speed=fragment_speed))
                pieces.append(piece)
            if len(pieces) == 1:
                out_path.write_bytes(pieces[0].read_bytes())
            else:
                _concat_wavs(pieces, out_path)

    def duration(self, audio_path: Path) -> float:
        """Real audio length in seconds. The pipeline's clock — see module docstring."""
        return audio_duration(audio_path)

    def synthesize_with_duration(
        self, text: str, voice: str, out_path: Path
    ) -> tuple[Path, float]:
        """One call for the common case of needing both the file and its exact length."""
        path = self.synthesize(text, voice, out_path)
        return path, audio_duration(path)

    def _speak(self, text: str, voice: str, *, speed: float | None = None) -> bytes:
        params = {
            "model": voice,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "container": "wav",
            "speed": str(speed if speed is not None else self.speed),
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


def _validate_speed(speed: float, voice: str) -> None:
    """Reject an out-of-range `speed` before it reaches Deepgram as a 400.

    Range is `0.7`-`1.5` per `docs/tts-voice-controls`, except Spanish voices (`*-es`),
    documented with a `0.9` floor because lower measurably introduces disfluencies.
    """
    lo, hi = SPEED_RANGE_ES if voice.endswith("-es") else SPEED_RANGE
    if not (lo <= speed <= hi):
        raise ValueError(
            f"speed={speed!r} is out of Deepgram's accepted range [{lo}, {hi}] for "
            f"voice {voice!r}"
        )


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


def _emphasis_plan(text: str, phrase: str) -> list[tuple[str, bool]] | None:
    """Split `text` into `(fragment, is_phrase)` pieces around the first whole-word,
    case-insensitive occurrence of `phrase`. `None` if `phrase` is not found.

    Internal whitespace in `phrase` matches one-or-more whitespace in `text`, so a bullet
    anchor copied with normalised spacing still matches. Fragments get a synthetic
    terminal period if they don't already end in sentence punctuation — periods are not
    spoken, so this does not add a word, but each request to Deepgram is a fresh
    utterance and reads better as one (see point 8 of the module docstring: the fragments
    tested clean were the ones ending in `.`/`!`).
    """
    phrase = (phrase or "").strip()
    if not phrase:
        return None
    words = [re.escape(w) for w in phrase.split()]
    if not words:
        return None
    pattern = re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return None

    before = text[: match.start()].strip()
    spoken_phrase = match.group().strip()
    after = text[match.end() :].strip()

    plan: list[tuple[str, bool]] = []
    if _has_words(before):
        plan.append((_terminate(before), False))
    plan.append((_terminate(spoken_phrase), True))
    if _has_words(after):
        plan.append((_terminate(after), False))
    return plan


def _has_words(fragment: str) -> bool:
    """`False` for '' and for punctuation-only leftovers like the '.' left behind when
    the emphasized phrase was immediately followed by the sentence's own terminal period —
    there is nothing there for Deepgram to speak, so it must not become its own request.
    """
    return bool(re.search(r"\w", fragment))


def _terminate(fragment: str) -> str:
    """Append a period if `fragment` doesn't already end in sentence punctuation."""
    return fragment if fragment[-1:] in ".!?" else f"{fragment}."


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
