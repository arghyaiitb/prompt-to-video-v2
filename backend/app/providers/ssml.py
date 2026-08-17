"""SSML markup for narration that sounds *performed* rather than read.

Plain text hands the engine no idea where a thought ends, which phrase the slide is
about, or that "MFA" is three letters and not a word. This module adds that information
back — breaks at sentence boundaries, a longer beat before an instruction, stress on the
exact phrase a bullet echoes, a role-appropriate rate, and spelled-out readings for the
acronyms a phishing-training script is full of.

THE INVARIANT (read this before changing anything here)
-------------------------------------------------------
Bullets are anchored to REAL word timings: narration is synthesised, `DeepgramAligner`
aligns the audio against the **plain reference narration**, and
`app.providers.bullet_timing` matches each bullet to a verbatim n-gram in that text. Two
rules fall out, and both are enforced in code rather than by convention:

1. The aligner is *never* given SSML. `Scene.narration` stays the plain-text truth and is
   what `align()` and the bullet matcher see. Use :func:`text_for_synthesis` at the TTS
   boundary so the choice is made once, from `SpeechSynthesizer.supports_ssml`, instead of
   at every call site.
2. SSML may add pauses, stress, rate and pronunciation, but it may **not change the
   words**. :func:`build_ssml` round-trips its own output through :func:`strip_ssml` and
   compares against `deepgram_align.tokenize` — the aligner's own tokenizer — and raises
   :class:`SsmlInvariantError` if a single token moved. A silent rewrite here would drift
   every bullet anchor in the video, which is invisible in review and obvious on screen.

`<sub>` is therefore banned outright (it substitutes spoken text for written text, i.e. it
breaks rule 2 by design) and :func:`validate_ssml` reports it as an error on every engine.

`<say-as interpret-as="characters">` deserves a note, because it *does* change what the
STT pass hears: "MFA" comes back as three tokens. That is survivable — `align_tokens`
treats the mismatch as a gap and bounds the interpolation by the span of the transcript
words inside it, so the reference token "MFA" still gets the start of "M" and the end of
"A". Timing stays correct; only the reported confidence softens. Verified in
`tests/test_ssml.py::test_spelled_acronym_keeps_anchor_timing`.

ENGINE SUPPORT IS NOT UNIFORM
-----------------------------
Deepgram Aura does not parse SSML — it *vocalises* the tags, reading "break time equals
eight hundred milliseconds" aloud (measured). So SSML is strictly opt-in per engine, and
:func:`sanitise_for_plain_tts` is the belt-and-braces guard for anything that reaches a
plain-text engine anyway. Polly's supported tag set additionally varies by engine tier;
see :data:`CAPABILITIES`.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.core.models import SceneRole, Word
from app.providers.bullet_timing import find_anchors
from app.providers.deepgram_align import tokenize


class SsmlInvariantError(RuntimeError):
    """Generated SSML would change the narration's words.

    A hard error by design. The alternative — logging a warning and shipping — silently
    decouples the on-screen bullets from the audio, and nothing downstream can detect it.
    """


# --------------------------------------------------------------------------- capabilities


@dataclass(frozen=True)
class Capability:
    """What one engine (or Polly engine tier) will actually accept.

    Sourced from AWS's own per-tag pages and then MEASURED against a live Polly account
    (see `tests/test_ssml.py` and the module docstring). Where the two disagreed the
    measurement won; where AWS documents a behaviour we cannot detect from audio alone
    (the neural `say-as characters` voice fallback) the doc is followed conservatively.
    """

    engine: str
    supports_ssml: bool
    tags: frozenset[str]
    """Element names accepted inside `<speak>`. Empty when the engine takes no SSML."""

    prosody_attrs: frozenset[str] = frozenset()
    """`<prosody>` attributes that MEASURABLY change the audio on this engine.

    Not the same as "attributes the API accepts". Polly's generative tier accepts
    `<prosody rate>` and returns byte-identical audio for anything from 88% to 100%, then
    snaps 85%/80%/`slow` to a single 1.25x step — so `rate` is listed as unsupported there,
    because emitting it would ship either a no-op or a jarring drawl with nothing in
    between. See the measurements quoted in :data:`CAPABILITIES`.
    """

    say_as: frozenset[str] = frozenset()
    effect_names: frozenset[str] = frozenset()
    max_break_seconds: float = 10.0
    notes: str = ""

    def emphasis_style(self) -> str:
        """How this engine can stress a phrase.

        ``"tag"``      `<emphasis>` — standard voices only; every other tier answers
                       `InvalidSsmlException: Unsupported <tier> feature`.
        ``"prosody"``  `<prosody>` around the phrase — neural, long-form and generative.
        ``"none"``     no mechanism at all.
        """
        if "emphasis" in self.tags:
            return "tag"
        if "prosody" in self.tags and {"volume", "rate"} & self.prosody_attrs:
            return "prosody"
        return "none"

    def stress_attrs(self) -> str:
        """`<prosody>` attributes for a stressed phrase, using only what this tier honours.

        Volume is the load-bearing one: it is honoured on every prosody-capable tier and is
        what makes a phrase *land*. Rate is added where it is continuous, because a phrase
        that is both slightly louder and slightly slower reads as deliberate rather than
        merely loud.
        """
        parts = []
        if "volume" in self.prosody_attrs:
            parts.append(f'volume="{STRESS_VOLUME}"')
        if "rate" in self.prosody_attrs:
            parts.append(f'rate="{STRESS_RATE}"')
        return " ".join(parts)


_POLLY_SAY_AS = frozenset(
    """characters spell-out cardinal number ordinal digits fraction unit date time
    address expletive telephone""".split()
)

# Neural accepts the <say-as> element but NOT characters/spell-out: AWS documents that a
# neural voice meeting `characters` "will be synthesized using the related standard
# voice" for the affected SENTENCE. That is a silent, mid-scene voice change, not an
# error — worse for a training video than mispronouncing "MFA", so it is withheld.
_NEURAL_SAY_AS = _POLLY_SAY_AS - {"characters", "spell-out"}

_POLLY_STANDARD_TAGS = frozenset(
    """speak break emphasis lang mark p s phoneme prosody say-as sub w
    amazon:effect amazon:auto-breaths amazon:breath""".split()
)

# Neural drops the standard-only extensions and, decisively for us, <emphasis>.
_POLLY_NEURAL_TAGS = frozenset(
    """speak break lang mark p s phoneme prosody say-as sub w amazon:effect
    amazon:domain""".split()
)

_POLLY_LONGFORM_TAGS = frozenset(
    """speak break lang mark p s phoneme prosody say-as sub w amazon:effect""".split()
)

# Generative additionally loses amazon:effect (including drc).
_POLLY_GENERATIVE_TAGS = frozenset(
    """speak break lang mark p s phoneme prosody say-as sub w""".split()
)

CAPABILITIES: dict[str, Capability] = {
    "deepgram": Capability(
        engine="deepgram",
        supports_ssml=False,
        tags=frozenset(),
        notes=(
            "Aura vocalises markup: <break time='800ms'/> is read aloud as "
            "'break time equals eight hundred milliseconds'. Send plain text only."
        ),
    ),
    "polly-standard": Capability(
        engine="polly-standard",
        supports_ssml=True,
        tags=_POLLY_STANDARD_TAGS,
        prosody_attrs=frozenset({"rate", "pitch", "volume"}),
        say_as=_POLLY_SAY_AS,
        effect_names=frozenset({"drc", "whispered"}),
        notes="The only tier with <emphasis>, <prosody pitch> and the amazon:* extensions.",
    ),
    "polly-neural": Capability(
        engine="polly-neural",
        supports_ssml=True,
        tags=_POLLY_NEURAL_TAGS,
        prosody_attrs=frozenset({"rate", "volume"}),
        say_as=_NEURAL_SAY_AS,
        effect_names=frozenset({"drc"}),
        notes=(
            "No <emphasis> (InvalidSsmlException), no <prosody pitch>, and say-as "
            "characters/spell-out silently re-synthesises the sentence with the "
            "STANDARD voice. Stress via <prosody rate/volume>."
        ),
    ),
    "polly-long-form": Capability(
        engine="polly-long-form",
        supports_ssml=True,
        tags=_POLLY_LONGFORM_TAGS,
        prosody_attrs=frozenset({"rate", "volume"}),
        say_as=_POLLY_SAY_AS,
        effect_names=frozenset({"drc"}),
        notes=(
            "No <emphasis>, no <prosody pitch>. NOTE: long-form is voice-limited "
            "(Danielle, Gregory, Ruth, Patrick...); asking for a non-long-form voice "
            "fails the whole request and looks like a tag problem."
        ),
    ),
    "polly-generative": Capability(
        engine="polly-generative",
        supports_ssml=True,
        tags=_POLLY_GENERATIVE_TAGS,
        prosody_attrs=frozenset({"volume"}),
        say_as=_POLLY_SAY_AS,
        notes=(
            "No <emphasis>, no <prosody pitch>, no amazon:*, <mark> is a no-op. "
            "<prosody rate> is QUANTISED to the point of uselessness — 88%-100% all "
            "return byte-identical audio, and 85%/80%/slow all snap to one 1.25x step — "
            "so rate is withheld. <prosody volume> IS honoured per phrase (+2.0 dB for "
            "'loud', measured), and is how stress is expressed on this tier."
        ),
    ),
    "elevenlabs": Capability(
        engine="elevenlabs",
        supports_ssml=True,
        tags=frozenset({"speak", "break", "phoneme"}),
        say_as=frozenset(),
        max_break_seconds=3.0,
        notes=(
            "Documented subset only: <break time='1.5s'/> (max 3s; NOT on eleven_v3) "
            "and <phoneme> (model-dependent). Nothing else is documented as parsed, and "
            "ElevenLabs does not document whether an unknown tag is ignored or spoken — "
            "so treat anything outside this set as unsupported."
        ),
    ),
}

_ENGINE_ALIASES = {
    "polly": "polly-neural",
    "aws": "polly-neural",
    "aws-polly": "polly-neural",
    "neural": "polly-neural",
    "standard": "polly-standard",
    "long-form": "polly-long-form",
    "longform": "polly-long-form",
    "generative": "polly-generative",
    "aura": "deepgram",
    "deepgram-aura": "deepgram",
    "eleven": "elevenlabs",
    "eleven_labs": "elevenlabs",
}

DEFAULT_ENGINE = "polly-neural"


def capability(engine: str | None) -> Capability:
    """Capability record for `engine`, tolerating the common aliases.

    An unknown engine is treated as *no SSML support* rather than as full support: the
    failure mode of withholding markup is flat narration, and the failure mode of
    guessing wrong is an engine reading tag names aloud.
    """
    key = (engine or DEFAULT_ENGINE).strip().lower()
    key = _ENGINE_ALIASES.get(key, key)
    found = CAPABILITIES.get(key)
    if found is not None:
        return found
    return Capability(
        engine=key,
        supports_ssml=False,
        tags=frozenset(),
        notes="Unknown engine — assumed plain-text only.",
    )


# --------------------------------------------------------------------------- tuning


SENTENCE_BREAK_MS = 250
"""Beat between sentences. Long enough to punctuate a thought, short enough to keep pace."""

INSTRUCTION_BREAK_MS = 600
"""Beat before the sentence that tells the viewer what to DO. This is the one pause a
viewer should actually notice."""

DOMAIN_BREAK_MS = 120
"""Weak beat before a domain, so a lookalike has a moment to register on screen."""

MIN_BREAK_MS = 100
"""Floor when breaks are scaled to fit a role's budget; below this it is not a pause."""

PAUSE_BUDGET_FRACTION = 0.10
"""Share of a role's *minimum* target duration that inserted silence may consume.

A title card gets 3-6s; 800ms of breaks is a fifth of that and reads as a stall, so the
budget is derived from the role rather than being one global constant. 10% of the floor
keeps pauses felt-but-not-waited-on: 0.30s for a title, 1.40s for a content scene.
"""

ROLE_RATE: dict[SceneRole, str] = {
    SceneRole.TITLE: "92%",
    SceneRole.CONTENT: "100%",
    SceneRole.SUMMARY: "97%",
    SceneRole.CLOSING: "95%",
}
"""Speaking rate per role. The opener and the call-to-action are the two places a slower,
more deliberate read pays off; the teaching body stays at normal pace so a 24s scene does
not become a 28s one."""

EMPHASIS_LEVEL = "moderate"
"""`<emphasis>` level for standard voices. "strong" over-sells a training script."""

STRESS_VOLUME = "loud"
STRESS_RATE = "95%"
"""Prosody values standing in for `<emphasis>` wherever that tag is rejected.

Chosen from measurement on a live Polly account, not from taste. Against the phrase's own
sentence as the control, `volume="loud" rate="95%"` on the anchored phrase moves the
phrase-versus-context loudness gap from +3.0 dB to +4.8 dB and stretches the phrase from
0.72s to 0.80s on a neural voice — i.e. slightly louder AND slightly slower, on exactly the
words the bullet is about. `x-loud` reaches +7.5 dB, which oversells a training script.

Anything gentler is not worth emitting: 98% rate is a 1.9% change no listener resolves.
"""

EMPHASIS_TOKEN_FRACTION = 0.40
"""Ceiling on the share of a scene's tokens that may be emphasised. Stress everything and
nothing is stressed — the flat read we are trying to fix, with extra tags."""

MAX_EMPHASIS_SPAN = 6
"""Tokens. A whole-sentence "emphasis" is just a louder sentence."""

BREAK_STRENGTH_SECONDS = {
    "none": 0.0,
    "x-weak": 0.05,
    "weak": 0.1,
    "medium": 0.3,
    "strong": 0.5,
    "x-strong": 0.8,
}
"""Approximate durations for `strength`-style breaks, for estimation only. Polly does not
document exact values; these are used to *budget*, never to place audio."""

ACRONYMS = frozenset(
    """URL MFA 2FA SSO VPN DNS IT HR CEO CFO CIO CISO CTO PDF OTP SMS API IP TLS SSL
    HTTPS HTTP PII QR ATM DM AI IOC SPF DKIM DMARC EDR MDM SIEM""".split()
)
"""Initialisms a training script uses constantly and that TTS reads badly by default
("MFA" becomes a mumbled word). Spelled out via `<say-as interpret-as="characters">`.

Deliberately excludes acronyms that are *pronounced* as words — PIN, SIM, SPAM, CAPTCHA —
because spelling those out would make the audio worse, not better.
"""

_CAPS_NOT_ACRONYM = frozenset(
    """NEVER ALWAYS ALL NOT DO DONT STOP MUST ONLY NOW NO YES AND OR THE IF IS BE WE
    YOU THIS THAT ANY EVERY REAL FAKE""".split()
)
"""All-caps words that are shouting, not initialisms. The heuristic must not spell these."""

_PRONOUNCED_AS_WORD = frozenset("PIN SIM SPAM SCAM HTML JSON SAAS NATO".split())

_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc e.g i.e approx inc ltd corp dept fig no vol
    u.s u.k a.m p.m""".split()
)
"""Tokens ending in "." that do not end a sentence. Without this, "e.g. hover the link"
gets a 250ms hole in the middle of a clause."""

IMPERATIVES = frozenset(
    """check verify confirm inspect hover report forward delete pause stop avoid never
    always call ask enable use type look read watch question think trust do treat click
    open reply send scan compare examine remember start pick choose take keep make
    dont don't and let's""".split()
)
"""Sentence-opening verbs that mark an instruction. Kept local rather than imported from
`bullet_timing`: that module's private set is tuned for picking an emphasised bullet, and
coupling the two would make a change there silently retime narration."""


# --------------------------------------------------------------------------- lexing


_TOKEN_RE = re.compile(r"\S+")

# Matches an element, a self-closing element, a closing tag, a comment or a PI. Attribute
# values are consumed as quoted runs so a ">" inside one does not end the match early.
_TAG_RE = re.compile(
    r"""<!--.*?-->            # comment
      | <\?.*?\?>             # processing instruction
      | <!\[CDATA\[.*?\]\]>   # cdata
      | <\s*/?\s*(?P<name>[A-Za-z_][\w:.-]*)
            (?:"[^"]*"|'[^']*'|[^>"'])*
        >
    """,
    re.VERBOSE | re.DOTALL,
)

# Removing these must leave a space behind: they sit *between* words, and deleting them
# outright would fuse "sender.<break/>Then" into "sender.Then" — one token where the
# aligner expects two. Inline tags are the opposite case: they hug a token's core
# ("<say-as ...>MFA</say-as>,") so they must vanish without a trace.
_SEPARATING_TAGS = frozenset({"break", "mark", "s", "p", "paragraph", "sentence"})

_WHITESPACE_RE = re.compile(r"\s+")

_CORE_RE = re.compile(r"^(?P<lead>[^\w]*)(?P<core>.*?)(?P<trail>[^\w]*)$", re.DOTALL)

_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"(?:com|net|org|io|co|gov|edu|info|biz|ru|cn|xyz|top|link|app|dev|uk|de|fr|us)"
    r"(?:/\S*)?$",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(r"^\+?\d[\d().\-]{5,}\d$")

_SENTENCE_END_RE = re.compile(r"[.!?][\"'’”)\]]*$")


@dataclass
class _Token:
    """One whitespace-delimited token of the narration, plus the markup chosen for it."""

    text: str
    lead: str
    core: str
    trail: str
    matchable: bool
    """True when the aligner would keep this token (it has a letter or digit)."""

    sentence: int = 0
    sentence_start: bool = False
    open_tags: list[str] = field(default_factory=list)
    close_tags: list[str] = field(default_factory=list)
    """Wrappers around the token's CORE, so punctuation stays outside — a `<say-as>` that
    swallowed the comma in "MFA," would try to spell the comma."""

    span_open: list[str] = field(default_factory=list)
    span_close: list[str] = field(default_factory=list)
    """Wrappers around the WHOLE token, punctuation included. A span that ends a sentence
    has to enclose the full stop: Polly's generative tier only accepts a `<prosody>` around
    a complete sentence, and "…the message" without its period is a fragment."""

    breaks_before: list[str] = field(default_factory=list)


def _split_tokens(narration: str) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _TOKEN_RE.finditer(narration or ""):
        raw = match.group(0)
        parts = _CORE_RE.match(raw)
        assert parts is not None  # the pattern is total over any string
        tokens.append(
            _Token(
                text=raw,
                lead=parts.group("lead"),
                core=parts.group("core"),
                trail=parts.group("trail"),
                matchable=bool(re.search(r"[A-Za-z0-9]", raw)),
            )
        )
    return tokens


def _mark_sentences(tokens: list[_Token]) -> int:
    """Number the sentences and flag their first tokens. Returns the sentence count."""
    index = 0
    starting = True
    for position, token in enumerate(tokens):
        token.sentence = index
        token.sentence_start = starting
        starting = False
        if _ends_sentence(tokens, position):
            index += 1
            starting = True
    return index + 1 if tokens else 0


def _ends_sentence(tokens: list[_Token], position: int) -> bool:
    token = tokens[position]
    if not _SENTENCE_END_RE.search(token.text):
        return False
    bare = token.text.rstrip("\"'’”)]").lower()
    if bare.rstrip(".") in _ABBREVIATIONS or bare in _ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", token.text):
        return False  # "U.S." — an initialism, not a full stop
    following = next((t for t in tokens[position + 1 :] if t.matchable), None)
    if following is not None and following.core[:1].islower():
        return False  # a real sentence would not resume in lower case
    return True


# --------------------------------------------------------------------------- public API


def build_ssml(
    narration: str,
    *,
    role: SceneRole | None = None,
    bullets: Sequence[str] = (),
    engine: str = DEFAULT_ENGINE,
    emphasise_anchors: bool = True,
    sentence_breaks: bool = True,
    detect_acronyms: bool = True,
    spell_out: Iterable[str] = (),
    max_pause_seconds: float | None = None,
) -> str:
    """Wrap plain `narration` in SSML the engine will actually use.

    `bullets` are the scene's on-screen points. They are located in the narration with
    `bullet_timing.find_anchors` — the *same* matcher that later assigns reveal times — so
    the spoken stress and the appearing text land on the same words by construction rather
    than by coincidence. Only verbatim (`ngram`) anchors are emphasised; a fuzzy match
    would put the stress near, but not on, the phrase.

    `role` sets the speaking rate (see :data:`ROLE_RATE`) and the default pause budget
    (:data:`PAUSE_BUDGET_FRACTION` of the role's minimum target duration), because a
    3-second title card cannot afford the silence a 20-second content scene can.

    `spell_out` forces `<say-as interpret-as="characters">` on specific strings — the
    lookalike domain in a phishing example, where the individual characters *are* the
    lesson. `max_pause_seconds` overrides the role budget.

    Returns a `<speak>` document. Raises :class:`SsmlInvariantError` if the result would
    not strip back to the same tokens, and `ValueError` on empty narration or an engine
    that does not take SSML at all — a caller must not silently hand SSML to Deepgram.
    """
    text = (narration or "").strip()
    if not text:
        raise ValueError("cannot build SSML from empty narration")

    caps = capability(engine)
    if not caps.supports_ssml:
        raise ValueError(
            f"engine {caps.engine!r} does not parse SSML ({caps.notes}) — "
            "send plain narration, or route it through text_for_synthesis()"
        )

    tokens = _split_tokens(text)
    sentences = _mark_sentences(tokens)

    if emphasise_anchors:
        _apply_emphasis(tokens, bullets, caps)
    if detect_acronyms or spell_out:
        _apply_say_as(tokens, caps, spell_out=spell_out, detect=detect_acronyms)
    if sentence_breaks and "break" in caps.tags:
        budget = max_pause_seconds if max_pause_seconds is not None else _role_pause_budget(role)
        _apply_breaks(tokens, sentences, budget)

    body = _render(tokens)
    rate = ROLE_RATE.get(role or SceneRole.CONTENT, "100%")
    if rate != "100%" and "prosody" in caps.tags and "rate" in caps.prosody_attrs:
        body = f'<prosody rate="{rate}">{body}</prosody>'

    ssml = f"<speak>{body}</speak>"
    _assert_round_trip(text, ssml)
    return ssml


def strip_ssml(ssml: str) -> str:
    """Recover the plain text an SSML document speaks.

    Separating tags (`<break>`, `<mark>`, `<s>`, `<p>`) become a space so removing them
    cannot fuse two words; inline tags (`<emphasis>`, `<prosody>`, `<say-as>`, ...) are
    removed outright so `<say-as ...>MFA</say-as>,` comes back as "MFA," rather than
    "MFA ,". Entities are then unescaped and whitespace collapsed.

    Note `<sub alias="...">` cannot be honoured — it substitutes the *spoken* text, which
    this pipeline forbids (see the module docstring). The written text is returned.
    """
    if not ssml:
        return ""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name and name.lower() in _SEPARATING_TAGS:
            return " "
        if name is None:
            return " "  # comment / PI / CDATA wrapper
        return ""

    text = _TAG_RE.sub(replace, ssml)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def sanitise_for_plain_tts(text: str) -> str:
    """Make `text` safe for an engine that does not parse SSML.

    This is the guard for the measured Deepgram failure: handed
    ``<speak>Check the sender.<break time="800ms"/>Then hover the link.</speak>`` Aura
    speaks the tags, producing "Speak. Check the sender. Break time equals eight hundred
    milliseconds. Then hover the link." Stripping first costs nothing and removes the
    entire class of bug — including SSML an LLM emitted unasked.

    Plain text passes through unchanged apart from whitespace collapsing and entity
    unescaping (an engine reads "&amp;" aloud as literally as it reads "<break/>").
    """
    if not text:
        return ""
    if "<" not in text and "&" not in text:
        return _WHITESPACE_RE.sub(" ", text).strip()
    return strip_ssml(text)


def text_for_synthesis(narration: str, ssml: str | None, *, supports_ssml: bool) -> str:
    """What to hand the TTS engine, decided once from `SpeechSynthesizer.supports_ssml`.

    An SSML-capable engine gets the markup (falling back to narration when a scene has
    none); everything else gets plain narration, run through
    :func:`sanitise_for_plain_tts` in case what arrived was not plain after all.

    The aligner is a separate question and has one answer: it always gets `narration`.
    """
    if supports_ssml and ssml and ssml.strip():
        return ssml.strip()
    return sanitise_for_plain_tts(narration)


def validate_ssml(ssml: str, *, engine: str) -> list[str]:
    """Problems that would make `ssml` fail on `engine`. Empty list means it will work.

    Covers XML well-formedness (an unescaped ``&`` invalidates the whole document and
    Polly rejects it outright), the per-tier tag matrix in :data:`CAPABILITIES`, `<prosody>`
    attributes the tier does not accept, `say-as interpret-as` values, break durations over
    the engine's ceiling, and `<sub>`, which breaks this pipeline's word invariant on every
    engine.
    """
    problems: list[str] = []
    caps = capability(engine)
    if not ssml or not ssml.strip():
        return ["empty document"]
    if not caps.supports_ssml:
        return [f"engine {caps.engine!r} does not parse SSML: {caps.notes}"]

    body = ssml.strip()
    if not body.startswith("<speak"):
        problems.append("document must be rooted at <speak>")

    problems.extend(_wellformedness_problems(body))

    seen: list[tuple[str, str]] = [
        (match.group("name"), match.group(0))
        for match in _TAG_RE.finditer(body)
        if match.group("name")
    ]
    for name, raw in seen:
        lowered = name.lower()
        if lowered == "sub":
            problems.append(
                "<sub> substitutes the spoken words for the written ones, which breaks "
                "the aligner's reference-text invariant — remove it"
            )
            continue
        if lowered not in caps.tags:
            problems.append(f"<{name}> is not supported on {caps.engine} ({caps.notes})")
            continue
        if lowered == "prosody":
            for attr in re.findall(r"\b(rate|pitch|volume)\s*=", raw):
                if attr not in caps.prosody_attrs:
                    problems.append(f"<prosody {attr}> is not supported on {caps.engine}")
        elif lowered == "amazon:effect":
            for value in re.findall(r"\b(?:name|phonation)\s*=\s*[\"']([^\"']*)[\"']", raw):
                if value not in caps.effect_names:
                    problems.append(
                        f'amazon:effect "{value}" is not supported on {caps.engine}'
                    )
        elif lowered == "say-as":
            for value in re.findall(r"interpret-as\s*=\s*[\"']([^\"']*)[\"']", raw):
                if caps.say_as and value not in caps.say_as:
                    problems.append(
                        f'say-as interpret-as="{value}" is not supported on '
                        f"{caps.engine} ({caps.notes})"
                    )
        elif lowered == "break":
            for value in re.findall(r"time\s*=\s*[\"']([^\"']*)[\"']", raw):
                seconds = _break_seconds(value)
                if seconds is None:
                    problems.append(f'break time="{value}" is not a duration')
                elif seconds > caps.max_break_seconds:
                    problems.append(
                        f'break time="{value}" exceeds the {caps.max_break_seconds}s '
                        f"maximum on {caps.engine}"
                    )

    return problems


def estimate_pause_seconds(ssml: str) -> float:
    """Total inserted silence in `ssml`, in seconds.

    Scene durations come from *measured* audio, so longer narration is self-correcting for
    sync — but not for budget: a title card is allowed 3-6s and pauses spend that budget
    without conveying anything. A caller sizing narration against
    `SceneRole.target_duration` should subtract this from the words' share.

    `strength`-style breaks are approximated from :data:`BREAK_STRENGTH_SECONDS`; `time`
    values are exact.
    """
    total = 0.0
    for match in _TAG_RE.finditer(ssml or ""):
        if (match.group("name") or "").lower() != "break":
            continue
        raw = match.group(0)
        times = re.findall(r"time\s*=\s*[\"']([^\"']*)[\"']", raw)
        if times:
            total += sum(_break_seconds(value) or 0.0 for value in times)
            continue
        for strength in re.findall(r"strength\s*=\s*[\"']([^\"']*)[\"']", raw):
            total += BREAK_STRENGTH_SECONDS.get(strength, 0.0)
        if not times and "strength" not in raw:
            total += BREAK_STRENGTH_SECONDS["medium"]  # Polly's default
    return round(total, 3)


BREAK_COST_FACTOR = {
    "polly-generative": 0.919,
    "polly-neural": 1.05,
    "polly-long-form": 1.08,
}
"""Wall-clock cost of a `<break>` as a multiple of its nominal duration, MEASURED.

A/B on identical text across five scenes per tier, plain versus marked-up. Generative was
remarkably consistent (0.918-0.920 over 1 and 3 break scenes); neural and long-form came in
slightly OVER nominal once the role rate is accounted for, because the engine also stretches
the delivery around a pause. Unlisted engines fall back to 1.0.
"""

SPELL_OUT_SECONDS_PER_CHAR = 0.28
"""Extra wall-clock per character of a `<say-as interpret-as="characters">` run.

Measured on generative: "MFA" cost +0.896s over the same sentence unmarked (3 chars), and
"IT" +0.486s (2 chars). This is NOT a pause, so it is excluded from
:func:`estimate_pause_seconds` — but it is real duration, and it was the entire explanation
for a scene overshooting its predicted length by 0.49s during verification.
"""


def estimate_duration(plain_seconds: float, ssml: str, *, engine: str | None = None) -> float:
    """Predict the spoken length of `ssml`, given how long the plain narration takes.

    This is the budgeting call: compare the result against `SceneRole.target_duration` before
    committing to markup. Three effects are modelled, in the order they matter — the role's
    `<prosody rate>` (exactly proportional), inserted breaks (see
    :data:`BREAK_COST_FACTOR`), and spelled-out acronyms
    (:data:`SPELL_OUT_SECONDS_PER_CHAR`).

    Verified against the live measurements to within ~3%: a neural title card measured
    1.753s against 1.766s predicted, and a 3-break generative content scene measured
    11.123s against 11.123s predicted.

    Scene boundaries are still derived from REAL audio, so an error here never desyncs
    anything — it only means a scene lands outside the length its role wanted.
    """
    factor = BREAK_COST_FACTOR.get(capability(engine).engine, 1.0)
    seconds = max(0.0, plain_seconds) / _outer_rate(ssml)
    seconds += estimate_pause_seconds(ssml) * factor
    seconds += _spelled_chars(ssml) * SPELL_OUT_SECONDS_PER_CHAR
    return round(seconds, 3)


def _outer_rate(ssml: str) -> float:
    """The document-level `<prosody rate="N%">` as a multiplier, or 1.0."""
    match = re.search(r"<speak[^>]*>\s*<prosody[^>]*\brate\s*=\s*[\"'](\d+)%[\"']", ssml or "")
    if not match:
        return 1.0
    percent = int(match.group(1))
    return percent / 100.0 if percent > 0 else 1.0


def _spelled_chars(ssml: str) -> int:
    """Characters inside `say-as characters`/`spell-out` runs, which are read one by one."""
    pattern = re.compile(
        r"<say-as[^>]*interpret-as\s*=\s*[\"'](?:characters|spell-out)[\"'][^>]*>(.*?)</say-as>",
        re.DOTALL | re.IGNORECASE,
    )
    return sum(len(re.sub(r"[^A-Za-z0-9]", "", body)) for body in pattern.findall(ssml or ""))


def pause_budget(role: SceneRole | None) -> float:
    """Seconds of inserted silence a role can absorb. See :data:`PAUSE_BUDGET_FRACTION`."""
    return _role_pause_budget(role)


def capability_matrix() -> dict[str, dict[str, object]]:
    """The support matrix as plain data, for logging or a diagnostics endpoint."""
    return {
        name: {
            "supports_ssml": caps.supports_ssml,
            "tags": sorted(caps.tags),
            "prosody_attrs": sorted(caps.prosody_attrs),
            "emphasis": caps.emphasis_style(),
            "max_break_seconds": caps.max_break_seconds,
            "notes": caps.notes,
        }
        for name, caps in CAPABILITIES.items()
    }


# --------------------------------------------------------------------------- markup steps


def _role_pause_budget(role: SceneRole | None) -> float:
    floor = (role or SceneRole.CONTENT).target_duration[0]
    return round(floor * PAUSE_BUDGET_FRACTION, 3)


def _apply_emphasis(tokens: list[_Token], bullets: Sequence[str], caps: Capability) -> None:
    """Stress the phrase each bullet anchors to, using the bullet matcher's own spans.

    The mechanism is chosen from the engine, not from preference — see
    :meth:`Capability.emphasis_style` and :meth:`Capability.stress_attrs`.
    """
    style = caps.emphasis_style()
    if style == "none":
        return
    spans = anchor_spans([t.text for t in tokens if t.matchable], bullets)
    if not spans:
        return

    # Map indices in the matchable-only sequence (what the aligner and bullet matcher see)
    # back onto positions in the full token list, which still holds unmatchable tokens.
    positions = [index for index, token in enumerate(tokens) if token.matchable]
    open_tag = (
        f'<emphasis level="{EMPHASIS_LEVEL}">'
        if style == "tag"
        else f"<prosody {caps.stress_attrs()}>"
    )
    close_tag = "</emphasis>" if style == "tag" else "</prosody>"

    ceiling = int(len(positions) * EMPHASIS_TOKEN_FRACTION)
    spent = 0
    for start, end in spans:
        width = end - start
        if spent + width > max(width, ceiling):
            continue
        spent += width
        tokens[positions[start]].span_open.append(open_tag)
        tokens[positions[end - 1]].span_close.insert(0, close_tag)


def anchor_spans(narration_tokens: Sequence[str], bullets: Sequence[str]) -> list[tuple[int, int]]:
    """`(start, end)` token spans in `narration_tokens` that `bullets` verbatim-anchor to.

    Delegates to `bullet_timing.find_anchors` with synthetic one-second words so the spans
    are exactly the ones the timer will use on real audio. Overlaps are merged, spans are
    capped at :data:`MAX_EMPHASIS_SPAN` tokens, and non-verbatim (fuzzy or proportional)
    anchors are dropped — stress near the phrase is worse than no stress.

    Exposed because it is the useful half of this module for anyone reasoning about where
    the audio and the on-screen text are supposed to meet.
    """
    texts = [b.strip() for b in bullets or () if b and b.strip()]
    if not texts or not narration_tokens:
        return []

    words = [
        Word(word=token, start=float(index), end=float(index) + 0.9)
        for index, token in enumerate(narration_tokens)
    ]
    raw: list[tuple[int, int]] = []
    for anchor in find_anchors(texts, words):
        if anchor.method != "ngram" or anchor.word_index < 0:
            continue
        start = anchor.word_index
        end = min(start + max(1, len(anchor.matched_words)), len(narration_tokens))
        end = min(end, start + MAX_EMPHASIS_SPAN)
        if end > start:
            raw.append((start, end))

    raw.sort()
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            widest = previous_start + MAX_EMPHASIS_SPAN
            merged[-1] = (previous_start, min(max(previous_end, end), widest))
        else:
            merged.append((start, end))
    return merged


def _apply_say_as(
    tokens: list[_Token], caps: Capability, *, spell_out: Iterable[str], detect: bool
) -> None:
    """Spell out initialisms, and slow down domains so a lookalike can be read."""
    forced = {value.strip().lower() for value in spell_out if value and value.strip()}
    can_say_as = "say-as" in caps.tags and "characters" in (caps.say_as or {"characters"})
    # Generative honours no usable `rate`, so a domain there gets the weak break alone —
    # less help than a slowdown, but better than a tag that measurably does nothing.
    can_prosody = "prosody" in caps.tags and "rate" in caps.prosody_attrs

    for token in tokens:
        core = token.core
        if not core:
            continue
        lowered = core.lower()

        if can_say_as and (lowered in forced or (detect and _is_initialism(core))):
            _wrap_core(token, '<say-as interpret-as="characters">', "</say-as>")
            continue

        if can_say_as and _PHONE_RE.match(core):
            _wrap_core(token, '<say-as interpret-as="telephone">', "</say-as>")
            continue

        if _DOMAIN_RE.match(core):
            # NOT spelled out by default: "e x a m p l e dot c o m" is worse than the
            # engine's own reading. What a domain needs is time on the ear, so it gets a
            # slower rate and a weak beat in front of it. Pass the domain in `spell_out`
            # when the individual characters are the point (a lookalike).
            if can_prosody:
                _wrap_core(token, '<prosody rate="slow">', "</prosody>")
            if "break" in caps.tags and not token.sentence_start:
                token.breaks_before.append(f'<break time="{DOMAIN_BREAK_MS}ms"/>')


def _is_initialism(core: str) -> bool:
    if core in ACRONYMS:
        return True
    if core in _PRONOUNCED_AS_WORD or core in _CAPS_NOT_ACRONYM:
        return False
    # Heuristic for initialisms not on the list: short, all caps, at most one digit lead.
    return bool(re.fullmatch(r"[A-Z]{2,5}|[0-9][A-Z]{1,4}", core))


def _wrap_core(token: _Token, open_tag: str, close_tag: str) -> None:
    """Wrap a token's core, leaving its punctuation outside.

    "MFA," must become `<say-as ...>MFA</say-as>,` — inside the tag the engine would try
    to spell the comma. Inline tags strip to nothing, so the token round-trips exactly.
    """
    token.open_tags.append(open_tag)
    token.close_tags.insert(0, close_tag)


def _apply_breaks(tokens: list[_Token], sentences: int, budget: float) -> None:
    """Place sentence breaks and one instruction break, then fit them to `budget`.

    No break is placed before the first sentence (nothing to separate) or after the last
    (the scene ends there — the pause would be spent on silence nobody hears as a beat).
    """
    if sentences < 2:
        return

    candidates: list[tuple[_Token, int, bool]] = []  # token, ms, is_instruction
    for token in tokens:
        if not token.sentence_start or token.sentence == 0:
            continue
        instruction = token.core.lower() in IMPERATIVES
        span = INSTRUCTION_BREAK_MS if instruction else SENTENCE_BREAK_MS
        candidates.append((token, span, instruction))

    if not candidates:
        return

    # Only the first instruction in a scene gets the long beat; a scene of five long
    # pauses is not emphatic, it is slow.
    promoted = False
    fitted: list[tuple[_Token, int, bool]] = []
    for token, ms, instruction in candidates:
        if instruction and promoted:
            ms, instruction = SENTENCE_BREAK_MS, False
        promoted = promoted or instruction
        fitted.append((token, ms, instruction))

    total = sum(ms for _, ms, _ in fitted) / 1000.0
    if total > budget:
        # A budget of zero is a legitimate instruction ("no pauses at all"), so this is
        # deliberately not guarded by `budget > 0` — _fit_breaks drops every break.
        fitted = _fit_breaks(fitted, budget)

    for token, ms, _ in fitted:
        if ms > 0:
            token.breaks_before.append(f'<break time="{ms}ms"/>')


def _fit_breaks(
    breaks: list[tuple[_Token, int, bool]], budget: float
) -> list[tuple[_Token, int, bool]]:
    """Scale breaks into `budget`, then drop the least meaningful ones if still over.

    Scaling first keeps the *rhythm* — every boundary still gets a beat, just a shorter
    one. Only when even :data:`MIN_BREAK_MS` everywhere would overrun do breaks get
    dropped, and then plain sentence boundaries go before the instruction beat, which is
    the one pause carrying meaning.
    """
    total_ms = sum(ms for _, ms, _ in breaks)
    budget_ms = max(0.0, budget * 1000.0)
    scale = budget_ms / total_ms if total_ms else 0.0
    scaled = [
        (token, max(MIN_BREAK_MS, int(ms * scale)), instruction)
        for token, ms, instruction in breaks
    ]

    if sum(ms for _, ms, _ in scaled) <= budget_ms:
        return scaled

    # Keep as many beats as fit, spending the budget on the instruction break first.
    order = sorted(range(len(scaled)), key=lambda i: (not scaled[i][2], i))
    kept: set[int] = set()
    spent = 0.0
    for index in order:
        cost = scaled[index][1]
        if spent + cost <= budget_ms:
            kept.add(index)
            spent += cost
    return [
        (token, ms if index in kept else 0, instruction)
        for index, (token, ms, instruction) in enumerate(scaled)
    ]


def _render(tokens: list[_Token]) -> str:
    """Emit the document body, keeping one space between every pair of tokens."""
    pieces: list[str] = []
    for index, token in enumerate(tokens):
        if index:
            pieces.append(" ")
        pieces.extend(token.breaks_before)
        pieces.extend(token.span_open)
        pieces.append(_escape(token.lead))
        pieces.extend(token.open_tags)
        pieces.append(_escape(token.core))
        pieces.extend(token.close_tags)
        pieces.append(_escape(token.trail))
        pieces.extend(token.span_close)
    return "".join(pieces)


def _escape(text: str) -> str:
    """XML-escape character data. A bare "&" invalidates the document and Polly 400s.

    Quotes and apostrophes are legal in character data and are left alone — narration
    never reaches an attribute value, which is the only place they would need escaping.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- checks


def _assert_round_trip(narration: str, ssml: str) -> None:
    """The invariant. Compared with the aligner's own tokenizer, not a private one."""
    expected = tokenize(narration)
    actual = tokenize(strip_ssml(ssml))
    if expected != actual:
        first = next(
            (
                index
                for index, (left, right) in enumerate(zip(expected, actual, strict=False))
                if left != right
            ),
            min(len(expected), len(actual)),
        )
        raise SsmlInvariantError(
            "SSML would change the narration's words, which would drift every bullet "
            f"anchor. First divergence at token {first}: "
            f"expected {expected[first : first + 4]!r}, got {actual[first : first + 4]!r}"
        )


def _wellformedness_problems(ssml: str) -> list[str]:
    """XML parse errors, with Polly's undeclared `amazon:` prefix accounted for."""
    from xml.etree import ElementTree  # noqa: PLC0415 — stdlib, kept out of import cost

    probe = ssml
    if "amazon:" in ssml and "xmlns:amazon" not in ssml:
        # Polly documents <amazon:effect> without a namespace declaration, so a strict
        # parser rejects valid Polly SSML. Declare it for the check only.
        probe = ssml.replace(
            "<speak", '<speak xmlns:amazon="http://amazon.com/ssml"', 1
        )
    try:
        ElementTree.fromstring(probe)
    except ElementTree.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    return []


def _break_seconds(value: str) -> float | None:
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(ms|s)\s*", value or "")
    if not match:
        return None
    amount = float(match.group(1))
    return amount / 1000.0 if match.group(2) == "ms" else amount
