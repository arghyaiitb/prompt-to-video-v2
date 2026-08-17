"""Narration via Amazon Polly — the SSML-capable arm of `SpeechSynthesizer`.

`supports_ssml = True` here, and that flag is the whole reason this module exists. Deepgram
Aura *vocalises* SSML tags instead of parsing them (send it ``<break time="800ms"/>`` and
you hear "break time equals eight hundred milliseconds"), so `DeepgramSynthesizer` declares
False. Polly parses SSML properly, which lets the caller control pauses and stress at the
exact word an on-screen bullet anchors to.

MEASURED ENGINE / SSML MATRIX
-----------------------------
Everything below was probed live against account 943436369047 in ``us-east-1`` — not read
off a doc page. ``OK`` means Polly returned audio; ``--`` means it returned 400.

    tag / attribute                standard   neural   long-form   generative
                                   (Joanna)  (Matthew) (Danielle)  (Matthew)
    <break>                           OK        OK        OK          OK
    <emphasis>                        OK        --        --          --
    <prosody rate|volume>             OK        OK        OK          OK
    <prosody pitch>                   OK        --        --          --
    <say-as>                          OK        OK        OK          OK
    <phoneme>                         OK        OK        OK          OK
    <sub>                             OK        OK        OK          OK
    <mark>                            OK        OK        OK          OK
    <lang>                            OK        OK        OK          OK
    <p> / <s>                         OK        OK        OK          OK
    <amazon:effect name="drc">        OK        OK        OK          --
    <amazon:domain name="news">      --        OK        --          --

Two results worth calling out, because both contradict what you would assume:

  * The failures are ``InvalidSsmlException: Unsupported <Engine> feature`` — the engine
    tier decides, not the voice. Re-probing ``long-form`` with a genuinely long-form voice
    (Danielle) showed it behaves exactly like neural; an earlier probe that used Matthew
    made long-form look like it rejected *every* tag, but Matthew is simply not a
    long-form voice, so all four calls died on the voice/engine mismatch instead.
  * AWS documents ``<prosody>`` on generative voices as usable "only around full
    sentences". Measured, mid-sentence ``<prosody>`` works on all four engines. That
    matters a lot — see below.

WHY THE DEFAULT ENGINE IS ``generative``
----------------------------------------
The highest-value SSML use in this pipeline is stressing the phrase a bullet points at,
and ``<emphasis>`` is unavailable on every engine worth shipping. Rather than drop to
``standard`` to keep one tag, this module keeps the best voices and rewrites the tag:
``adapt_ssml`` turns ``<emphasis>`` into the ``<prosody rate/volume>`` equivalent and drops
``pitch``, both of which are confirmed working mid-sentence on generative. Perceived
production value is dominated by voice quality, and prosody gets most of the way to
emphasis; losing ``<amazon:effect name="drc">`` costs us nothing since we do not use it.
Pass ``engine=`` to override — the tier is a constructor argument because the UI exposes it
— or ``engine="auto"`` to pick the best tier each voice actually supports.

Set ``strict_ssml=True`` and nothing is rewritten: unsupported markup raises instead, which
is what you want in a test that is *checking* engine capability.

CREDENTIALS
-----------
Env secrets win over the ambient profile, deliberately: ``AWS_PROFILE`` is set on dev boxes
and boto3's default chain would silently prefer that SSO role over the keys someone put in
``.env``. Resolution order is constructor args, then `Settings` attributes, then
``os.environ``, then the repo ``.env`` file, then boto3's own chain. Whatever is found is
passed to `boto3.client` explicitly, which is what stops the profile winning.

`Settings` carries ``aws_access_key_id`` / ``aws_secret_access_key`` / ``aws_session_token``
/ ``aws_region``, so step two normally answers. Steps three and four exist because that was
not always true: `Settings` sets ``extra="ignore"``, so before those fields existed
pydantic-settings read ``AWS_ACCESS_KEY_ID`` out of ``.env`` and discarded it without ever
touching ``os.environ``, making keys placed there invisible. The direct ``.env`` read keeps
this module working if a field is renamed or dropped again.

The session token is not optional decoration: this account issues temporary STS keys
(``ASIA...``), and passing the pair without the token yields ``InvalidClientTokenId``, which
points nowhere near the real cause.

OUTPUT FORMAT
-------------
Polly has no wav output, so ``.wav`` is assembled locally. ``pcm`` is signed 16-bit mono
little-endian and caps at 16 kHz (24000 returns ``InvalidSampleRateException``, measured),
so the default path requests pcm at 16 kHz and wraps it with the stdlib ``wave`` module —
lossless, no ffmpeg, byte-exact. Ask for ``sample_rate=24000`` and it switches to mp3 (whose
neural/long-form/generative default is 24 kHz) and transcodes, matching the 24 kHz wav that
`DeepgramSynthesizer` emits at the cost of one lossy hop. ``.mp3`` and ``.ogg`` out paths are
written straight through. Duration always comes from ffprobe, never from a character count.
"""

from __future__ import annotations

import io
import logging
import os
import random
import re
import time
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app.core.config import REPO_ROOT, get_settings
from app.providers._media import audio_duration, run_ffmpeg

logger = logging.getLogger(__name__)

#: Tiers, best first. `engine="auto"` walks this and takes the first the voice supports.
ENGINE_PREFERENCE = ("generative", "long-form", "neural", "standard")

ENGINES = frozenset(ENGINE_PREFERENCE)

DEFAULT_ENGINE = "generative"
DEFAULT_VOICE = "Matthew"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_REGION = "us-east-1"

#: SynthesizeSpeech caps at 6000 total / 3000 billed characters; SSML tags are not billed.
#: Verified: 6500 characters returns TextLengthExceededException.
MAX_TOTAL_CHARS = 6000
MAX_BILLED_CHARS = 3000

#: pcm is the only lossless format and SampleRate=24000 is rejected for it (measured).
PCM_SAMPLE_RATES = frozenset({8000, 16000})
MP3_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})

_RETRYABLE = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "ThrottledException",
        "TooManyRequestsException",
        "RequestThrottled",
        "RequestThrottledException",
        "ServiceFailureException",
        "ServiceUnavailable",
        "InternalFailure",
        "RequestTimeout",
        "SlowDown",
    }
)

_THROTTLE_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "ThrottledException",
        "TooManyRequestsException",
        "RequestThrottled",
        "RequestThrottledException",
        "SlowDown",
    }
)

_CREDENTIAL_CODES = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "ExpiredToken",
        "ExpiredTokenException",
        "IncompleteSignature",
        "InvalidAccessKeyId",
        "InvalidClientTokenId",
        "InvalidSignatureException",
        "MissingAuthenticationToken",
        "SignatureDoesNotMatch",
        "UnrecognizedClientException",
    }
)

# --------------------------------------------------------------- SSML capability data
#
# Sources are marked because they are not equally trustworthy: "measured" rows were probed
# against the live API, "documented" rows come from the AWS SSML tag table. Do not silently
# promote a documented row to measured.

#: Polly rejects these outright, on every engine (documented).
UNSUPPORTED_EVERYWHERE = frozenset({"audio", "lexicon", "lookup", "voice"})

#: Engines that return "Unsupported <Engine> feature" for <emphasis> (measured).
NO_EMPHASIS = frozenset({"neural", "long-form", "generative"})

#: Engines that reject the prosody `pitch` attribute (measured). rate/volume are fine.
NO_PROSODY_PITCH = frozenset({"neural", "long-form", "generative"})

#: <amazon:effect name="drc"> (measured).
NO_DRC = frozenset({"generative"})

#: Newscaster style. Measured OK on neural/Matthew, 400 on the other three tiers.
DOMAIN_NEWS_ENGINES = frozenset({"neural"})

#: Never available outside `standard` (documented; not probed).
STANDARD_ONLY_EFFECTS = frozenset({"phonation", "vocal-tract-length", "whispered"})
STANDARD_ONLY_TAGS = frozenset({"amazon:auto-breaths"})

#: How <emphasis level=...> is approximated when the engine has no <emphasis>.
#: pitch is deliberately absent — it is rejected on exactly the same engines.
EMPHASIS_TO_PROSODY: dict[str, dict[str, str] | None] = {
    "strong": {"rate": "90%", "volume": "+4dB"},
    "moderate": {"rate": "95%", "volume": "+2dB"},
    "reduced": {"rate": "105%", "volume": "-3dB"},
    "none": None,  # unwrap; no prosody at all
}

_AMAZON_NS = "http://polly.amazon.com/ssml"

_SSML_PREFIX = re.compile(r"^\s*<speak\b", re.IGNORECASE)
_TAG_STRIP = re.compile(r"<[^>]*>")


# --------------------------------------------------------------------------- errors


class PollyError(RuntimeError):
    """Base for every failure this module raises. Never leaks a botocore traceback."""


class PollyCredentialsError(PollyError):
    """No usable credentials, or the ones we have were rejected/expired."""


class PollyVoiceError(PollyError):
    """Unknown voice, or a voice the requested engine does not support."""


class PollyThrottledError(PollyError):
    """Rate limited after exhausting retries. Polly allows 8 tps on generative."""


class PollyRegionError(PollyError):
    """No region configured, or Polly/this voice is unavailable in the region."""


class SsmlError(PollyError):
    """SSML rejected locally, before the call, naming the offending markup."""


class TextTooLongError(PollyError):
    """Over the SynthesizeSpeech character cap. We refuse rather than truncate."""


# ---------------------------------------------------------------------- credentials


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=value`` reader. Needed because `Settings` drops undeclared keys."""
    values: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def resolve_credentials(
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region_name: str | None = None,
    env_file: Path | None = None,
) -> dict[str, str | None]:
    """Credentials for boto3, env secrets ahead of the ambient profile.

    Order: explicit args, `Settings` attributes, ``os.environ``, the repo ``.env``, then
    all-None meaning "let boto3 use its own chain" (profile / SSO / instance role).
    Partial credentials are returned as-is so `make_client` can report *which* half is
    missing instead of raising a generic NoCredentialsError.

    `Settings` now declares the ``aws_*`` fields, so in practice step two answers and
    pydantic-settings has already merged ``os.environ`` and ``.env`` into it. Steps three
    and four remain as a safety net for a field that gets renamed or dropped, and for the
    alternative spelling ``AWS_DEFAULT_REGION`` that pydantic does not map. Note that
    ``Settings.aws_region`` defaults to a real region rather than empty, so it wins over
    ``AWS_DEFAULT_REGION`` unless someone blanks it.
    """
    settings = get_settings()

    def _settings_value(name: str) -> str | None:
        value = getattr(settings, name, None)
        return str(value) if value else None

    dotenv = _parse_env_file(env_file if env_file is not None else REPO_ROOT / ".env")

    def pick(arg: str | None, field: str, *env_keys: str) -> str | None:
        if arg:
            return arg
        from_settings = _settings_value(field)
        if from_settings:
            return from_settings
        for key in env_keys:
            if os.environ.get(key):
                return os.environ[key]
        for key in env_keys:
            if dotenv.get(key):
                return dotenv[key]
        return None

    return {
        "aws_access_key_id": pick(aws_access_key_id, "aws_access_key_id", "AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": pick(
            aws_secret_access_key, "aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"
        ),
        "aws_session_token": pick(aws_session_token, "aws_session_token", "AWS_SESSION_TOKEN"),
        "region_name": pick(region_name, "aws_region", "AWS_REGION", "AWS_DEFAULT_REGION"),
    }


def make_client(
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region_name: str | None = None,
    attempts: int = 1,
    service: str = "polly",
) -> Any:
    """A boto3 client. SDK retries are off by default — we retry in `_call` so the backoff
    is visible and testable instead of buried in botocore."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PollyError(
            "boto3 is not installed — add it to backend/pyproject.toml and run `uv sync`"
        ) from exc

    creds = resolve_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    key, secret = creds["aws_access_key_id"], creds["aws_secret_access_key"]
    if bool(key) != bool(secret):
        present = "AWS_ACCESS_KEY_ID" if key else "AWS_SECRET_ACCESS_KEY"
        missing = "AWS_SECRET_ACCESS_KEY" if key else "AWS_ACCESS_KEY_ID"
        raise PollyCredentialsError(
            f"incomplete AWS credentials: {present} is set but {missing} is not. "
            f"Set both in .env (with AWS_SESSION_TOKEN too if they are temporary "
            f"STS keys starting 'ASIA'), or unset both to use the AWS_PROFILE chain."
        )

    kwargs: dict[str, Any] = {"config": Config(retries={"max_attempts": max(attempts, 1)})}
    if key and secret:
        kwargs["aws_access_key_id"] = key
        kwargs["aws_secret_access_key"] = secret
        # STS keys are useless without it, and omitting it yields a baffling
        # "InvalidClientTokenId" rather than anything pointing at the token.
        if creds["aws_session_token"]:
            kwargs["aws_session_token"] = creds["aws_session_token"]
    if creds["region_name"]:
        kwargs["region_name"] = creds["region_name"]

    try:
        return boto3.client(service, **kwargs)
    except Exception as exc:  # noqa: BLE001 - botocore raises a zoo of init errors
        raise _translate(exc, context=f"creating the {service} client") from exc


# ------------------------------------------------------------------ error translation


def _translate(exc: BaseException, *, context: str, payload: str | None = None) -> PollyError:
    """Turn a botocore exception into something a developer can act on."""
    name = type(exc).__name__

    if name in {"NoCredentialsError", "PartialCredentialsError", "TokenRetrievalError"}:
        return PollyCredentialsError(
            f"no usable AWS credentials while {context}: {exc}. Set AWS_ACCESS_KEY_ID, "
            f"AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN for temporary keys) in .env, "
            f"or make sure AWS_PROFILE points at a logged-in SSO profile."
        )
    if name in {"SSOTokenLoadError", "UnauthorizedSSOTokenError"}:
        return PollyCredentialsError(
            f"the AWS_PROFILE SSO session is not logged in ({exc}). Run `aws sso login`, "
            f"or put explicit keys in .env — env secrets take precedence over the profile."
        )
    if name == "NoRegionError":
        return PollyRegionError(
            f"no AWS region while {context}. Set AWS_REGION in .env "
            f"(Polly generative and long-form voices are region-limited; "
            f"{DEFAULT_REGION} has both)."
        )
    if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
        return PollyRegionError(
            f"could not reach Polly while {context}: {exc}. Check AWS_REGION is a real "
            f"region and that the network allows outbound HTTPS."
        )

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return PollyError(f"unexpected failure while {context}: {name}: {exc}")

    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    message = str(error.get("Message") or exc)

    if code in _CREDENTIAL_CODES:
        hint = (
            " Temporary STS keys expire in hours — refresh AWS_SESSION_TOKEN in .env."
            if code.startswith("ExpiredToken")
            else " Check AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and that the role has"
            " polly:SynthesizeSpeech."
        )
        return PollyCredentialsError(
            f"AWS rejected the credentials ({code}) while {context}: {message}.{hint}"
        )
    if code in _THROTTLE_CODES:
        return PollyThrottledError(
            f"Polly throttled the request ({code}) while {context}: {message}. "
            f"Generative and long-form allow 8 tps — lower VIDEO_API_CONCURRENCY."
        )
    if code == "InvalidSsmlException":
        return SsmlError(
            f"Polly rejected the SSML ({message}) while {context}. This is an engine "
            f"capability limit, not malformed XML — see the matrix in polly_tts's "
            f"docstring; <emphasis> and <prosody pitch> need engine='standard'."
            + (f" Payload: {payload[:400]!r}" if payload else "")
        )
    if code == "TextLengthExceededException":
        return TextTooLongError(f"text over the Polly limit while {context}: {message}")
    if code in {"EngineNotSupportedException", "InvalidSampleRateException"}:
        return PollyVoiceError(f"{code} while {context}: {message}")
    if code == "ValidationException":
        return PollyVoiceError(
            f"Polly refused the request ({code}) while {context}: {message}. Usually the "
            f"voice does not support the chosen engine or an SSML feature in the payload."
        )
    if code in {"LanguageNotSupportedException", "UnsupportedPlsLanguageException"}:
        return PollyVoiceError(f"{code} while {context}: {message}")
    if code == "ServiceFailureException":
        return PollyError(f"Polly server error while {context}: {message}")
    return PollyError(f"Polly error {code} while {context}: {message}")


# --------------------------------------------------------------------------- SSML


def is_ssml(text: str) -> bool:
    """True when the payload opens with a ``<speak>`` root."""
    return bool(_SSML_PREFIX.match(text or ""))


def billed_chars(text: str) -> int:
    """Characters AWS bills: SSML tags are excluded, everything else counts."""
    return len(_TAG_STRIP.sub("", text)) if is_ssml(text) else len(text)


def _parseable(ssml: str) -> str:
    """Polly SSML uses the ``amazon:`` prefix without declaring it, which ElementTree
    rejects as an unbound prefix. Declare it for the parser only."""
    if "amazon:" not in ssml or "xmlns:amazon" in ssml:
        return ssml
    return re.sub(
        r"<speak\b", f'<speak xmlns:amazon="{_AMAZON_NS}"', ssml, count=1, flags=re.IGNORECASE
    )


def _local(tag: str) -> str:
    """`{ns}effect` -> `amazon:effect`; bare tags pass through."""
    if tag.startswith("{"):
        return f"amazon:{tag.split('}', 1)[1]}"
    return tag


def _parse(ssml: str) -> ET.Element:
    try:
        return ET.fromstring(_parseable(ssml))
    except ET.ParseError as exc:
        line, column = getattr(exc, "position", (0, 0))
        excerpt = ""
        lines = ssml.splitlines()
        if 0 < line <= len(lines):
            offending = lines[line - 1]
            excerpt = f" near {offending[max(0, column - 40) : column + 40]!r}"
        raise SsmlError(
            f"malformed SSML at line {line} column {column}: {exc}.{excerpt} "
            f"(a bare '&' or an unclosed tag is the usual cause; "
            f"escape it as '&amp;'). Full payload: {ssml[:400]!r}"
        ) from exc


def validate_ssml(ssml: str, engine: str = DEFAULT_ENGINE) -> None:
    """Reject bad SSML locally, so the error names the markup.

    A wire ``InvalidSsmlException`` says only "Unsupported Generative feature" with no
    indication of which tag, which is miserable to debug from a job log.
    """
    if not is_ssml(ssml):
        raise SsmlError(
            f"SSML must have a <speak> root element; got {ssml[:80]!r}. "
            f"Plain text is fine — just do not open it with a tag."
        )

    root = _parse(ssml)
    if _local(root.tag) != "speak":
        raise SsmlError(f"SSML root must be <speak>, found <{_local(root.tag)}>")

    for element in root.iter():
        tag = _local(element.tag)

        if tag in UNSUPPORTED_EVERYWHERE:
            raise SsmlError(f"<{tag}> is not supported by Amazon Polly on any engine")
        if tag in STANDARD_ONLY_TAGS and engine != "standard":
            raise SsmlError(f"<{tag}> requires engine='standard'; engine is {engine!r}")

        if tag == "emphasis" and engine in NO_EMPHASIS:
            level = element.get("level", "moderate")
            raise SsmlError(
                f"<emphasis level={level!r}> is rejected by the {engine} engine "
                f"(measured: InvalidSsmlException 'Unsupported {engine} feature'). Use "
                f"engine='standard', or let adapt_ssml rewrite it as <prosody> by leaving "
                f"strict_ssml off."
            )

        if tag == "prosody":
            if not element.attrib:
                raise SsmlError("<prosody> needs at least one of rate, volume, pitch")
            if "amazon:max-duration" in element.attrib and engine != "standard":
                raise SsmlError(
                    f"<prosody amazon:max-duration> requires engine='standard'; "
                    f"engine is {engine!r}"
                )
            if "pitch" in element.attrib and engine in NO_PROSODY_PITCH:
                raise SsmlError(
                    f"<prosody pitch={element.get('pitch')!r}> is rejected by the {engine} "
                    f"engine (measured). rate and volume do work — adapt_ssml drops pitch "
                    f"for you when strict_ssml is off."
                )

        if tag == "amazon:effect":
            name = element.get("name")
            if name == "drc" and engine in NO_DRC:
                raise SsmlError(
                    f'<amazon:effect name="drc"> is rejected by the {engine} engine '
                    f"(measured); it works on standard, neural and long-form."
                )
            offenders = STANDARD_ONLY_EFFECTS & set(element.attrib)
            if name in STANDARD_ONLY_EFFECTS:
                offenders = offenders | {name}
            if offenders and engine != "standard":
                raise SsmlError(
                    f"<amazon:effect {'/'.join(sorted(offenders))}> requires "
                    f"engine='standard'; engine is {engine!r}"
                )

        if tag == "amazon:domain" and element.get("name") == "news":
            if engine not in DOMAIN_NEWS_ENGINES:
                raise SsmlError(
                    f'<amazon:domain name="news"> works only on the neural engine '
                    f"(measured); engine is {engine!r}"
                )

        if tag == "break":
            _validate_break(element.get("time"))


def _validate_break(value: str | None) -> None:
    if not value:
        return
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*", value)
    if not match:
        raise SsmlError(f'<break time="{value}"> is not a duration; use e.g. "800ms" or "1s"')
    seconds = float(match.group(1)) / (1000.0 if match.group(2) == "ms" else 1.0)
    if seconds > 10.0:
        raise SsmlError(f'<break time="{value}"> exceeds the 10 second Polly maximum')


def adapt_ssml(ssml: str, engine: str) -> tuple[str, list[str]]:
    """Rewrite markup the engine cannot take into its closest supported equivalent.

    Returns ``(ssml, notes)``; ``notes`` is empty when nothing changed. Two rewrites, both
    driven by the measured matrix:

      * ``<emphasis>`` -> ``<prosody rate/volume>`` per `EMPHASIS_TO_PROSODY`
      * ``<prosody pitch=...>`` -> the same tag minus ``pitch``

    Text-only and already-valid payloads are returned untouched, so this is safe to call
    unconditionally.
    """
    if not is_ssml(ssml):
        return ssml, []

    notes: list[str] = []
    result = ssml

    if engine in NO_EMPHASIS and re.search(r"<\s*emphasis", result, re.IGNORECASE):
        result, notes_e = _rewrite_emphasis(result)
        notes.extend(notes_e)

    if engine in NO_PROSODY_PITCH and re.search(r"pitch\s*=", result, re.IGNORECASE):
        result, notes_p = _strip_prosody_pitch(result)
        notes.extend(notes_p)

    if notes:
        logger.info("polly: adapted SSML for the %s engine — %s", engine, "; ".join(notes))
    return result, notes


_EMPHASIS_TAG = re.compile(
    r"""<\s*(?P<close>/)?\s*emphasis\b(?P<attrs>[^>]*?)(?P<selfclose>/)?\s*>""", re.IGNORECASE
)
_LEVEL_ATTR = re.compile(r"""level\s*=\s*["']([^"']*)["']""", re.IGNORECASE)


def _rewrite_emphasis(ssml: str) -> tuple[str, list[str]]:
    """One pass, tracking a stack so each ``</emphasis>`` closes what its open tag became.

    ``level="none"`` maps to no prosody at all, so that pair has to disappear entirely —
    emitting a stray ``</prosody>`` would turn a capability problem into malformed XML.
    """
    notes: list[str] = []
    out: list[str] = []
    stack: list[str | None] = []
    cursor = 0

    for match in _EMPHASIS_TAG.finditer(ssml):
        out.append(ssml[cursor : match.start()])
        cursor = match.end()

        if match.group("close"):
            replacement = stack.pop() if stack else "</prosody>"
            out.append(replacement or "")
            continue

        level = _LEVEL_ATTR.search(match.group("attrs") or "") or None
        name = (level.group(1) if level else "moderate").lower()
        mapping = EMPHASIS_TO_PROSODY.get(name, EMPHASIS_TO_PROSODY["moderate"])

        if mapping is None:
            notes.append(f'<emphasis level="{name}"> unwrapped (no prosody equivalent)')
            opened, closer = "", None
        else:
            rendered = " ".join(f'{key}="{value}"' for key, value in mapping.items())
            notes.append(f'<emphasis level="{name}"> -> <prosody {rendered}>')
            opened, closer = f"<prosody {rendered}>", "</prosody>"

        # Self-closing <emphasis/> wraps nothing, so it needs no close tag either.
        if match.group("selfclose"):
            continue
        out.append(opened)
        stack.append(closer)

    out.append(ssml[cursor:])
    return "".join(out), notes


def _strip_prosody_pitch(ssml: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    def fix(match: re.Match[str]) -> str:
        attrs = match.group(1)
        pitch = re.search(r"""pitch\s*=\s*["']([^"']*)["']""", attrs, re.IGNORECASE)
        if not pitch:
            return match.group(0)
        cleaned = re.sub(
            r"""\s*pitch\s*=\s*["'][^"']*["']""", "", attrs, flags=re.IGNORECASE
        ).strip()
        notes.append(f'<prosody pitch="{pitch.group(1)}"> dropped (unsupported on engine)')
        if not cleaned:
            # A pitch-only <prosody> has nothing left to say; keep the element valid by
            # asking for the default rate rather than emitting an attribute-less tag.
            cleaned = 'rate="100%"'
        return f"<prosody {cleaned}>"

    return re.sub(r"<\s*prosody\b([^>]*?)\s*>", fix, ssml, flags=re.IGNORECASE), notes


# --------------------------------------------------------------------- voice catalogue

#: en-* language code -> the accent label `app/api/voices.py` puts in its `accent` field.
ACCENTS = {
    "en-US": "American",
    "en-GB": "British",
    "en-GB-WLS": "Welsh",
    "en-AU": "Australian",
    "en-IN": "Indian",
    "en-IE": "Irish",
    "en-NZ": "New Zealand",
    "en-ZA": "South African",
    "en-SG": "Singaporean",
}

_voice_cache: dict[str, list[dict[str, Any]]] = {}


def reset_voice_cache() -> None:
    """Test seam, mirroring `app.api.voices.reset_cache`."""
    _voice_cache.clear()


def shape_voice(entry: dict[str, Any]) -> dict[str, Any]:
    """One DescribeVoices entry in the shape `app/api/voices.py` serves.

    ``id``/``name``/``accent``/``tags``/``use_cases`` are the keys that endpoint's `Voice`
    model declares, so these dicts drop into the same response with no adapter. The extra
    keys (``provider``, ``engines``, ``language_code``, ...) are what a unified cross-engine
    picker needs: response_model would strip them, so widen the model when the UI wants them.
    """
    language = str(entry.get("LanguageCode") or "")
    gender = str(entry.get("Gender") or "")
    engines = sorted(str(e) for e in (entry.get("SupportedEngines") or []))
    tags = [
        *(["feminine"] if gender == "Female" else ["masculine"] if gender == "Male" else []),
        *engines,
    ]
    return {
        "id": str(entry.get("Id") or ""),
        "name": str(entry.get("Name") or entry.get("Id") or ""),
        "accent": ACCENTS.get(language, str(entry.get("LanguageName") or "") or None),
        "tags": tags,
        "use_cases": ["Storytelling", "Informative"]
        if "generative" in engines
        else ["Informative"],
        "provider": "polly",
        "engines": engines,
        "best_engine": best_engine(engines),
        "language_code": language,
        "language_name": str(entry.get("LanguageName") or ""),
        "gender": gender,
        "additional_language_codes": [str(c) for c in (entry.get("AdditionalLanguageCodes") or [])],
    }


def best_engine(engines: list[str] | tuple[str, ...] | frozenset[str]) -> str | None:
    """Highest-quality tier in `ENGINE_PREFERENCE` that this voice supports."""
    available = set(engines)
    return next((e for e in ENGINE_PREFERENCE if e in available), None)


def _voice_sort_key(voice: dict[str, Any]) -> tuple[int, int, str]:
    # Best tier first (generative before neural), then A-Z, matching the intent of
    # api/voices.py: put the voices worth reading a script with at the top.
    order = {name: index for index, name in enumerate(ENGINE_PREFERENCE)}
    return (
        order.get(voice.get("best_engine") or "", len(ENGINE_PREFERENCE)),
        0 if "long-form" in voice.get("engines", []) else 1,
        voice["name"].lower(),
    )


def list_voices(
    *,
    language_code: str | None = DEFAULT_LANGUAGE,
    engine: str | None = None,
    client: Any = None,
    use_cache: bool = True,
    **client_kwargs: Any,
) -> list[dict[str, Any]]:
    """Available Polly voices, best engine tier first.

    ``engine`` filters to voices supporting that tier. Results are cached per
    (language, engine) for the process lifetime — the catalogue does not change under us.
    """
    if engine is not None and engine not in ENGINES:
        raise PollyVoiceError(f"unknown engine {engine!r}; expected one of {sorted(ENGINES)}")

    key = f"{language_code or 'all'}:{engine or 'all'}"
    if use_cache and key in _voice_cache:
        return list(_voice_cache[key])

    polly = client or make_client(**client_kwargs)
    params: dict[str, Any] = {"IncludeAdditionalLanguageCodes": True}
    if language_code:
        params["LanguageCode"] = language_code
    if engine:
        params["Engine"] = engine

    try:
        entries: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            page = polly.describe_voices(**params, **({"NextToken": token} if token else {}))
            entries.extend(page.get("Voices") or [])
            token = page.get("NextToken")
            if not token:
                break
    except PollyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc, context="listing Polly voices") from exc

    voices = [shape_voice(entry) for entry in entries]
    voices.sort(key=_voice_sort_key)
    if use_cache:
        _voice_cache[key] = list(voices)
    return voices


# ------------------------------------------------------------------------ audio glue


def pcm_to_wav(pcm: bytes, out_path: Path, sample_rate: int) -> Path:
    """Wrap raw Polly pcm in a RIFF container. Signed 16-bit, mono, little-endian.

    stdlib `wave` rather than ffmpeg: the payload already *is* the samples, so this is a
    44-byte header and no decode step to lose anything in.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    out_path.write_bytes(buffer.getvalue())
    return out_path


def _transcode_to_wav(source: bytes, out_path: Path, suffix: str) -> Path:
    """mp3/ogg -> wav, for callers who want the 24 kHz that pcm cannot give us."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = out_path.parent / f".{out_path.stem}.polly{suffix}"
    scratch.write_bytes(source)
    try:
        run_ffmpeg(["-i", str(scratch), "-ac", "1", str(out_path)])
    finally:
        scratch.unlink(missing_ok=True)
    return out_path


# ------------------------------------------------------------------------- provider


class PollySynthesizer:
    """Satisfies `SpeechSynthesizer`, with SSML actually honoured.

    All constructor arguments are keyword-only with defaults, and ``voice`` / ``settings``
    are named to match what `app.worker.factory._construct` offers, so the factory can
    build this class without a signature change.
    """

    #: Not advisory — see `app.core.ports.SpeechSynthesizer`. Polly parses SSML; Aura
    #: reads the tags aloud.
    supports_ssml: bool = True

    def __init__(
        self,
        *,
        voice: str | None = None,
        engine: str | None = None,
        sample_rate: int | None = None,
        source_format: str = "auto",
        strict_ssml: bool = False,
        attempts: int = 4,
        base_delay: float = 0.5,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        client: Any = None,
        settings: Any = None,
    ) -> None:
        # `VIDEO_POLLY_ENGINE` is the deployment-level knob; DEFAULT_ENGINE is the answer
        # when config has nothing to say.
        if engine is None:
            engine = getattr(settings or get_settings(), "video_polly_engine", "") or DEFAULT_ENGINE
        if engine != "auto" and engine not in ENGINES:
            raise PollyVoiceError(
                f"unknown engine {engine!r}; expected 'auto' or one of {sorted(ENGINES)}"
            )
        if source_format not in {"auto", "pcm", "mp3", "ogg_vorbis"}:
            raise PollyError(
                f"source_format must be auto, pcm, mp3 or ogg_vorbis; got {source_format!r}"
            )

        self.engine = engine
        self.strict_ssml = strict_ssml
        self.attempts = max(1, attempts)
        self.base_delay = base_delay
        self.source_format = source_format
        self.sample_rate = sample_rate
        self.default_voice = self._resolve_default_voice(voice, settings)

        self._client = client
        self._client_kwargs = {
            "region_name": region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }
        #: What `adapt_ssml` changed on the last call. Handy in a job log.
        self.last_adaptations: list[str] = []
        #: The engine actually sent, which differs from `self.engine` when it is "auto".
        self.last_engine: str | None = None

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def _resolve_default_voice(voice: str | None, settings: Any) -> str:
        """Fall back to `VIDEO_DEFAULT_POLLY_VOICE`, refusing another vendor's id loudly.

        `app.worker.factory.SPEECH_ENGINES` gives this engine its own
        ``default_voice_setting``, so the Deepgram default never leaks in by accident. It
        can still arrive explicitly — a stored job row, or a caller reaching for
        ``VIDEO_DEFAULT_TTS_VOICE`` — and ``aura-2-draco-en`` would otherwise come back
        from the wire as a ValidationException listing a hundred voice names.
        """
        if not voice:
            config = settings or get_settings()
            voice = getattr(config, "video_default_polly_voice", "") or DEFAULT_VOICE
        if not re.fullmatch(r"[A-Z][A-Za-zÀ-ſ]*", voice):
            raise PollyVoiceError(
                f"{voice!r} is not an Amazon Polly voice id. Polly ids are capitalised "
                f"given names such as 'Matthew', 'Danielle' or 'Ruth' — "
                f"{voice!r} looks like another vendor's model name. Set "
                f"VIDEO_DEFAULT_POLLY_VOICE to a Polly voice, or pass voice= explicitly; "
                f"call list_voices() for the catalogue."
            )
        return voice

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(**self._client_kwargs)
        return self._client

    def _engine_for(self, voice: str) -> str:
        if self.engine != "auto":
            return self.engine
        for candidate in list_voices(language_code=None, client=self.client):
            if candidate["id"] == voice:
                chosen = candidate["best_engine"]
                if not chosen:
                    raise PollyVoiceError(f"voice {voice!r} supports no known engine")
                return chosen
        raise PollyVoiceError(
            f"voice {voice!r} is not offered by Polly in this region. "
            f"call list_voices() to see what is."
        )

    # -- the port ------------------------------------------------------------

    def synthesize(self, text: str, voice: str, out_path: Path) -> Path:
        """Text or SSML to an audio file on disk. See the module docstring for formats."""
        payload = (text or "").strip()
        if not payload:
            raise ValueError("cannot synthesize empty text")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        voice = voice or self.default_voice
        engine = self._engine_for(voice)
        self.last_engine = engine

        ssml = is_ssml(payload)
        if ssml:
            if self.strict_ssml:
                self.last_adaptations = []
            else:
                payload, self.last_adaptations = adapt_ssml(payload, engine)
            # Validate the payload we are actually sending, after any rewrite.
            validate_ssml(payload, engine)
        else:
            self.last_adaptations = []

        self._check_length(payload)

        output_format, sample_rate = self._format_for(out_path)
        audio = self._call(
            Text=payload,
            TextType="ssml" if ssml else "text",
            VoiceId=voice,
            Engine=engine,
            OutputFormat=output_format,
            SampleRate=str(sample_rate),
        )

        suffix = out_path.suffix.lower()
        if output_format == "pcm":
            # `_format_for` only chooses pcm for a wav (or extension-less) target.
            return pcm_to_wav(audio, out_path, sample_rate)
        if suffix in {"", ".wav"}:
            return _transcode_to_wav(audio, out_path, ".mp3" if output_format == "mp3" else ".ogg")
        out_path.write_bytes(audio)
        return out_path

    def duration(self, audio_path: Path) -> float:
        """Real audio length in seconds, from ffprobe — the pipeline's clock."""
        return audio_duration(audio_path)

    def synthesize_with_duration(self, text: str, voice: str, out_path: Path) -> tuple[Path, float]:
        """Mirrors `DeepgramSynthesizer` so the two are interchangeable at every call site."""
        path = self.synthesize(text, voice, out_path)
        return path, audio_duration(path)

    # -- internals -----------------------------------------------------------

    def _check_length(self, payload: str) -> None:
        total, billed = len(payload), billed_chars(payload)
        if total > MAX_TOTAL_CHARS or billed > MAX_BILLED_CHARS:
            raise TextTooLongError(
                f"input is {total} characters ({billed} billed); SynthesizeSpeech allows "
                f"{MAX_TOTAL_CHARS} total and {MAX_BILLED_CHARS} billed. Split it per "
                f"scene, or use StartSpeechSynthesisTask for long audio. Refusing rather "
                f"than truncating, which would silently drop narration."
            )

    def _format_for(self, out_path: Path) -> tuple[str, int]:
        """Pick OutputFormat/SampleRate from the requested file and the configured rate."""
        suffix = out_path.suffix.lower()
        rate = self.sample_rate

        if suffix in {".mp3", ".mpga"}:
            chosen = "mp3"
        elif suffix in {".ogg", ".oga"}:
            chosen = "ogg_vorbis"
        elif self.source_format != "auto":
            chosen = self.source_format
        else:
            # wav target: pcm is lossless but caps at 16 kHz, so a higher requested rate
            # means going via mp3 and transcoding.
            chosen = "pcm" if rate is None or rate <= 16000 else "mp3"

        if chosen == "pcm":
            rate = rate or 16000
            if rate not in PCM_SAMPLE_RATES:
                raise PollyError(
                    f"pcm supports only {sorted(PCM_SAMPLE_RATES)} Hz (measured: 24000 "
                    f"returns InvalidSampleRateException); got {rate}. Use source_format="
                    f"'mp3' for a higher rate."
                )
        else:
            rate = rate or 24000
            if rate not in MP3_SAMPLE_RATES:
                raise PollyError(
                    f"{chosen} supports only {sorted(MP3_SAMPLE_RATES)} Hz; got {rate}"
                )
        return chosen, rate

    def _call(self, **params: Any) -> bytes:
        """synthesize_speech with explicit backoff.

        botocore's own retries are disabled (`make_client` sets max_attempts=1) so that a
        429 is retried here, where the jitter is visible and a test can drive it.
        """
        last: PollyError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = self.client.synthesize_speech(**params)
            except PollyError as exc:
                raise exc
            except Exception as exc:  # noqa: BLE001 - botocore ClientError and friends
                error = _translate(
                    exc,
                    context=f"synthesizing {params.get('VoiceId')!r} on the "
                    f"{params.get('Engine')} engine",
                    payload=params.get("Text"),
                )
                code = str(((getattr(exc, "response", None) or {}).get("Error") or {}).get("Code"))
                if code in _RETRYABLE and attempt < self.attempts:
                    last = error
                    delay = self.base_delay * (2 ** (attempt - 1))
                    time.sleep(delay + random.uniform(0, self.base_delay))
                    logger.warning(
                        "polly: %s on attempt %d/%d, retrying", code, attempt, self.attempts
                    )
                    continue
                raise error from exc

            stream = response.get("AudioStream")
            if stream is None:
                raise PollyError(f"Polly returned no AudioStream: {response!r}")
            audio = stream.read()
            if not audio:
                raise PollyError("Polly returned an empty AudioStream")
            return audio

        raise last or PollyThrottledError(f"exhausted {self.attempts} attempts")
