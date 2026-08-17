"""Voice catalogue, per speech engine.

Deepgram's `/v1/models` returns ~100 TTS voices across languages. We keep the English
Aura voices (`canonical_name` ends in `-en`) and surface the curated `metadata` the
picker needs. Narration voices come first: a voice Deepgram tags for storytelling or
informative content reads an explainer far better than one tuned for drive-thru orders.

Polly's catalogue comes from `app.providers.polly_tts`, filtered to the voice tier in
`VIDEO_POLLY_ENGINE` — Polly refuses `Engine=generative` for a voice that only supports
neural, so offering one would hand the user a choice that fails at synthesis.

Each engine's list is immutable for the life of the process, so it is fetched once and
cached per engine. If an upstream is unreachable the endpoint still answers with a
documented fallback — an offline dev box should not lose its voice picker. Availability of
the engine itself is reported separately and honestly by GET /api/engines; a fallback list
here is a usable picker, not a claim that the engine works.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import get_settings
from app.worker import factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["voices"])

DEEPGRAM_MODELS_URL = "https://api.deepgram.com/v1/models"

DEEPGRAM_ENGINE = "deepgram"
POLLY_ENGINE = "polly"

#: Polly serves ~100 voices across languages; the narration pipeline is English.
POLLY_LANGUAGE = "en-US"

#: use_cases that mean "good at reading a narration script"
PREFERRED_USE_CASES = ("storytelling", "informative")

FALLBACK_VOICES: list[dict[str, Any]] = [
    {
        "id": "aura-2-draco-en",
        "name": "Draco",
        "accent": "British",
        "tags": ["masculine", "warm", "trustworthy"],
        "use_cases": ["Storytelling", "Informative"],
    },
    {
        "id": "aura-2-pluto-en",
        "name": "Pluto",
        "accent": "American",
        "tags": ["masculine", "calm", "baritone"],
        "use_cases": ["Storytelling", "Informative"],
    },
    {
        "id": "aura-2-hera-en",
        "name": "Hera",
        "accent": "American",
        "tags": ["feminine", "warm", "confident"],
        "use_cases": ["Informative", "Customer Service"],
    },
]

#: Polly's en-US voices as `polly:DescribeVoices` returned them on 2026-08-17, with the
#: tiers each one actually supports. Two uses:
#:   * the fallback catalogue when `app.providers.polly_tts` cannot be imported;
#:   * voice->engine attribution for POST /api/jobs, which must recognise a Polly name
#:     even for tiers this deployment does not offer.
POLLY_VOICES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Danielle", "feminine", ("generative", "long-form", "neural")),
    ("Gregory", "masculine", ("long-form", "neural")),
    ("Ivy", "feminine", ("neural", "standard")),
    ("Joanna", "feminine", ("generative", "neural", "standard")),
    ("Joey", "masculine", ("neural", "standard")),
    ("Justin", "masculine", ("neural", "standard")),
    ("Kendra", "feminine", ("neural", "standard")),
    ("Kevin", "masculine", ("neural",)),
    ("Kimberly", "feminine", ("neural", "standard")),
    ("Matthew", "masculine", ("generative", "neural", "standard")),
    ("Patrick", "masculine", ("long-form",)),
    ("Ruth", "feminine", ("generative", "long-form", "neural")),
    ("Salli", "feminine", ("generative", "neural", "standard")),
    ("Stephen", "masculine", ("generative", "neural")),
    ("Tiffany", "feminine", ("generative",)),
)

#: Attribute names a Polly provider might expose its catalogue under. The module is owned
#: by another branch, so probe rather than hard-code one spelling.
POLLY_CATALOGUE_ATTRS = (
    "list_voices",
    "voice_catalogue",
    "available_voices",
    "list_polly_voices",
    "describe_voices",
    "catalogue",
    "voices",
    "POLLY_VOICES",
    "VOICES",
)

#: engine id -> catalogue. Populated on first request per engine.
_cache: dict[str, list[dict[str, Any]]] = {}


class Voice(BaseModel):
    id: str
    name: str
    accent: str | None = None
    tags: list[str] = []
    use_cases: list[str] = []


def reset_cache() -> None:
    """Test seam."""
    _cache.clear()


def _is_preferred(use_cases: list[str]) -> bool:
    lowered = " ".join(use_cases).lower()
    return any(kind in lowered for kind in PREFERRED_USE_CASES)


def _shape(entry: dict[str, Any]) -> dict[str, Any]:
    meta = entry.get("metadata") or {}
    canonical = str(entry.get("canonical_name", ""))
    # Deepgram sends metadata.display_name: null for the English voices, so fall back to
    # the bare model name ("arcas") and title-case it for the picker.
    raw_name = str(entry.get("name") or canonical)
    return {
        "id": canonical,
        "name": meta.get("display_name") or raw_name.replace("_", " ").title(),
        "accent": meta.get("accent"),
        "tags": [str(t) for t in (meta.get("tags") or [])],
        "use_cases": [str(u) for u in (meta.get("use_cases") or [])],
    }


def _sort_key(voice: dict[str, Any]) -> tuple[int, int, str]:
    # narration-suited first, then the current-generation aura-2 models, then A-Z
    return (
        0 if _is_preferred(voice["use_cases"]) else 1,
        0 if voice["id"].startswith("aura-2-") else 1,
        voice["name"].lower(),
    )


def parse_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """English TTS voices, narration-friendly ones first. Pure — unit testable."""
    voices = [
        _shape(entry)
        for entry in payload.get("tts") or []
        if str(entry.get("canonical_name", "")).endswith("-en")
    ]
    voices.sort(key=_sort_key)
    return voices


def _shape_polly(entry: Any) -> dict[str, Any] | None:
    """Normalise one provider catalogue entry.

    Accepts our own shape (`id`/`name`), Polly's raw `DescribeVoices` shape (`Id`/`Gender`
    /`SupportedEngines`) or a bare voice-id string, because the provider module is written
    on another branch and only its data matters here, not its spelling.
    """
    if isinstance(entry, str):
        return {"id": entry, "name": entry, "accent": None, "tags": [], "use_cases": []}
    if not isinstance(entry, dict):
        entry = getattr(entry, "__dict__", None) or {}
        if not entry:
            return None

    voice_id = str(entry.get("id") or entry.get("Id") or entry.get("voice") or "")
    if not voice_id:
        return None
    gender = str(entry.get("gender") or entry.get("Gender") or "").lower()
    tiers = entry.get("engines") or entry.get("SupportedEngines") or entry.get("tiers") or ()
    tags = [str(t) for t in (entry.get("tags") or [])]
    if not tags:
        if gender.startswith("f"):
            tags.append("feminine")
        elif gender.startswith("m"):
            tags.append("masculine")
        tags += [str(t) for t in tiers]
    return {
        "id": voice_id,
        "name": str(entry.get("name") or entry.get("Name") or voice_id),
        "accent": entry.get("accent")
        or ("American" if str(entry.get("LanguageCode", "")).startswith("en-US") else None),
        "tags": tags,
        "use_cases": [str(u) for u in (entry.get("use_cases") or [])],
    }


def _polly_fallback(tier: str) -> list[dict[str, Any]]:
    """Measured en-US voices supporting `tier`, default voice first."""
    default = get_settings().video_default_polly_voice
    voices = [
        {
            "id": name,
            "name": name,
            "accent": "American",
            "tags": [gender, *tiers],
            "use_cases": [],
        }
        for name, gender, tiers in POLLY_VOICES
        if tier in tiers
    ]
    voices.sort(key=lambda v: (0 if v["id"] == default else 1, v["name"]))
    return voices


def _polly_provider_voices(tier: str) -> list[dict[str, Any]] | None:
    """Polly's catalogue from its provider module, or None while that is unavailable.

    Filtered to `tier`, and that filter is load-bearing: `PollySynthesizer` sends the
    configured tier verbatim, so offering a standard-only voice while `VIDEO_POLLY_ENGINE`
    is `generative` hands the user a choice Polly rejects with a ValidationException.
    """
    try:
        import app.providers.polly_tts as polly
    except Exception as exc:  # noqa: BLE001 - the whole providers package imports eagerly
        logger.info("polly voice catalogue unavailable (%s); serving measured fallback", exc)
        return None

    for attr in POLLY_CATALOGUE_ATTRS:
        source = getattr(polly, attr, None)
        if source is None:
            continue
        try:
            raw = _call_catalogue(source, tier)
        except Exception as exc:  # noqa: BLE001 - a live AWS call can fail any number of ways
            logger.warning("polly_tts.%s failed (%s); serving measured fallback", attr, exc)
            return None
        shaped = [v for v in (_shape_polly(entry) for entry in raw or []) if v]
        if shaped:
            logger.info("polly voice catalogue from polly_tts.%s (%d voices)", attr, len(shaped))
            return shaped
    logger.info(
        "polly_tts exposes no catalogue (looked for %s); serving measured fallback",
        ", ".join(POLLY_CATALOGUE_ATTRS),
    )
    return None


def _call_catalogue(source: Any, tier: str) -> Any:
    """Invoke a provider catalogue, preferring the tier-filtered call.

    `polly_tts.list_voices(language_code=..., engine=...)` is the shape that exists today;
    the bare call is the fallback for a provider that spells it differently, and a plain
    list constant is accepted as-is.
    """
    if not callable(source):
        return source
    engine = tier if tier and tier != "auto" else None
    try:
        return source(language_code=POLLY_LANGUAGE, engine=engine)
    except TypeError:
        return source()


async def fetch_polly_voices() -> list[dict[str, Any]]:
    cached = _cache.get(POLLY_ENGINE)
    if cached is not None:
        return cached

    tier = get_settings().video_polly_engine
    voices = _polly_provider_voices(tier)
    if voices is None:
        # Not cached: the provider module may land while this process is running, and a
        # cached fallback would outlive the reason for it.
        return _polly_fallback(tier)
    _cache[POLLY_ENGINE] = voices
    return voices


def engine_for_voice(voice: str) -> str | None:
    """Which engine a voice id belongs to, or None if it cannot be attributed.

    Attribution is by ownership, not catalogue membership, and that is deliberate. The
    Deepgram list is fetched live and degrades to three entries when the network is down,
    so rejecting anything absent from it would reject 50 valid voices on a bad DNS day.
    A prefix/name match, by contrast, only ever fires on the real bug: sending
    `aura-2-draco-en` to Polly, or `Matthew` to Deepgram.
    """
    candidate = (voice or "").strip()
    if not candidate:
        return None
    if candidate.lower().startswith("aura"):
        return DEEPGRAM_ENGINE
    if candidate.lower() in {name.lower() for name, _, _ in POLLY_VOICES}:
        return POLLY_ENGINE
    return None


async def fetch_voices() -> list[dict[str, Any]]:
    cached = _cache.get(DEEPGRAM_ENGINE)
    if cached is not None:
        return cached

    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                DEEPGRAM_MODELS_URL,
                headers={"Authorization": f"Token {settings.deepgram_api_key}"},
            )
            response.raise_for_status()
            voices = parse_models(response.json())
        if not voices:
            raise ValueError("Deepgram returned no English TTS voices")
    except Exception as exc:  # noqa: BLE001 - any upstream problem degrades gracefully
        logger.warning("voice list unavailable (%s); serving fallback", exc)
        return list(FALLBACK_VOICES)

    _cache[DEEPGRAM_ENGINE] = voices
    return voices


@router.get("/voices", response_model=list[Voice])
async def list_voices(
    engine: str | None = Query(
        default=None,
        description="Speech engine id from GET /api/engines. Omit for the default engine.",
    ),
) -> list[dict[str, Any]]:
    """Voices for one engine.

    An unknown engine is a 422 rather than a silent fall back to the default: this list
    populates a picker, and quietly answering with another engine's voices would guarantee
    the user selects one that POST /api/jobs then rejects.
    """
    requested = (engine or "").strip().lower()
    if not requested:
        requested = factory.default_speech_engine()
    elif requested not in factory.speech_engine_ids():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_engine",
                "message": f"unknown speech engine {engine!r}",
                "known_engines": list(factory.speech_engine_ids()),
            },
        )

    if requested == POLLY_ENGINE:
        return await fetch_polly_voices()
    return await fetch_voices()
