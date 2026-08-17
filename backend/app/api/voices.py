"""Voice catalogue, sourced from Deepgram's live model list.

Deepgram's `/v1/models` returns ~100 TTS voices across languages. We keep the English
Aura voices (`canonical_name` ends in `-en`) and surface the curated `metadata` the
picker needs. Narration voices come first: a voice Deepgram tags for storytelling or
informative content reads an explainer far better than one tuned for drive-thru orders.

The list is immutable for the life of the process, so it is fetched once and cached. If
Deepgram is unreachable the endpoint still answers with the three voices the pipeline
defaults document — an offline dev box should not lose its voice picker.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["voices"])

DEEPGRAM_MODELS_URL = "https://api.deepgram.com/v1/models"

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

_cache: list[dict[str, Any]] | None = None


class Voice(BaseModel):
    id: str
    name: str
    accent: str | None = None
    tags: list[str] = []
    use_cases: list[str] = []


def reset_cache() -> None:
    """Test seam."""
    global _cache
    _cache = None


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


async def fetch_voices() -> list[dict[str, Any]]:
    global _cache
    if _cache is not None:
        return _cache

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

    _cache = voices
    return _cache


@router.get("/voices", response_model=list[Voice])
async def list_voices() -> list[dict[str, Any]]:
    return await fetch_voices()
