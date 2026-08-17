"""Shared transport for the three Google Generative Language endpoints we use.

Script, image, and music all go through `models/{model}:generateContent`; only the
request body and which part type we harvest differ. Centralising the call means the
retry policy and the `parts` quirks are fixed in exactly one place.

VERIFIED parts quirk (2026-08-17, live key): a single part may carry `inlineData`
*and* `thoughtSignature` side by side, and text responses may include parts with no
`text` key at all. Never index parts positionally; always filter by key presence.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class GeminiError(RuntimeError):
    """Non-retryable API failure. Message includes the server's own explanation."""


def generate_content(
    model: str,
    body: dict[str, Any],
    api_key: str,
    *,
    timeout: float = 180.0,
    attempts: int = 3,
) -> dict[str, Any]:
    """POST to generateContent, retrying transient statuses with linear backoff."""
    if not api_key:
        raise GeminiError("gemini_api_key is empty — set GEMINI_API_KEY in .env")

    url = f"{BASE_URL}/{model}:generateContent"
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(
                url,
                params={"key": api_key},
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            last_error = f"transport error: {exc}"
            if attempt == attempts:
                raise GeminiError(f"{model}: {last_error}") from exc
            time.sleep(2.0 * attempt)
            continue

        if response.status_code == 200:
            return response.json()

        last_error = f"HTTP {response.status_code}: {response.text[:800]}"
        if response.status_code in RETRY_STATUS and attempt < attempts:
            time.sleep(2.0 * attempt)
            continue
        raise GeminiError(f"{model}: {last_error}")

    raise GeminiError(f"{model}: exhausted {attempts} attempts — {last_error}")


def _parts(response: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = response.get("candidates") or []
    if not candidates:
        feedback = response.get("promptFeedback") or {}
        raise GeminiError(f"no candidates returned (promptFeedback={feedback})")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts")
    if not parts:
        raise GeminiError(
            f"candidate had no parts (finishReason={candidate.get('finishReason')!r})"
        )
    return parts


def text_from(response: dict[str, Any]) -> str:
    """Concatenate only the parts that actually carry text.

    Reasoning models interleave metadata-only parts (`thoughtSignature`); treating
    parts[0] as the answer silently yields an empty string or a KeyError.
    """
    chunks = [p["text"] for p in _parts(response) if isinstance(p.get("text"), str)]
    if not chunks:
        raise GeminiError("response contained no text parts")
    return "".join(chunks)


def inline_data_from(response: dict[str, Any]) -> tuple[str, bytes]:
    """First binary payload as (mime_type, decoded bytes).

    Accepts both camelCase (REST) and snake_case (SDK-style) key spellings.
    """
    for part in _parts(response):
        inline = part.get("inlineData") or part.get("inline_data")
        if not inline:
            continue
        data = inline.get("data")
        if not data:
            continue
        mime = inline.get("mimeType") or inline.get("mime_type") or ""
        return mime, base64.b64decode(data)
    raise GeminiError("response contained no inlineData parts")
