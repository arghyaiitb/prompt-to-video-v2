"""Speech engine catalogue for the narration picker.

Two fields here carry weight far beyond their size.

`supports_ssml` tells the caller which text the pipeline will send. Deepgram Aura does not
parse SSML, it VOCALISES it: `<speak>Check the sender.<break time="800ms"/>Then hover the
link.</speak>` came back through STT as *"Speak. Check the sender. Break time equals eight
hundred milliseconds. Then hover the link."* So `Scene.ssml` goes only to an engine that
declares support, and everything else gets `Scene.narration`.

`available` is a measured fact, not a declaration. It is false unless the credentials that
engine needs are actually present, its SDK imports, and its provider module loads. An
engine offered but broken costs the user a multi-minute render before it fails, which is
strictly worse than an engine that is greyed out with a reason attached.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.worker import factory

router = APIRouter(prefix="/api", tags=["engines"])


class EngineOut(BaseModel):
    id: str
    name: str

    supports_ssml: bool
    """Whether this engine parses SSML. False means it would read the tags aloud."""

    available: bool
    """Verified: credentials present, SDK importable, provider module loadable."""

    default: bool
    """True for the engine a job gets when it names none."""

    default_voice: str
    """Voice used when a job selects this engine without naming one."""

    reason: str | None = None
    """Why `available` is false — shown as the picker's disabled-option tooltip."""


@router.get("/engines", response_model=list[EngineOut])
def list_engines() -> list[EngineOut]:
    settings = get_settings()
    default_engine = factory.default_speech_engine(settings)
    out: list[EngineOut] = []
    for spec in factory.speech_engines():
        available, reason = factory.speech_engine_status(spec.id, settings)
        out.append(
            EngineOut(
                id=spec.id,
                name=spec.name,
                supports_ssml=spec.supports_ssml,
                available=available,
                default=spec.id == default_engine,
                default_voice=factory.default_voice(spec.id, settings),
                reason=reason,
            )
        )
    return out
