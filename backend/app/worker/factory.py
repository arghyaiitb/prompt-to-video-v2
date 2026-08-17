"""Provider resolution. Selection is config data; wiring is one lazy import each.

Every import happens *inside* a function on purpose. The provider and render packages are
developed independently, so importing them at module scope would make the whole API
un-importable whenever one of them is mid-edit. A missing provider must surface as a
failed job with a readable message, not as a dead web server.

Constructor contract we assume (see `_construct`): concrete classes either take no
required arguments (reading `get_settings()` themselves) or accept some subset of
``settings`` / ``api_key`` / ``model`` / ``voice``. Anything else raises
ProviderUnavailableError with the offending parameter named.
"""

from __future__ import annotations

import inspect
from typing import Any

from app.core.config import Settings, get_settings
from app.core.ports import (
    Aligner,
    ImageProvider,
    MusicProvider,
    ScriptProvider,
    SpeechSynthesizer,
    VideoBackend,
    VisualPlanner,
)


class ProviderUnavailableError(RuntimeError):
    """A configured provider could not be imported or constructed."""


def _load(module_path: str, class_name: str, role: str) -> type:
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as exc:  # module not written yet / broken dependency
        raise ProviderUnavailableError(
            f"{role} provider not yet available: cannot import {module_path} ({exc})"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ProviderUnavailableError(
            f"{role} provider not yet available: {module_path} has no {class_name}"
        ) from exc


def _construct(cls: type, role: str, **candidates: Any) -> Any:
    """Instantiate `cls`, passing only the kwargs its signature actually declares."""
    settings = get_settings()
    pool: dict[str, Any] = {"settings": settings, **candidates}
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # builtins / C types — just try it bare
        params = {}

    kwargs = {name: value for name, value in pool.items() if name in params}
    missing = [
        name
        for name, p in params.items()
        if name not in kwargs
        and p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if missing:
        raise ProviderUnavailableError(
            f"{role} provider {cls.__name__} needs constructor arg(s) "
            f"{missing} that the factory cannot supply"
        )
    try:
        return cls(**kwargs)
    except ProviderUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any init failure as one error type
        raise ProviderUnavailableError(
            f"{role} provider {cls.__name__} failed to initialise: {exc}"
        ) from exc


def script_provider(settings: Settings | None = None) -> ScriptProvider:
    settings = settings or get_settings()
    if settings.video_default_llm_provider != "gemini":
        raise ProviderUnavailableError(
            f"no script provider for VIDEO_DEFAULT_LLM_PROVIDER="
            f"{settings.video_default_llm_provider!r}"
        )
    cls = _load("app.providers.gemini_script", "GeminiScriptProvider", "script")
    return _construct(
        cls, "script", api_key=settings.gemini_api_key, model=settings.video_default_llm_model
    )


def image_provider(settings: Settings | None = None) -> ImageProvider:
    settings = settings or get_settings()
    cls = _load("app.providers.gemini_image", "GeminiImageProvider", "image")
    return _construct(
        cls, "image", api_key=settings.gemini_api_key, model=settings.video_default_image_model
    )


def speech_synthesizer(settings: Settings | None = None) -> SpeechSynthesizer:
    settings = settings or get_settings()
    if settings.video_default_tts_provider != "deepgram":
        raise ProviderUnavailableError(
            f"no TTS provider for VIDEO_DEFAULT_TTS_PROVIDER="
            f"{settings.video_default_tts_provider!r}"
        )
    cls = _load("app.providers.deepgram_tts", "DeepgramSynthesizer", "tts")
    return _construct(
        cls, "tts", api_key=settings.deepgram_api_key, voice=settings.video_default_tts_voice
    )


def aligner(settings: Settings | None = None) -> Aligner:
    settings = settings or get_settings()
    if settings.video_default_aligner != "deepgram":
        raise ProviderUnavailableError(
            f"no aligner for VIDEO_DEFAULT_ALIGNER={settings.video_default_aligner!r}"
        )
    cls = _load("app.providers.deepgram_align", "DeepgramAligner", "aligner")
    return _construct(
        cls,
        "aligner",
        api_key=settings.deepgram_api_key,
        model=settings.video_default_aligner_model,
    )


def music_provider(settings: Settings | None = None) -> MusicProvider:
    settings = settings or get_settings()
    cls = _load("app.providers.lyria_music", "LyriaMusicProvider", "music")
    return _construct(
        cls, "music", api_key=settings.gemini_api_key, model=settings.video_default_music_model
    )


def visual_planner(settings: Settings | None = None) -> VisualPlanner:
    cls = _load("app.render.planner", "RuleBasedPlanner", "planner")
    return _construct(cls, "planner")


def video_backend(settings: Settings | None = None) -> VideoBackend:
    cls = _load("app.render.ffmpeg_backend", "FFmpegBackend", "render")
    return _construct(cls, "render")
