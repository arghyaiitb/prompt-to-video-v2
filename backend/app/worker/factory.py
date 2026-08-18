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

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.core.models import Theme
from app.core.ports import (
    Aligner,
    ImageProvider,
    MusicProvider,
    ScriptProvider,
    SpeechSynthesizer,
    VideoBackend,
    VideoClipProvider,
    VisualPlanner,
)

logger = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    """A configured provider could not be imported or constructed."""


@dataclass(frozen=True)
class SpeechEngine:
    """One selectable narration engine, declared without importing its provider.

    `supports_ssml` is duplicated here from the provider class on purpose: the engine
    list is served over HTTP to a browser, and answering it must not require credentials,
    a boto3 install, or a provider module that another branch is still writing. The
    *authoritative* read at synthesis time is the instance attribute — see
    `app.core.ports.SpeechSynthesizer` — and the two must agree.
    """

    id: str
    name: str
    module: str
    class_name: str
    supports_ssml: bool
    default_voice_setting: str
    """Name of the `Settings` field holding this engine's default voice."""


#: Every engine POST /api/jobs will accept. Order is the order the picker shows.
SPEECH_ENGINES: tuple[SpeechEngine, ...] = (
    SpeechEngine(
        id="deepgram",
        name="Deepgram Aura 2",
        module="app.providers.deepgram_tts",
        class_name="DeepgramSynthesizer",
        # Measured: Aura does not parse SSML, it reads the tags out loud.
        supports_ssml=False,
        default_voice_setting="video_default_tts_voice",
    ),
    SpeechEngine(
        id="polly",
        name="AWS Polly",
        module="app.providers.polly_tts",
        class_name="PollySynthesizer",
        supports_ssml=True,
        default_voice_setting="video_default_polly_voice",
    ),
)

_ENGINES_BY_ID: dict[str, SpeechEngine] = {engine.id: engine for engine in SPEECH_ENGINES}

#: Used when `VIDEO_DEFAULT_TTS_ENGINE` names an engine that does not exist. Deepgram
#: because it is the engine every other default in this file is measured against.
FALLBACK_SPEECH_ENGINE = "deepgram"


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


# --------------------------------------------------------------------- speech engines


def speech_engines() -> tuple[SpeechEngine, ...]:
    return SPEECH_ENGINES


def speech_engine_ids() -> tuple[str, ...]:
    return tuple(_ENGINES_BY_ID)


def speech_engine(engine_id: str) -> SpeechEngine | None:
    return _ENGINES_BY_ID.get((engine_id or "").strip().lower())


def default_speech_engine(settings: Settings | None = None) -> str:
    """The engine id new jobs get, guaranteed to be one we can actually resolve."""
    settings = settings or get_settings()
    requested = (settings.video_default_tts_engine or "").strip().lower()
    if requested in _ENGINES_BY_ID:
        return requested
    logger.warning(
        "VIDEO_DEFAULT_TTS_ENGINE=%r is not a known engine; using %r",
        settings.video_default_tts_engine,
        FALLBACK_SPEECH_ENGINE,
    )
    return FALLBACK_SPEECH_ENGINE


def resolve_speech_engine(requested: str | None, settings: Settings | None = None) -> str:
    """Normalise a caller-supplied engine id. Unknown ids fall back to the default.

    Same contract as `app.api.jobs._resolve_theme_name`: the *resolved* value is what gets
    stored, so a job row never claims an engine that was not used.
    """
    settings = settings or get_settings()
    normalised = (requested or "").strip().lower()
    if not normalised:
        return default_speech_engine(settings)
    if normalised not in _ENGINES_BY_ID:
        return default_speech_engine(settings)
    return normalised


def default_voice(engine_id: str | None = None, settings: Settings | None = None) -> str:
    """This engine's default voice. Falls back to the Deepgram default for unknown ids."""
    settings = settings or get_settings()
    spec = speech_engine(engine_id or "") or _ENGINES_BY_ID[FALLBACK_SPEECH_ENGINE]
    return str(getattr(settings, spec.default_voice_setting, "") or "")


def _import_error(module_path: str) -> str | None:
    """None if `module_path` imports, else a one-line reason.

    Broad except on purpose: `app.providers.__init__` imports every adapter eagerly, so a
    sibling module mid-edit can raise SyntaxError here. An availability probe must answer
    "no, because X" rather than take the API down — hence the exception text is kept, or
    the reason would blame the wrong module.
    """
    try:
        importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - see docstring
        return f"cannot import {module_path}: {type(exc).__name__}: {exc}"
    return None


def speech_engine_status(
    engine_id: str, settings: Settings | None = None
) -> tuple[bool, str | None]:
    """``(available, reason_unavailable)`` — a real check, not a declaration.

    GET /api/engines renders this, and the frontend enables an engine because of it.
    Telling a user an engine works when its credentials are absent buys a job that fails
    minutes later, so every branch here verifies something concrete.
    """
    settings = settings or get_settings()
    spec = speech_engine(engine_id)
    if spec is None:
        return False, f"unknown engine {engine_id!r}"

    if spec.id == "deepgram" and not settings.deepgram_api_key:
        return False, "DEEPGRAM_API_KEY is not set"

    if spec.id == "polly":
        if not settings.aws_configured:
            return False, "AWS credentials are not configured (AWS_ACCESS_KEY_ID/SECRET)"
        if _import_error("boto3") is not None:
            return False, "boto3 is not installed"

    reason = _import_error(spec.module)
    if reason is not None:
        return False, reason
    return True, None


def speech_engine_available(engine_id: str, settings: Settings | None = None) -> bool:
    return speech_engine_status(engine_id, settings)[0]


def speech_synthesizer(
    settings: Settings | None = None, engine: str | None = None
) -> SpeechSynthesizer:
    """The narration engine for this job.

    `engine` is the per-job choice persisted on `Job.tts_engine`; None means "the
    configured default". Callers must consult `.supports_ssml` on the returned object
    before handing it `Scene.ssml` — see the `SpeechSynthesizer` docstring.
    """
    # Tolerate a positional engine id. The documented call is `speech_synthesizer(engine=...)`
    # but every other factory function here takes `settings` first, so both spellings turn
    # up in callers; silently treating "polly" as a Settings object would fail obscurely.
    if isinstance(settings, str):
        settings, engine = None, settings
    settings = settings or get_settings()

    requested = (engine or "").strip().lower() or default_speech_engine(settings)
    spec = speech_engine(requested)
    if spec is None:
        raise ProviderUnavailableError(
            f"unknown speech engine {requested!r}; known engines are "
            f"{', '.join(speech_engine_ids())}"
        )

    available, reason = speech_engine_status(spec.id, settings)
    if not available:
        raise ProviderUnavailableError(f"speech engine {spec.id!r} is not available: {reason}")

    cls = _load(spec.module, spec.class_name, f"tts:{spec.id}")
    return _construct(
        cls,
        f"tts:{spec.id}",
        api_key=settings.deepgram_api_key if spec.id == "deepgram" else None,
        # Deepgram only: 48 kHz is real bandwidth there (see the setting's docstring).
        # Polly's PCM path maxes at 16 kHz, so it gets None and keeps its own default.
        sample_rate=(
            settings.video_deepgram_sample_rate if spec.id == "deepgram" else None
        ),
        voice=default_voice(spec.id, settings),
        default_voice=default_voice(spec.id, settings),
        region=settings.aws_region,
        region_name=settings.aws_region,
        # Polly's own name for the voice tier. `_construct` drops any kwarg the
        # constructor does not declare, so this is inert for Deepgram.
        engine=settings.video_polly_engine,
        polly_engine=settings.video_polly_engine,
        tier=settings.video_polly_engine,
    )


def video_clip_provider(settings: Settings | None = None) -> VideoClipProvider:
    """Generated moving footage. Gated on `video_enable_veo`, which defaults to False.

    The gate is here rather than inside the provider because constructing this class IS
    the decision to spend: Veo is the most expensive call in the pipeline. A caller that
    wants a still should never reach this function.
    """
    settings = settings or get_settings()
    if not settings.video_enable_veo:
        raise ProviderUnavailableError(
            "video clips are disabled; set VIDEO_ENABLE_VEO=true to generate them"
        )
    if not settings.gemini_api_key:
        raise ProviderUnavailableError("GEMINI_API_KEY is not set — Veo cannot authenticate")
    cls = _load("app.providers.veo_video", "VeoVideoProvider", "video clip")
    return _construct(
        cls,
        "video clip",
        api_key=settings.gemini_api_key,
        model=settings.video_default_video_model,
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


def video_backend(settings: Settings | None = None, theme: Theme | None = None) -> VideoBackend:
    """The render backend, optionally on a caller-chosen palette.

    ``theme`` is passed through ``_construct``, so a backend whose constructor does not
    declare it simply renders on its own default — the same graceful-degradation rule as
    every other provider argument here.
    """
    cls = _load("app.render.ffmpeg_backend", "FFmpegBackend", "render")
    return _construct(cls, "render", theme=theme)
