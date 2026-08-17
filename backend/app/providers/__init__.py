"""Concrete adapters for the external services behind `app.core.ports`.

Nothing in here is imported by the core. Each class satisfies one Protocol structurally,
so a vendor swap is a new module plus a config value — see `app/core/ports.py`.

    ScriptProvider      GeminiScriptProvider, VerbatimScriptProvider
    ImageProvider       GeminiImageProvider, PlaceholderImageProvider
    SpeechSynthesizer   DeepgramSynthesizer (text only), PollySynthesizer (SSML)
    Aligner             DeepgramAligner
    MusicProvider       LyriaMusicProvider
"""

from __future__ import annotations

from app.providers._gemini import GeminiError
from app.providers._media import MediaError, audio_duration, image_dimensions
from app.providers.bullet_timing import BulletAnchor, anchor_position, find_anchors, time_bullets
from app.providers.deepgram_align import (
    AlignmentError,
    DeepgramAligner,
    align_tokens,
    normalize,
    tokenize,
)
from app.providers.deepgram_tts import DeepgramSynthesizer, SynthesisError, probe_duration
from app.providers.gemini_image import (
    GeminiImageProvider,
    PlaceholderImageProvider,
    nearest_ratio,
)
from app.providers.gemini_script import (
    GeminiScriptProvider,
    VerbatimScriptProvider,
)
from app.providers.lyria_music import LyriaMusicProvider, MusicError, loop_count
from app.providers.polly_tts import (
    PollyCredentialsError,
    PollyError,
    PollyRegionError,
    PollySynthesizer,
    PollyThrottledError,
    PollyVoiceError,
    SsmlError,
    TextTooLongError,
    adapt_ssml,
    best_engine,
    billed_chars,
    is_ssml,
    validate_ssml,
)

# Aliased on the way out: `list_voices` is already the name of the FastAPI route handler in
# `app.api.voices`, and a bare re-export would read as that endpoint.
from app.providers.polly_tts import list_voices as list_polly_voices
from app.providers.veo_video import (
    PlaceholderVideoProvider,
    VeoVideoProvider,
    VideoClipError,
    clip_budget,
    veo_enabled,
)

__all__ = [
    "AlignmentError",
    "BulletAnchor",
    "DeepgramAligner",
    "DeepgramSynthesizer",
    "GeminiError",
    "GeminiImageProvider",
    "GeminiScriptProvider",
    "LyriaMusicProvider",
    "MediaError",
    "MusicError",
    "PlaceholderImageProvider",
    "PlaceholderVideoProvider",
    "PollyCredentialsError",
    "PollyError",
    "PollyRegionError",
    "PollySynthesizer",
    "PollyThrottledError",
    "PollyVoiceError",
    "SsmlError",
    "SynthesisError",
    "TextTooLongError",
    "VeoVideoProvider",
    "VerbatimScriptProvider",
    "VideoClipError",
    "adapt_ssml",
    "align_tokens",
    "anchor_position",
    "audio_duration",
    "best_engine",
    "billed_chars",
    "clip_budget",
    "find_anchors",
    "image_dimensions",
    "is_ssml",
    "list_polly_voices",
    "loop_count",
    "nearest_ratio",
    "normalize",
    "probe_duration",
    "time_bullets",
    "tokenize",
    "validate_ssml",
    "veo_enabled",
]
