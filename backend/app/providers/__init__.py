"""Concrete adapters for the external services behind `app.core.ports`.

Nothing in here is imported by the core. Each class satisfies one Protocol structurally,
so a vendor swap is a new module plus a config value — see `app/core/ports.py`.

    ScriptProvider      GeminiScriptProvider, VerbatimScriptProvider
    ImageProvider       GeminiImageProvider, PlaceholderImageProvider
    SpeechSynthesizer   DeepgramSynthesizer
    Aligner             DeepgramAligner
    MusicProvider       LyriaMusicProvider
"""

from __future__ import annotations

from app.providers._gemini import GeminiError
from app.providers._media import MediaError, audio_duration, image_dimensions
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

__all__ = [
    "AlignmentError",
    "DeepgramAligner",
    "DeepgramSynthesizer",
    "GeminiError",
    "GeminiImageProvider",
    "GeminiScriptProvider",
    "LyriaMusicProvider",
    "MediaError",
    "MusicError",
    "PlaceholderImageProvider",
    "SynthesisError",
    "VerbatimScriptProvider",
    "align_tokens",
    "audio_duration",
    "image_dimensions",
    "loop_count",
    "nearest_ratio",
    "normalize",
    "probe_duration",
    "tokenize",
]
