"""Config from .env at the repo root. Provider selection is data, not code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    """Set only for temporary STS credentials, which is what this account issues.

    Empty is legitimate (long-lived IAM keys), so it is deliberately NOT part of
    :meth:`aws_configured` — requiring it would hide a perfectly usable key pair.
    """

    aws_region: str = "us-east-1"

    video_default_llm_provider: str = "gemini"
    video_default_llm_model: str = "gemini-3.7-flash"
    video_default_tts_provider: str = "deepgram"
    video_default_tts_voice: str = "aura-2-draco-en"

    video_default_tts_engine: str = "deepgram"
    """Speech engine id for new jobs — see `app.worker.factory.SPEECH_ENGINES`.

    Distinct from `video_default_tts_provider`, which selects the single process-wide
    provider. The engine is a *per-job* choice (POST /api/jobs takes `tts_engine`), and
    it decides whether `Scene.ssml` is used at all: only an engine that declares
    `supports_ssml` is given the marked-up narration. Deepgram Aura vocalises SSML tags
    rather than parsing them, so this is a correctness switch, not a preference.
    """

    video_default_polly_voice: str = "Matthew"
    """Polly voice id used when a job selects `polly` without naming a voice."""

    video_polly_engine: str = "generative"
    """Polly voice tier: `generative`, `neural`, `long-form` or `standard`.

    Also filters the voice catalogue — a voice is only offered if it supports this tier,
    because Polly rejects `Engine=generative` for a neural-only voice at synthesis time.
    Measured on this account: `emphasis` is silently dropped by the generative and neural
    tiers and only honoured by `standard`; `break`, `prosody` and `say-as` work on all.
    """

    video_enable_veo: bool = False
    """Gate on generated video clips. Off by default: Veo is the most expensive call in
    the pipeline, so spending has to be opted into explicitly."""

    video_default_video_model: str = "veo-3.1-fast-generate-preview"

    video_default_aligner: str = "deepgram"
    video_default_aligner_model: str = "nova-3"
    video_default_image_model: str = "gemini-3.1-flash-image"
    video_default_music_model: str = "lyria-3-clip-preview"
    video_music_duck_db: int = -18
    video_logo_max_bytes: int = 4 * 1024 * 1024
    """Upload ceiling. Generous for a logo, small enough to bound decode cost."""

    video_logo_max_dimension: int = 4096
    """Reject larger sources: a small file can decode to an enormous bitmap, and the
    renderer only ever needs ~120px of logo height."""

    @property
    def logo_dir(self) -> Path:
        """Where uploaded brand marks live. Outside `out/` so job cleanup can't wipe them."""
        d = self.video_cache_dir / "logos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    video_logo_path: Path | None = None
    """Branding watermark, composited bottom-left for the whole video.

    Defaults to the app's own mark. SVG or PNG-with-alpha; rasterised once per render.
    Set empty in .env to disable branding.
    """

    video_scene_pause_s: float = 1.0
    """Audible silence between one scene's last word and the next scene's first.

    Back-to-back narration reads as rushed. This is the *heard* gap — the pipeline adds
    the crossfade duration on top, because xfade consumes overlap and would otherwise
    eat half the pause.
    """

    video_api_concurrency: int = 4
    """Concurrent calls per provider during the fan-out stages (images, TTS, alignment).

    Raise for faster jobs, lower if a provider starts returning 429.
    """

    video_output_dir: Path = REPO_ROOT / "out"
    video_cache_dir: Path = REPO_ROOT / "cache"

    @field_validator("video_output_dir", "video_cache_dir", mode="after")
    @classmethod
    def _anchor_to_repo(cls, v: Path) -> Path:
        """`.env` holds `./out`, which would otherwise resolve against the launch CWD —
        putting artifacts in `backend/out` when started there and `<repo>/out` otherwise.
        """
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @property
    def aws_configured(self) -> bool:
        """True when Polly could actually authenticate.

        The engine catalogue reports availability from this, so it must not be optimistic:
        claiming an engine works and then failing six minutes into a render is worse than
        not offering it. Note the creds on this account are temporary and DO expire —
        this says "configured", not "unexpired".
        """
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def db_path(self) -> Path:
        return REPO_ROOT / "backend" / "videos.db"

    def job_dir(self, job_id: str) -> Path:
        d = self.video_output_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
