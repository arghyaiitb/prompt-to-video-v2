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

    video_default_llm_provider: str = "gemini"
    video_default_llm_model: str = "gemini-3.7-flash"
    video_default_tts_provider: str = "deepgram"
    video_default_tts_voice: str = "aura-2-draco-en"
    video_default_aligner: str = "deepgram"
    video_default_aligner_model: str = "nova-3"
    video_default_image_model: str = "gemini-3.1-flash-image"
    video_default_music_model: str = "lyria-3-clip-preview"
    video_music_duck_db: int = -18

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
    def db_path(self) -> Path:
        return REPO_ROOT / "backend" / "videos.db"

    def job_dir(self, job_id: str) -> Path:
        d = self.video_output_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
