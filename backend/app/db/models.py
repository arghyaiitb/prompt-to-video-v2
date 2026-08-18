"""The Job row: everything the frontend polls for, plus the Timeline snapshot.

`timeline_json` is written after every pipeline stage. It is the debug trail and the
resume point — a job that died during rendering still has its narration timings on disk.

The row also carries the *request*, not just its progress: theme, bullet budget and tone
are what the caller asked for, so a finished job can say what it was actually rendered
with. `Timeline` does not carry a Theme yet (see `resolved_theme`), so the row is the
only place the choice survives.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, text
from sqlmodel import Field, SQLModel

from app.core.models import JobStatus

if TYPE_CHECKING:
    from app.core.models import Theme

logger = logging.getLogger(__name__)

#: Fallback when `app.core.themes` is unavailable — matches `Theme.name`'s default.
FALLBACK_THEME_NAME = "midnight"

#: Sentinel `Job.theme` value meaning "the palette is in `theme_custom`".
CUSTOM_THEME_NAME = "custom"

DEFAULT_BULLETS_PER_SLIDE = 4

#: `Job.logo_id` value meaning "render with no brand mark at all".
#:
#: Distinct from NULL, which means "the bundled default mark" — the behaviour of every job
#: created before uploads existed. It lives here rather than in `app.api.logos` because the
#: DB layer is what both the API and the worker already import, and it is a stored value.
#: It is also exactly the spelling `render.ffmpeg_backend.resolve_logo_source` already
#: reads as an explicit opt-out, so the pipeline can stamp it straight onto the Timeline.
NO_LOGO_ID = "none"

#: Fallback `Job.tts_engine` — also the literal baked into the ALTER TABLE default, so it
#: has to be a constant rather than a settings read.
FALLBACK_TTS_ENGINE = "deepgram"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def default_theme_name() -> str:
    """Preset id new jobs get. Lazy import: `app.core.themes` may not exist yet."""
    try:
        from app.core.themes import DEFAULT_THEME_NAME
    except ImportError:
        return FALLBACK_THEME_NAME
    return DEFAULT_THEME_NAME


def default_tts_engine() -> str:
    """Narration engine a job gets when the caller names none.

    Deliberately a plain settings read and not `factory.resolve_speech_engine`: the DB
    layer must not import the worker. API callers store the factory-resolved id, so a
    typo in `.env` can only reach a row created outside the HTTP surface.
    """
    try:
        from app.core.config import get_settings
    except ImportError:  # pragma: no cover - config is a hard dependency of everything
        return FALLBACK_TTS_ENGINE
    return get_settings().video_default_tts_engine or FALLBACK_TTS_ENGINE


class Job(SQLModel, table=True):
    """One render request. Status/progress are advanced by app.worker.pipeline."""

    id: str = Field(default_factory=_new_id, primary_key=True)

    topic: str
    slide_count: int
    voice: str
    music: bool = False

    tts_engine: str = Field(default_factory=default_tts_engine)
    """Which narration engine renders this job — see `app.worker.factory.SPEECH_ENGINES`.

    Persisted rather than read from config at render time for two reasons: a job that
    finished must be able to say what actually narrated it, and this value decides whether
    `Scene.ssml` is used at all. Only an engine declaring `supports_ssml` receives the
    marked-up narration; Deepgram Aura would read the tags aloud.
    """

    logo_id: str | None = None

    reuse_from: str | None = None
    """Reuse another job's script and imagery instead of generating new ones.

    Makes an honest A/B possible: change only the voice, engine or theme and everything
    else — words, bullets, pictures — is byte-identical to the source render.
    """
    """Uploaded brand mark for this video — an id from POST /api/logos.

    Three states, all meaningful: NULL is "the bundled default mark" (which is what every
    job did before uploads existed), `NO_LOGO_ID` is an explicit "no branding at all", and
    an id names a file in `settings.logo_dir`. Stored as the id rather than a path so the
    store can be relocated, and so DELETE /api/logos/{id} can see that a job needs it.
    """

    theme: str = Field(default_factory=default_theme_name)
    """Preset id from `app.core.themes.PRESETS`, or `custom` — see `theme_custom`."""

    theme_custom: str | None = None
    """JSON palette when the caller supplied their own colours. Overrides `theme`."""

    bullets_per_slide: int = DEFAULT_BULLETS_PER_SLIDE
    tone: str | None = None
    """Audience register for the script — `new_hires`, `all_staff`, `technical`, ..."""

    status: str = Field(default=JobStatus.QUEUED.value, index=True)
    progress: int = 0
    current_stage: str | None = None
    error: str | None = None

    timeline_json: str | None = None
    video_path: str | None = None

    created_at: datetime = Field(default_factory=_now, index=True)
    updated_at: datetime = Field(default_factory=_now)

    def custom_palette(self) -> dict[str, Any] | None:
        """`theme_custom` decoded, or None. Never raises — a bad blob is not fatal."""
        if not self.theme_custom:
            return None
        try:
            palette = json.loads(self.theme_custom)
        except ValueError:
            logger.warning("job %s has unparseable theme_custom; ignoring", self.id)
            return None
        return palette if isinstance(palette, dict) else None

    def resolved_theme(self) -> Theme:
        """The palette this job renders with.

        Custom colours win over the preset id. `app.core.themes` is imported lazily and
        its absence is not an error: a job still renders, on the default palette.
        """
        from app.core.models import Theme

        try:
            from app.core.themes import get_theme
        except ImportError:
            base = Theme()
        else:
            base = get_theme(self.theme)

        palette = self.custom_palette()
        if palette is None:
            return base
        # model_validate, not model_copy(update=...): copy skips validation, so a blob
        # holding `{"bg": 12}` would sail through here and blow up deep inside the
        # renderer's colour maths instead.
        merged = base.model_dump() | palette
        try:
            return Theme.model_validate(merged)
        except Exception:  # noqa: BLE001 - an unrenderable palette must not kill the job
            logger.warning("job %s has an invalid custom palette; using %s", self.id, base.name)
            return base


#: Columns added after the first release, with the SQL needed to backfill them.
#:
#: SQLite has no migration tool here and `create_all` only ever creates whole tables, so
#: an existing dev DB would keep its old five-column `job` table and every INSERT would
#: fail with "table job has no column named theme". Each entry is applied only when the
#: column is absent, which makes `migrate()` safe to run on every startup.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("theme", f"VARCHAR NOT NULL DEFAULT '{FALLBACK_THEME_NAME}'"),
    ("theme_custom", "VARCHAR"),
    ("bullets_per_slide", f"INTEGER NOT NULL DEFAULT {DEFAULT_BULLETS_PER_SLIDE}"),
    ("tone", "VARCHAR"),
    ("tts_engine", f"VARCHAR NOT NULL DEFAULT '{FALLBACK_TTS_ENGINE}'"),
    # Nullable with no default on purpose: NULL is the meaningful "bundled default mark"
    # state, so every pre-existing row keeps rendering exactly the branding it had.
    ("logo_id", "VARCHAR"),
    # Job id whose script and imagery this render reuses. NULL = generate fresh.
    ("reuse_from", "VARCHAR"),
)


def migrate(engine: Engine) -> list[str]:
    """Add any post-release `job` columns that this database is missing.

    Idempotent and additive: returns the columns it actually added, so startup can say
    so out loud. Nothing here drops or rewrites data.
    """
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info('job')").fetchall()
        if not rows:  # table doesn't exist yet — create_all owns that case
            return []
        existing = {row[1] for row in rows}
        added = []
        for name, ddl in _ADDED_COLUMNS:
            if name in existing:
                continue
            conn.execute(text(f"ALTER TABLE job ADD COLUMN {name} {ddl}"))
            added.append(name)
    if added:
        logger.info("migrated job table: added column(s) %s", ", ".join(added))
    return added
