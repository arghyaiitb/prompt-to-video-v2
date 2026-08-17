"""Persistence + pipeline bookkeeping. No network, no ffmpeg."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import select

from app.core.config import get_settings
from app.core.models import BulletPoint, JobStatus, Scene, SceneRole, Timeline, Word
from app.db import models as db_models
from app.db.models import Job, default_theme_name
from app.db.session import get_engine, init_db, make_engine, session_scope, set_engine
from app.worker import factory, pipeline


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Path]:
    db_path = tmp_path / "jobs.db"
    set_engine(make_engine(db_path))
    init_db()
    yield db_path
    set_engine(None)


def test_wal_mode_is_enabled(db: Path) -> None:
    """Status polling every second must not block on the worker's writes."""
    with get_engine().connect() as conn:
        from sqlalchemy import text

        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000


def test_job_defaults(db: Path) -> None:
    with session_scope() as session:
        job = Job(topic="Photosynthesis", slide_count=4, voice="aura-2-draco-en", music=True)
        session.add(job)
        session.commit()
        session.refresh(job)

        assert len(job.id) == 36  # uuid4
        assert job.status == JobStatus.QUEUED.value
        assert job.progress == 0
        assert job.current_stage is None
        assert job.error is None
        assert job.timeline_json is None
        assert job.video_path is None
        assert job.created_at is not None
        assert job.updated_at is not None


def test_job_theme_and_script_choices_default(db: Path) -> None:
    with session_scope() as session:
        job = Job(topic="Phishing", slide_count=4, voice="v")
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.theme == default_theme_name()
        assert job.theme_custom is None
        assert job.custom_palette() is None
        assert job.bullets_per_slide == 4
        assert job.tone is None
        assert job.resolved_theme().name == default_theme_name()


def test_resolved_theme_falls_back_for_an_unknown_preset() -> None:
    """The last line of defence: a bogus id in the row still renders on a real palette."""
    theme = Job(topic="t", slide_count=2, voice="v", theme="chartreuse-disco").resolved_theme()
    assert theme.contrast_report()["text_on_bg"] >= 4.5


def test_resolved_theme_applies_the_custom_palette() -> None:
    palette = {
        "bg": "#FFFFFF",
        "surface": "#F1F5F9",
        "text": "#0B1220",
        "muted": "#475569",
        "accent": "#B45309",
    }
    job = Job(
        topic="t", slide_count=2, voice="v", theme="custom", theme_custom=json.dumps(palette)
    )
    theme = job.resolved_theme()
    assert theme.bg == "#FFFFFF"
    assert theme.text == "#0B1220"
    assert theme.is_light is True
    # untouched fields keep the preset's values, not pydantic's bare defaults being lost
    assert theme.image_radius > 0


@pytest.mark.parametrize("blob", ["{not json", "[]", "null", '{"bg": 12}'])
def test_resolved_theme_tolerates_a_broken_custom_palette(blob: str) -> None:
    """A bad blob costs the palette, never the render."""
    job = Job(topic="t", slide_count=2, voice="v", theme="custom", theme_custom=blob)
    assert job.resolved_theme().contrast_report()["text_on_bg"] >= 4.5


# --------------------------------------------------------------------------- migration

#: The `job` table exactly as it shipped before theme selection existed.
_OLD_SCHEMA = """
CREATE TABLE job (
    id VARCHAR NOT NULL,
    topic VARCHAR NOT NULL,
    slide_count INTEGER NOT NULL,
    voice VARCHAR NOT NULL,
    music BOOLEAN NOT NULL,
    status VARCHAR NOT NULL,
    progress INTEGER NOT NULL,
    current_stage VARCHAR,
    error VARCHAR,
    timeline_json VARCHAR,
    video_path VARCHAR,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)
"""


def test_init_db_migrates_a_database_created_with_the_old_schema(tmp_path: Path) -> None:
    """`create_all` never alters an existing table, so an old dev DB needs this.

    Without the migration, the first INSERT after the upgrade dies with
    "table job has no column named theme" — an OperationalError inside the request
    handler, on a database that looks perfectly fine.
    """
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as raw:
        raw.execute(_OLD_SCHEMA)
        raw.execute(
            "INSERT INTO job (id, topic, slide_count, voice, music, status, progress, "
            "created_at, updated_at) VALUES "
            "('legacy-1', 'Old job', 3, 'aura-2-draco-en', 0, 'done', 100, "
            "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
        )

    set_engine(make_engine(db_path))
    try:
        init_db()  # create_all + additive migration

        # a brand-new row inserts cleanly through the ORM: no OperationalError
        with session_scope() as session:
            job = Job(
                topic="New job",
                slide_count=2,
                voice="v",
                bullets_per_slide=5,
                tone="technical",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            assert job.theme == default_theme_name()
            assert job.bullets_per_slide == 5
            assert job.tone == "technical"
            assert job.tts_engine == db_models.default_tts_engine()

        # the pre-existing row was backfilled rather than left NULL
        with session_scope() as session:
            legacy = session.get(Job, "legacy-1")
            assert legacy is not None
            assert legacy.topic == "Old job"
            assert legacy.theme == db_models.FALLBACK_THEME_NAME
            assert legacy.theme_custom is None
            assert legacy.bullets_per_slide == 4
            assert legacy.tone is None
            # backfilled, not NULL: the column is NOT NULL and the pipeline reads it to
            # decide whether SSML may be sent at all
            assert legacy.tts_engine == db_models.FALLBACK_TTS_ENGINE
            assert legacy.resolved_theme().contrast_report()["text_on_bg"] >= 4.5

        # and it is idempotent — a second startup adds nothing and does not raise
        assert db_models.migrate(get_engine()) == []
        init_db()
    finally:
        set_engine(None)


def test_migrate_adds_every_new_column(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy2.db"
    with sqlite3.connect(db_path) as raw:
        raw.execute(_OLD_SCHEMA)

    engine = make_engine(db_path)
    added = db_models.migrate(engine)
    expected = ["theme", "theme_custom", "bullets_per_slide", "tone", "tts_engine", "logo_id"]
    assert added == expected

    with engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info('job')").fetchall()}
    assert set(expected) <= columns
    engine.dispose()


def test_logo_id_survives_the_migration_of_an_old_database(tmp_path: Path) -> None:
    """`backend/videos.db` is a live database with real rows in it.

    `create_all` never alters an existing table, so without the `logo_id` entry in
    `_ADDED_COLUMNS` the first POST /api/jobs carrying a logo dies with "table job has no
    column named logo_id" inside the handler. The pre-existing rows must also come back
    with NULL — which is the state that means "the bundled default mark", i.e. exactly the
    branding those videos already have.
    """
    db_path = tmp_path / "legacy-logo.db"
    with sqlite3.connect(db_path) as raw:
        raw.execute(_OLD_SCHEMA)
        raw.execute(
            "INSERT INTO job (id, topic, slide_count, voice, music, status, progress, "
            "created_at, updated_at) VALUES "
            "('legacy-logo', 'Old job', 3, 'v', 0, 'done', 100, "
            "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
        )

    set_engine(make_engine(db_path))
    try:
        init_db()
        with session_scope() as session:
            job = Job(topic="Branded", slide_count=2, voice="v", logo_id="a" * 32)
            session.add(job)
            session.commit()
            job_id = job.id

        with session_scope() as session:
            stored = session.get(Job, job_id)
            assert stored is not None
            assert stored.logo_id == "a" * 32

            legacy = session.get(Job, "legacy-logo")
            assert legacy is not None
            assert legacy.logo_id is None
            assert pipeline.resolve_job_logo(legacy) is None
    finally:
        set_engine(None)


def test_tts_engine_survives_the_migration_of_an_old_database(tmp_path: Path) -> None:
    """Writing a *chosen* engine into a pre-existing DB must not raise OperationalError.

    `create_all` never alters an existing table, so without the `tts_engine` entry in
    `_ADDED_COLUMNS` this INSERT dies with "table job has no column named tts_engine" —
    inside the POST handler, on a database that looks perfectly healthy. `backend/videos.db`
    is exactly such a database, with real rows in it.
    """
    db_path = tmp_path / "legacy-engine.db"
    with sqlite3.connect(db_path) as raw:
        raw.execute(_OLD_SCHEMA)

    set_engine(make_engine(db_path))
    try:
        init_db()
        with session_scope() as session:
            job = Job(topic="Polly job", slide_count=2, voice="Matthew", tts_engine="polly")
            session.add(job)
            session.commit()
            job_id = job.id

        # read back through a fresh session: the value is persisted, not just in the identity map
        with session_scope() as session:
            stored = session.get(Job, job_id)
            assert stored is not None
            assert stored.tts_engine == "polly"
            assert stored.voice == "Matthew"
    finally:
        set_engine(None)


def test_default_tts_engine_follows_settings(monkeypatch) -> None:
    """The row default tracks config, so `.env` alone can flip new jobs to another engine."""
    settings = get_settings()
    monkeypatch.setattr(settings, "video_default_tts_engine", "polly")
    assert db_models.default_tts_engine() == "polly"

    # an empty setting must not write "" into a NOT NULL column
    monkeypatch.setattr(settings, "video_default_tts_engine", "")
    assert db_models.default_tts_engine() == db_models.FALLBACK_TTS_ENGINE


def test_migrate_is_a_noop_when_the_table_does_not_exist(tmp_path: Path) -> None:
    """Fresh install: `create_all` owns table creation, the migration stays out of it."""
    engine = make_engine(tmp_path / "empty.db")
    assert db_models.migrate(engine) == []
    engine.dispose()


def test_ids_are_unique(db: Path) -> None:
    with session_scope() as session:
        for i in range(5):
            session.add(Job(topic=f"t{i}", slide_count=2, voice="v"))
        session.commit()
        ids = {j.id for j in session.exec(select(Job)).all()}
    assert len(ids) == 5


def test_patch_job_updates_row_and_timestamp(db: Path) -> None:
    with session_scope() as session:
        job = Job(topic="Tides", slide_count=3, voice="v")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id, before = job.id, job.updated_at

    pipeline._patch_job(
        job_id, status=JobStatus.IMAGING.value, progress=30, current_stage="imaging"
    )

    with session_scope() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == "imaging"
        assert row.progress == 30
        assert row.current_stage == "imaging"
        assert row.updated_at >= before


def test_patch_job_on_missing_row_is_a_noop(db: Path) -> None:
    pipeline._patch_job("nope", status="failed")  # must not raise


def test_load_job_raises_for_unknown_id(db: Path) -> None:
    with pytest.raises(LookupError):
        pipeline._load_job("nope")


def test_stage_progress_sequence_is_monotonic() -> None:
    order = [
        JobStatus.SCRIPTING,
        JobStatus.IMAGING,
        JobStatus.NARRATING,
        JobStatus.ALIGNING,
        JobStatus.SCORING,
        JobStatus.RENDERING,
        JobStatus.ASSEMBLING,
        JobStatus.DONE,
    ]
    values = [pipeline.STAGE_PROGRESS[s] for s in order]
    assert values == [10, 30, 50, 60, 70, 90, 95, 100]
    assert values == sorted(values)


async def test_run_job_records_failure_when_providers_are_missing(db: Path) -> None:
    """A missing provider must fail the job, never leave it 'running'."""
    with session_scope() as session:
        job = Job(topic="Anything", slide_count=2, voice="v")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    def _unavailable(*_args, **_kwargs):
        raise factory.ProviderUnavailableError("script provider not yet available: boom")

    original = factory.script_provider
    factory.script_provider = _unavailable  # type: ignore[assignment]
    try:
        await pipeline.run_job(job_id)
    finally:
        factory.script_provider = original  # type: ignore[assignment]

    with session_scope() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == JobStatus.FAILED.value
        assert "not yet available" in (row.error or "")


async def test_stage_script_threads_bullets_and_tone_and_stamps_the_theme(
    tmp_path: Path, monkeypatch
) -> None:
    from app.core.models import SceneScript, Script

    seen: dict[str, object] = {}

    class _WidenedProvider:
        def generate(
            self,
            topic: str,
            slide_count: int,
            *,
            bullets_per_slide: int = 4,
            tone: str | None = None,
        ) -> Script:
            seen.update(
                topic=topic, slide_count=slide_count, bullets=bullets_per_slide, tone=tone
            )
            return Script(
                topic=topic,
                title="T",
                scenes=[
                    SceneScript(
                        id=1, narration="n", heading="H", bullets=["A", "B"], image_prompt="p"
                    )
                ],
            )

    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _WidenedProvider())
    job = Job(
        topic="Phishing", slide_count=1, voice="v", bullets_per_slide=5, tone="technical"
    )
    theme = Job(topic="t", slide_count=1, voice="v", theme="daylight").resolved_theme()

    timeline = await pipeline._stage_script(job, tmp_path, theme)

    assert seen == {"topic": "Phishing", "slide_count": 1, "bullets": 5, "tone": "technical"}
    assert timeline.theme.bg == theme.bg
    assert [b.text for b in timeline.scenes[0].bullets] == ["A", "B"]


async def test_stage_script_still_works_with_a_two_arg_provider(
    tmp_path: Path, monkeypatch
) -> None:
    """`gemini_script.GeminiScriptProvider` is still on `generate(topic, slide_count)`.

    Until it widens, the new choices are dropped with a warning rather than crashing the
    job on an unexpected keyword argument.
    """
    from app.core.models import SceneScript, Script

    calls: list[tuple] = []

    class _LegacyProvider:
        def generate(self, topic: str, slide_count: int) -> Script:
            calls.append((topic, slide_count))
            return Script(
                topic=topic,
                title="T",
                scenes=[SceneScript(id=1, narration="n", heading="H", image_prompt="p")],
            )

    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _LegacyProvider())
    job = Job(topic="Badges", slide_count=2, voice="v", bullets_per_slide=5, tone="all_staff")

    timeline = await pipeline._stage_script(job, tmp_path)

    assert calls == [("Badges", 2)]
    assert timeline.title == "T"


async def test_stage_script_carries_the_role_and_the_motion_prompt_onto_every_scene(
    tmp_path: Path, monkeypatch
) -> None:
    """The script decides the video's shape; the pipeline's job is not to lose it.

    ``Scene.role`` drives duration, layout, type scale and bullet budget downstream, and
    ``clip_prompt`` is what a video model is handed instead of a still. Both live only on
    the script until this stage copies them across.
    """
    from app.core.models import Script
    from app.providers.gemini_script import StructuredSceneScript

    class _StructuredProvider:
        def generate(
            self,
            topic: str,
            slide_count: int,
            *,
            bullets_per_slide: int = 4,
            tone: str | None = None,
        ) -> Script:
            return Script(
                topic=topic,
                title="How Phishing Works",
                scenes=[
                    StructuredSceneScript(
                        id=1,
                        role=SceneRole.TITLE,
                        narration="This module shows how a phishing email is built.",
                        heading="How Phishing Works",
                        bullets=[],
                        image_prompt="p1",
                        clip_prompt="Slow push in on a lit keyboard",
                    ),
                    StructuredSceneScript(
                        id=2,
                        role=SceneRole.CLOSING,
                        narration="Report anything suspicious to the security team.",
                        heading="If in doubt, report it",
                        bullets=["Report anything suspicious"],
                        image_prompt="p2",
                        clip_prompt="Handheld follow as a hand taps report",
                    ),
                ],
            )

    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _StructuredProvider())
    job = Job(topic="Phishing", slide_count=2, voice="v")

    timeline = await pipeline._stage_script(job, tmp_path)

    assert [s.role for s in timeline.scenes] == [SceneRole.TITLE, SceneRole.CLOSING]
    assert timeline.scenes[0].clip_prompt == "Slow push in on a lit keyboard"
    assert [b.text for b in timeline.scenes[1].bullets] == ["Report anything suspicious"]


async def test_stage_script_defaults_to_content_for_a_provider_that_plans_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """An unstructured script IS a queue of content scenes — that is the honest label."""
    from app.core.models import SceneScript, Script

    class _PlainProvider:
        def generate(self, topic: str, slide_count: int) -> Script:
            return Script(
                topic=topic,
                title="T",
                scenes=[SceneScript(id=1, narration="n", heading="H", image_prompt="p")],
            )

    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _PlainProvider())
    timeline = await pipeline._stage_script(Job(topic="t", slide_count=1, voice="v"), tmp_path)

    assert timeline.scenes[0].role is SceneRole.CONTENT
    assert timeline.scenes[0].clip_prompt is None


async def test_stage_script_strips_bullets_a_title_card_may_not_show(
    tmp_path: Path, monkeypatch
) -> None:
    """The worst defect in the rejected output: scene 1 rendered as a title card carrying a
    content heading and four bullets. The budget is enforced here as well as in the
    provider, because the provider is not the only way scenes reach a Timeline."""
    from app.core.models import Script
    from app.providers.gemini_script import StructuredSceneScript

    class _SloppyProvider:
        def generate(self, topic: str, slide_count: int) -> Script:
            return Script(
                topic=topic,
                title="T",
                scenes=[
                    StructuredSceneScript(
                        id=1,
                        role=SceneRole.TITLE,
                        narration="n",
                        heading="H",
                        bullets=["Inspect the sender", "Hover over links"],
                        image_prompt="p",
                    )
                ],
            )

    monkeypatch.setattr(factory, "script_provider", lambda *a, **k: _SloppyProvider())
    timeline = await pipeline._stage_script(Job(topic="t", slide_count=1, voice="v"), tmp_path)

    assert timeline.scenes[0].bullets == []


def test_stage_bullets_enforces_the_budget_of_each_scenes_own_role() -> None:
    """The budget is per ROLE, not per video. A title card gets none, a closing gets two,
    however many points arrived — and the survivors are still timed against real words.
    """
    narration = (
        "Check the sender domain before you trust the display name. Hover over every link "
        "to reveal the real destination. Report anything suspicious straight away."
    )
    tokens = narration.split()
    points = [
        "Check the sender domain",
        "Hover over every link",
        "Reveal the real destination",
        "Report anything suspicious",
    ]

    def _scene(scene_id: int, role: SceneRole, start: float) -> Scene:
        return Scene(
            id=scene_id,
            role=role,
            narration=narration,
            heading="H",
            image_prompt="p",
            start=start,
            end=start + 20.0,
            words=[
                Word(word=t, start=start + i * 0.4, end=start + i * 0.4 + 0.35)
                for i, t in enumerate(tokens)
            ],
            bullets=[BulletPoint(text=text) for text in points],
        )

    timeline = Timeline(
        job_id="j",
        topic="t",
        title="t",
        voice="v",
        scenes=[
            _scene(1, SceneRole.TITLE, 0.0),
            _scene(2, SceneRole.CONTENT, 20.0),
            _scene(3, SceneRole.SUMMARY, 40.0),
            _scene(4, SceneRole.CLOSING, 60.0),
        ],
    )

    pipeline._stage_bullets(timeline)

    assert [len(s.bullets) for s in timeline.scenes] == [
        s.role.bullet_budget for s in timeline.scenes
    ]
    assert timeline.scenes[0].bullets == []
    closing = timeline.scenes[-1]
    # Kept in the order they were given, and still timed against the spoken words.
    assert [b.text for b in closing.bullets] == points[:2]
    assert [b.appear_at for b in closing.bullets] == sorted(
        b.appear_at for b in closing.bullets
    )
    assert closing.bullets[0].appear_at < closing.duration


def test_video_backend_receives_the_resolved_theme(monkeypatch) -> None:
    """The palette reaches the renderer by constructor — that is where it is read from."""
    captured: dict[str, object] = {}

    class _FakeBackend:
        def __init__(self, *, theme=None) -> None:  # noqa: ANN001
            captured["theme"] = theme

    monkeypatch.setattr(factory, "_load", lambda *a, **k: _FakeBackend)
    theme = Job(topic="t", slide_count=1, voice="v", theme="forest").resolved_theme()
    factory.video_backend(theme=theme)
    assert captured["theme"] is theme


# ---------------------------------------------------------------- speech engine factory


def test_speech_engine_registry_declares_ssml_support_per_engine() -> None:
    """Deepgram False, Polly True. Getting this backwards ships narrated markup."""
    support = {spec.id: spec.supports_ssml for spec in factory.speech_engines()}
    assert support == {"deepgram": False, "polly": True}


def test_registry_ssml_support_matches_the_provider_classes() -> None:
    """The catalogue's `supports_ssml` must not drift from the class the factory builds.

    GET /api/engines answers from the registry (no credentials, no provider import), the
    pipeline reads the instance attribute. If those two disagree the pipeline sends SSML to
    an engine that vocalises it, or withholds it from one that parses it — and nothing else
    in the system would notice.

    Read with a False default deliberately: a provider that does not declare the attribute
    is by definition not SSML-capable, which is how `DeepgramSynthesizer` reads today.
    """
    for spec in factory.speech_engines():
        try:
            cls = factory._load(spec.module, spec.class_name, "test")
        except factory.ProviderUnavailableError:
            pytest.skip(f"{spec.module} not importable in this tree")
        assert getattr(cls, "supports_ssml", False) == spec.supports_ssml, spec.id


def test_default_speech_engine_falls_back_for_an_unknown_config_value(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "video_default_tts_engine", "elevenlabs")
    assert factory.default_speech_engine(settings) == factory.FALLBACK_SPEECH_ENGINE

    monkeypatch.setattr(settings, "video_default_tts_engine", "POLLY  ")
    assert factory.default_speech_engine(settings) == "polly"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("polly", "polly"),
        ("Polly", "polly"),
        (None, "deepgram"),
        ("", "deepgram"),
        ("nope", "deepgram"),
    ],
)
def test_resolve_speech_engine(requested, expected, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(get_settings(), "video_default_tts_engine", "deepgram")
    assert factory.resolve_speech_engine(requested) == expected


def test_default_voice_is_per_engine(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "video_default_tts_voice", "aura-2-draco-en")
    monkeypatch.setattr(settings, "video_default_polly_voice", "Matthew")
    assert factory.default_voice("deepgram", settings) == "aura-2-draco-en"
    assert factory.default_voice("polly", settings) == "Matthew"
    # unknown id degrades to the fallback engine's voice rather than returning ""
    assert factory.default_voice("nope", settings) == "aura-2-draco-en"


def test_polly_is_unavailable_without_aws_credentials(monkeypatch) -> None:
    """`available` is a measured fact — see app/api/engines.py."""
    settings = get_settings()
    monkeypatch.setattr(settings, "aws_access_key_id", "")
    monkeypatch.setattr(settings, "aws_secret_access_key", "")
    available, reason = factory.speech_engine_status("polly", settings)
    assert available is False
    assert "AWS credentials" in (reason or "")


def test_deepgram_is_unavailable_without_an_api_key(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "deepgram_api_key", "")
    available, reason = factory.speech_engine_status("deepgram", settings)
    assert available is False
    assert "DEEPGRAM_API_KEY" in (reason or "")


def test_unknown_engine_is_not_available() -> None:
    available, reason = factory.speech_engine_status("elevenlabs")
    assert available is False
    assert "unknown engine" in (reason or "")


def test_speech_synthesizer_refuses_an_unavailable_engine(monkeypatch) -> None:
    """A missing credential must fail loudly at construction, not mid-render."""
    settings = get_settings()
    monkeypatch.setattr(settings, "aws_access_key_id", "")
    monkeypatch.setattr(settings, "aws_secret_access_key", "")
    with pytest.raises(factory.ProviderUnavailableError, match="not available"):
        factory.speech_synthesizer(engine="polly")


def test_speech_synthesizer_rejects_an_unknown_engine() -> None:
    with pytest.raises(factory.ProviderUnavailableError, match="unknown speech engine"):
        factory.speech_synthesizer(engine="elevenlabs")


def test_speech_synthesizer_resolves_the_requested_engine(monkeypatch) -> None:
    """The engine id selects the class; both the keyword and positional spellings work."""
    loaded: list[str] = []

    class _Fake:
        supports_ssml = True

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    def _fake_load(module_path: str, class_name: str, role: str) -> type:
        loaded.append(f"{module_path}.{class_name}")
        return _Fake

    monkeypatch.setattr(factory, "_load", _fake_load)
    monkeypatch.setattr(factory, "speech_engine_status", lambda *a, **k: (True, None))

    factory.speech_synthesizer(engine="polly")
    factory.speech_synthesizer("polly")  # positional engine id, not a Settings object
    factory.speech_synthesizer()
    assert loaded == [
        "app.providers.polly_tts.PollySynthesizer",
        "app.providers.polly_tts.PollySynthesizer",
        "app.providers.deepgram_tts.DeepgramSynthesizer",
    ]


def test_polly_synthesizer_gets_the_configured_voice_and_tier(monkeypatch) -> None:
    """Polly rejects `Engine=generative` for a neural-only voice, so the tier must arrive."""
    captured: dict[str, object] = {}
    settings = get_settings()
    monkeypatch.setattr(settings, "video_default_polly_voice", "Ruth")
    monkeypatch.setattr(settings, "video_polly_engine", "neural")
    monkeypatch.setattr(settings, "aws_region", "eu-west-2")

    class _FakePolly:
        supports_ssml = True

        def __init__(self, *, voice=None, engine=None, region_name=None, settings=None) -> None:  # noqa: ANN001
            captured.update(voice=voice, engine=engine, region_name=region_name)

    monkeypatch.setattr(factory, "_load", lambda *a, **k: _FakePolly)
    monkeypatch.setattr(factory, "speech_engine_status", lambda *a, **k: (True, None))
    factory.speech_synthesizer(engine="polly")
    assert captured == {"voice": "Ruth", "engine": "neural", "region_name": "eu-west-2"}


def test_video_clip_provider_is_gated_on_the_veo_flag(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "video_enable_veo", False)
    with pytest.raises(factory.ProviderUnavailableError, match="VIDEO_ENABLE_VEO"):
        factory.video_clip_provider(settings)

    captured: dict[str, object] = {}

    class _FakeVeo:
        def __init__(self, *, api_key=None, model=None) -> None:  # noqa: ANN001
            captured.update(api_key=api_key, model=model)

    monkeypatch.setattr(settings, "video_enable_veo", True)
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "video_default_video_model", "veo-3.1-fast-generate-preview")
    monkeypatch.setattr(factory, "_load", lambda *a, **k: _FakeVeo)
    factory.video_clip_provider(settings)
    assert captured == {"api_key": "k", "model": "veo-3.1-fast-generate-preview"}


def test_timeline_round_trips_through_the_job_row(db: Path) -> None:
    timeline = Timeline(
        job_id="abc",
        topic="Tides",
        title="Why Tides Happen",
        voice="aura-2-draco-en",
        scenes=[
            Scene(id=1, narration="one", heading="One", image_prompt="p1", start=0.0, end=3.5),
            Scene(id=2, narration="two", heading="Two", image_prompt="p2", start=3.5, end=7.0),
        ],
    )
    job = Job(
        topic="Tides", slide_count=2, voice="v", timeline_json=timeline.model_dump_json()
    )
    restored = pipeline.timeline_from_job(job)
    assert restored is not None
    assert restored.narration_duration == 7.0
    assert [s.duration for s in restored.scenes] == [3.5, 3.5]


def test_timeline_from_job_tolerates_garbage(db: Path) -> None:
    assert pipeline.timeline_from_job(Job(topic="t", slide_count=2, voice="v")) is None
    assert (
        pipeline.timeline_from_job(
            Job(topic="t", slide_count=2, voice="v", timeline_json="{not json")
        )
        is None
    )


def test_measured_duration_falls_back_to_word_timings(tmp_path: Path) -> None:
    """No probe-able audio -> last aligned word wins; never zero-length."""
    missing = tmp_path / "nope.mp3"
    words = [Word(word="hello", start=0.1, end=0.6), Word(word="world", start=0.7, end=1.4)]
    assert pipeline._measured_duration(missing, words) == pytest.approx(
        1.4 + pipeline._scene_tail_pad()
    )
    assert pipeline._measured_duration(missing, []) == pipeline._MIN_SCENE_DURATION


async def test_align_stage_lays_scenes_on_the_audio_clock(tmp_path: Path, monkeypatch) -> None:
    """The load-bearing invariant: durations come from audio, not word counts.

    Scene 1's narration is 3x longer in words but its audio is shorter — the timeline must
    follow the audio.
    """
    timeline = Timeline(
        job_id="j",
        topic="t",
        title="t",
        voice="v",
        scenes=[
            Scene(
                id=1,
                narration="a b c d e f g h i",
                heading="H1",
                image_prompt="p",
                audio_path=str(tmp_path / "s1.mp3"),
            ),
            Scene(
                id=2,
                narration="short",
                heading="H2",
                image_prompt="p",
                audio_path=str(tmp_path / "s2.mp3"),
            ),
        ],
    )
    fake_words = {
        "s1.mp3": [Word(word="a", start=0.0, end=1.0)],
        "s2.mp3": [Word(word="short", start=0.0, end=4.0)],
    }
    durations = {"s1.mp3": 2.0, "s2.mp3": 5.0}

    class _FakeAligner:
        def align(self, audio_path: Path, reference_text: str) -> list[Word]:
            return fake_words[audio_path.name]

    monkeypatch.setattr(factory, "aligner", lambda *a, **k: _FakeAligner())
    monkeypatch.setattr(pipeline, "_probe_duration", lambda path: durations[Path(path).name])

    await pipeline._stage_align(timeline)

    pad = pipeline._scene_tail_pad()
    s1, s2 = timeline.scenes
    assert (s1.start, s1.end) == (0.0, 2.0 + pad)
    assert (s2.start, s2.end) == (2.0 + pad, 2.0 + pad + 5.0 + pad)
    assert timeline.narration_duration == pytest.approx(7.0 + 2 * pad)
    # word timings are rebased onto the global clock
    assert s2.words[0].start == pytest.approx(2.0 + pad)
    assert s2.words[0].end == pytest.approx(2.0 + pad + 4.0)
