"""Persistence + pipeline bookkeeping. No network, no ffmpeg."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import select

from app.core.models import JobStatus, Scene, Timeline
from app.db.models import Job
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
    from app.core.models import Word

    missing = tmp_path / "nope.mp3"
    words = [Word(word="hello", start=0.1, end=0.6), Word(word="world", start=0.7, end=1.4)]
    assert pipeline._measured_duration(missing, words) == pytest.approx(
        1.4 + pipeline._SCENE_TAIL_PAD
    )
    assert pipeline._measured_duration(missing, []) == pipeline._MIN_SCENE_DURATION


async def test_align_stage_lays_scenes_on_the_audio_clock(tmp_path: Path, monkeypatch) -> None:
    """The load-bearing invariant: durations come from audio, not word counts.

    Scene 1's narration is 3x longer in words but its audio is shorter — the timeline must
    follow the audio.
    """
    from app.core.models import Word

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

    pad = pipeline._SCENE_TAIL_PAD
    s1, s2 = timeline.scenes
    assert (s1.start, s1.end) == (0.0, 2.0 + pad)
    assert (s2.start, s2.end) == (2.0 + pad, 2.0 + pad + 5.0 + pad)
    assert timeline.narration_duration == pytest.approx(7.0 + 2 * pad)
    # word timings are rebased onto the global clock
    assert s2.words[0].start == pytest.approx(2.0 + pad)
    assert s2.words[0].end == pytest.approx(2.0 + pad + 4.0)
