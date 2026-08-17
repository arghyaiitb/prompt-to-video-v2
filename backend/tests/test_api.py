"""HTTP-level tests. Real app, real SQLite (a throwaway file), no network, no ffmpeg."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import voices as voices_api
from app.core.config import get_settings
from app.db.session import init_db, make_engine, set_engine
from app.main import app
from app.worker import pipeline


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """App wired to a temp DB and temp output dir, with the renderer stubbed out."""
    engine = make_engine(tmp_path / "test.db")
    set_engine(engine)
    init_db()

    settings = get_settings()
    monkeypatch.setattr(settings, "video_output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "video_cache_dir", tmp_path / "cache")

    spawned: list[str] = []

    async def _fake_run_job(job_id: str) -> None:
        spawned.append(job_id)

    monkeypatch.setattr(pipeline, "run_job", _fake_run_job)
    voices_api.reset_cache()

    with TestClient(app) as test_client:
        test_client.spawned = spawned  # type: ignore[attr-defined]
        yield test_client

    set_engine(None)
    voices_api.reset_cache()


def _wait_for(predicate, timeout: float = 2.0):
    """Poll a background-task side effect; the task runs on the portal's event loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return predicate()


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job_returns_id_and_queues_work(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        json={"topic": "How rainbows form", "slide_count": 3, "voice": "aura-2-draco-en"},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert job_id

    status = client.get(f"/api/jobs/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["job_id"] == job_id
    assert body["topic"] == "How rainbows form"
    assert body["status"] == "queued"
    assert body["progress"] == 0
    assert body["video_url"] is None
    assert body["error"] is None
    # the background render was handed off, not executed inline
    assert _wait_for(lambda: client.spawned) == [job_id]  # type: ignore[attr-defined]


def test_create_job_defaults_voice_from_settings(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"topic": "Tides"})
    job_id = response.json()["job_id"]
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["status"] == "queued"
    assert get_settings().video_default_tts_voice


@pytest.mark.parametrize("slide_count", [1, 0, 11, -3])
def test_slide_count_out_of_range_is_rejected(client: TestClient, slide_count: int) -> None:
    response = client.post("/api/jobs", json={"topic": "Bees", "slide_count": slide_count})
    assert response.status_code == 422


def test_slide_count_bounds_are_inclusive(client: TestClient) -> None:
    for slide_count in (2, 10):
        response = client.post("/api/jobs", json={"topic": "Bees", "slide_count": slide_count})
        assert response.status_code == 202, response.text


def test_empty_topic_is_rejected(client: TestClient) -> None:
    assert client.post("/api/jobs", json={"topic": ""}).status_code == 422


def test_list_jobs_is_newest_first_and_capped(client: TestClient) -> None:
    for i in range(3):
        client.post("/api/jobs", json={"topic": f"topic {i}", "slide_count": 2})

    rows = client.get("/api/jobs").json()
    assert len(rows) == 3
    assert [r["topic"] for r in rows] == ["topic 2", "topic 1", "topic 0"]
    assert all(set(r) >= {"job_id", "status", "progress", "video_url"} for r in rows)


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/does-not-exist/video").status_code == 404
    assert client.delete("/api/jobs/does-not-exist").status_code == 404


def test_video_url_appears_only_when_done(client: TestClient, tmp_path: Path) -> None:
    from app.db.models import Job
    from app.db.session import session_scope

    job_id = client.post("/api/jobs", json={"topic": "Volcanoes"}).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["video_url"] is None
    # done, but nothing on disk yet -> still 404 on the file, no crash
    assert client.get(f"/api/jobs/{job_id}/video").status_code == 404

    video = tmp_path / "out" / job_id / "video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"not really an mp4")

    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.status = "done"
        job.progress = 100
        job.current_stage = "done"
        job.video_path = str(video)
        session.add(job)
        session.commit()

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["video_url"] == f"/api/jobs/{job_id}/video"

    served = client.get(detail["video_url"])
    assert served.status_code == 200
    assert served.content == b"not really an mp4"
    assert served.headers["content-type"] == "video/mp4"


def test_delete_removes_row_and_output_dir(client: TestClient, tmp_path: Path) -> None:
    job_id = client.post("/api/jobs", json={"topic": "Glaciers"}).json()["job_id"]
    job_dir = tmp_path / "out" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "scene_01.png").write_bytes(b"png")

    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not job_dir.exists()


def test_timeline_endpoint_404s_until_pipeline_writes_one(client: TestClient) -> None:
    job_id = client.post("/api/jobs", json={"topic": "Kites"}).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}/timeline").status_code == 404


def test_timeline_endpoint_serves_persisted_timeline(client: TestClient) -> None:
    from app.core.models import Scene, Timeline
    from app.db.models import Job
    from app.db.session import session_scope

    job_id = client.post("/api/jobs", json={"topic": "Kites"}).json()["job_id"]
    timeline = Timeline(
        job_id=job_id,
        topic="Kites",
        title="Kites",
        voice="aura-2-draco-en",
        scenes=[Scene(id=1, narration="hi", heading="Hi", image_prompt="a kite", end=2.0)],
    )
    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.timeline_json = timeline.model_dump_json()
        session.add(job)
        session.commit()

    body = client.get(f"/api/jobs/{job_id}/timeline").json()
    assert body["title"] == "Kites"
    assert body["scenes"][0]["end"] == 2.0


def test_cors_allows_the_vite_dev_server(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# ------------------------------------------------------------------------------ voices


def test_voices_uses_fallback_when_deepgram_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", _boom)
    voices_api.reset_cache()

    response = client.get("/api/voices")
    assert response.status_code == 200
    ids = [v["id"] for v in response.json()]
    assert {"aura-2-draco-en", "aura-2-pluto-en", "aura-2-hera-en"} <= set(ids)


def test_voices_fallback_is_not_cached(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient outage must not poison the cache for the rest of the process."""

    async def _boom(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", _boom)
    voices_api.reset_cache()
    assert client.get("/api/voices").status_code == 200
    assert voices_api._cache is None


def test_voices_parses_and_sorts_deepgram_payload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "tts": [
            {
                "name": "agathe",
                "canonical_name": "aura-2-agathe-fr",
                "metadata": {"display_name": "Agathe", "accent": "French", "use_cases": ["Chat"]},
            },
            {
                "name": "zeus",
                "canonical_name": "aura-2-zeus-en",
                "metadata": {
                    "display_name": "Zeus",
                    "accent": "American",
                    "tags": ["masculine"],
                    "use_cases": ["Customer Service"],
                },
            },
            {
                # display_name is null for real English voices -> title-cased fallback
                "name": "draco",
                "canonical_name": "aura-2-draco-en",
                "metadata": {
                    "display_name": None,
                    "accent": "British",
                    "tags": ["masculine", "warm"],
                    "use_cases": ["storytelling", "informative"],
                },
            },
            {
                "name": "arcas",
                "canonical_name": "aura-arcas-en",
                "metadata": {
                    "display_name": None,
                    "accent": "American",
                    "tags": ["masculine"],
                    "use_cases": ["storytelling"],
                },
            },
        ]
    }

    async def _ok(self, url, **kwargs):  # noqa: ANN001, ARG001
        return httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _ok)
    voices_api.reset_cache()

    body = client.get("/api/voices").json()
    # French voice dropped; narration voices first, aura-2 ahead of aura-1
    assert [v["id"] for v in body] == ["aura-2-draco-en", "aura-arcas-en", "aura-2-zeus-en"]
    assert body[0] == {
        "id": "aura-2-draco-en",
        "name": "Draco",
        "accent": "British",
        "tags": ["masculine", "warm"],
        "use_cases": ["storytelling", "informative"],
    }

    # second call is served from cache — upstream is not hit again
    async def _fail(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("cache miss: Deepgram was called twice")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fail)
    assert client.get("/api/voices").json() == body
