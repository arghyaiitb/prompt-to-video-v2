"""HTTP-level tests. Real app, real SQLite (a throwaway file), no network, no ffmpeg."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import jobs as jobs_api
from app.api import voices as voices_api
from app.api.themes import HEX_COLOUR, default_theme_name, validate_palette
from app.core.config import get_settings
from app.core.models import Theme
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


def test_a_job_created_without_a_logo_id_keeps_the_bundled_mark(client: TestClient) -> None:
    """The default-preservation contract: an untouched request renders as it always did.

    `logo_id: null` is what "the bundled mark" looks like on the wire, and it is what every
    row created before uploads existed holds. See test_logos.py for the upload surface.
    """
    job_id = client.post("/api/jobs", json={"topic": "Tides", "slide_count": 2}).json()["job_id"]
    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["logo_id"] is None
    assert body["logo_url"] is None
    assert client.get("/api/jobs").json()[0]["logo_id"] is None


def test_an_unknown_logo_id_does_not_silently_fall_back(client: TestClient) -> None:
    response = client.post(
        "/api/jobs", json={"topic": "Tides", "slide_count": 2, "logo_id": "deadbeef" * 4}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "unknown_logo"
    assert client.get("/api/jobs").json() == []


def test_the_logo_catalogue_is_empty_until_something_is_uploaded(client: TestClient) -> None:
    assert client.get("/api/logos").json() == []


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


# ------------------------------------------------------------------------------ themes


def test_themes_endpoint_lists_presets_with_contrast(client: TestClient) -> None:
    """The picker needs swatches *and* numbers — a ratio-free picker invites bad palettes."""
    response = client.get("/api/themes")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body, "no themes offered"

    ids = [entry["id"] for entry in body]
    assert len(ids) == len(set(ids))
    assert default_theme_name() in ids

    for entry in body:
        assert entry["name"]
        assert set(entry["swatches"]) >= {"bg", "surface", "text", "muted", "accent"}
        assert all(re.fullmatch(HEX_COLOUR, c) for c in entry["swatches"].values())
        assert entry["contrast"]["text_on_bg"] >= 4.5, entry["id"]
        assert isinstance(entry["is_light"], bool)


def test_offered_themes_are_all_accepted_by_post_jobs(client: TestClient) -> None:
    """Anything the picker can show must be a legal choice — no dead options."""
    for entry in client.get("/api/themes").json():
        created = client.post("/api/jobs", json={"topic": "Phishing", "theme": entry["id"]})
        assert created.status_code == 202, created.text
        detail = client.get(f"/api/jobs/{created.json()['job_id']}").json()
        assert detail["theme"] == entry["id"]


def test_job_defaults_to_the_default_theme(client: TestClient) -> None:
    job_id = client.post("/api/jobs", json={"topic": "Data handling"}).json()["job_id"]
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["theme"] == default_theme_name()
    assert detail["theme_custom"] is None
    assert detail["bullets_per_slide"] == 4
    assert detail["tone"] is None


def test_unknown_theme_id_falls_back_to_the_default(client: TestClient) -> None:
    """A stale id from an old frontend build must not be echoed back as if it rendered."""
    created = client.post("/api/jobs", json={"topic": "Badges", "theme": "chartreuse-disco"})
    assert created.status_code == 202, created.text
    detail = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert detail["theme"] == default_theme_name()


def test_unknown_theme_falls_back_even_without_the_preset_catalogue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a known catalogue, normalisation is on the id; the resolved Theme is the
    backstop either way (see test_db.py)."""
    monkeypatch.setattr(jobs_api, "known_theme_ids", lambda: {"midnight", "daylight"})
    monkeypatch.setattr(jobs_api, "default_theme_name", lambda: "midnight")

    job_id = client.post("/api/jobs", json={"topic": "Badges", "theme": "nope"}).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["theme"] == "midnight"

    job_id = client.post("/api/jobs", json={"topic": "Badges", "theme": "daylight"}).json()[
        "job_id"
    ]
    assert client.get(f"/api/jobs/{job_id}").json()["theme"] == "daylight"


def test_custom_palette_is_stored_when_it_passes_contrast(client: TestClient) -> None:
    palette = client.get("/api/themes").json()[0]["swatches"]
    created = client.post("/api/jobs", json={"topic": "Brand", "theme_custom": palette})
    assert created.status_code == 202, created.text

    detail = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert detail["theme"] == "custom"
    assert detail["theme_custom"]["bg"] == palette["bg"]
    assert detail["theme_custom"]["text"] == palette["text"]


def test_custom_palette_failing_contrast_is_422_with_specific_failures(
    client: TestClient,
) -> None:
    """Unreadable text is burned into pixels — this has to fail before the render."""
    palette = {
        "bg": "#808080",
        "surface": "#8A8A8A",
        "text": "#7A7A7A",
        "muted": "#888888",
        "accent": "#909090",
    }
    response = client.post("/api/jobs", json={"topic": "Brand", "theme_custom": palette})
    assert response.status_code == 422, response.text

    detail = response.json()["detail"]
    assert detail["error"] == "theme_contrast_failed"
    failures = detail["failures"]
    assert failures, "rejected without saying why"
    # Every failure names a pair and a measured ratio, not just "bad contrast".
    assert all(":1" in message for message in failures), failures
    assert any("text_on_bg" in message for message in failures), failures
    assert detail["contrast"]["text_on_bg"] < 4.5

    # ...and the UI is handed a palette that actually passes, for one-click correction.
    fix = detail["suggested_fix"]
    assert set(fix) == {"bg", "surface", "text", "muted", "accent"}
    assert all(re.fullmatch(HEX_COLOUR, c) for c in fix.values())
    assert detail["suggested_contrast"]["text_on_bg"] > detail["contrast"]["text_on_bg"]
    assert validate_palette(Theme(**fix)) == []

    # nothing was queued
    assert client.get("/api/jobs").json() == []


def test_custom_palette_rejects_short_hex_and_junk(client: TestClient) -> None:
    """`Theme._luminance` slices fixed digit pairs, so `#fff` would crash the renderer."""
    base = {
        "bg": "#0B1220",
        "surface": "#131F35",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "accent": "#F5A524",
    }
    for bad in ("#fff", "white", "0B1220", "#GGGGGG", ""):
        response = client.post(
            "/api/jobs", json={"topic": "Brand", "theme_custom": {**base, "bg": bad}}
        )
        assert response.status_code == 422, f"{bad!r} was accepted"


def test_custom_palette_must_be_complete(client: TestClient) -> None:
    """A half-palette mixed with preset defaults is a combination nobody checked."""
    response = client.post(
        "/api/jobs", json={"topic": "Brand", "theme_custom": {"bg": "#0B1220"}}
    )
    assert response.status_code == 422


# ------------------------------------------------------- bullets_per_slide and tone


@pytest.mark.parametrize("bullets", [3, 4, 5])
def test_bullets_per_slide_in_range_is_stored(client: TestClient, bullets: int) -> None:
    created = client.post("/api/jobs", json={"topic": "Passwords", "bullets_per_slide": bullets})
    assert created.status_code == 202, created.text
    detail = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert detail["bullets_per_slide"] == bullets


@pytest.mark.parametrize("bullets", [2, 6, 0, -1, 99])
def test_bullets_per_slide_out_of_range_is_rejected(client: TestClient, bullets: int) -> None:
    response = client.post("/api/jobs", json={"topic": "Passwords", "bullets_per_slide": bullets})
    assert response.status_code == 422


@pytest.mark.parametrize("tone", ["new_hires", "all_staff", "technical", "executives"])
def test_tone_is_stored_and_reflected(client: TestClient, tone: str) -> None:
    created = client.post("/api/jobs", json={"topic": "Incidents", "tone": tone})
    assert created.status_code == 202, created.text
    detail = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert detail["tone"] == tone


def test_unknown_tone_is_rejected(client: TestClient) -> None:
    """A free-text tone would be stored and then silently ignored by the prompt."""
    assert (
        client.post("/api/jobs", json={"topic": "Incidents", "tone": "pirate"}).status_code == 422
    )


def test_list_jobs_reports_theme_and_script_choices(client: TestClient) -> None:
    client.post(
        "/api/jobs",
        json={"topic": "Clean desk", "bullets_per_slide": 5, "tone": "executives"},
    )
    row = client.get("/api/jobs").json()[0]
    assert row["theme"] == default_theme_name()
    assert row["bullets_per_slide"] == 5
    assert row["tone"] == "executives"


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
    assert "deepgram" not in voices_api._cache


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


# ----------------------------------------------------------------------------- engines


def test_engines_endpoint_reports_ssml_support_and_the_default(client: TestClient) -> None:
    """The two fields the frontend acts on: which engine to preselect, and SSML support.

    `supports_ssml` here must match the provider class attribute the pipeline reads, or
    the picker promises pauses the engine will read out loud as words.
    """
    response = client.get("/api/engines")
    assert response.status_code == 200, response.text
    body = response.json()

    by_id = {engine["id"]: engine for engine in body}
    assert set(by_id) == {"deepgram", "polly"}
    assert by_id["deepgram"]["supports_ssml"] is False
    assert by_id["polly"]["supports_ssml"] is True
    assert [engine["id"] for engine in body if engine["default"]] == ["deepgram"]
    assert by_id["polly"]["default_voice"] == get_settings().video_default_polly_voice
    assert by_id["deepgram"]["default_voice"] == get_settings().video_default_tts_voice


def test_engines_available_is_false_when_credentials_are_absent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Availability is measured, never assumed.

    A greyed-out engine with a reason costs a user nothing; an engine offered without
    credentials costs them a render that dies partway through.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "deepgram_api_key", "")
    monkeypatch.setattr(settings, "aws_access_key_id", "")
    monkeypatch.setattr(settings, "aws_secret_access_key", "")

    by_id = {engine["id"]: engine for engine in client.get("/api/engines").json()}
    assert by_id["deepgram"]["available"] is False
    assert "DEEPGRAM_API_KEY" in by_id["deepgram"]["reason"]
    assert by_id["polly"]["available"] is False
    assert "AWS credentials" in by_id["polly"]["reason"]
    # still listed and still labelled — the picker shows why, it does not silently drop them
    assert by_id["polly"]["supports_ssml"] is True


def test_engines_reports_polly_available_when_creds_and_provider_are_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "aws_access_key_id", "AKIAFAKE")
    monkeypatch.setattr(settings, "aws_secret_access_key", "secret")

    polly = {engine["id"]: engine for engine in client.get("/api/engines").json()}["polly"]
    # boto3 is a hard dependency; the provider module is the only remaining variable
    if polly["available"]:
        assert polly["reason"] is None
    else:
        assert "polly_tts" in polly["reason"]


# --------------------------------------------------------------- engine-scoped voices


def test_voices_are_engine_scoped(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """`?engine=polly` must not serve Aura ids — they are what the mismatch 422 rejects."""

    async def _boom(self, url, **kwargs):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", _boom)
    # both engines offline: this asserts the *scoping*, not either upstream
    monkeypatch.setattr(voices_api, "_polly_provider_voices", lambda tier: None)
    voices_api.reset_cache()

    deepgram_ids = [v["id"] for v in client.get("/api/voices?engine=deepgram").json()]
    assert all(vid.startswith("aura") for vid in deepgram_ids)
    # no-param call is unchanged: it is the default engine's list
    assert client.get("/api/voices").json() == client.get("/api/voices?engine=deepgram").json()

    polly = client.get("/api/voices?engine=polly").json()
    polly_ids = [v["id"] for v in polly]
    assert polly_ids, "polly must offer at least one voice"
    assert not any(vid.startswith("aura") for vid in polly_ids)
    assert get_settings().video_default_polly_voice in polly_ids
    assert set(polly_ids).isdisjoint(deepgram_ids)


def test_polly_voices_all_support_the_configured_tier(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polly sends the configured tier verbatim, so a voice that lacks it would 400."""
    monkeypatch.setattr(get_settings(), "video_polly_engine", "generative")
    monkeypatch.setattr(voices_api, "_polly_provider_voices", lambda tier: None)
    voices_api.reset_cache()
    supported = {name: tiers for name, _, tiers in voices_api.POLLY_VOICES}

    offered = client.get("/api/voices?engine=polly").json()
    assert offered
    for voice in offered:
        assert "generative" in supported[voice["id"]], voice["id"]


def test_polly_voices_come_from_the_provider_catalogue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider owns the catalogue; this endpoint only reshapes what it returns.

    Also pins the tier filter: `list_voices` is asked for the configured tier, because
    `PollySynthesizer` sends that tier verbatim and Polly 400s a voice that lacks it.
    """
    import app.providers.polly_tts as polly_tts

    calls: list[dict] = []

    def _fake_list_voices(*, language_code=None, engine=None, **kwargs):  # noqa: ANN001, ANN003
        calls.append({"language_code": language_code, "engine": engine})
        # raw DescribeVoices shape, to prove the endpoint normalises AWS's own keys
        return [
            {
                "Id": "Ruth",
                "Name": "Ruth",
                "Gender": "Female",
                "LanguageCode": "en-US",
                "LanguageName": "US English",
                "SupportedEngines": ["generative", "long-form", "neural"],
            }
        ]

    monkeypatch.setattr(polly_tts, "list_voices", _fake_list_voices)
    monkeypatch.setattr(get_settings(), "video_polly_engine", "generative")
    voices_api.reset_cache()

    body = client.get("/api/voices?engine=polly").json()
    assert calls == [{"language_code": "en-US", "engine": "generative"}]
    assert body == [
        {
            "id": "Ruth",
            "name": "Ruth",
            "accent": "American",
            "tags": ["feminine", "generative", "long-form", "neural"],
            # empty because AWS does not publish use cases: gender and tier are derived
            # here, editorial labels are the provider's to add (`polly_tts.shape_voice`)
            "use_cases": [],
        }
    ]

    # served from cache on the second call — the provider is not asked twice
    body_again = client.get("/api/voices?engine=polly").json()
    assert body_again == body
    assert len(calls) == 1


def test_polly_voices_fall_back_when_the_provider_is_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unimportable provider degrades to the measured catalogue, never a 500."""
    monkeypatch.setattr(voices_api, "_polly_provider_voices", lambda tier: None)
    voices_api.reset_cache()

    body = client.get("/api/voices?engine=polly").json()
    assert [v["id"] for v in body][0] == get_settings().video_default_polly_voice
    # and the fallback is not cached: the provider may land while this process runs
    assert "polly" not in voices_api._cache


def test_unknown_voice_engine_is_422_and_names_the_valid_ids(client: TestClient) -> None:
    response = client.get("/api/voices?engine=elevenlabs")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "unknown_engine"
    assert set(detail["known_engines"]) == {"deepgram", "polly"}


# ------------------------------------------------------------- tts_engine on POST /jobs


def test_job_defaults_to_the_default_engine(client: TestClient) -> None:
    job_id = client.post("/api/jobs", json={"topic": "Tides"}).json()["job_id"]
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["tts_engine"] == "deepgram"
    # and the voice is filled in from that engine rather than left empty
    assert detail["voice"] == get_settings().video_default_tts_voice


def test_requested_engine_is_stored_and_exposed(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"topic": "Phishing", "tts_engine": "polly"})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["tts_engine"] == "polly"
    # the voice defaults to Polly's, NOT to the Deepgram default in video_default_tts_voice
    assert detail["voice"] == get_settings().video_default_polly_voice
    assert detail["voice"] in [v["id"] for v in client.get("/api/voices?engine=polly").json()]


@pytest.mark.parametrize("requested", ["POLLY", " polly "])
def test_engine_id_is_case_and_space_insensitive(client: TestClient, requested: str) -> None:
    job_id = client.post("/api/jobs", json={"topic": "T", "tts_engine": requested}).json()[
        "job_id"
    ]
    assert client.get(f"/api/jobs/{job_id}").json()["tts_engine"] == "polly"


def test_unknown_engine_falls_back_to_the_default(client: TestClient) -> None:
    """Same contract as an unknown theme: render on the default, and STORE the default.

    Storing the raw request would make GET /api/jobs/{id} claim narration came from an
    engine that does not exist — and the engine decides whether SSML was sent at all.
    """
    response = client.post(
        "/api/jobs", json={"topic": "Tides", "tts_engine": "elevenlabs", "voice": ""}
    )
    assert response.status_code == 202, response.text
    detail = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert detail["tts_engine"] == "deepgram"
    assert detail["voice"] == get_settings().video_default_tts_voice


def test_offered_engines_are_all_accepted_by_post_jobs(client: TestClient) -> None:
    """Anything GET /api/engines lists must be creatable — no advertising a dead id."""
    for engine in client.get("/api/engines").json():
        response = client.post("/api/jobs", json={"topic": "T", "tts_engine": engine["id"]})
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}").json()["tts_engine"] == engine["id"]


def test_deepgram_voice_with_polly_engine_is_422_naming_both(client: TestClient) -> None:
    """The single most likely user-facing bug: engine switched, stale voice submitted.

    Rejected rather than normalised. The voice is the most audible property of a
    multi-minute render, the caller can list valid voices from this same API, and Polly
    would answer an Aura model id with a ValidationException listing 100 voice names.
    """
    response = client.post(
        "/api/jobs",
        json={"topic": "Phishing", "tts_engine": "polly", "voice": "aura-2-draco-en"},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "voice_engine_mismatch"
    # names BOTH sides, plus the way out
    assert detail["voice"] == "aura-2-draco-en"
    assert detail["voice_engine"] == "deepgram"
    assert detail["tts_engine"] == "polly"
    assert detail["engine_default_voice"] == get_settings().video_default_polly_voice
    assert "aura-2-draco-en" in detail["message"] and "polly" in detail["message"]


def test_polly_voice_with_deepgram_engine_is_422(client: TestClient) -> None:
    """The mirror case. Deepgram would treat `Matthew` as an unknown model id."""
    response = client.post(
        "/api/jobs",
        json={"topic": "Phishing", "tts_engine": "deepgram", "voice": "Matthew"},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["voice_engine"] == "polly"
    assert detail["tts_engine"] == "deepgram"


def test_mismatched_voice_creates_no_job(client: TestClient) -> None:
    """The 422 is raised before the INSERT — a rejected request leaves no row behind."""
    before = len(client.get("/api/jobs").json())
    client.post(
        "/api/jobs",
        json={"topic": "Phishing", "tts_engine": "polly", "voice": "aura-2-draco-en"},
    )
    assert len(client.get("/api/jobs").json()) == before


def test_matching_voice_and_engine_are_accepted(client: TestClient) -> None:
    for engine, voice in (("polly", "Ruth"), ("deepgram", "aura-2-hera-en")):
        response = client.post(
            "/api/jobs", json={"topic": "T", "tts_engine": engine, "voice": voice}
        )
        assert response.status_code == 202, response.text
        detail = client.get(f"/api/jobs/{response.json()['job_id']}").json()
        assert (detail["tts_engine"], detail["voice"]) == (engine, voice)


def test_an_unattributable_voice_is_passed_through(client: TestClient) -> None:
    """Only a voice that demonstrably belongs to another engine is rejected.

    The Deepgram catalogue is fetched live and degrades to three entries offline, so
    validating against catalogue membership would reject 50 valid voices on a bad DNS day.
    """
    response = client.post(
        "/api/jobs", json={"topic": "T", "tts_engine": "deepgram", "voice": "some-new-model-en"}
    )
    assert response.status_code == 202, response.text
    detail = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert detail["voice"] == "some-new-model-en"


def test_engine_is_reported_in_the_job_list(client: TestClient) -> None:
    client.post("/api/jobs", json={"topic": "A", "tts_engine": "polly"})
    rows = client.get("/api/jobs").json()
    assert rows[0]["tts_engine"] == "polly"


def test_polly_fallback_is_never_empty_for_an_odd_tier(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty picker is a worse failure than an unfiltered one."""
    monkeypatch.setattr(voices_api, "_polly_provider_voices", lambda tier: None)
    for tier in ("auto", "wobble", ""):
        monkeypatch.setattr(get_settings(), "video_polly_engine", tier)
        voices_api.reset_cache()
        body = client.get("/api/voices?engine=polly").json()
        assert body, tier
        assert get_settings().video_default_polly_voice in [v["id"] for v in body]
