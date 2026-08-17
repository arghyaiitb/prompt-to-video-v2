"""Tests for the Veo 3.1 motion-clip provider.

Two tiers, matching tests/test_providers.py:

  * Everything outside `TestLive*` runs offline. The HTTP layer is stubbed at
    `httpx.post` / `httpx.get`, so the submit -> poll -> download state machine, the
    failure shapes, and the ffmpeg audio strip are all covered without a 90-second
    generation. The "downloaded" bytes are a locally built stand-in with the measured Veo
    properties (8s, 1280x720, 24 fps, AAC stereo), so the strip is exercised for real.
  * `TestLive*` hits the real API and is skipped unless RUN_LIVE_API=1:

        uv run pytest tests/test_veo_video.py                      # offline only
        RUN_LIVE_API=1 uv run pytest tests/test_veo_video.py        # everything

Artifacts go to pytest's tmp_path, never into the repo.
"""

from __future__ import annotations

import inspect
import logging
import os
import subprocess
from pathlib import Path

import pytest

from app.core.ports import VideoClipProvider
from app.providers import veo_video
from app.providers.veo_video import (
    CLIP_FPS,
    CLIP_HEIGHT,
    CLIP_SECONDS,
    CLIP_WIDTH,
    DEFAULT_MODEL,
    ClipBudgetError,
    PlaceholderVideoProvider,
    VeoTimeoutError,
    VeoVideoProvider,
    VideoClipError,
    clip_budget,
    clips_needed,
    compose_prompt,
    probe_clip,
    strip_audio,
    veo_enabled,
    video_uri_from,
)

LIVE = os.environ.get("RUN_LIVE_API") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_API=1 to hit real APIs")

OPERATION = "models/veo-3.1-fast-generate-preview/operations/2wuadveixogd"
VIDEO_URI = "https://generativelanguage.googleapis.com/v1beta/files/9awrnhhzmich:download?alt=media"

# The clip this suite's shapes were taken from, kept out of the repo. Present only on the
# machine that captured it; the one test that reads it skips elsewhere.
REAL_VEO_CLIP = Path(
    "/private/tmp/claude-501/-Users-argo-ab-prompt-to-video-v2/"
    "17f5789b-d93a-4c4f-af36-254d779b6e1c/scratchpad/veo_clip.mp4"
)

PROMPT = "Slow push-in on an email inbox on a laptop in a dim office"


def _done_payload(uri: str = VIDEO_URI) -> dict:
    """The verified completed-operation body."""
    return {
        "name": OPERATION,
        "done": True,
        "response": {
            "@type": (
                "type.googleapis.com/google.ai.generativelanguage.v1beta."
                "PredictLongRunningResponse"
            ),
            "generateVideoResponse": {"generatedSamples": [{"video": {"uri": uri}}]},
        },
    }


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeVeo:
    """Routes the three calls the provider makes, recording each one.

    `polls` is the sequence of operation bodies handed back in order; the last is repeated
    if the provider polls again. `submits` counts resubmissions, which is how the retry
    and no-resubmit-on-timeout behaviours are asserted.
    """

    def __init__(self, *, polls: list[dict], video: bytes, post_status: int = 200) -> None:
        self.polls = polls
        self.video = video
        self.post_status = post_status
        self.submits: list[dict] = []
        self.poll_count = 0
        self.downloads: list[dict] = []

    def post(self, url, *, params=None, json=None, headers=None, timeout=None):
        assert ":predictLongRunning" in url, url
        self.submits.append({"url": url, "params": params, "body": json})
        if self.post_status != 200:
            return _FakeResponse(status_code=self.post_status, payload={"error": "nope"})
        return _FakeResponse(payload={"name": OPERATION})

    def get(self, url, *, params=None, headers=None, follow_redirects=False, timeout=None):
        if "/files/" in url:
            self.downloads.append(
                {"url": url, "headers": headers, "follow_redirects": follow_redirects}
            )
            return _FakeResponse(content=self.video)
        assert "/operations/" in url, url
        index = min(self.poll_count, len(self.polls) - 1)
        self.poll_count += 1
        return _FakeResponse(payload=self.polls[index])

    def install(self, monkeypatch) -> FakeVeo:
        monkeypatch.setattr(veo_video.httpx, "post", self.post)
        monkeypatch.setattr(veo_video.httpx, "get", self.get)
        return self


@pytest.fixture(autouse=True)
def _no_sleep(request, monkeypatch):
    """Retry backoff and poll gaps are real seconds; the state machine does not need them.

    Deliberately NOT applied to the live tests: without the gap the poll loop would
    hammer the real operations endpoint for the ~50s the generation takes.
    """
    if request.cls is not None and request.cls.__name__.startswith("TestLive"):
        return
    monkeypatch.setattr(veo_video.time, "sleep", lambda _seconds: None)


@pytest.fixture(scope="session")
def veo_like_clip(tmp_path_factory) -> bytes:
    """An 8s 1280x720 24fps h264 clip WITH an AAC stereo track — Veo's measured shape."""
    path = tmp_path_factory.mktemp("veolike") / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate=24:duration=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "2", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path.read_bytes()


def _provider(**kwargs) -> VeoVideoProvider:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("poll_interval", 0.001)
    kwargs.setdefault("max_wait", 5.0)
    return VeoVideoProvider(**kwargs)


# =========================================================== protocol conformance


class TestProtocolConformance:
    def test_veo_provider_satisfies_the_port(self):
        assert isinstance(_provider(), VideoClipProvider)

    def test_placeholder_satisfies_the_port(self):
        assert isinstance(PlaceholderVideoProvider(), VideoClipProvider)

    @pytest.mark.parametrize("cls", [VeoVideoProvider, PlaceholderVideoProvider])
    def test_generate_signature_matches_the_port(self, cls):
        expected = list(inspect.signature(VideoClipProvider.generate).parameters)
        assert list(inspect.signature(cls.generate).parameters) == expected

    def test_default_model_is_the_cost_tier(self):
        assert DEFAULT_MODEL == "veo-3.1-fast-generate-preview"
        assert _provider().model == DEFAULT_MODEL

    def test_explicit_model_wins(self):
        provider = _provider(model="veo-3.1-lite-generate-preview")
        assert provider.model == "veo-3.1-lite-generate-preview"


# =========================================================== happy path


class TestHappyPath:
    def test_submit_poll_download_writes_a_clip(self, monkeypatch, tmp_path, veo_like_clip):
        fake = FakeVeo(
            polls=[{"name": OPERATION}, {"name": OPERATION}, _done_payload()],
            video=veo_like_clip,
        ).install(monkeypatch)

        out = tmp_path / "scene1.mp4"
        result = _provider().generate(PROMPT, 8.0, out)

        assert result == out
        assert out.exists()
        probe = probe_clip(out)
        assert probe.has_video
        assert (probe.width, probe.height) == (CLIP_WIDTH, CLIP_HEIGHT)
        assert probe.fps == pytest.approx(CLIP_FPS, abs=0.01)
        assert probe.duration == pytest.approx(CLIP_SECONDS, abs=0.05)
        assert fake.submits and fake.poll_count == 3 and len(fake.downloads) == 1

    def test_request_body_is_the_verified_shape(self, monkeypatch, tmp_path, veo_like_clip):
        fake = FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        _provider(aspect_ratio="9:16").generate(PROMPT, 8.0, tmp_path / "a.mp4")

        body = fake.submits[0]["body"]
        assert set(body) == {"instances", "parameters"}
        assert body["parameters"] == {"aspectRatio": "9:16"}
        text = body["instances"][0]["prompt"]
        assert PROMPT in text
        assert "no text" in text.lower()
        assert fake.submits[0]["params"] == {"key": "test-key"}
        assert ":predictLongRunning" in fake.submits[0]["url"]

    def test_download_uses_the_header_auth_and_follows_redirects(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        """The file endpoint wants x-goog-api-key, not ?key=, and it 302s to storage."""
        fake = FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")

        call = fake.downloads[0]
        assert call["headers"] == {"x-goog-api-key": "test-key"}
        assert call["follow_redirects"] is True
        assert call["url"] == VIDEO_URI

    def test_last_clip_info_reports_the_measured_clip(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider()
        provider.generate(PROMPT, 8.0, tmp_path / "a.mp4")

        info = provider.last_clip_info
        assert info is not None
        assert info.operation == OPERATION
        assert info.model == DEFAULT_MODEL
        assert info.duration == pytest.approx(CLIP_SECONDS, abs=0.05)
        assert (info.width, info.height) == (CLIP_WIDTH, CLIP_HEIGHT)
        assert info.fps == pytest.approx(CLIP_FPS, abs=0.01)
        assert info.has_audio is False
        assert info.covers_request is True
        assert info.shortfall == 0.0

    def test_native_frame_rate_is_not_resampled(self, monkeypatch, tmp_path, veo_like_clip):
        """24 fps in, 24 fps out: matching the render profile is the renderer's job."""
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider()
        provider.generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert provider.last_clip_info is not None
        assert provider.last_clip_info.fps == pytest.approx(24.0, abs=0.01)

    def test_nested_out_dir_is_created(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        out = tmp_path / "clips" / "deep" / "a.mp4"
        _provider().generate(PROMPT, 8.0, out)
        assert out.exists()

    def test_no_temp_files_left_behind(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        out = tmp_path / "a.mp4"
        _provider().generate(PROMPT, 8.0, out)
        assert [p.name for p in tmp_path.iterdir()] == ["a.mp4"]


# =========================================================== the audio track


class TestAudioIsStripped:
    def test_source_really_has_audio(self, veo_like_clip, tmp_path):
        """Guards the guard: if the fixture lost its audio the strip test proves nothing."""
        src = tmp_path / "src.mp4"
        src.write_bytes(veo_like_clip)
        assert probe_clip(src).has_audio is True

    def test_generated_clip_has_no_audio_stream(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        out = tmp_path / "a.mp4"
        _provider().generate(PROMPT, 8.0, out)

        probe = probe_clip(out)
        assert probe.has_audio is False
        assert probe.has_video is True
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert streams == ["video"]

    def test_strip_audio_preserves_the_video_stream_exactly(self, tmp_path, veo_like_clip):
        src = tmp_path / "src.mp4"
        src.write_bytes(veo_like_clip)
        out = strip_audio(src, tmp_path / "out.mp4")

        before, after = probe_clip(src), probe_clip(out)
        assert (after.width, after.height) == (before.width, before.height)
        assert after.fps == pytest.approx(before.fps, abs=0.01)
        assert after.duration == pytest.approx(before.duration, abs=0.05)
        assert before.has_audio and not after.has_audio

    def test_strip_audio_falls_back_to_reencode_when_a_copy_is_impossible(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        """A container that cannot hold the stream as-is must not fail the whole call."""
        real = veo_video.run_ffmpeg
        calls: list[list[str]] = []

        def no_copy(args, **kwargs):
            calls.append(args)
            if "copy" in args:
                raise veo_video.MediaError("simulated: container cannot hold this codec")
            real(args, **kwargs)

        monkeypatch.setattr(veo_video, "run_ffmpeg", no_copy)
        src = tmp_path / "src.mp4"
        src.write_bytes(veo_like_clip)
        out = strip_audio(src, tmp_path / "out.mp4")

        assert len(calls) == 2 and "copy" not in calls[1]
        probe = probe_clip(out)
        assert probe.has_video and not probe.has_audio


# =========================================================== target_duration honesty


class TestTargetDurationIsARequest:
    def test_short_clip_against_a_long_scene_is_reported_not_hidden(
        self, monkeypatch, tmp_path, veo_like_clip, caplog
    ):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider()
        with caplog.at_level(logging.WARNING, logger="app.providers.veo_video"):
            provider.generate(PROMPT, 14.0, tmp_path / "a.mp4")

        info = provider.last_clip_info
        assert info is not None
        assert info.requested_duration == 14.0
        assert info.duration == pytest.approx(8.0, abs=0.05)
        assert info.shortfall == pytest.approx(6.0, abs=0.05)
        assert info.covers_request is False
        assert "NOT covered" in caplog.text

    def test_no_warning_when_the_clip_covers_the_scene(
        self, monkeypatch, tmp_path, veo_like_clip, caplog
    ):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="app.providers.veo_video"):
            _provider().generate(PROMPT, 5.0, tmp_path / "a.mp4")
        assert "NOT covered" not in caplog.text

    @pytest.mark.parametrize("target", [0.0, -1.0])
    def test_non_positive_target_rejected(self, tmp_path, target):
        with pytest.raises(ValueError, match="must be positive"):
            _provider().generate(PROMPT, target, tmp_path / "a.mp4")


# =========================================================== failure shapes


class TestFailureShapes:
    def test_done_with_an_error_payload_raises_after_retrying(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        errored = {
            "name": OPERATION,
            "done": True,
            "error": {"code": 3, "message": "prompt violates policy"},
        }
        fake = FakeVeo(polls=[errored], video=veo_like_clip).install(monkeypatch)

        with pytest.raises(VideoClipError, match="no video after 3 attempts"):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert len(fake.submits) == veo_video.EMPTY_RESULT_ATTEMPTS

    def test_error_payload_message_reaches_the_log(
        self, monkeypatch, tmp_path, veo_like_clip, caplog
    ):
        errored = {"name": OPERATION, "done": True, "error": {"code": 3, "message": "boom"}}
        FakeVeo(polls=[errored], video=veo_like_clip).install(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="app.providers.veo_video"):
            with pytest.raises(VideoClipError):
                _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert "boom" in caplog.text
        assert "code=3" in caplog.text

    def test_empty_generated_samples_raises_after_retrying(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        empty = {
            "name": OPERATION,
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [],
                    "raiMediaFilteredReasons": ["blocked"],
                }
            },
        }
        fake = FakeVeo(polls=[empty], video=veo_like_clip).install(monkeypatch)
        with pytest.raises(VideoClipError, match="no video after 3 attempts"):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert len(fake.submits) == 3
        assert len(fake.downloads) == 0

    def test_retry_uses_a_varied_prompt(self, monkeypatch, tmp_path, veo_like_clip):
        """A refusal can be deterministic per wording, so the identical string is useless."""
        empty = {"name": OPERATION, "done": True, "response": {"generateVideoResponse": {}}}
        fake = FakeVeo(polls=[empty], video=veo_like_clip).install(monkeypatch)
        with pytest.raises(VideoClipError):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        prompts = [s["body"]["instances"][0]["prompt"] for s in fake.submits]
        assert len(set(prompts)) == len(prompts) == 3

    def test_a_retry_that_succeeds_returns_the_clip(self, monkeypatch, tmp_path, veo_like_clip):
        empty = {"name": OPERATION, "done": True, "response": {"generateVideoResponse": {}}}
        fake = FakeVeo(polls=[empty, _done_payload()], video=veo_like_clip).install(monkeypatch)
        out = _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert out.exists()
        assert len(fake.submits) == 2

    def test_timeout_names_the_operation_and_does_not_resubmit(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        fake = FakeVeo(polls=[{"name": OPERATION}], video=veo_like_clip).install(monkeypatch)
        provider = _provider(max_wait=0.05, poll_interval=0.01)

        with pytest.raises(VeoTimeoutError) as excinfo:
            provider.generate(PROMPT, 8.0, tmp_path / "a.mp4")

        assert excinfo.value.operation == OPERATION
        assert OPERATION in str(excinfo.value)
        assert "fetch_completed" in str(excinfo.value)
        # Resubmitting a job that is probably still running would double the bill.
        assert len(fake.submits) == 1
        assert provider.last_clip_info is None
        assert not (tmp_path / "a.mp4").exists()

    def test_timeout_is_a_video_clip_error(self):
        assert issubclass(VeoTimeoutError, VideoClipError)

    def test_submit_without_an_operation_name_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            veo_video.httpx, "post", lambda *a, **k: _FakeResponse(payload={"metadata": {}})
        )
        with pytest.raises(VideoClipError, match="no operation name"):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")

    def test_client_error_status_is_not_retried(self, monkeypatch, tmp_path, veo_like_clip):
        fake = FakeVeo(polls=[_done_payload()], video=veo_like_clip, post_status=400)
        fake.install(monkeypatch)
        with pytest.raises(VideoClipError, match="HTTP 400"):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert len(fake.submits) == 1

    def test_transient_status_is_retried(self, monkeypatch, tmp_path, veo_like_clip):
        calls = {"n": 0}

        def flaky_post(url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(status_code=503, payload={"error": "busy"})
            return _FakeResponse(payload={"name": OPERATION})

        fake = FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        monkeypatch.setattr(veo_video.httpx, "post", flaky_post)
        out = _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert out.exists() and calls["n"] == 2 and len(fake.downloads) == 1

    def test_empty_download_body_raises(self, monkeypatch, tmp_path):
        FakeVeo(polls=[_done_payload()], video=b"").install(monkeypatch)
        with pytest.raises(VideoClipError, match="empty body"):
            _provider().generate(PROMPT, 8.0, tmp_path / "a.mp4")

    def test_missing_api_key_fails_before_any_call(self, monkeypatch, tmp_path):
        def explode(*args, **kwargs):
            raise AssertionError("should not have called the API")

        monkeypatch.setattr(veo_video.httpx, "post", explode)
        with pytest.raises(VideoClipError, match="GEMINI_API_KEY"):
            VeoVideoProvider(api_key="").generate(PROMPT, 8.0, tmp_path / "a.mp4")


class TestVideoUriFrom:
    def test_extracts_the_verified_shape(self):
        assert video_uri_from(_done_payload()) == VIDEO_URI

    def test_error_object_beats_a_present_response(self):
        payload = _done_payload()
        payload["error"] = {"code": 7, "message": "denied"}
        with pytest.raises(VideoClipError, match="code=7"):
            video_uri_from(payload)

    def test_sample_without_a_uri(self):
        payload = {"response": {"generateVideoResponse": {"generatedSamples": [{"video": {}}]}}}
        with pytest.raises(VideoClipError, match="no video uri"):
            video_uri_from(payload)

    def test_no_response_key_at_all(self):
        with pytest.raises(VideoClipError, match="no generatedSamples"):
            video_uri_from({"name": OPERATION, "done": True})


class TestFetchCompleted:
    def test_downloads_a_finished_operation_without_resubmitting(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        fake = FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider()
        out = provider.fetch_completed(OPERATION, tmp_path / "a.mp4", requested_duration=12.0)
        assert out.exists()
        assert fake.submits == []
        assert provider.last_clip_info is not None
        assert provider.last_clip_info.shortfall == pytest.approx(4.0, abs=0.05)

    def test_still_running_operation_raises_with_the_id(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[{"name": OPERATION}], video=veo_like_clip).install(monkeypatch)
        with pytest.raises(VeoTimeoutError, match="still running"):
            _provider().fetch_completed(OPERATION, tmp_path / "a.mp4", requested_duration=8.0)


# =========================================================== cost guard


class TestCostGuard:
    def test_clip_budget_bills_the_fixed_length_not_the_scene_length(self):
        budget = clip_budget([4.0, 8.0, 14.0])
        assert budget.clips == 3
        assert budget.billed_seconds == 24.0
        assert budget.requested_seconds == 26.0
        assert budget.uncovered_seconds == pytest.approx(6.0)

    def test_clip_budget_ignores_empty_scenes(self):
        assert clip_budget([]).clips == 0
        assert clip_budget([0.0, -3.0, 5.0]).clips == 1

    def test_clip_budget_summary_is_readable(self):
        text = clip_budget([14.0]).summary()
        assert "1 Veo clip" in text and "8s billed" in text and "6.0s not covered" in text

    def test_clips_per_scene_multiplies_the_bill(self):
        budget = clip_budget([14.0], clips_per_scene=2)
        assert budget.clips == 2
        assert budget.billed_seconds == 16.0
        assert budget.uncovered_seconds == 0.0

    def test_clips_per_scene_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            clip_budget([8.0], clips_per_scene=0)

    @pytest.mark.parametrize(
        ("duration", "expected"), [(0.0, 0), (1.0, 1), (8.0, 1), (8.5, 2), (17.0, 3)]
    )
    def test_clips_needed(self, duration, expected):
        assert clips_needed(duration) == expected

    def test_max_clips_caps_a_runaway_caller(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider(max_clips=1)
        provider.generate(PROMPT, 8.0, tmp_path / "a.mp4")
        with pytest.raises(ClipBudgetError, match="max_clips=1"):
            provider.generate(PROMPT, 8.0, tmp_path / "b.mp4")
        assert provider.clips_generated == 1

    def test_cap_can_be_disabled(self, monkeypatch, tmp_path, veo_like_clip):
        FakeVeo(polls=[_done_payload()], video=veo_like_clip).install(monkeypatch)
        provider = _provider(max_clips=None)
        for index in range(2):
            provider.generate(PROMPT, 8.0, tmp_path / f"{index}.mp4")
        assert provider.clips_generated == 2

    def test_failed_generations_do_not_count_against_the_cap(
        self, monkeypatch, tmp_path, veo_like_clip
    ):
        empty = {"name": OPERATION, "done": True, "response": {"generateVideoResponse": {}}}
        FakeVeo(polls=[empty], video=veo_like_clip).install(monkeypatch)
        provider = _provider(max_clips=1)
        with pytest.raises(VideoClipError):
            provider.generate(PROMPT, 8.0, tmp_path / "a.mp4")
        assert provider.clips_generated == 0

    def test_veo_enabled_defaults_to_false_when_the_flag_is_absent(self):
        class NoFlag:
            pass

        assert veo_enabled(NoFlag()) is False

    @pytest.mark.parametrize(("flag", "expected"), [(True, True), (False, False)])
    def test_veo_enabled_reads_the_flag(self, flag, expected):
        class WithFlag:
            video_enable_veo = flag

        assert veo_enabled(WithFlag()) is expected

    def test_real_settings_do_not_crash_the_flag_lookup(self):
        assert isinstance(veo_enabled(), bool)


# =========================================================== prompt composition


class TestComposePrompt:
    def test_first_attempt_adds_style_and_no_text_guards(self):
        text = compose_prompt("a quiet server room")
        assert text.startswith("a quiet server room.")
        assert "continuous shot" in text
        assert "No text" in text

    def test_third_attempt_falls_back_to_a_generic_prompt(self):
        text = compose_prompt("a quiet server room", attempt=3)
        assert "quiet server room" not in text
        assert "No text" in text

    def test_attempts_differ(self):
        variants = {compose_prompt("a quiet server room", attempt=n) for n in (1, 2, 3)}
        assert len(variants) == 3

    def test_existing_no_text_clause_is_not_duplicated(self):
        text = compose_prompt("a lab, no text anywhere")
        assert text.lower().count("no text") == 1

    def test_empty_prompt_still_produces_something_generatable(self):
        assert compose_prompt("").strip()
        assert compose_prompt("   ", attempt=1).strip()


# =========================================================== placeholder


class TestPlaceholderVideoProvider:
    def test_produces_a_playable_silent_clip(self, tmp_path):
        out = PlaceholderVideoProvider().generate("intro scene", 8.0, tmp_path / "p.mp4")
        probe = probe_clip(out)
        assert probe.has_video and not probe.has_audio
        assert (probe.width, probe.height) == (CLIP_WIDTH, CLIP_HEIGHT)
        assert probe.fps == pytest.approx(CLIP_FPS, abs=0.5)

    def test_mirrors_veos_fixed_length_by_default(self, tmp_path):
        provider = PlaceholderVideoProvider()
        provider.generate("intro", 20.0, tmp_path / "p.mp4")
        info = provider.last_clip_info
        assert info is not None
        assert info.duration == pytest.approx(CLIP_SECONDS, abs=0.15)
        assert info.covers_request is False

    def test_can_honour_the_requested_duration_instead(self, tmp_path):
        out = PlaceholderVideoProvider(fixed_duration=None).generate(
            "intro", 3.0, tmp_path / "p.mp4"
        )
        assert probe_clip(out).duration == pytest.approx(3.0, abs=0.15)

    def test_the_clip_actually_moves(self, tmp_path):
        """A still placeholder would not exercise the renderer's motion path."""
        out = PlaceholderVideoProvider(fixed_duration=4.0).generate(
            "drifting gradient", 4.0, tmp_path / "p.mp4"
        )
        frames = []
        for index, timestamp in enumerate(("0", "3.5")):
            frame = tmp_path / f"f{index}.png"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", timestamp,
                 "-i", str(out), "-frames:v", "1", str(frame)],
                check=True, capture_output=True,
            )
            frames.append(frame.read_bytes())
        assert frames[0] != frames[1]

    def test_same_prompt_gives_the_same_placeholder(self, tmp_path):
        provider = PlaceholderVideoProvider(fixed_duration=1.0)
        a = provider.generate("scene one", 1.0, tmp_path / "a.mp4").read_bytes()
        b = provider.generate("scene one", 1.0, tmp_path / "b.mp4").read_bytes()
        assert a == b

    def test_different_prompts_look_different(self, tmp_path):
        provider = PlaceholderVideoProvider(fixed_duration=1.0)
        a = provider.generate("scene one", 1.0, tmp_path / "a.mp4").read_bytes()
        b = provider.generate("a completely different scene", 1.0, tmp_path / "b.mp4").read_bytes()
        assert a != b

    def test_makes_no_network_calls(self, monkeypatch, tmp_path):
        def explode(*args, **kwargs):
            raise AssertionError("placeholder must not touch the network")

        monkeypatch.setattr(veo_video.httpx, "post", explode)
        monkeypatch.setattr(veo_video.httpx, "get", explode)
        assert PlaceholderVideoProvider(fixed_duration=1.0).generate(
            "x", 1.0, tmp_path / "p.mp4"
        ).exists()

    def test_non_positive_target_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be positive"):
            PlaceholderVideoProvider().generate("x", 0.0, tmp_path / "p.mp4")


# =========================================================== the captured artifact


@pytest.mark.skipif(not REAL_VEO_CLIP.exists(), reason="captured Veo clip not on this machine")
class TestRealCapturedClip:
    """The measured constants, checked against an actual Veo download."""

    def test_measured_properties_hold(self):
        probe = probe_clip(REAL_VEO_CLIP)
        assert probe.duration == pytest.approx(8.0, abs=0.01)
        assert (probe.width, probe.height) == (1280, 720)
        assert probe.fps == pytest.approx(24.0, abs=0.01)
        assert probe.has_audio is True

    def test_strip_leaves_video_only(self, tmp_path):
        out = strip_audio(REAL_VEO_CLIP, tmp_path / "silent.mp4")
        probe = probe_clip(out)
        assert probe.has_video and not probe.has_audio
        assert probe.duration == pytest.approx(8.0, abs=0.05)
        assert probe.fps == pytest.approx(24.0, abs=0.01)


# =========================================================== live API


@live_only
class TestLiveVeo:
    def test_generates_real_motion_footage_without_audio(self, tmp_path):
        import time as _time

        provider = VeoVideoProvider(model="veo-3.1-fast-generate-preview")
        started = _time.monotonic()
        out = provider.generate(
            "Slow cinematic push-in on a laptop screen showing an email inbox in a dim "
            "modern office, warm key light, dust in the air",
            14.0,
            tmp_path / "live.mp4",
        )
        elapsed = _time.monotonic() - started

        info = provider.last_clip_info
        assert info is not None
        print(
            f"\nlive veo: {elapsed:.1f}s wall clock, {out.stat().st_size / 1e6:.2f} MB, "
            f"{info.duration:.3f}s {info.width}x{info.height} @{info.fps:g}fps, "
            f"audio={info.has_audio}, shortfall={info.shortfall:.3f}s, op={info.operation}"
        )
        assert out.exists() and out.stat().st_size > 100_000
        assert info.has_audio is False
        assert (info.width, info.height) == (CLIP_WIDTH, CLIP_HEIGHT)
        assert info.duration == pytest.approx(CLIP_SECONDS, abs=0.2)
        assert info.fps == pytest.approx(CLIP_FPS, abs=0.5)
        # The whole point of last_clip_info: 14s asked for, 8s delivered.
        assert info.covers_request is False
        assert info.shortfall == pytest.approx(6.0, abs=0.2)
