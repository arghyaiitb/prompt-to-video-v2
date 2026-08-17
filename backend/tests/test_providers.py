"""Tests for the external-service adapters.

Two tiers:

  * Everything outside `TestLive*` classes runs offline. Network transport is stubbed at
    the `generate_content` / `httpx.post` seam, so the response-parsing quirks and the
    ffmpeg fitting paths are covered without spending an API call.
  * `TestLive*` classes hit the real vendors and are skipped unless RUN_LIVE_API=1:

        uv run pytest tests/test_providers.py                     # offline only
        RUN_LIVE_API=1 uv run pytest tests/test_providers.py      # everything
        RUN_LIVE_API=1 uv run pytest tests/test_providers.py \
            -k TestLive                                           # live only

    Note that `-k` matching is case-insensitive, so "-k TestLive" is the precise selector;
    a bare "-k live" also catches offline tests with "delivery" in the name.

Artifacts go to pytest's tmp_path, never into the repo.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from app.core.models import Motion, Word
from app.core.ports import (
    Aligner,
    ImageProvider,
    MusicProvider,
    ScriptProvider,
    SpeechSynthesizer,
)
from app.providers import (
    DeepgramAligner,
    DeepgramSynthesizer,
    GeminiError,
    GeminiImageProvider,
    GeminiScriptProvider,
    LyriaMusicProvider,
    PlaceholderImageProvider,
    VerbatimScriptProvider,
    _gemini,
    align_tokens,
    audio_duration,
    deepgram_align,
    deepgram_tts,
    gemini_image,
    image_dimensions,
    loop_count,
    lyria_music,
    nearest_ratio,
    normalize,
    tokenize,
)
from app.providers.gemini_script import _split_into_segments

LIVE = os.environ.get("RUN_LIVE_API") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_API=1 to hit real APIs")

SAMPLE_NARRATION = (
    "Construction begins beneath the surface, where engineers drive steel cylinders into "
    "the riverbed. They pump out the water to pour concrete piers."
)


# =========================================================== protocol conformance


class TestProtocolConformance:
    """Structural typing is the whole design; assert the shapes actually match."""

    @pytest.mark.parametrize(
        ("factory", "protocol"),
        [
            (lambda: GeminiScriptProvider(api_key="x"), ScriptProvider),
            (lambda: VerbatimScriptProvider("One. Two. Three."), ScriptProvider),
            (lambda: GeminiImageProvider(api_key="x"), ImageProvider),
            (lambda: PlaceholderImageProvider(), ImageProvider),
            (lambda: DeepgramSynthesizer(api_key="x"), SpeechSynthesizer),
            (lambda: DeepgramAligner(api_key="x"), Aligner),
            (lambda: LyriaMusicProvider(api_key="x"), MusicProvider),
        ],
    )
    def test_satisfies_protocol(self, factory, protocol):
        assert isinstance(factory(), protocol)


# =========================================================== gemini transport quirks


class TestGeminiPartsHandling:
    """The `parts` list is not a single blob and not positionally stable."""

    def test_text_skips_non_text_parts(self):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thoughtSignature": "abc"},
                            {"text": '{"a":'},
                            {"text": "1}"},
                        ]
                    }
                }
            ]
        }
        assert _gemini.text_from(response) == '{"a":1}'

    def test_inline_data_found_alongside_thought_signature(self):
        """The live image model returns inlineData and thoughtSignature on ONE part."""
        payload = b"\x89PNG\r\n\x1a\nfake"
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(payload).decode(),
                                },
                                "thoughtSignature": "zzz",
                            }
                        ]
                    }
                }
            ]
        }
        mime, data = _gemini.inline_data_from(response)
        assert mime == "image/png"
        assert data == payload

    def test_snake_case_inline_data_accepted(self):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inline_data": {"mime_type": "audio/mpeg", "data": "aGk="}}
                        ]
                    }
                }
            ]
        }
        assert _gemini.inline_data_from(response) == ("audio/mpeg", b"hi")

    def test_no_candidates_raises_with_feedback(self):
        with pytest.raises(GeminiError, match="no candidates"):
            _gemini.text_from({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})

    def test_no_text_parts_raises(self):
        with pytest.raises(GeminiError, match="no text parts"):
            _gemini.text_from({"candidates": [{"content": {"parts": [{"thoughtSignature": "x"}]}}]})

    def test_empty_api_key_raises_before_network(self):
        with pytest.raises(GeminiError, match="gemini_api_key is empty"):
            _gemini.generate_content("m", {}, "")


# =========================================================== script: gemini path


def _script_response(scenes: list[dict], title: str = "How Bridges Are Built") -> dict:
    """Canned generateContent reply, including a metadata-only part."""
    body = json.dumps({"title": title, "scenes": scenes})
    return {
        "candidates": [
            {"content": {"parts": [{"thoughtSignature": "sig"}, {"text": body}]}}
        ]
    }


def _scene(i: int, motion: str = "zoom_in", narration: str | None = None) -> dict:
    return {
        "id": i,
        "narration": narration
        or f"Scene {i} explains one step of the build. It runs long enough to sound natural.",
        "heading": f"Step Number {i}",
        "image_prompt": f"A photograph of step {i}. No text anywhere in the image.",
        "motion": motion,
    }


class TestGeminiScriptProvider:
    def test_parses_structured_output(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(1), _scene(2, "pan_left")]),
        )
        script = GeminiScriptProvider(api_key="x").generate("how bridges are built", 2)
        assert script.topic == "how bridges are built"
        assert script.title == "How Bridges Are Built"
        assert [s.id for s in script.scenes] == [1, 2]
        assert script.scenes[1].motion is Motion.PAN_LEFT

    def test_request_body_uses_uppercase_schema_and_json_mime(self, monkeypatch):
        captured: dict = {}

        def fake(model, body, api_key, **kwargs):
            captured["model"] = model
            captured["body"] = body
            return _script_response([_scene(1)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x", model="m").generate("topic", 1)

        config = captured["body"]["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        schema = config["responseSchema"]
        assert schema["type"] == "OBJECT"
        assert schema["properties"]["scenes"]["type"] == "ARRAY"
        assert schema["properties"]["scenes"]["items"]["type"] == "OBJECT"
        motion = schema["properties"]["scenes"]["items"]["properties"]["motion"]
        assert motion["enum"] == [m.value for m in Motion]
        # Every SceneScript field must be required or the model omits it.
        assert set(schema["properties"]["scenes"]["items"]["required"]) == {
            "id",
            "narration",
            "heading",
            "image_prompt",
            "motion",
        }

    def test_prompt_states_the_slide_count_and_no_text_rule(self, monkeypatch):
        captured: dict = {}

        def fake(model, body, api_key, **kwargs):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _script_response([_scene(1)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x").generate("bridges", 4)
        prompt = captured["text"]
        assert "EXACTLY 4 scenes" in prompt
        assert "No text, no letters" in prompt
        assert "text-to-speech" in prompt

    def test_over_delivery_is_truncated(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(i) for i in range(1, 7)]),
        )
        script = GeminiScriptProvider(api_key="x").generate("t", 3)
        assert len(script.scenes) == 3

    def test_under_delivery_is_backfilled_by_splitting(self, monkeypatch):
        long_narration = (
            "Engineers survey the ground first. They drill into bedrock to test the soil. "
            "Then the foundations are poured. Steel towers rise from the piers."
        )
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(1, narration=long_narration)]),
        )
        script = GeminiScriptProvider(api_key="x").generate("t", 3)
        assert len(script.scenes) == 3
        assert [s.id for s in script.scenes] == [1, 2, 3]
        assert all(s.narration.strip() for s in script.scenes)

    def test_adjacent_duplicate_motions_are_broken_up(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(i, "zoom_in") for i in range(1, 5)]),
        )
        script = GeminiScriptProvider(api_key="x").generate("t", 4)
        motions = [s.motion for s in script.scenes]
        assert all(a != b for a, b in zip(motions, motions[1:], strict=False)), motions

    def test_off_schema_motion_falls_back_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(1, "spin_around")]),
        )
        script = GeminiScriptProvider(api_key="x").generate("t", 1)
        assert isinstance(script.scenes[0].motion, Motion)

    def test_markdown_and_emoji_stripped_from_narration(self, monkeypatch):
        dirty = "**Bold** start here now. \U0001f600 It keeps going for a while too."
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(1, narration=dirty)]),
        )
        narration = GeminiScriptProvider(api_key="x").generate("t", 1).scenes[0].narration
        assert "*" not in narration
        assert "\U0001f600" not in narration
        assert "  " not in narration

    def test_non_json_body_raises_clearly(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: {"candidates": [{"content": {"parts": [{"text": "sorry!"}]}}]},
        )
        with pytest.raises(GeminiError, match="not valid JSON"):
            GeminiScriptProvider(api_key="x").generate("t", 1)

    def test_zero_scenes_raises(self, monkeypatch):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([]),
        )
        with pytest.raises(GeminiError, match="zero scenes"):
            GeminiScriptProvider(api_key="x").generate("t", 2)

    def test_invalid_slide_count_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            GeminiScriptProvider(api_key="x").generate("t", 0)


# =========================================================== script: verbatim path


VERBATIM_TEXT = (
    "Bridges begin with a survey of the ground. Engineers drill deep into bedrock to test "
    "what the soil can carry. Then the foundations are poured, huge concrete piers that hold "
    "the structure. Steel towers rise from those piers, lifted piece by piece by crane. "
    "Finally the deck is threaded between them, and the road is complete."
)


class TestVerbatimScriptProvider:
    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
    def test_returns_exactly_the_requested_scene_count(self, count):
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", count)
        assert len(script.scenes) == count
        assert [s.id for s in script.scenes] == list(range(1, count + 1))

    def test_narration_is_verbatim_no_words_lost_or_invented(self):
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 3)
        rejoined = " ".join(s.narration for s in script.scenes)
        assert rejoined.split() == VERBATIM_TEXT.split()

    def test_no_llm_call_is_made(self, monkeypatch):
        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("verbatim path must not call the network")

        monkeypatch.setattr("app.providers.gemini_script.generate_content", explode)
        monkeypatch.setattr("httpx.post", explode)
        assert VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 3)

    def test_headings_are_short_and_image_prompts_forbid_text(self):
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 4)
        for scene in script.scenes:
            assert 3 <= len(scene.heading.split()) <= 7, scene.heading
            assert not scene.heading.endswith(".")
            assert "No text, no letters" in scene.image_prompt

    def test_motion_varies_between_consecutive_scenes(self):
        motions = [s.motion for s in VerbatimScriptProvider(VERBATIM_TEXT).generate("b", 5).scenes]
        assert all(a != b for a, b in zip(motions, motions[1:], strict=False))

    def test_explicit_title_wins(self):
        script = VerbatimScriptProvider(VERBATIM_TEXT, title="My Title").generate("b", 2)
        assert script.title == "My Title"

    def test_word_counts_are_roughly_balanced(self):
        scenes = VerbatimScriptProvider(VERBATIM_TEXT).generate("b", 3).scenes
        counts = [len(s.narration.split()) for s in scenes]
        assert min(counts) >= 5
        # No scene should be more than ~3x another, or narration pacing lurches.
        assert max(counts) <= 3 * min(counts), counts

    def test_empty_script_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            VerbatimScriptProvider("   ")

    def test_script_too_short_for_slide_count_raises(self):
        with pytest.raises(ValueError, match="cannot be split"):
            VerbatimScriptProvider("Two words.").generate("t", 6)

    def test_single_scene_keeps_whole_text(self):
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("b", 1)
        assert script.scenes[0].narration.split() == VERBATIM_TEXT.split()


class TestSegmentSplitter:
    def test_splits_on_sentence_boundaries(self):
        segments = _split_into_segments("One two. Three four. Five six.", 3)
        assert segments == ["One two.", "Three four.", "Five six."]

    def test_subdivides_clauses_when_sentences_are_scarce(self):
        segments = _split_into_segments("First part, second part, third part.", 3)
        assert len(segments) == 3

    def test_never_returns_empty_segments(self):
        for count in range(1, 6):
            assert all(s.strip() for s in _split_into_segments(VERBATIM_TEXT, count))


# =========================================================== image


class TestImagePromptComposition:
    def test_no_text_guard_appended_when_missing(self):
        composed = gemini_image._compose_prompt("A photo of a bridge")
        assert "No text, no letters" in composed
        assert composed.startswith("A photo of a bridge.")

    def test_no_text_guard_not_duplicated(self):
        original = "A photo. No text, no letters, no words anywhere in the image."
        assert gemini_image._compose_prompt(original).count("No text") == 1

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (1920, 1080, "16:9"),
            (960, 540, "16:9"),
            (1080, 1920, "9:16"),
            (1024, 1024, "1:1"),
            (2560, 1080, "21:9"),
            (0, 0, "16:9"),
        ],
    )
    def test_nearest_ratio(self, width, height, expected):
        assert nearest_ratio(width, height) == expected

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [(960, 540, "1K"), (1376, 768, "1K"), (1920, 1080, "2K"), (3840, 2160, "2K")],
    )
    def test_2k_requested_whenever_1k_would_undersize_the_frame(self, width, height, expected):
        assert gemini_image._image_size_tier(width, height) == expected


class TestGeminiImageProvider:
    def test_writes_bytes_and_requests_landscape(self, monkeypatch, tmp_path):
        captured: dict = {}
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

        def fake(model, body, api_key, **kwargs):
            captured["body"] = body
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(png).decode(),
                                    },
                                    "thoughtSignature": "sig",
                                }
                            ]
                        }
                    }
                ]
            }

        monkeypatch.setattr("app.providers.gemini_image.generate_content", fake)
        out = tmp_path / "scene.png"
        result = GeminiImageProvider(api_key="x").generate("a bridge", out, 1920, 1080)

        assert result == out
        assert out.read_bytes() == png
        image_config = captured["body"]["generationConfig"]["imageConfig"]
        assert image_config["aspectRatio"] == "16:9"
        assert image_config["imageSize"] == "2K"
        assert captured["body"]["generationConfig"]["responseModalities"] == ["IMAGE"]

    def test_jpeg_bytes_for_a_png_path_are_remuxed_not_mislabelled(self, monkeypatch, tmp_path):
        """The live model returns image/jpeg; a .png path must not hold jpeg bytes."""
        source = tmp_path / "src.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=teal:s=320x180:d=1",
                "-frames:v", "1", str(source),
            ],
            check=True,
        )
        jpeg = source.read_bytes()

        monkeypatch.setattr(
            "app.providers.gemini_image.generate_content",
            lambda *a, **k: {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/jpeg",
                                        "data": base64.b64encode(jpeg).decode(),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )
        out = tmp_path / "scene.png"
        GeminiImageProvider(api_key="x").generate("a bridge", out, 1920, 1080)

        assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        # Remux only: dimensions must be untouched.
        assert image_dimensions(out) == (320, 180)
        assert not list(tmp_path.glob("*.raw.*"))

    def test_creates_missing_parent_directories(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "app.providers.gemini_image.generate_content",
            lambda *a, **k: {
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png",
                                                           "data": "aGk="}}]}}
                ]
            },
        )
        out = tmp_path / "deep" / "nested" / "scene.png"
        GeminiImageProvider(api_key="x").generate("p", out, 1920, 1080)
        assert out.exists()


class TestPlaceholderImageProvider:
    def test_produces_an_image_at_the_requested_size(self, tmp_path):
        out = PlaceholderImageProvider().generate("bridges", tmp_path / "ph.png", 1280, 720)
        assert image_dimensions(out) == (1280, 720)

    def test_deterministic_for_the_same_prompt(self, tmp_path):
        a = PlaceholderImageProvider().generate("same", tmp_path / "a.png", 320, 180)
        b = PlaceholderImageProvider().generate("same", tmp_path / "b.png", 320, 180)
        assert a.read_bytes() == b.read_bytes()

    def test_different_prompts_differ(self, tmp_path):
        a = PlaceholderImageProvider().generate("alpha", tmp_path / "a.png", 320, 180)
        b = PlaceholderImageProvider().generate("beta", tmp_path / "b.png", 320, 180)
        assert a.read_bytes() != b.read_bytes()

    def test_makes_no_network_call(self, monkeypatch, tmp_path):
        def explode(*args, **kwargs):  # pragma: no cover
            raise AssertionError("placeholder must stay offline")

        monkeypatch.setattr("httpx.post", explode)
        assert PlaceholderImageProvider().generate("x", tmp_path / "p.png", 64, 36).exists()


# =========================================================== tts


class _FakeResponse:
    def __init__(self, status_code=200, content=b"", text="", headers=None):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.headers = headers or {}


def _wav_bytes(seconds: float, path: Path) -> bytes:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path.read_bytes()


class TestDeepgramSynthesizer:
    def test_writes_wav_body_straight_to_disk(self, monkeypatch, tmp_path):
        wav = _wav_bytes(1.0, tmp_path / "src.wav")
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs["params"]
            captured["headers"] = kwargs["headers"]
            captured["json"] = kwargs["json"]
            return _FakeResponse(content=wav)

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        out = tmp_path / "n.wav"
        result = DeepgramSynthesizer(api_key="k").synthesize("Hello there.", "aura-2-draco-en", out)

        assert result == out
        assert out.read_bytes() == wav
        assert captured["url"] == "https://api.deepgram.com/v1/speak"
        assert captured["params"] == {
            "model": "aura-2-draco-en",
            "encoding": "linear16",
            "sample_rate": "24000",
            "container": "wav",
        }
        assert captured["headers"]["Authorization"] == "Token k"
        assert captured["headers"]["Content-Type"] == "application/json"
        assert captured["json"] == {"text": "Hello there."}

    def test_duration_helper_reads_real_length(self, monkeypatch, tmp_path):
        wav = _wav_bytes(2.5, tmp_path / "src.wav")
        monkeypatch.setattr(
            deepgram_tts.httpx, "post", lambda url, **kw: _FakeResponse(content=wav)
        )
        synth = DeepgramSynthesizer(api_key="k")
        path, duration = synth.synthesize_with_duration("hi", "v", tmp_path / "o.wav")
        assert duration == pytest.approx(2.5, abs=0.05)
        assert synth.duration(path) == pytest.approx(2.5, abs=0.05)
        assert deepgram_tts.probe_duration(path) == pytest.approx(2.5, abs=0.05)

    def test_default_voice_used_when_blank(self, monkeypatch, tmp_path):
        wav = _wav_bytes(0.5, tmp_path / "src.wav")
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["model"] = kwargs["params"]["model"]
            return _FakeResponse(content=wav)

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        DeepgramSynthesizer(api_key="k", default_voice="aura-2-thalia-en").synthesize(
            "hi", "", tmp_path / "o.wav"
        )
        assert captured["model"] == "aura-2-thalia-en"

    def test_non_audio_body_is_rejected_loudly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            deepgram_tts.httpx,
            "post",
            lambda url, **kw: _FakeResponse(
                content=b'{"err":"nope"}', headers={"content-type": "application/json"}
            ),
        )
        with pytest.raises(deepgram_tts.SynthesisError, match="expected a wav body"):
            DeepgramSynthesizer(api_key="k").synthesize("hi", "v", tmp_path / "o.wav")

    def test_client_error_is_not_retried(self, monkeypatch, tmp_path):
        calls: list[int] = []

        def fake_post(url, **kwargs):
            calls.append(1)
            return _FakeResponse(status_code=401, text="unauthorized")

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        with pytest.raises(deepgram_tts.SynthesisError, match="401"):
            DeepgramSynthesizer(api_key="k").synthesize("hi", "v", tmp_path / "o.wav")
        assert len(calls) == 1

    def test_transient_error_is_retried(self, monkeypatch, tmp_path):
        wav = _wav_bytes(0.5, tmp_path / "src.wav")
        calls: list[int] = []

        def fake_post(url, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return _FakeResponse(status_code=503, text="busy")
            return _FakeResponse(content=wav)

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        DeepgramSynthesizer(api_key="k").synthesize("hi", "v", tmp_path / "o.wav")
        assert len(calls) == 2

    def test_empty_text_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            DeepgramSynthesizer(api_key="k").synthesize("  ", "v", tmp_path / "o.wav")

    def test_missing_key_rejected(self, tmp_path):
        with pytest.raises(deepgram_tts.SynthesisError, match="deepgram_api_key is empty"):
            DeepgramSynthesizer(api_key="").synthesize("hi", "v", tmp_path / "o.wav")

    def test_long_text_is_chunked_and_concatenated(self, monkeypatch, tmp_path):
        """A caller handing over a whole script must get one continuous wav back."""
        wav = _wav_bytes(1.0, tmp_path / "src.wav")
        requests: list[str] = []

        def fake_post(url, **kwargs):
            requests.append(kwargs["json"]["text"])
            return _FakeResponse(content=wav)

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        long_text = " ".join(["This sentence is a filler sentence used for chunking."] * 80)
        out = tmp_path / "long.wav"
        DeepgramSynthesizer(api_key="k").synthesize(long_text, "v", out)

        assert len(requests) > 1
        assert all(len(r) <= deepgram_tts.MAX_CHARS for r in requests)
        assert audio_duration(out) == pytest.approx(len(requests) * 1.0, abs=0.15)
        assert not list(tmp_path.glob(".*concat.txt"))

    def test_chunker_respects_the_limit(self):
        text = " ".join(["Word"] * 2000)
        chunks = deepgram_tts._chunk(text, 100)
        assert all(len(c) <= 100 for c in chunks)
        assert " ".join(chunks).split() == text.split()


# =========================================================== aligner


def _listen_payload(words: list[tuple[str, float, float]], duration: float) -> dict:
    return {
        "metadata": {"duration": duration},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": " ".join(w[0] for w in words),
                            "words": [
                                {
                                    "word": w,
                                    "start": s,
                                    "end": e,
                                    "confidence": 0.99,
                                    "punctuated_word": w,
                                }
                                for w, s, e in words
                            ],
                        }
                    ]
                }
            ]
        },
    }


def _even_stt(text: str, step: float = 0.4) -> list[Word]:
    return [
        Word(word=t, start=round(i * step, 3), end=round(i * step + step * 0.9, 3), confidence=0.99)
        for i, t in enumerate(text.split())
    ]


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Ground,", "ground"),
            ("bedrock.", "bedrock"),
            ("don't", "dont"),
            ("world-class", "worldclass"),
            ('"Quoted"', "quoted"),
            ("25", "25"),
            ("—", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize(raw) == expected

    def test_tokenize_drops_punctuation_only_tokens(self):
        assert tokenize("Hello — world , ok") == ["Hello", "world", "ok"]

    def test_tokenize_keeps_original_punctuation_on_words(self):
        assert tokenize("Ground, and bedrock.") == ["Ground,", "and", "bedrock."]

    def test_tokenize_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestAlignTokens:
    """The reference text is authoritative for display; STT only supplies the clock."""

    def test_perfect_match_carries_exact_timings(self):
        reference = "Bridges begin with a survey."
        stt = _even_stt("bridges begin with a survey")
        aligned = align_tokens(tokenize(reference), stt, 2.0)
        assert [w.display for w in aligned] == ["Bridges", "begin", "with", "a", "survey."]
        assert aligned[0].start == pytest.approx(stt[0].start)
        assert aligned[-1].end == pytest.approx(stt[-1].end)
        assert all(w.confidence == pytest.approx(0.99) for w in aligned)

    def test_punctuation_and_casing_survive_the_round_trip(self):
        reference = 'She said, "Go now" — really, don\'t wait!'
        stt = _even_stt("she said go now really dont wait")
        aligned = align_tokens(tokenize(reference), stt, 3.0)
        assert " ".join(w.display for w in aligned) == 'She said, "Go now" really, don\'t wait!'
        assert any("," in w.display for w in aligned)
        assert any('"' in w.display for w in aligned)

    def test_stt_words_are_never_returned_as_display_text(self):
        reference = "Engineers drill into bedrock."
        stt = _even_stt("engineers drilled into bedrock")  # misheard tense
        aligned = align_tokens(tokenize(reference), stt, 2.0)
        assert [w.display for w in aligned] == ["Engineers", "drill", "into", "bedrock."]
        assert "drilled" not in [w.display for w in aligned]

    def test_same_count_mishearing_keeps_measured_timing(self):
        reference = "Engineers drill into bedrock."
        stt = _even_stt("engineers drilled into bedrock")
        aligned = align_tokens(tokenize(reference), stt, 2.0)
        assert aligned[1].start == pytest.approx(stt[1].start)
        assert aligned[1].end == pytest.approx(stt[1].end)

    def test_smart_format_digits_collapse_onto_one_reference_token(self):
        """smart_format turns "twenty five" into "25"; counts differ, timings must still fit."""
        reference = "Drill 25 feet down."
        stt = _even_stt("drill twenty five feet down")
        aligned = align_tokens(tokenize(reference), stt, 2.5)
        assert [w.display for w in aligned] == ["Drill", "25", "feet", "down."]
        digit = next(w for w in aligned if w.display == "25")
        assert digit.start >= aligned[0].end - 1e-6
        assert digit.end <= aligned[2].start + 1e-6

    def test_reference_words_missing_from_transcript_are_interpolated(self):
        reference = "One two three four five six."
        stt = _even_stt("one six")  # everything in the middle dropped
        aligned = align_tokens(tokenize(reference), stt, 3.0)
        assert len(aligned) == 6
        assert [w.display for w in aligned] == ["One", "two", "three", "four", "five", "six."]
        middle = aligned[1:5]
        assert all(w.end > w.start for w in middle)
        assert all(w.confidence == 0.0 for w in middle)

    def test_transcript_hallucinations_do_not_add_words(self):
        reference = "One two three."
        stt = _even_stt("one uh um two three")
        aligned = align_tokens(tokenize(reference), stt, 2.0)
        assert [w.display for w in aligned] == ["One", "two", "three."]

    def test_empty_transcript_spreads_reference_across_the_clip(self):
        aligned = align_tokens(tokenize("One two three four."), [], 4.0)
        assert len(aligned) == 4
        assert aligned[0].start == 0.0
        assert aligned[-1].end == pytest.approx(4.0)
        assert all(a.end <= b.start + 1e-6 for a, b in zip(aligned, aligned[1:], strict=False))

    def test_empty_reference_returns_empty(self):
        assert align_tokens([], _even_stt("anything"), 1.0) == []

    def test_timings_are_monotonic_and_within_the_clip(self):
        reference = "Alpha bravo charlie delta echo foxtrot golf hotel."
        stt = _even_stt("alpha charlie delta zulu echo golf hotel")
        aligned = align_tokens(tokenize(reference), stt, 3.0)
        assert all(w.start <= w.end for w in aligned)
        assert all(a.end <= b.start + 1e-6 for a, b in zip(aligned, aligned[1:], strict=False))
        assert aligned[-1].end <= 3.0 + 1e-6

    def test_interpolation_weights_by_word_length(self):
        """An even split drifts; "a" should not occupy as long as "infrastructure"."""
        aligned = align_tokens(tokenize("a infrastructure"), [], 4.0)
        assert (aligned[1].end - aligned[1].start) > 3 * (aligned[0].end - aligned[0].start)

    def test_long_script_does_not_degrade_via_autojunk(self):
        """SequenceMatcher's autojunk would treat common words as noise past 200 tokens."""
        reference = " ".join(["the quick brown fox jumps over the lazy dog."] * 40)
        stt = _even_stt(" ".join(["the quick brown fox jumps over the lazy dog"] * 40))
        aligned = align_tokens(tokenize(reference), stt, 200.0)
        assert " ".join(w.display for w in aligned) == reference
        measured = [w for w in aligned if w.confidence > 0.5]
        assert len(measured) == len(aligned)

    def test_word_field_is_normalized_and_display_is_original(self):
        aligned = align_tokens(tokenize("Ground,"), _even_stt("ground"), 1.0)
        assert aligned[0].word == "ground"
        assert aligned[0].punctuated_word == "Ground,"
        assert aligned[0].display == "Ground,"


class TestDeepgramAligner:
    def test_request_shape_and_response_parsing(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.wav"
        _wav_bytes(1.0, audio)
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs["params"]
            captured["headers"] = kwargs["headers"]
            captured["content"] = kwargs["content"]
            return _FakeJson(_listen_payload([("hello", 0.1, 0.5), ("world", 0.5, 1.0)], 1.0))

        monkeypatch.setattr(deepgram_align.httpx, "post", fake_post)
        aligned = DeepgramAligner(api_key="k", model="nova-3").align(audio, "Hello, world!")

        assert captured["url"] == "https://api.deepgram.com/v1/listen"
        assert captured["params"] == {
            "model": "nova-3",
            "smart_format": "true",
            "punctuate": "true",
        }
        assert captured["headers"]["Authorization"] == "Token k"
        assert captured["headers"]["Content-Type"] == "audio/wav"
        assert captured["content"] == audio.read_bytes()
        assert [w.display for w in aligned] == ["Hello,", "world!"]
        assert aligned[0].start == pytest.approx(0.1)

    def test_transcribe_returns_metadata_duration(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.wav"
        _wav_bytes(1.0, audio)
        monkeypatch.setattr(
            deepgram_align.httpx,
            "post",
            lambda url, **kw: _FakeJson(_listen_payload([("hi", 0.0, 0.4)], 7.5)),
        )
        words, duration = DeepgramAligner(api_key="k").transcribe(audio)
        assert duration == pytest.approx(7.5)
        assert words[0].word == "hi"

    def test_empty_reference_short_circuits_without_network(self, monkeypatch, tmp_path):
        def explode(*args, **kwargs):  # pragma: no cover
            raise AssertionError("must not call the API for empty reference text")

        monkeypatch.setattr(deepgram_align.httpx, "post", explode)
        assert DeepgramAligner(api_key="k").align(tmp_path / "missing.wav", "") == []

    def test_missing_channels_raises(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.wav"
        _wav_bytes(0.5, audio)
        monkeypatch.setattr(
            deepgram_align.httpx, "post", lambda url, **kw: _FakeJson({"results": {"channels": []}})
        )
        with pytest.raises(deepgram_align.AlignmentError, match="no channels"):
            DeepgramAligner(api_key="k").align(audio, "hello")

    def test_missing_key_rejected(self, tmp_path):
        with pytest.raises(deepgram_align.AlignmentError, match="deepgram_api_key is empty"):
            DeepgramAligner(api_key="").align(tmp_path / "a.wav", "hello")


class _FakeJson:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


# =========================================================== music


@pytest.fixture(scope="module")
def fixed_clip(tmp_path_factory) -> bytes:
    """A ~29.57s mp3 standing in for Lyria's fixed-length clip."""
    path = tmp_path_factory.mktemp("music") / "clip.mp3"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=29.57",
            "-c:a", "libmp3lame", "-b:a", "192k", str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def _music_response(clip: bytes) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/mpeg",
                                "data": base64.b64encode(clip).decode(),
                            },
                            "thoughtSignature": "sig",
                        }
                    ]
                }
            }
        ]
    }


class TestLoopMath:
    @pytest.mark.parametrize(
        ("target", "expected"), [(5.0, 1), (29.0, 1), (29.57, 1), (45.0, 2), (60.0, 3)]
    )
    def test_loop_count_accounts_for_crossfade_overlap(self, target, expected):
        assert loop_count(29.57, target, 2.5) == expected

    def test_copies_always_cover_the_target(self):
        for target in (31.0, 45.0, 90.0, 200.0, 605.0):
            n = loop_count(29.57, target, 2.5)
            assert n * 29.57 - (n - 1) * 2.5 >= target


class TestLyriaMusicProvider:
    @pytest.mark.parametrize("target", [12.0, 29.0, 45.0, 75.0])
    def test_output_matches_target_duration(self, monkeypatch, tmp_path, fixed_clip, target):
        monkeypatch.setattr(
            "app.providers.lyria_music.generate_content",
            lambda *a, **k: _music_response(fixed_clip),
        )
        out = tmp_path / f"bed{target}.mp3"
        LyriaMusicProvider(api_key="x").generate("calm ambient", target, out)
        assert audio_duration(out) == pytest.approx(target, abs=lyria_music.DURATION_TOLERANCE)

    def test_45s_target_is_looped_not_truncated_to_the_clip(
        self, monkeypatch, tmp_path, fixed_clip
    ):
        """The regression that matters: a fixed ~30s clip must not cap a 45s bed."""
        monkeypatch.setattr(
            "app.providers.lyria_music.generate_content",
            lambda *a, **k: _music_response(fixed_clip),
        )
        out = tmp_path / "bed.mp3"
        LyriaMusicProvider(api_key="x").generate("calm ambient", 45.0, out)
        duration = audio_duration(out)
        assert duration > 40.0
        assert duration == pytest.approx(45.0, abs=0.1)

    def test_prompt_steers_away_from_vocals_and_dynamics(self, monkeypatch, tmp_path, fixed_clip):
        captured: dict = {}

        def fake(model, body, api_key, **kwargs):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _music_response(fixed_clip)

        monkeypatch.setattr("app.providers.lyria_music.generate_content", fake)
        LyriaMusicProvider(api_key="x").generate("warm documentary underscore", 10.0,
                                                 tmp_path / "b.mp3")
        prompt = captured["text"].lower()
        assert "warm documentary underscore" in prompt
        assert "no vocals" in prompt
        assert "instrumental" in prompt
        assert "unobtrusive" in prompt

    def test_wav_output_extension_is_honoured(self, monkeypatch, tmp_path, fixed_clip):
        monkeypatch.setattr(
            "app.providers.lyria_music.generate_content",
            lambda *a, **k: _music_response(fixed_clip),
        )
        out = tmp_path / "bed.wav"
        LyriaMusicProvider(api_key="x").generate("calm", 8.0, out)
        assert out.read_bytes().startswith(b"RIFF")
        assert audio_duration(out) == pytest.approx(8.0, abs=0.1)

    def test_target_longer_than_many_loops(self, monkeypatch, tmp_path, fixed_clip):
        monkeypatch.setattr(
            "app.providers.lyria_music.generate_content",
            lambda *a, **k: _music_response(fixed_clip),
        )
        out = tmp_path / "long.mp3"
        LyriaMusicProvider(api_key="x").generate("calm", 150.0, out)
        assert audio_duration(out) == pytest.approx(150.0, abs=0.1)

    def test_non_positive_target_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be positive"):
            LyriaMusicProvider(api_key="x").generate("calm", 0.0, tmp_path / "b.mp3")

    def test_no_temp_files_left_behind(self, monkeypatch, tmp_path, fixed_clip):
        monkeypatch.setattr(
            "app.providers.lyria_music.generate_content",
            lambda *a, **k: _music_response(fixed_clip),
        )
        out = tmp_path / "bed.mp3"
        LyriaMusicProvider(api_key="x").generate("calm", 40.0, out)
        assert [p.name for p in tmp_path.iterdir()] == ["bed.mp3"]


# =========================================================== live API tests


@live_only
class TestLiveScript:
    def test_generates_the_requested_number_of_speakable_scenes(self):
        script = GeminiScriptProvider().generate("how bridges are built", 3)
        assert len(script.scenes) == 3
        assert script.title
        for scene in script.scenes:
            words = len(scene.narration.split())
            assert 10 <= words <= 60, (words, scene.narration)
            assert 2 <= len(scene.heading.split()) <= 9, scene.heading
            for banned in ("*", "#", "•", "\n"):
                assert banned not in scene.narration
            assert scene.image_prompt
        motions = [s.motion for s in script.scenes]
        assert all(a != b for a, b in zip(motions, motions[1:], strict=False)), motions


@live_only
class TestLiveImage:
    def test_generates_a_landscape_image(self, tmp_path):
        out = tmp_path / "scene.png"
        prompt = (
            "A cinematic wide photograph of a suspension bridge under construction at "
            "golden hour, cranes silhouetted against a soft sky, open space in the lower "
            "third. No text, no letters, no words anywhere in the image."
        )
        GeminiImageProvider().generate(prompt, out, 1920, 1080)
        assert out.exists()
        assert out.stat().st_size > 100_000
        width, height = image_dimensions(out)
        # The model's "16:9" is really ~1.79:1; assert landscape and adequate resolution
        # rather than an exact ratio, since the renderer fits rather than stretches.
        assert width > height
        assert 1.7 < width / height < 1.85
        assert width >= 1920, f"{width}x{height} would be upscaled by the renderer"


@live_only
class TestLiveSpeechAndAlignment:
    def test_synthesize_then_align_preserves_reference_punctuation(self, tmp_path):
        out = tmp_path / "narration.wav"
        synth = DeepgramSynthesizer()
        path, duration = synth.synthesize_with_duration(
            SAMPLE_NARRATION, "aura-2-draco-en", out
        )
        assert path.read_bytes().startswith(b"RIFF")
        assert duration > 3.0

        aligner = DeepgramAligner()
        raw_words, _ = aligner.transcribe(path)
        raw_transcript = " ".join(w.word for w in raw_words)
        # Precondition for the whole design: STT really does lose the punctuation.
        assert "," not in raw_transcript
        assert "." not in raw_transcript

        aligned = aligner.align(path, SAMPLE_NARRATION)
        assert len(aligned) == len(SAMPLE_NARRATION.split())
        assert " ".join(w.display for w in aligned) == SAMPLE_NARRATION
        assert all(a.end <= b.start + 1e-6 for a, b in zip(aligned, aligned[1:], strict=False))
        assert aligned[-1].end <= duration + 0.05
        assert aligned[0].start >= 0.0


@live_only
class TestLiveMusic:
    def test_45_second_bed_is_looped_from_the_fixed_clip(self, tmp_path):
        out = tmp_path / "music.mp3"
        LyriaMusicProvider().generate("calm ambient documentary underscore", 45.0, out)
        duration = audio_duration(out)
        assert duration == pytest.approx(45.0, abs=0.1)
        assert duration > 40.0, "looping did not happen — clip length leaked through"
