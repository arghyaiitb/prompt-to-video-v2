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
import inspect
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from app.core.models import Language, Motion, SceneRole, SceneScript, Word
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
from app.providers.bullet_timing import (
    LEAD,
    TAIL_GUARD,
    anchor_position,
    find_anchors,
    time_bullets,
)
from app.providers.gemini_script import (
    BULLET_CHAR_MAX,
    BULLET_DEFAULT,
    BULLET_MIN,
    DANDA,
    HEADING_CHAR_MAX,
    LANGUAGE_ANCHOR_NOTES,
    LANGUAGE_CLAUSES,
    LANGUAGE_NAMES,
    LANGUAGE_WORD_FACTOR,
    LANGUAGE_WPM,
    ROLE_NARRATION_WORDS,
    STRUCTURE_FROM_SLIDES,
    SUMMARY_FROM_SLIDES,
    TITLE_CHAR_MAX,
    TONE_CLAUSES,
    WORDS_PER_MINUTE,
    StructuredSceneScript,
    _bullets_from,
    _clean_bullet,
    _clean_bullets,
    _clean_heading,
    _fragment_from,
    _heading_from,
    _split_into_segments,
    _words,
    anchoring_supported,
    coerce_language,
    language_word_factor,
    language_wpm,
    narration_words,
    role_bullet_target,
    role_plan,
    role_sentences,
    scene_clip_prompt,
    scene_role,
    words_spoken_in,
)

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
        # The role drives duration, layout and bullet budget, so it is an enum like motion
        # rather than a free string that could come back as prose.
        role = schema["properties"]["scenes"]["items"]["properties"]["role"]
        assert role["enum"] == [r.value for r in SceneRole]
        # Every scene field must be required or the model omits it.
        assert set(schema["properties"]["scenes"]["items"]["required"]) == {
            "id",
            "role",
            "narration",
            "heading",
            "bullets",
            "image_prompt",
            "clip_prompt",
            "motion",
        }
        bullets = schema["properties"]["scenes"]["items"]["properties"]["bullets"]
        assert bullets == {"type": "ARRAY", "items": {"type": "STRING"}}

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


class TestSceneBullets:
    """Bullets must arrive anchored: a point with no phrase in its own narration cannot
    be timed, so both providers guarantee the anchor rather than hoping for it."""

    BULLET_NARRATION = (
        "Check the sender domain before you trust the display name. Hover over every link "
        "to reveal the real destination it points to. Report anything suspicious to the "
        "security team straight away."
    )

    def _generate(self, monkeypatch, bullets, narration=None, bullets_per_slide=4):
        scene = _scene(1, narration=narration or self.BULLET_NARRATION)
        scene["bullets"] = bullets
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([scene]),
        )
        return (
            GeminiScriptProvider(api_key="x")
            .generate("phishing", 1, bullets_per_slide=bullets_per_slide)
            .scenes[0]
        )

    def test_prompt_demands_anchored_bullets_and_a_topical_image(self, monkeypatch):
        captured: dict = {}

        def fake(model, body, api_key, **kwargs):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _script_response([_scene(1)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x").generate("how phishing attacks work", 4)
        prompt = captured["text"]
        assert "CONSECUTIVE CONTENT WORDS" in prompt
        assert "SAME ORDER" in prompt
        # The content scene's pacing contract, from docs/DIRECTION.md §1.2.
        assert "narration 34 words (25-43)" in prompt
        assert "EXACTLY 4 short on-screen points" in prompt
        # The observed defect: a generic atrium for a security video.
        assert "VIDEO'S TOPIC" in prompt
        assert "how phishing attacks work" in prompt
        assert "atrium" in prompt
        assert "No text, no letters" in prompt

    def test_model_bullets_are_carried_through(self, monkeypatch):
        given = ["Check The Sender Domain", "Reveal The Real Destination"]
        scene = self._generate(
            monkeypatch, given + ["Report Anything Suspicious"], bullets_per_slide=3
        )
        assert scene.bullets[:2] == given

    def test_glyphs_markdown_and_terminal_periods_are_stripped(self, monkeypatch):
        scene = self._generate(
            monkeypatch,
            ["- Check the sender domain.", "**Hover over every link**", "3. Report anything!"],
            bullets_per_slide=3,
        )
        assert scene.bullets == [
            "Check the sender domain",
            "Hover over every link",
            "Report anything",
        ]

    def test_bullets_are_reordered_to_follow_the_narration(self, monkeypatch):
        scene = self._generate(
            monkeypatch,
            ["Report Anything Suspicious", "Check The Sender Domain", "Hover Over Every Link"],
            bullets_per_slide=3,
        )
        assert scene.bullets == [
            "Check The Sender Domain",
            "Hover Over Every Link",
            "Report Anything Suspicious",
        ]

    def test_shortfall_is_topped_up_from_the_narration(self, monkeypatch):
        scene = self._generate(monkeypatch, ["Check The Sender Domain"])
        assert len(scene.bullets) >= 3
        for bullet in scene.bullets:
            assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_missing_bullets_field_still_yields_anchored_bullets(self, monkeypatch):
        scene = self._generate(monkeypatch, None)
        assert 3 <= len(scene.bullets) <= 5
        for bullet in scene.bullets:
            assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_over_delivery_is_capped_at_the_content_budget(self, monkeypatch):
        """A content slide shows four points however many the model sends.

        `docs/DIRECTION.md` §9 fixes `CONTENT.bullet_budget` at 4: at the 11s duration floor
        the usable reveal window is 6.27s and five points need 6.4s at the spec stagger.
        """
        anchored = [
            "Check The Sender Domain",
            "Trust The Display Name",
            "Hover Over Every Link",
            "Reveal The Real Destination",
            "Report Anything Suspicious",
            "Security Team Straight Away",
            "Points To",
        ]
        scene = self._generate(monkeypatch, anchored, bullets_per_slide=5)
        budget = SceneRole.CONTENT.bullet_budget
        assert len(scene.bullets) == budget
        assert scene.bullets == anchored[:budget]

    def test_unanchored_bullets_yield_to_phrases_from_the_narration(self, monkeypatch):
        scene = self._generate(monkeypatch, [f"Abstract Point {i}" for i in range(4)])
        assert 3 <= len(scene.bullets) <= 5
        for bullet in scene.bullets:
            assert "Abstract Point" not in bullet
            assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_an_unanchored_bullet_survives_when_it_is_needed_to_reach_three(self, monkeypatch):
        # Short narration yields few derivable phrases; content is not silently lost.
        scene = self._generate(
            monkeypatch,
            ["Check The Sender Domain", "Zero Trust Posture"],
            narration="Check the sender domain first.",
        )
        assert "Zero Trust Posture" in scene.bullets

    def test_duplicates_are_dropped(self, monkeypatch):
        scene = self._generate(
            monkeypatch, ["Check The Sender Domain", "check the sender domain"]
        )
        assert len([b for b in scene.bullets if "sender" in b.lower()]) == 1

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_verbatim_provider_derives_anchored_bullets(self, count):
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", count)
        for scene in script.scenes:
            # A segment shorter than the 35-55 word narration target carries fewer points
            # by design: the alternative is scraps like "Check The".
            assert 2 <= len(scene.bullets) <= 5, scene.bullets
            for bullet in scene.bullets:
                assert 2 <= len(bullet.split()) <= 6, bullet
                assert not bullet.endswith(".")
                assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_narration_of_the_target_length_yields_three_points(self):
        scene = (
            VerbatimScriptProvider(self.BULLET_NARRATION)
            .generate("phishing", 1, bullets_per_slide=3)
            .scenes[0]
        )
        assert len(scene.bullets) == 3, scene.bullets

    def test_split_scenes_keep_the_bullets_their_own_half_says(self, monkeypatch):
        long_narration = (
            "Check the sender domain before you trust the display name shown in your "
            "inbox. Report anything suspicious to the security team straight away."
        )
        scene = _scene(1, narration=long_narration)
        scene["bullets"] = ["Check The Sender Domain", "Report Anything Suspicious"]
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([scene]),
        )
        script = GeminiScriptProvider(api_key="x").generate("phishing", 2)
        assert len(script.scenes) == 2
        assert "Check The Sender Domain" in script.scenes[0].bullets
        assert "Report Anything Suspicious" in script.scenes[1].bullets
        for part in script.scenes:
            for bullet in part.bullets:
                assert anchor_position(bullet, part.narration) is not None, (part.id, bullet)

    def test_bullets_can_be_timed_against_aligned_words(self):
        scene = VerbatimScriptProvider(self.BULLET_NARRATION).generate("phishing", 1).scenes[0]
        tokens = scene.narration.split()
        words = [Word(word=t, start=i * 0.4, end=i * 0.4 + 0.35) for i, t in enumerate(tokens)]
        points = time_bullets(scene.bullets, words, 0.0, len(tokens) * 0.4 + 0.3)
        assert [p.text for p in points] == scene.bullets
        times = [p.appear_at for p in points]
        assert times == sorted(times)
        assert sum(p.emphasis for p in points) == 1


class TestBulletBudgetAndTone:
    """The two UI knobs — `bullets_per_slide` and `tone` — must reach the model and come
    back honoured. Both were previously dropped between the API and the provider, so these
    cover the whole path: the signature the pipeline inspects, the prompt text, and the
    bullet count on the returned scenes.
    """

    NARRATION = (
        "Check the sender domain before you trust the display name shown in your inbox. "
        "Hover over every link to reveal the real destination it points to. Read the "
        "greeting closely because a generic salutation is a warning sign. Watch for urgent "
        "deadlines designed to rush your decision. Report anything suspicious to the "
        "security team straight away."
    )

    def _prompt(self, monkeypatch, *, slide_count: int = 2, **kwargs) -> str:
        """The prompt text the provider would send for `kwargs`."""
        captured: dict = {}

        def fake(model, body, api_key, **_):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _script_response([_scene(1, narration=self.NARRATION)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x").generate("phishing", slide_count, **kwargs)
        return captured["text"]

    def _scene_for(self, monkeypatch, bullets, **kwargs):
        scene = _scene(1, narration=self.NARRATION)
        scene["bullets"] = bullets
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([scene]),
        )
        return GeminiScriptProvider(api_key="x").generate("phishing", 1, **kwargs).scenes[0]

    # ------------------------------------------------------------- signature conformance

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: GeminiScriptProvider(api_key="x"),
            lambda: VerbatimScriptProvider(VERBATIM_TEXT),
        ],
    )
    def test_signature_matches_the_protocol_exactly(self, factory):
        """`pipeline._script_kwargs` inspects the real signature, so name, kind and
        default all have to match the Protocol or the choices are silently dropped.

        A provider must accept EVERY parameter the Protocol declares, positionally in the
        same order, with the same kind and default. It may accept more: `_script_kwargs`
        passes a fixed set of knobs, so an extra keyword-only parameter with a default is
        invisible to it, and `language` is exactly that — it is on the providers ahead of
        the Protocol so the API can be wired to it. What the assertion still forbids is the
        real failure mode: a provider MISSING a knob the Protocol promises, or renaming or
        reordering one, which is how `bullets_per_slide`/`tone` were dropped before.
        """
        provider = factory()
        expected = inspect.signature(ScriptProvider.generate).parameters
        actual = inspect.signature(provider.generate).parameters
        # `self` is bound away on the instance method but present on the Protocol.
        expected = {k: v for k, v in expected.items() if k != "self"}
        assert list(actual)[: len(expected)] == list(expected)
        for name, param in expected.items():
            assert actual[name].kind is param.kind, name
            assert actual[name].default == param.default, name
        for extra in set(actual) - set(expected):
            # An extra parameter is only invisible to the pipeline if it is keyword-only
            # AND defaulted; a required extra would make every existing call site a
            # TypeError.
            assert actual[extra].kind is inspect.Parameter.KEYWORD_ONLY, extra
            assert actual[extra].default is not inspect.Parameter.empty, extra
        assert actual["bullets_per_slide"].kind is inspect.Parameter.KEYWORD_ONLY
        assert actual["tone"].kind is inspect.Parameter.KEYWORD_ONLY
        assert isinstance(provider, ScriptProvider)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: GeminiScriptProvider(api_key="x"),
            lambda: VerbatimScriptProvider(VERBATIM_TEXT),
        ],
    )
    def test_pipeline_passes_both_knobs_through(self, factory):
        """The regression this whole change exists for: `_script_kwargs` used to return
        {} and log "those choices are being dropped"."""
        from app.db.models import Job
        from app.worker.pipeline import _script_kwargs

        job = Job(topic="phishing", slide_count=3, bullets_per_slide=5, tone="executives")
        assert _script_kwargs(factory(), job) == {
            "bullets_per_slide": 5,
            "tone": "executives",
        }

    # ------------------------------------------------------------------ count in prompt

    @pytest.mark.parametrize("count", [3, 4, 5])
    def test_requested_bullet_count_reaches_the_prompt(self, monkeypatch, count):
        """The knob reaches the prompt, capped by what a content slide may show.

        `docs/DIRECTION.md` §9 caps CONTENT at four points, so a request for five is honoured
        as four. Three is a deliberate, legal choice under that ceiling and survives.
        """
        prompt = self._prompt(monkeypatch, bullets_per_slide=count)
        expected = role_bullet_target(SceneRole.CONTENT, count)
        assert f"EXACTLY {expected} short on-screen points" in prompt
        assert f"{expected - 1} is too few and {expected + 1} is too many" in prompt
        # The old fixed range must be gone, not merely supplemented.
        assert "3 to 5 short on-screen points" not in prompt

    @pytest.mark.parametrize("count", [3, 4, 5])
    def test_the_narration_word_budget_does_not_move_with_the_bullet_count(
        self, monkeypatch, count
    ):
        """Audio is the clock, so a scene's word budget is its DURATION.

        A content scene is 11-19s whatever it carries (`docs/DIRECTION.md` §1.2), so asking
        for one fewer point buys a slower reveal, not a shorter scene. The word budget used
        to scale with the bullet count, which made the same role two different lengths.
        """
        prompt = self._prompt(monkeypatch, bullets_per_slide=count)
        low, target, high = narration_words(SceneRole.CONTENT)
        assert f"narration {target} words ({low}-{high})" in prompt
        assert "35-55 words" not in prompt

    def test_every_role_in_the_plan_states_its_own_word_budget(self, monkeypatch):
        prompt = self._prompt(monkeypatch, slide_count=SUMMARY_FROM_SLIDES)
        for role in SceneRole:
            low, target, high = narration_words(role)
            assert f"role: {role.value} — narration {target} words ({low}-{high})" in prompt

    @pytest.mark.parametrize("requested", [0, 1, 2, -7])
    def test_counts_below_the_legible_range_are_clamped_up(self, monkeypatch, requested):
        prompt = self._prompt(monkeypatch, bullets_per_slide=requested)
        assert f"EXACTLY {BULLET_MIN} short on-screen points" in prompt

    @pytest.mark.parametrize("requested", [6, 12, 400])
    def test_counts_above_the_legible_range_are_clamped_down(self, monkeypatch, requested):
        prompt = self._prompt(monkeypatch, bullets_per_slide=requested)
        assert f"EXACTLY {SceneRole.CONTENT.bullet_budget} short on-screen points" in prompt

    def test_a_nonsense_count_falls_back_to_the_default(self, monkeypatch):
        prompt = self._prompt(monkeypatch, bullets_per_slide="lots")  # type: ignore[arg-type]
        expected = role_bullet_target(SceneRole.CONTENT, BULLET_DEFAULT)
        assert f"EXACTLY {expected} short on-screen points" in prompt

    # ------------------------------------------------------------------- count honoured

    @pytest.mark.parametrize("count", [3, 4, 5])
    def test_returned_bullet_count_matches_the_request(self, monkeypatch, count):
        """Enforced on the way back too — a model that miscounts must not overflow or
        under-fill the panel the user sized, and must not exceed the role's budget."""
        over_delivered = [
            "Check The Sender Domain",
            "Trust The Display Name",
            "Hover Over Every Link",
            "Reveal The Real Destination",
            "Generic Salutation",
            "Urgent Deadlines",
            "Report Anything Suspicious",
        ]
        scene = self._scene_for(monkeypatch, over_delivered, bullets_per_slide=count)
        assert len(scene.bullets) == role_bullet_target(SceneRole.CONTENT, count), scene.bullets

    @pytest.mark.parametrize("count", [3, 4, 5])
    def test_under_delivery_is_topped_up_to_the_requested_count(self, monkeypatch, count):
        scene = self._scene_for(
            monkeypatch, ["Check The Sender Domain"], bullets_per_slide=count
        )
        assert len(scene.bullets) == role_bullet_target(SceneRole.CONTENT, count), scene.bullets
        for bullet in scene.bullets:
            # Top-ups are verbatim runs of the narration, so they stay timeable.
            assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_clamping_applies_to_the_returned_bullets_too(self, monkeypatch):
        scene = self._scene_for(monkeypatch, None, bullets_per_slide=99)
        assert len(scene.bullets) == SceneRole.CONTENT.bullet_budget

    # ------------------------------------------------------------------------- tone

    TONE_MARKERS = {
        "new_hires": "brand-new hires",
        "all_staff": "zero jargon",
        "technical": "MECHANISM",
        "executives": "business impact",
    }

    def test_every_documented_tone_has_a_clause(self):
        assert set(TONE_CLAUSES) == {"new_hires", "all_staff", "technical", "executives"}

    @pytest.mark.parametrize("tone", sorted(TONE_MARKERS))
    def test_each_tone_injects_its_own_clause_and_no_others(self, monkeypatch, tone):
        prompt = self._prompt(monkeypatch, tone=tone)
        assert "AUDIENCE" in prompt
        assert TONE_CLAUSES[tone] in prompt
        assert self.TONE_MARKERS[tone] in prompt
        for other, marker in self.TONE_MARKERS.items():
            if other != tone:
                assert marker not in prompt, (tone, other)

    @pytest.mark.parametrize("tone", sorted(TONE_MARKERS))
    def test_tone_clauses_change_the_instruction_not_just_an_adjective(self, tone):
        """Each clause must tell the model something substantive about what to say."""
        clause = TONE_CLAUSES[tone]
        assert len(clause.split()) >= 40, tone
        assert clause.startswith("AUDIENCE"), tone

    @pytest.mark.parametrize("tone", [None, "", "casual", "Pirate", "  "])
    def test_unknown_or_absent_tone_is_a_no_op(self, monkeypatch, tone):
        """Byte-identical to the untoned prompt, so nothing regresses for callers that
        never set a tone."""
        baseline = self._prompt(monkeypatch)
        assert self._prompt(monkeypatch, tone=tone) == baseline
        assert "AUDIENCE" not in baseline

    def test_tone_is_matched_case_and_space_insensitively(self, monkeypatch):
        assert TONE_CLAUSES["technical"] in self._prompt(monkeypatch, tone="  Technical ")

    def test_tone_and_count_compose(self, monkeypatch):
        prompt = self._prompt(monkeypatch, bullets_per_slide=3, tone="new_hires")
        assert "EXACTLY 3 short on-screen points" in prompt
        assert "narration 34 words (25-43)" in prompt
        assert TONE_CLAUSES["new_hires"] in prompt

    # -------------------------------------------------------------- verbatim provider

    @pytest.mark.parametrize("count", [3, 4, 5])
    def test_verbatim_provider_honours_the_bullet_budget(self, count):
        scene = (
            VerbatimScriptProvider(self.NARRATION)
            .generate("phishing", 1, bullets_per_slide=count)
            .scenes[0]
        )
        assert len(scene.bullets) == count, scene.bullets
        for bullet in scene.bullets:
            assert anchor_position(bullet, scene.narration) is not None, bullet

    def test_verbatim_provider_never_cuts_scraps_to_hit_the_budget(self):
        """A budget is a ceiling, not a quota: thin source text yields fewer points rather
        than two-word fragments."""
        scene = (
            VerbatimScriptProvider("Check the sender domain first.")
            .generate("phishing", 1, bullets_per_slide=5)
            .scenes[0]
        )
        assert len(scene.bullets) < 5
        for bullet in scene.bullets:
            assert len(bullet.split()) >= 2, bullet

    @pytest.mark.parametrize("tone", [None, "new_hires", "executives", "technical"])
    def test_verbatim_provider_accepts_tone_and_ignores_it(self, tone):
        """Documented asymmetry: it writes no prose of its own, so there is nothing for a
        register to change. Accepting the argument keeps the Protocol satisfiable."""
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 3, tone=tone)
        baseline = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 3)
        assert script.model_dump() == baseline.model_dump()

    def test_verbatim_narration_is_still_verbatim_at_every_budget(self):
        for count in (3, 4, 5):
            script = VerbatimScriptProvider(VERBATIM_TEXT).generate(
                "bridges", 3, bullets_per_slide=count, tone="executives"
            )
            rejoined = " ".join(s.narration for s in script.scenes)
            assert rejoined.split() == VERBATIM_TEXT.split(), count


class TestScriptStructure:
    """The video's SHAPE: a short title card, teaching scenes, an optional recap, an ending.

    Every number asserted here comes from `docs/DIRECTION.md` — §1.1 for the role sequence
    per slide count, §1.2 for the per-role word and bullet budgets, §9 for the corrections
    to the committed `SceneRole` values. Where DIRECTION and the older calibration disagree,
    DIRECTION wins; these tests are what pins that down.
    """

    NARRATION = TestBulletBudgetAndTone.NARRATION

    # docs/DIRECTION.md §1.1, transcribed. T=title C=content S=summary X=closing.
    DIRECTION_TABLE = {
        4: "TCCX",
        5: "TCCCX",
        6: "TCCCCX",
        7: "TCCCCSX",
        8: "TCCCCCSX",
        9: "TCCCCCCSX",
        10: "TCCCCCCCSX",
    }

    def _prompt(self, monkeypatch, slide_count: int, **kwargs) -> str:
        captured: dict = {}

        def fake(model, body, api_key, **_):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _script_response([_scene(1, narration=self.NARRATION)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x").generate("phishing", slide_count, **kwargs)
        return captured["text"]

    def _script(self, monkeypatch, scenes: list[dict], slide_count: int, **kwargs):
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response(scenes),
        )
        return GeminiScriptProvider(api_key="x").generate("phishing", slide_count, **kwargs)

    # -------------------------------------------------------------------------- the plan

    @pytest.mark.parametrize("slide_count", sorted(DIRECTION_TABLE))
    def test_the_plan_matches_the_direction_table(self, slide_count):
        letters = {"title": "T", "content": "C", "summary": "S", "closing": "X"}
        actual = "".join(letters[role.value] for role in role_plan(slide_count))
        assert actual == self.DIRECTION_TABLE[slide_count]

    @pytest.mark.parametrize("slide_count", range(STRUCTURE_FROM_SLIDES, 11))
    def test_exactly_one_title_first_and_one_closing_last(self, slide_count):
        plan = role_plan(slide_count)
        assert plan[0] is SceneRole.TITLE
        assert plan[-1] is SceneRole.CLOSING
        assert plan.count(SceneRole.TITLE) == 1
        assert plan.count(SceneRole.CLOSING) == 1

    @pytest.mark.parametrize("slide_count", range(1, 11))
    def test_summary_appears_only_from_the_spec_threshold_and_only_before_the_closing(
        self, slide_count
    ):
        """DIRECTION §1.1 is explicit that a six-slide video does not get a recap either:
        below ~100s there is nothing to recap and the closing already restates the point."""
        plan = role_plan(slide_count)
        summaries = [i for i, role in enumerate(plan) if role is SceneRole.SUMMARY]
        if slide_count < SUMMARY_FROM_SLIDES:
            assert summaries == []
        else:
            assert summaries == [len(plan) - 2], plan

    @pytest.mark.parametrize("slide_count", [1, 2])
    def test_below_the_structural_floor_the_ending_is_what_survives(self, slide_count):
        """A two-scene video cannot hold all three parts. The title card is what gives:
        stopping dead on the last content slide is the defect being fixed, and a title card
        would be half the runtime."""
        plan = role_plan(slide_count)
        assert len(plan) == slide_count
        assert SceneRole.TITLE not in plan
        assert plan[-1] is (SceneRole.CLOSING if slide_count > 1 else SceneRole.CONTENT)

    # ------------------------------------------------------------------------- budgets

    @pytest.mark.parametrize("role", list(SceneRole))
    def test_word_budgets_match_the_role_durations(self, role):
        """The word budget IS the duration, so the two must not drift apart.

        If someone retunes `SceneRole.target_duration` without retuning
        `ROLE_NARRATION_WORDS`, the script would keep writing to the old pacing and every
        scene would land at the wrong length. One word of slack for the doc's rounding.
        """
        low, target, high = narration_words(role)
        min_seconds, max_seconds = role.target_duration
        assert abs(low - words_spoken_in(min_seconds)) <= 1.0, role
        assert abs(high - words_spoken_in(max_seconds)) <= 1.0, role
        assert low <= target <= high, role

    @pytest.mark.parametrize("role", list(SceneRole))
    def test_the_bullet_knob_is_capped_by_every_roles_budget(self, role):
        for requested in (0, 3, 4, 5, 99):
            assert role_bullet_target(role, requested) <= role.bullet_budget
        assert role_bullet_target(SceneRole.TITLE, 5) == 0

    def test_a_deliberately_smaller_request_is_still_honoured(self):
        """The budget is a ceiling, not a quota: asking for three points gets three."""
        assert role_bullet_target(SceneRole.CONTENT, 3) == 3

    # -------------------------------------------------------------------------- prompt

    def test_the_prompt_names_every_scene_with_its_role_and_budgets(self, monkeypatch):
        prompt = self._prompt(monkeypatch, 7)
        for index, role in enumerate(role_plan(7), start=1):
            low, target, high = narration_words(role)
            bullets = role_bullet_target(role, BULLET_DEFAULT)
            plural = "" if bullets == 1 else "s"
            assert (
                f"scene {index} — role: {role.value} — narration {target} words "
                f"({low}-{high}) — {bullets} bullet{plural}"
            ) in prompt

    def test_the_prompt_forbids_bullets_on_the_title_card_in_words(self, monkeypatch):
        prompt = self._prompt(monkeypatch, 5)
        assert "ZERO bullets" in prompt
        assert "scene 1 — role: title" in prompt
        assert "0 bullets" in prompt

    def test_a_video_without_a_recap_is_never_told_what_a_recap_is(self, monkeypatch):
        """Describing the summary role in a five-slide prompt is an invitation to write
        one, and the shape is then wrong however good the description was."""
        prompt = self._prompt(monkeypatch, SUMMARY_FROM_SLIDES - 1)
        assert "summary — the RECAP" not in prompt
        assert "role: summary" not in prompt
        assert "summary — the RECAP" in self._prompt(monkeypatch, SUMMARY_FROM_SLIDES)

    def test_the_prompt_states_the_one_line_copy_caps(self, monkeypatch):
        """DIRECTION §2.1/§2.2: a wrapped heading moves the first bullet's baseline, so the
        stack starts at a different height on every slide."""
        prompt = self._prompt(monkeypatch, 5)
        assert f"at most {HEADING_CHAR_MAX} characters" in prompt
        assert f"at most {BULLET_CHAR_MAX} characters" in prompt
        assert f"at most {TITLE_CHAR_MAX} characters" in prompt
        assert "Sentence case, NOT Title Case" in prompt

    def test_the_prompt_asks_for_motion_in_clip_prompt_not_composition(self, monkeypatch):
        prompt = self._prompt(monkeypatch, 5)
        assert "clip_prompt" in prompt
        assert "described as MOTION instead of composition" in prompt
        assert "camera move" in prompt

    # ------------------------------------------------------------------ returned script

    def test_the_returned_script_carries_the_planned_shape(self, monkeypatch):
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 8)]
        script = self._script(monkeypatch, scenes, 7)
        assert [scene_role(s) for s in script.scenes] == role_plan(7)

    def test_the_title_card_loses_bullets_the_model_sent_anyway(self, monkeypatch):
        """The observed defect: scene 1 rendered as a title card carrying a content heading
        and four bullets. DIRECTION §1.3 calls it the single worst defect in the output."""
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 6)]
        for scene in scenes:
            scene["bullets"] = ["Check The Sender Domain", "Hover Over Every Link"]
        script = self._script(monkeypatch, scenes, 5)
        assert scene_role(script.scenes[0]) is SceneRole.TITLE
        assert script.scenes[0].bullets == []

    def test_the_closing_is_cut_to_two_points_in_spoken_order(self, monkeypatch):
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 6)]
        for scene in scenes:
            scene["bullets"] = [
                "Report Anything Suspicious",
                "Check The Sender Domain",
                "Hover Over Every Link",
                "Urgent Deadlines",
            ]
        closing = self._script(monkeypatch, scenes, 5).scenes[-1]
        assert scene_role(closing) is SceneRole.CLOSING
        assert closing.bullets == ["Check The Sender Domain", "Hover Over Every Link"]

    def test_a_model_that_labels_every_scene_content_is_overridden_by_position(
        self, monkeypatch
    ):
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 8)]
        for scene in scenes:
            scene["role"] = "content"
        script = self._script(monkeypatch, scenes, 7)
        assert [scene_role(s).value for s in script.scenes] == [
            r.value for r in role_plan(7)
        ]

    def test_an_off_schema_role_does_not_crash_the_parse(self, monkeypatch):
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 5)]
        scenes[0]["role"] = "interlude"
        script = self._script(monkeypatch, scenes, 4)
        assert scene_role(script.scenes[0]) is SceneRole.TITLE

    def test_roles_are_assigned_after_a_scene_count_backfill(self, monkeypatch):
        """`_fit_scene_count` can split one scene into several, so a scene's role depends on
        where it ends up — not on what the model called it before the split."""
        long_narration = " ".join([self.NARRATION] * 2)
        script = self._script(monkeypatch, [_scene(1, narration=long_narration)], 4)
        assert len(script.scenes) == 4
        assert [scene_role(s) for s in script.scenes] == role_plan(4)
        assert script.scenes[0].bullets == []

    def test_clip_prompt_is_carried_through_and_missing_becomes_none(self, monkeypatch):
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 5)]
        scenes[1]["clip_prompt"] = "Slow push in as a **cursor** hovers over the link"
        script = self._script(monkeypatch, scenes, 4)
        # Markdown is stripped: this text may be forwarded to a video model verbatim.
        assert scene_clip_prompt(script.scenes[1]) == (
            "Slow push in as a cursor hovers over the link"
        )
        assert scene_clip_prompt(script.scenes[0]) is None

    def test_every_bullet_still_anchors_in_its_own_narration_at_every_role(
        self, monkeypatch
    ):
        """The load-bearing invariant, unchanged by the new shape: an unanchored bullet
        falls back to proportional placement and lands on the wrong words."""
        scenes = [_scene(i, narration=self.NARRATION) for i in range(1, 8)]
        for scene in scenes:
            scene["bullets"] = ["Check The Sender Domain", "Hover Over Every Link"]
        for scene in self._script(monkeypatch, scenes, 7).scenes:
            for bullet in scene.bullets:
                assert anchor_position(bullet, scene.narration) is not None, (
                    scene.id,
                    scene_role(scene).value,
                    bullet,
                )

    # ------------------------------------------------------------- providers and helpers

    def test_verbatim_scripts_are_all_content(self):
        """It cannot resize a slice of the user's own words to fit a 4.5s title card, and
        labelling a forty-word segment "title" would tell the renderer it is one."""
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 5)
        assert [scene_role(s) for s in script.scenes] == [SceneRole.CONTENT] * 5

    def test_verbatim_scenes_still_offer_a_motion_prompt(self):
        for scene in VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 3).scenes:
            clip = scene_clip_prompt(scene)
            assert clip and "push in" in clip
            assert "No text, no letters" in clip

    def test_the_role_helpers_default_for_a_plain_scene_script(self):
        """A provider that knows nothing about roles still satisfies the Protocol, and the
        pipeline reads its scenes through these two helpers."""
        plain = SceneScript(id=1, narration="n", heading="H", image_prompt="p")
        assert scene_role(plain) is SceneRole.CONTENT
        assert scene_clip_prompt(plain) is None

    def test_a_structured_scene_survives_pydantic_validation_inside_a_script(self):
        """`Script.scenes` is typed `list[SceneScript]`; the subclass must not be coerced
        away, or role and clip_prompt would silently vanish between provider and pipeline."""
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate("bridges", 2)
        assert all(isinstance(s, StructuredSceneScript) for s in script.scenes)


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
            # 48 kHz, not 24: at 24 kHz the band above 12 kHz is empty (12 kHz IS the
            # Nyquist limit); at 48 kHz it carries real sibilance and breath. Speed 0.9
            # lands ~146 wpm against the pipeline's 135 target — 1.0 overshoots by ~22%.
            "sample_rate": "48000",
            "container": "wav",
            "speed": "0.9",
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

    def test_declares_no_ssml_support(self):
        """Aura vocalises markup rather than parsing it, and there is no flag to change
        that — see `app/providers/deepgram_tts` for the Deepgram statement and the two API
        errors proving no SSML request shape exists. Full coverage in test_deepgram_ssml.py.
        """
        assert DeepgramSynthesizer.supports_ssml is False

    def test_markup_never_reaches_the_api_even_if_a_caller_ignores_the_flag(
        self, monkeypatch, tmp_path
    ):
        """Defence in depth: tags in the request would be spoken aloud in the video."""
        wav = _wav_bytes(1.0, tmp_path / "src.wav")
        requests: list[str] = []

        def fake_post(url, **kwargs):
            requests.append(kwargs["json"]["text"])
            return _FakeResponse(content=wav)

        monkeypatch.setattr(deepgram_tts.httpx, "post", fake_post)
        DeepgramSynthesizer(api_key="k").synthesize(
            '<speak>Check the sender.<break time="1s"/>Then hover the link.</speak>',
            "v",
            tmp_path / "a.wav",
        )
        assert requests == ["Check the sender. Then hover the link."]


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
class TestLiveBulletTiming:
    """The whole chain: real script, real speech, real alignment, real reveal times."""

    def test_generated_bullets_anchor_to_the_words_the_narrator_says(self, tmp_path):
        script = GeminiScriptProvider().generate(
            "How phishing attacks work and how to spot them", 4
        )
        assert len(script.scenes) == 4
        for scene in script.scenes:
            assert 3 <= len(scene.bullets) <= 5, scene.bullets
            for bullet in scene.bullets:
                assert not bullet.endswith(".")
                assert bullet[0] not in "-*•"
                assert anchor_position(bullet, scene.narration) is not None, (
                    scene.id,
                    bullet,
                    scene.narration,
                )

        scene = script.scenes[0]
        out = tmp_path / "scene1.wav"
        path, duration = DeepgramSynthesizer().synthesize_with_duration(
            scene.narration, "aura-2-draco-en", out
        )
        words = DeepgramAligner().align(path, scene.narration)
        assert words

        points = time_bullets(scene.bullets, words, 0.0, duration)
        assert [p.text for p in points] == scene.bullets
        times = [p.appear_at for p in points]
        assert times == sorted(times), times
        assert all(t >= 0.0 for t in times)
        assert max(times) <= duration - TAIL_GUARD + 1e-6
        assert sum(p.emphasis for p in points) == 1

        # Every reveal must be a real anchor, not a proportional guess, and must sit next
        # to the words it quotes rather than a second away from them.
        anchors = find_anchors(scene.bullets, words, 0.0, duration)
        for point, anchor in zip(points, anchors, strict=True):
            assert anchor.method == "ngram", (anchor.text, anchor.method)
            assert anchor.match_len >= 2 or len(anchor.text.split()) == 1
            quoted = " ".join(anchor.matched_words).lower()
            assert quoted in scene.narration.lower(), (quoted, scene.narration)
            assert abs(point.appear_at - max(0.0, anchor.anchor_time - LEAD)) <= 1.0


@live_only
class TestLiveMusic:
    def test_45_second_bed_is_looped_from_the_fixed_clip(self, tmp_path):
        out = tmp_path / "music.mp3"
        LyriaMusicProvider().generate("calm ambient documentary underscore", 45.0, out)
        duration = audio_duration(out)
        assert duration == pytest.approx(45.0, abs=0.1)
        assert duration > 40.0, "looping did not happen — clip length leaked through"


# =========================================================== language


class TestScriptLanguage:
    """`generate(..., language=...)` writes the script natively in the target language.

    The three things worth guarding, in order of how expensive they are to get wrong:

      * ENGLISH DOES NOT MOVE. English is the measured baseline (22/22 bullets anchored),
        so the English prompt must stay byte-identical and the English parse path must be
        unchanged. Every language-derived insert is empty for `Language.EN`.
      * A word budget is a DURATION. Reusing English's word count in another language puts
        the scene outside its role's window, so the budgets scale by measured speaking rate
        and the test asserts the DURATION is preserved, not the words.
      * The English/in-language SPLIT. Narration, heading and bullets are in-language;
        `image_prompt` and `clip_prompt` are English, because the models consuming them are.
    """

    NARRATION_ES = (
        "Revise con cuidado la dirección del remitente para confirmar su autenticidad. "
        "Desconfíe siempre de un tono de urgencia que exija acciones inmediatas. "
        "Detecte cualquier saludo genérico en lugar de su nombre real. "
        "Examine los enlaces sospechosos pasando el cursor encima antes de abrirlos."
    )
    NARRATION_HI = (
        "संदेश मिलते ही सबसे पहले भेजने वाले का पता ध्यान से देखें। "
        "जालसाज अक्सर डोमेन नाम में गड़बड़ी करके असली जैसे दिखने वाले ईमेल भेजते हैं। "
        "यदि कोई ईमेल तुरंत कार्रवाई का दबाव बनाए, तो सतर्क हो जाएं।"
    )

    def _prompt(self, monkeypatch, *, slide_count: int = 5, **kwargs) -> str:
        captured: dict = {}

        def fake(model, body, api_key, **_):
            captured["text"] = body["contents"][0]["parts"][0]["text"]
            return _script_response([_scene(1)])

        monkeypatch.setattr("app.providers.gemini_script.generate_content", fake)
        GeminiScriptProvider(api_key="x").generate("phishing", slide_count, **kwargs)
        return captured["text"]

    # -------------------------------------------------------------- signature

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: GeminiScriptProvider(api_key="x"),
            lambda: VerbatimScriptProvider(VERBATIM_TEXT),
        ],
    )
    def test_language_is_keyword_only_with_an_english_default(self, factory):
        """Keyword-only and defaulted is what keeps every existing caller — and Protocol
        conformance — working without being touched."""
        param = inspect.signature(factory().generate).parameters["language"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is Language.EN

    def test_existing_callers_are_unaffected(self, monkeypatch):
        """The two-positional-argument call still works and still returns English."""
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([_scene(1), _scene(2)]),
        )
        script = GeminiScriptProvider(api_key="x").generate("phishing", 2)
        assert len(script.scenes) == 2

    # -------------------------------------------------------------- english does not move

    @pytest.mark.parametrize("slide_count", [1, 2, 4, 5, 7, 9])
    @pytest.mark.parametrize("tone", [None, "technical", "executives"])
    def test_the_english_prompt_is_unchanged_by_the_existence_of_languages(
        self, monkeypatch, slide_count, tone
    ):
        """Every language-derived insert is empty for English, so there is no new text in
        the English prompt that could move the measured English result."""
        prompt = self._prompt(
            monkeypatch, slide_count=slide_count, tone=tone, language=Language.EN
        )
        assert "LANGUAGE — write this video in" not in prompt
        assert "IN ENGLISH" not in prompt
        assert "Check the language too" not in prompt
        # The pre-language wording of the two visual fields, verbatim.
        assert "image_prompt — describe ONE photographic" in prompt
        assert "clip_prompt — the SAME shot as image_prompt" in prompt
        assert f"about {int(WORDS_PER_MINUTE)} words per minute" in prompt

    def test_english_is_the_default_and_the_default_is_english(self, monkeypatch):
        assert self._prompt(monkeypatch) == self._prompt(monkeypatch, language=Language.EN)

    def test_english_word_budgets_are_exactly_the_direction_table(self):
        for role in SceneRole:
            assert narration_words(role) == narration_words(role, Language.EN)
            assert narration_words(role, Language.EN) == ROLE_NARRATION_WORDS[role]

    # -------------------------------------------------------------- the prompt

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_a_non_english_prompt_asks_for_native_composition_not_translation(
        self, monkeypatch, language
    ):
        prompt = self._prompt(monkeypatch, language=language)
        assert f"LANGUAGE — write this video in {LANGUAGE_NAMES[language]}" in prompt
        assert "Compose it NATIVELY" in prompt
        assert "Do NOT draft in English and translate" in prompt

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_the_visual_prompts_are_pinned_to_english_at_the_field(
        self, monkeypatch, language
    ):
        """Stated where the model is deciding what to write, not only in a block up top."""
        prompt = self._prompt(monkeypatch, language=language)
        assert "image_prompt — IN ENGLISH, not in the language of the narration —" in prompt
        assert "clip_prompt — IN ENGLISH, not in the language of the narration —" in prompt
        assert "image_prompt and clip_prompt stay in ENGLISH" in prompt
        assert "image_prompt and clip_prompt in English on every single scene" in prompt

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_each_language_injects_only_its_own_conventions(self, monkeypatch, language):
        prompt = self._prompt(monkeypatch, language=language)
        assert LANGUAGE_CLAUSES[language].strip() in prompt
        for other in (Language.ES, Language.HI):
            if other is not language:
                assert LANGUAGE_CLAUSES[other].strip() not in prompt

    def test_the_spanish_clause_demands_accents_and_inverted_punctuation(self, monkeypatch):
        """Unaccented Spanish reads as machine output, and "está" is not "esta"."""
        prompt = self._prompt(monkeypatch, language=Language.ES)
        assert "¿" in prompt and "¡" in prompt
        assert "fully accented Spanish" in prompt

    def test_the_hindi_clause_demands_the_danda_and_drops_the_casing_rule(self, monkeypatch):
        """Devanagari is unicameral, so a sentence-case instruction is noise at best; and a
        Hindi sentence ending in "." is a visible defect, not a nit."""
        prompt = self._prompt(monkeypatch, language=Language.HI)
        assert DANDA in prompt
        assert 'End every\nsentence with the danda' in prompt
        assert "Devanagari has no upper and lower case" in prompt

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_the_anchor_rule_is_qualified_for_an_inflected_language(
        self, monkeypatch, language
    ):
        """An inflected language can restate a phrase's meaning in a form that shares no
        word with the narration, which is a silently broken anchor."""
        prompt = self._prompt(monkeypatch, language=language)
        # The base rule is still there...
        assert "CONSECUTIVE CONTENT WORDS" in prompt
        # ...and now says the echo must be the SAME SURFACE FORM.
        assert "SURFACE FORM, not meaning" in prompt
        assert LANGUAGE_ANCHOR_NOTES[language].strip() in prompt

    # -------------------------------------------------------------- word budgets

    @pytest.mark.parametrize("language", list(Language))
    @pytest.mark.parametrize("role", list(SceneRole))
    def test_the_word_budget_range_stays_ordered_after_scaling(self, language, role):
        low, target, high = narration_words(role, language)
        assert 1 <= low <= target <= high

    @pytest.mark.parametrize("language", list(Language))
    @pytest.mark.parametrize("role", list(SceneRole))
    def test_scaling_preserves_the_DURATION_not_the_word_count(self, language, role):
        """The point of the whole exercise. A word budget is a duration, so the scaled
        budget must take the same time to speak as the English one does — spoken at that
        language's own pace."""
        english = narration_words(role, Language.EN)
        scaled = narration_words(role, language)
        for en_words, lang_words in zip(english, scaled, strict=True):
            en_seconds = en_words / WORDS_PER_MINUTE * 60
            lang_seconds = lang_words / language_wpm(language) * 60
            # Within a word of rounding at the language's own rate.
            assert abs(en_seconds - lang_seconds) <= 60 / language_wpm(language)

    @pytest.mark.parametrize("language", list(Language))
    @pytest.mark.parametrize("role", list(SceneRole))
    def test_every_scaled_budget_tracks_its_roles_window(self, language, role):
        """The per-language generalisation of `test_word_budgets_match_the_role_durations`.

        Same invariant and the same tolerance: the budget's floor CORRESPONDS to the window's
        floor and its ceiling to the window's ceiling, within one word of rounding at that
        language's pace. Deliberately not "strictly inside" — the committed English edges are
        themselves a word either side of the window (a 20-word closing is 8.89s against a 9.0s
        ceiling), so demanding strict containment would fail English too and would be a claim
        about `docs/DIRECTION.md`, not about this scaling.
        """
        low, target, high = narration_words(role, language)
        floor, ceiling = role.target_duration
        slack = 60 / language_wpm(language)  # one word, at this language's rate
        assert abs(low - words_spoken_in(floor, language)) <= 1.0 + slack, (role, language)
        assert abs(high - words_spoken_in(ceiling, language)) <= 1.0 + slack, (role, language)
        assert low <= target <= high

    @pytest.mark.parametrize("language", list(Language))
    @pytest.mark.parametrize("role", list(SceneRole))
    def test_the_target_word_count_is_comfortably_inside_the_window(self, language, role):
        """The TARGET is the number the prompt leads with and the one scenes actually land
        on, so unlike the range edges it must sit strictly inside the role's window."""
        target = narration_words(role, language)[1]
        floor, ceiling = role.target_duration
        seconds = target / language_wpm(language) * 60
        assert floor < seconds < ceiling, (role, language, target, seconds)

    def test_a_language_that_speaks_faster_gets_more_words(self):
        """Spanish and Hindi speak faster than English word-for-word, so re-using English's
        budget makes the scene SHORT, not long — the opposite of the intuition that Spanish
        "runs longer". Both are true: es needs 1.12x the words to say the same thing, and gets
        1.04x the budget, which is `LANGUAGES.md` §6.3's content-per-slide loss.

        Monotonic but not strictly so at every role: at 1.04x the Spanish title target rounds
        back onto English's 10, which is the documented table, not a bug.
        """
        assert language_word_factor(Language.EN) == 1.0
        assert language_word_factor(Language.ES) > 1.0
        assert language_word_factor(Language.HI) > language_word_factor(Language.ES)
        for role in SceneRole:
            en_low, en_target, en_high = narration_words(role, Language.EN)
            es_low, es_target, es_high = narration_words(role, Language.ES)
            hi_low, hi_target, hi_high = narration_words(role, Language.HI)
            assert es_target >= en_target and es_high >= en_high, role
            assert hi_target > en_target and hi_high > en_high, role
            assert hi_target >= es_target and hi_low >= es_low, role
        # ...and strictly more somewhere, or the scaling is doing nothing at all.
        assert narration_words(SceneRole.CONTENT, Language.ES)[1] > (
            narration_words(SceneRole.CONTENT, Language.EN)[1]
        )

    def test_the_budgets_are_the_ones_in_LANGUAGES_md(self):
        """`docs/LANGUAGES.md` §6.2 is the authority; these are transcribed from it, not
        recomputed, because multiply-then-round does not reproduce its table exactly."""
        assert narration_words(SceneRole.CONTENT, Language.ES) == (26, 35, 45)
        assert narration_words(SceneRole.CONTENT, Language.HI) == (29, 39, 50)
        assert narration_words(SceneRole.TITLE, Language.ES) == (9, 10, 15)
        assert narration_words(SceneRole.TITLE, Language.HI) == (10, 12, 16)
        assert narration_words(SceneRole.SUMMARY, Language.ES) == (21, 28, 32)
        assert narration_words(SceneRole.SUMMARY, Language.HI) == (23, 31, 36)
        assert narration_words(SceneRole.CLOSING, Language.ES) == (14, 18, 21)
        assert narration_words(SceneRole.CLOSING, Language.HI) == (15, 20, 23)
        # Effective pace, §6.2. English is DIRECTION §5's 135, untouched.
        assert language_wpm(Language.EN) == WORDS_PER_MINUTE == 135.0
        assert language_wpm(Language.ES) == 140.0
        assert language_wpm(Language.HI) == 155.0
        assert LANGUAGE_WORD_FACTOR == {Language.EN: 1.0, Language.ES: 1.04, Language.HI: 1.15}
        assert LANGUAGE_WPM[Language.EN] == WORDS_PER_MINUTE

    @pytest.mark.parametrize("language", list(Language))
    def test_the_prompt_states_the_languages_own_budget_and_pace(self, monkeypatch, language):
        prompt = self._prompt(monkeypatch, slide_count=7, language=language)
        assert f"about {int(language_wpm(language))} words per minute" in prompt
        for index, role in enumerate(role_plan(7), start=1):
            low, target, high = narration_words(role, language)
            assert (
                f"scene {index} — role: {role.value} — narration {target} words "
                f"({low}-{high})"
            ) in prompt

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_a_non_english_scene_gets_a_sentence_budget_too(self, monkeypatch, language):
        """`docs/LANGUAGES.md` §6.3: the same 34-word budget produced 20.2-23.5s of staccato
        English and 12.8-13.0s of Hindi, purely on sentence count. A word budget with no
        sentence budget is not a duration, so the two travel together."""
        prompt = self._prompt(monkeypatch, slide_count=7, language=language)
        for index, role in enumerate(role_plan(7), start=1):
            assert f"scene {index} — role: {role.value}" in prompt
            assert role_sentences(role) in prompt

    def test_english_is_left_out_of_the_sentence_budget_on_purpose(self, monkeypatch):
        """§6.3 says English needs this MORE than the others — its 34-word content scenes run
        20-23s against a 19.0s max. But retuning English pacing inside a language change would
        move a number DIRECTION §5 owns and other tests measure, so it is reported, not done.
        """
        prompt = self._prompt(monkeypatch, slide_count=7, language=Language.EN)
        for role in SceneRole:
            assert role_sentences(role) not in prompt

    def test_words_spoken_in_follows_the_language(self):
        assert words_spoken_in(60.0) == pytest.approx(WORDS_PER_MINUTE)
        assert words_spoken_in(60.0, Language.ES) == pytest.approx(language_wpm(Language.ES))
        assert words_spoken_in(60.0, Language.HI) > words_spoken_in(60.0, Language.ES)

    # -------------------------------------------------------------- coercion

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Language.HI, Language.HI),
            ("es", Language.ES),
            (" ES ", Language.ES),
            ("hi", Language.HI),
            ("en", Language.EN),
            ("klingon", Language.EN),
            ("", Language.EN),
            (None, Language.EN),
            (7, Language.EN),
        ],
    )
    def test_a_language_code_or_junk_never_fails_the_job(self, value, expected):
        """A bad language code is not worth failing a job over when the alternative is a
        perfectly good English script."""
        assert coerce_language(value) is expected

    def test_an_unknown_language_string_reaches_the_model_as_english(self, monkeypatch):
        assert self._prompt(monkeypatch, language="klingon") == self._prompt(monkeypatch)

    # -------------------------------------------------------------- typography

    def test_a_hindi_heading_loses_its_danda_and_an_english_one_its_full_stop(self):
        assert _clean_heading("संदेश की जांच करें।", Language.HI) == "संदेश की जांच करें"
        assert _clean_heading("Check the domain.", Language.EN) == "Check the domain"
        # The danda is not English punctuation, so English leaves it alone rather than
        # silently editing a character it does not understand.
        assert _clean_heading("Check the domain।", Language.EN).endswith(DANDA)

    def test_a_hindi_bullet_loses_its_danda(self):
        assert _clean_bullet("डोमेन नाम जांचें।", Language.HI) == "डोमेन नाम जांचें"

    def test_spanish_opening_punctuation_survives_cleaning(self):
        """Only the TRAILING end is stripped — an opened question that never closes would
        be worse than the terminal mark."""
        assert _clean_heading("¿Reconoce el remitente?", Language.ES).startswith("¿")
        assert _clean_bullet("¡No haga clic!", Language.ES) == "¡No haga clic"

    def test_devanagari_sentences_split_on_the_danda(self):
        """Without this a whole Hindi narration is one unsplittable sentence and every path
        that works by cutting on sentence boundaries quietly gives up."""
        segments = _split_into_segments(self.NARRATION_HI, 3)
        assert len(segments) == 3
        assert all(seg.strip() for seg in segments)
        # No words invented or lost.
        assert " ".join(segments).split() == self.NARRATION_HI.split()

    # -------------------------------------------------------------- tokenisation

    def test_the_tokenizer_sees_accents_and_devanagari(self):
        """The old ASCII-only pattern cut "protección" into "protecci"+"n" and matched
        nothing whatsoever in Devanagari, so every deterministic fallback returned []."""
        assert _words("protección") == ["protección"]
        assert _words("señales de diseño") == ["señales", "de", "diseño"]
        assert _words("डोमेन नाम जांचें") == ["डोमेन", "नाम", "जांचें"]
        # ASCII behaviour is untouched.
        assert _words("don't world-class") == ["don't", "world-class"]

    def test_the_tokenizer_keeps_devanagari_marks_attached(self):
        """The trap: Python's `\\w` covers L*/N* but NOT Mn/Mc, so a `[^\\W_]+` "Unicode"
        pattern DROPS every matra, anusvara and nukta — "जांचें" becomes "जच". That is the
        corruption `Language.needs_shaping` warns about, arriving at the tokeniser."""
        for word in ("जांचें", "फ़िशिंग", "गड़बड़ी", "व्यक्तिगत", "सुरक्षित"):
            assert _words(word) == [word], word
        # Two words that differ ONLY by nukta and anusvara must stay distinct.
        assert _words("फ़िशिंग") != _words("फिशिग")

    @pytest.mark.parametrize(
        ("text", "language"),
        [(NARRATION_ES, Language.ES), (NARRATION_HI, Language.HI)],
    )
    def test_fallback_bullets_are_derivable_in_every_language(self, text, language):
        bullets = _bullets_from(text, 3, language)
        assert len(bullets) >= 2
        for bullet in bullets:
            assert bullet.strip()
            # Every fragment is a verbatim run of the source, by construction.
            assert bullet.lower() in text.lower()

    def test_a_spanish_fragment_opens_on_a_content_word_not_an_article(self):
        """English stopwords applied to Spanish treat "el" as content and open a bullet on
        it, which reads as a template artefact in any language."""
        fragment = _fragment_from("el dominio del remitente es falso", Language.ES)
        assert not fragment.lower().startswith("el ")
        assert "dominio" in fragment.lower()

    def test_devanagari_headings_are_not_put_through_an_english_casing_rule(self):
        heading = _heading_from(self.NARRATION_HI, fallback="X", language=Language.HI)
        assert heading.strip()
        assert DANDA not in heading
        # Every word came from the narration, unmodified.
        for word in heading.split():
            assert word in self.NARRATION_HI

    # -------------------------------------------------------------- anchoring reality

    def test_anchoring_is_supported_for_latin_scripts(self):
        assert anchoring_supported(Language.EN) is True
        assert anchoring_supported(Language.ES) is True

    def test_devanagari_cannot_anchor_and_this_is_measured_not_assumed(self):
        """THE PRODUCT CONSTRAINT, asserted so it cannot regress silently in either
        direction.

        `deepgram_align.normalize` keeps only `[a-z0-9]`, so every Devanagari token
        normalises to "" and no Hindi bullet can ever match its own narration — even when the
        bullet is a verbatim copy of it, which is what the model actually produces. If this
        test starts failing because `anchoring_supported(HI)` became True, that is GOOD news:
        `normalize` was widened, and the Hindi fallback branches here are now dead code that
        should be removed.
        """
        assert anchoring_supported(Language.HI) is False
        # The cause, spelled out: it is the normaliser, not the model's wording.
        bullet = "वाले का डोमेन"
        narration = "भेजने वाले का डोमेन ध्यान से जांचें।"
        assert bullet in narration, "the bullet IS a verbatim run of the narration"
        assert anchor_position(bullet, narration) is None, "yet it cannot be anchored"
        assert normalize("डोमेन") == ""

    def test_hindi_bullets_are_kept_rather_than_shredded_by_a_check_that_cannot_run(self):
        """Applying the anchor defences to a script the matcher cannot read would demote
        every bullet the model wrote on the strength of a test that answered "no" to every
        question — replacing good copy with mechanically sliced fragments."""
        model_bullets = ["भेजने वाले का पता", "डोमेन नाम में गड़बड़ी", "तुरंत कार्रवाई का दबाव"]
        kept = _clean_bullets(model_bullets, self.NARRATION_HI, 3, language=Language.HI)
        assert kept == model_bullets

    def test_spanish_bullets_still_go_through_the_anchor_defences(self):
        """Spanish accents are stripped on BOTH sides by `normalize`, so they still compare
        equal and the invariant holds — an unanchored bullet must still yield."""
        bullets = ["La dirección del remitente", "Algo que no se dice en ninguna parte"]
        kept = _clean_bullets(bullets, self.NARRATION_ES, 2, language=Language.ES)
        assert "La dirección del remitente" in kept
        assert "Algo que no se dice en ninguna parte" not in kept
        for bullet in kept:
            assert anchor_position(bullet, self.NARRATION_ES) is not None

    # -------------------------------------------------------------- parse path

    @pytest.mark.parametrize("language", list(Language))
    def test_the_visual_prompts_pass_through_untranslated(self, monkeypatch, language):
        """Whatever the model returns for these two fields is carried verbatim — this
        provider never rewrites them, in any language."""
        scene = _scene(1)
        scene["image_prompt"] = "A photograph of a laptop. No text anywhere in the image."
        scene["clip_prompt"] = "Slow push in on a laptop screen."
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([scene]),
        )
        result = GeminiScriptProvider(api_key="x").generate(
            "phishing", 1, language=language
        ).scenes[0]
        assert result.image_prompt == scene["image_prompt"]
        assert scene_clip_prompt(result) == scene["clip_prompt"]

    @pytest.mark.parametrize(
        ("language", "narration", "heading"),
        [
            (Language.ES, NARRATION_ES, "Señales de alerta"),
            (Language.HI, NARRATION_HI, "संदेश की जांच करें।"),
        ],
    )
    def test_in_language_narration_survives_the_parse_unchanged(
        self, monkeypatch, language, narration, heading
    ):
        scene = _scene(1, narration=narration)
        scene["heading"] = heading
        monkeypatch.setattr(
            "app.providers.gemini_script.generate_content",
            lambda *a, **k: _script_response([scene]),
        )
        result = GeminiScriptProvider(api_key="x").generate(
            "phishing", 1, language=language
        ).scenes[0]
        assert result.narration == narration
        assert result.heading == heading.rstrip(DANDA)
        assert result.bullets, "a scene must still come back with on-screen points"

    # -------------------------------------------------------------- verbatim provider

    def test_the_verbatim_provider_never_translates_a_word(self):
        """`language` names what script the text is already in; it cannot change the text,
        because every narration word is the user's own."""
        script = VerbatimScriptProvider(VERBATIM_TEXT).generate(
            "bridges", 3, language=Language.ES
        )
        assert " ".join(s.narration for s in script.scenes).split() == VERBATIM_TEXT.split()

    def test_a_pasted_hindi_script_is_split_and_cleaned_as_hindi(self):
        script = VerbatimScriptProvider(self.NARRATION_HI).generate(
            "phishing email", 3, language=Language.HI
        )
        assert len(script.scenes) == 3
        joined = " ".join(s.narration for s in script.scenes)
        assert joined.split() == self.NARRATION_HI.split()
        for scene in script.scenes:
            assert scene.heading.strip()
            assert not scene.heading.endswith(DANDA)
            # Devanagari headings are not put through an English casing rule, so every word
            # is the source's own, untouched.
            for word in scene.heading.split():
                assert word in self.NARRATION_HI, word

    def test_the_verbatim_visual_prompt_never_carries_devanagari_from_the_narration(self):
        """There is no LLM on this path and so no translator. The English boilerplate stays
        English and the Devanagari NARRATION is kept out of it entirely — Devanagari keywords
        in an English-conditioned image prompt are noise, not subject matter.

        The topic is a different matter: it is the caller's own string and is passed through
        as written, because inventing a translation of it would be worse than quoting it.
        """
        scene = VerbatimScriptProvider(self.NARRATION_HI).generate(
            "phishing email", 2, language=Language.HI
        ).scenes[0]
        assert not re.search(r"[ऀ-ॿ]", scene.image_prompt)
        assert not re.search(r"[ऀ-ॿ]", scene_clip_prompt(scene) or "")
        assert "phishing email" in scene.image_prompt
        assert "No text, no letters" in scene.image_prompt


@live_only
class TestLiveScriptLanguage:
    """Real Gemini, all three languages. The numbers in the report come from here."""

    TOPIC = "spotting phishing email at work"
    DEVANAGARI = re.compile(r"[ऀ-ॿ]")

    def _words(self, narration: str, wpm: float) -> list[Word]:
        """Whole-word tokens at a uniform cadence — the token shape an aligner returns.
        `find_anchors` classifies from the strings, so the cadence cannot change the verdict.
        """
        step = 60.0 / wpm
        return [
            Word(word=tok, start=i * step, end=(i + 1) * step - 0.01)
            for i, tok in enumerate(narration.split())
        ]

    @pytest.mark.parametrize("language", list(Language))
    def test_a_live_script_is_in_language_paced_and_anchored(self, language):
        script = GeminiScriptProvider().generate(self.TOPIC, 4, language=language)
        assert len(script.scenes) == 4

        tally = {"ngram": 0, "fuzzy": 0, "proportional": 0}
        for scene in script.scenes:
            role = scene_role(scene)
            low, _, high = narration_words(role, language)
            count = len(scene.narration.split())
            seconds = count / language_wpm(language) * 60
            floor, ceiling = role.target_duration

            # The script is in the right script.
            expected_devanagari = language is Language.HI
            assert bool(self.DEVANAGARI.search(scene.narration)) is expected_devanagari
            assert bool(self.DEVANAGARI.search(scene.heading)) is expected_devanagari

            # The visual prompts are NOT.
            assert not self.DEVANAGARI.search(scene.image_prompt)
            assert not self.DEVANAGARI.search(scene_clip_prompt(scene) or "")

            # Pacing: the word budget is a duration, so the duration is what is checked.
            assert low - 2 <= count <= high + 2, (language, role, count, low, high)
            assert floor <= seconds <= ceiling, (language, role, seconds, floor, ceiling)

            if language is Language.HI:
                assert scene.narration.rstrip().endswith((DANDA, "?", "!"))
                assert not scene.heading.endswith(DANDA)

            for anchor in find_anchors(
                scene.bullets,
                self._words(scene.narration, language_wpm(language)),
                0.0,
                seconds,
            ):
                tally[anchor.method] += 1

        total = sum(tally.values())
        assert total > 0
        anchored = tally["ngram"] + tally["fuzzy"]
        if anchoring_supported(language):
            # The invariant: every bullet quotes its own narration verbatim.
            assert tally["proportional"] == 0, (language, tally)
            assert tally["ngram"] == total, (language, tally)
        else:
            # Hindi. Documented, measured, and NOT papered over: nothing anchors, because
            # `normalize` cannot see Devanagari at all.
            assert anchored == 0, (
                f"{language} unexpectedly anchored {anchored}/{total} — if `normalize` was "
                f"widened, delete the Hindi fallback branches in gemini_script"
            )

    @pytest.mark.parametrize("language", [Language.ES, Language.HI])
    def test_a_live_bullet_is_a_verbatim_run_of_its_own_narration(self, language):
        """The prompt-side half of the anchor invariant, and the half this file owns.

        It holds even for Hindi, where the MATCHER cannot confirm it — which is precisely
        why Hindi's 0/N anchoring is a normaliser limitation rather than a copy problem.
        """
        script = GeminiScriptProvider().generate(self.TOPIC, 4, language=language)
        total = verbatim = 0
        for scene in script.scenes:
            narration = re.sub(r"\s+", " ", scene.narration).lower()
            for bullet in scene.bullets:
                total += 1
                if re.sub(r"\s+", " ", bullet).lower() in narration:
                    verbatim += 1
        assert total > 0
        assert verbatim / total >= 0.75, f"{language}: only {verbatim}/{total} verbatim"

    def test_spanish_comes_back_with_real_spanish_orthography(self):
        """Accent-stripped Spanish reads as machine output, and it was what the model
        produced before the conventions clause existed."""
        script = GeminiScriptProvider().generate(self.TOPIC, 5, language=Language.ES)
        body = " ".join(
            [script.title]
            + [f"{s.heading} {s.narration} {' '.join(s.bullets)}" for s in script.scenes]
        )
        assert re.search(r"[áéíóúñÁÉÍÓÚÑ]", body), f"no Spanish diacritics at all: {body[:400]}"

    def test_hindi_uses_the_danda_and_not_the_full_stop(self):
        script = GeminiScriptProvider().generate(self.TOPIC, 5, language=Language.HI)
        for scene in script.scenes:
            assert DANDA in scene.narration, scene.narration
            # A stray Latin full stop mid-narration is the English-typography defect.
            assert not re.search(r"[ऀ-ॿ]\s*\.", scene.narration), scene.narration
