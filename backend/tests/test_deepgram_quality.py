"""Pins the quality/expressiveness levers documented in `app.providers.deepgram_tts`:
`sample_rate` default and validation, `speed` default/validation/override, the
`emphasize` phrase-splice mechanism, and the fallback when a phrase can't be located.

The empirical basis (live key, STT round-trips, click-safety calibration) is recorded in
the module docstring, points 1-2 and 7-9. Everything here mocks `httpx.post`; the live
probes that produced those numbers are not part of the suite.
"""

from __future__ import annotations

import pytest

from app.providers.deepgram_tts import (
    EMPHASIZE_SPEED_DEFAULT,
    EMPHASIZE_SPEED_DEFAULT_ES,
    SPEED_RANGE,
    SPEED_RANGE_ES,
    VALID_SAMPLE_RATES,
    DeepgramSynthesizer,
    _emphasis_plan,
    _terminate,
)

WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 20


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": "audio/wav"}
        self.text = ""


@pytest.fixture
def captured(monkeypatch):
    """Collect the (params, json) of every request that would have gone to Deepgram."""
    sent: list[dict] = []

    def fake_post(url, **kwargs):
        sent.append({"params": kwargs.get("params") or {}, "json": kwargs.get("json") or {}})
        return _FakeResponse(WAV)

    monkeypatch.setattr("httpx.post", fake_post)
    return sent


def synth(**kwargs) -> DeepgramSynthesizer:
    return DeepgramSynthesizer(api_key="test-key", **kwargs)


# ============================================================ sample_rate


class TestSampleRateDefaultsAndValidation:
    def test_default_is_48000(self):
        """See module docstring point 1: matches the final mux's AUDIO_RATE, no resample."""
        assert synth().sample_rate == 48000

    def test_explicit_override_is_honoured(self):
        assert synth(sample_rate=24000).sample_rate == 24000

    def test_every_documented_rate_is_accepted(self):
        for rate in VALID_SAMPLE_RATES:
            assert synth(sample_rate=rate).sample_rate == rate

    def test_an_undocumented_rate_is_rejected_before_any_request(self):
        with pytest.raises(ValueError, match="sample_rate"):
            synth(sample_rate=44100)

    def test_settings_default_is_read_when_present(self, monkeypatch):
        """`Settings` doesn't have `video_deepgram_sample_rate` yet — the field this
        module wants added, see the report. Read via `getattr` with a fallback so this
        module works whether or not the field exists; a bare namespace stands in for a
        `Settings` instance that does have it."""
        from types import SimpleNamespace

        from app.core.config import get_settings as real_get_settings

        fake = SimpleNamespace(**vars(real_get_settings()))
        fake.video_deepgram_sample_rate = 32000
        monkeypatch.setattr("app.providers.deepgram_tts.get_settings", lambda: fake)
        assert synth().sample_rate == 32000

    def test_sample_rate_is_sent_on_the_wire(self, captured, tmp_path):
        synth(sample_rate=16000).synthesize("Go.", "v", tmp_path / "a.wav")
        assert captured[0]["params"]["sample_rate"] == "16000"


# ============================================================ speed


class TestSpeedDefaultsAndValidation:
    def test_default_is_point_nine(self):
        """See module docstring point 2: 1.0 overshoots the 135 wpm target by ~22%."""
        assert synth().speed == 0.9

    def test_explicit_override_is_honoured(self):
        assert synth(speed=1.2).speed == 1.2

    @pytest.mark.parametrize("speed", [0.7, 0.85, 1.0, 1.5])
    def test_documented_range_is_accepted_for_english(self, speed, captured, tmp_path):
        synth(speed=speed).synthesize("Go.", "aura-2-jupiter-en", tmp_path / "a.wav")
        assert captured[0]["params"]["speed"] == str(speed)

    @pytest.mark.parametrize("speed", [0.69, 1.51, 2.0, 0.0, -1.0])
    def test_out_of_range_speed_is_rejected_before_any_request(self, speed, captured, tmp_path):
        with pytest.raises(ValueError, match="speed"):
            synth(speed=speed).synthesize("Go.", "aura-2-jupiter-en", tmp_path / "a.wav")
        assert captured == []

    def test_spanish_voice_has_a_tighter_floor(self, captured, tmp_path):
        """Docs: below 0.9 on a Spanish voice measurably introduces disfluencies."""
        with pytest.raises(ValueError, match="speed"):
            synth(speed=0.8).synthesize("Hola.", "aura-2-celeste-es", tmp_path / "a.wav")
        assert captured == []

    def test_spanish_voice_accepts_point_nine(self, captured, tmp_path):
        synth(speed=0.9).synthesize("Hola.", "aura-2-celeste-es", tmp_path / "a.wav")
        assert captured[0]["params"]["speed"] == "0.9"

    def test_per_call_speed_overrides_the_instance_default(self, captured, tmp_path):
        """Lets a caller pace a title card differently from a content scene without a
        second synthesizer instance."""
        s = synth(speed=0.9)
        s.synthesize("Go.", "aura-2-jupiter-en", tmp_path / "a.wav", speed=1.1)
        assert captured[0]["params"]["speed"] == "1.1"
        assert s.speed == 0.9  # the instance default is untouched

    def test_settings_default_is_read_when_present(self, monkeypatch):
        """`Settings` doesn't have `video_deepgram_speed` yet — the field this module
        wants added, see the report. See the `sample_rate` test above for why a
        `SimpleNamespace` stands in rather than mutating the real `Settings` instance."""
        from types import SimpleNamespace

        from app.core.config import get_settings as real_get_settings

        fake = SimpleNamespace(**vars(real_get_settings()))
        fake.video_deepgram_speed = 0.95
        monkeypatch.setattr("app.providers.deepgram_tts.get_settings", lambda: fake)
        assert synth().speed == 0.95

    def test_documented_ranges_match_the_docs(self):
        assert SPEED_RANGE == (0.7, 1.5)
        assert SPEED_RANGE_ES == (0.9, 1.5)


# ============================================================ _terminate


class TestTerminate:
    @pytest.mark.parametrize(
        ("fragment", "expected"),
        [
            ("Check the", "Check the."),
            ("Sender address", "Sender address."),
            ("Already done.", "Already done."),
            ("Really?!", "Really?!"),
            ("Wait!", "Wait!"),
        ],
    )
    def test_adds_a_period_only_when_missing(self, fragment, expected):
        assert _terminate(fragment) == expected


# ============================================================ _emphasis_plan


class TestEmphasisPlan:
    def test_splits_into_before_phrase_after(self):
        plan = _emphasis_plan(
            "Check the sender address before you click.", "sender address"
        )
        assert plan == [
            ("Check the.", False),
            ("sender address.", True),
            ("before you click.", False),
        ]

    def test_case_insensitive_whole_word_match(self):
        plan = _emphasis_plan("Enable MFA on your account.", "mfa")
        assert plan is not None
        assert plan[1] == ("MFA.", True)

    def test_phrase_at_the_start_has_no_before_fragment(self):
        """Original casing is preserved — `phrase` is only a case-insensitive locator,
        not the text that gets spoken."""
        plan = _emphasis_plan("Sender address matters most.", "sender address")
        assert plan[0] == ("Sender address.", True)
        assert all(not is_phrase for _, is_phrase in plan[1:])

    def test_phrase_at_the_end_has_no_after_fragment(self):
        """The sentence's own trailing period lands after the phrase match but carries
        no words, so it must not become a spurious punctuation-only fragment."""
        plan = _emphasis_plan("Always check the sender address.", "sender address")
        assert plan[-1] == ("sender address.", True)

    def test_phrase_is_the_whole_text(self):
        plan = _emphasis_plan("Sender address.", "sender address")
        assert plan == [("Sender address.", True)]

    def test_partial_word_is_not_matched(self):
        """'senders' must not match a request for 'sender' — whole words only."""
        assert _emphasis_plan("Check the senders list.", "sender") is None

    def test_phrase_not_present_returns_none(self):
        assert _emphasis_plan("Check the sender address.", "totally different") is None

    def test_empty_phrase_returns_none(self):
        assert _emphasis_plan("Check the sender address.", "") is None
        assert _emphasis_plan("Check the sender address.", "   ") is None

    def test_multi_word_phrase_tolerates_normalised_whitespace(self):
        plan = _emphasis_plan("Check the  sender   address now.", "sender address")
        assert plan is not None
        assert plan[1][1] is True


# ============================================================ synthesize(..., emphasize=)


@pytest.fixture
def no_real_ffmpeg(monkeypatch):
    """The shared `WAV` fixture is a bare RIFF magic, not a decodable file — real ffmpeg
    concat needs a valid header. Every other multi-piece test in this suite (see
    `test_deepgram_ssml.py::test_chunking_still_applies_after_stripping`) stubs
    `_concat_wavs` for the same reason; the wire-level `captured` assertions are what
    actually pin the behaviour here, not the on-disk bytes."""
    monkeypatch.setattr(
        "app.providers.deepgram_tts._concat_wavs",
        lambda pieces, out: out.write_bytes(WAV),
    )


class TestSynthesizeEmphasize:
    def test_emphasized_phrase_is_sent_at_the_slower_default_speed(
        self, captured, no_real_ffmpeg, tmp_path
    ):
        synth().synthesize(
            "Check the sender address before you click.",
            "aura-2-jupiter-en",
            tmp_path / "a.wav",
            emphasize="sender address",
        )
        assert len(captured) == 3
        speeds = [r["params"]["speed"] for r in captured]
        texts = [r["json"]["text"] for r in captured]
        assert texts == ["Check the.", "sender address.", "before you click."]
        assert speeds == [str(0.9), str(EMPHASIZE_SPEED_DEFAULT), str(0.9)]

    def test_emphasize_speed_override_is_honoured(self, captured, no_real_ffmpeg, tmp_path):
        synth().synthesize(
            "Check the sender address before you click.",
            "aura-2-jupiter-en",
            tmp_path / "a.wav",
            emphasize="sender address",
            emphasize_speed=0.8,
        )
        assert captured[1]["params"]["speed"] == "0.8"

    def test_spanish_voice_gets_the_higher_emphasize_default(
        self, captured, no_real_ffmpeg, tmp_path
    ):
        synth().synthesize(
            "Revisa la dirección del remitente antes de hacer clic.",
            "aura-2-celeste-es",
            tmp_path / "a.wav",
            emphasize="remitente",
        )
        phrase_call = next(c for c in captured if c["json"]["text"].lower().startswith("remitente"))
        assert phrase_call["params"]["speed"] == str(EMPHASIZE_SPEED_DEFAULT_ES)

    def test_out_of_range_emphasize_speed_is_rejected_before_any_request(
        self, captured, tmp_path
    ):
        with pytest.raises(ValueError, match="speed"):
            synth().synthesize(
                "Check the sender address before you click.",
                "aura-2-jupiter-en",
                tmp_path / "a.wav",
                emphasize="sender address",
                emphasize_speed=2.0,
            )
        assert captured == []

    def test_phrase_not_found_degrades_to_plain_single_speed_synthesis(
        self, captured, tmp_path, caplog
    ):
        import logging

        with caplog.at_level(logging.WARNING, logger="app.providers.deepgram_tts"):
            synth().synthesize(
                "Check the sender address before you click.",
                "aura-2-jupiter-en",
                tmp_path / "a.wav",
                emphasize="not in the sentence anywhere",
            )
        assert len(captured) == 1
        assert captured[0]["json"]["text"] == "Check the sender address before you click."
        assert captured[0]["params"]["speed"] == "0.9"
        assert any("not found" in r.message for r in caplog.records)

    def test_emphasize_on_text_that_needs_chunking_degrades_and_warns(
        self, captured, tmp_path, monkeypatch, caplog
    ):
        """Phrase-splicing only supports single-chunk text; combining it with the
        multi-request chunk path is not implemented (module docstring point 8)."""
        import logging

        monkeypatch.setattr("app.providers.deepgram_tts.MAX_CHARS", 40)
        monkeypatch.setattr(
            "app.providers.deepgram_tts._concat_wavs",
            lambda pieces, out: out.write_bytes(WAV),
        )
        sentence = "Check the sender address carefully every time. "
        with caplog.at_level(logging.WARNING, logger="app.providers.deepgram_tts"):
            synth().synthesize(
                sentence * 3,
                "aura-2-jupiter-en",
                tmp_path / "a.wav",
                emphasize="sender address",
            )
        assert len(captured) > 1
        assert any("only supports single-chunk" in r.message for r in caplog.records)

    def test_default_call_shape_is_unaffected_by_the_new_kwargs(self, captured, tmp_path):
        """Every existing caller does `synthesize(text, voice, out_path)` with no kwargs —
        confirms that call shape still produces exactly one plain request."""
        synth().synthesize("Check the sender address.", "aura-2-jupiter-en", tmp_path / "a.wav")
        assert len(captured) == 1
        assert captured[0]["json"]["text"] == "Check the sender address."

    def test_emphasized_audio_round_trips_to_a_real_file(
        self, captured, no_real_ffmpeg, tmp_path
    ):
        out = tmp_path / "a.wav"
        synth().synthesize(
            "Check the sender address before you click.",
            "aura-2-jupiter-en",
            out,
            emphasize="sender address",
        )
        assert out.read_bytes().startswith(b"RIFF")
