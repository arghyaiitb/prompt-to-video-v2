"""Deepgram Aura does not support SSML. These tests pin that down so it stops being
re-litigated, and pin the defensive strip that keeps markup out of the voice model.

The empirical basis (live key, audio round-tripped through `/v1/listen?model=nova-3`) is
recorded in the `app.providers.deepgram_tts` module docstring together with the Deepgram
statement and the two API errors that prove there is no SSML request shape to opt into.

Everything here mocks `httpx.post`; the live probes are not part of the suite.
"""

from __future__ import annotations

import logging
import re

import pytest

from app.core.ports import SpeechSynthesizer
from app.providers.deepgram_tts import DeepgramSynthesizer, strip_markup

# Minimal valid wav: the synthesizer only asserts the RIFF magic before writing.
WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 20


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": "audio/wav"}
        self.text = ""


@pytest.fixture
def captured(monkeypatch):
    """Collect the request bodies that would have gone to Deepgram."""
    sent: list[dict] = []

    def fake_post(url, **kwargs):
        sent.append(kwargs.get("json") or {})
        return _FakeResponse(WAV)

    monkeypatch.setattr("httpx.post", fake_post)
    return sent


def synth() -> DeepgramSynthesizer:
    return DeepgramSynthesizer(api_key="test-key")


# ============================================================ the declared capability


class TestCapabilityFlag:
    def test_supports_ssml_is_false(self):
        """Aura vocalises markup. See the module docstring for the doc citation."""
        assert DeepgramSynthesizer.supports_ssml is False
        assert synth().supports_ssml is False

    def test_still_satisfies_the_speech_synthesizer_protocol(self):
        """`supports_ssml` is part of the Protocol, so omitting it breaks `isinstance`."""
        assert isinstance(synth(), SpeechSynthesizer)

    def test_the_flag_is_readable_without_constructing_the_provider(self):
        """`app/worker/factory.py` advertises engine capability before instantiating."""
        assert getattr(DeepgramSynthesizer, "supports_ssml", None) is False


# ============================================================ the strip


class TestStripMarkupLeavesPlainTextAlone:
    @pytest.mark.parametrize(
        "text",
        [
            "Check the sender. Then hover the link.",
            "Report anything suspicious to the security team.",
            "Latency < 200ms is the target, and 5 < 6 always.",
            "Profit & loss, R&D spend, AT&T invoices.",
            "Use the arrow -> to continue.",
            "She said \"don't click it\" and she was right.",
        ],
    )
    def test_plain_text_is_returned_unchanged(self, text):
        """A bare `<` or `&` is prose, not markup — the strip must not touch it."""
        assert strip_markup(text) == (text, False)

    def test_a_bare_less_than_is_not_treated_as_a_tag(self):
        """Tag names must start with a letter; this is what protects arithmetic."""
        assert strip_markup("if x < y then stop")[1] is False


class TestStripMarkupRemovesSSML:
    def test_speak_wrapper_is_removed(self):
        plain, had = strip_markup("<speak>Check the sender.</speak>")
        assert had is True
        assert plain == "Check the sender."

    def test_break_becomes_whitespace_not_a_welded_word(self):
        """The measured failure: deleting the tag would produce 'carefullybefore'."""
        plain, had = strip_markup(
            '<speak>Verify the domain carefully<break time="1s"/>'
            "before you approve.</speak>"
        )
        assert had is True
        assert plain == "Verify the domain carefully before you approve."
        assert "carefullybefore" not in plain

    @pytest.mark.parametrize(
        ("ssml", "expected"),
        [
            ('<emphasis level="strong">domain</emphasis>', "domain"),
            ('<prosody rate="slow">Verify the domain.</prosody>', "Verify the domain."),
            ('<say-as interpret-as="characters">ABC</say-as>', "ABC"),
            ("<p>One.</p><p>Two.</p>", "One. Two."),
            ("<s>One.</s><s>Two.</s>", "One. Two."),
            ('<phoneme alphabet="ipa" ph="təmɑːto">tomato</phoneme>', "tomato"),
            ('<lang xml:lang="fr-FR">bonjour</lang>', "bonjour"),
            ('<mark name="here"/>Go.', "Go."),
            ('<amazon:effect name="whispered">quietly</amazon:effect>', "quietly"),
            ('<w role="amazon:VB">read</w>', "read"),
        ],
    )
    def test_every_tag_family_is_reduced_to_its_words(self, ssml, expected):
        assert strip_markup(f"<speak>{ssml}</speak>")[0] == expected

    def test_sub_keeps_the_written_form_not_the_alias(self):
        """The reference text the aligner sees contains 'Inc', so 'Inc' must be spoken."""
        plain, _ = strip_markup('<speak>Acme <sub alias="Incorporated">Inc</sub>.</speak>')
        assert plain == "Acme Inc."
        assert "Incorporated" not in plain

    def test_xml_comments_are_removed(self):
        assert strip_markup("<speak>Go.<!-- a note --> Stop.</speak>")[0] == "Go. Stop."

    def test_entities_are_unescaped_once_markup_is_gone(self):
        """Otherwise the narrator says 'amp' and 'lt' out loud."""
        plain, _ = strip_markup("<speak>R&amp;D spend &lt; budget &quot;target&quot;</speak>")
        assert plain == 'R&D spend < budget "target"'

    def test_amp_is_unescaped_last_so_nested_escapes_survive(self):
        assert strip_markup("<speak>&amp;lt; stays literal</speak>")[0] == "&lt; stays literal"

    def test_space_before_punctuation_is_tidied(self):
        """A tag sitting between a word and its period must not strand a space."""
        plain, _ = strip_markup('<speak>Check the sender<break time="500ms"/>.</speak>')
        assert plain == "Check the sender."

    def test_no_tag_name_survives_anywhere_in_the_output(self):
        ssml = (
            '<speak>Check the sender.<break time="800ms"/>'
            '<emphasis level="strong">Then</emphasis> '
            '<prosody rate="90%">hover the link.</prosody></speak>'
        )
        plain, had = strip_markup(ssml)
        assert had is True
        assert not re.search(
            r"\b(speak|break|emphasis|prosody|say-as|time|level|rate)\b", plain, re.I
        )
        assert "<" not in plain and ">" not in plain

    def test_the_words_and_their_order_are_preserved(self):
        """The invariant `bullet_timing` depends on: same words, same order."""
        reference = "Check the sender. Then hover the link."
        ssml = (
            '<speak>Check the sender.<break time="800ms"/>'
            '<emphasis level="strong">Then</emphasis> hover the link.</speak>'
        )
        assert strip_markup(ssml)[0].split() == reference.split()


# ============================================================ synthesize() integration


class TestSynthesizeNeverSendsMarkup:
    def test_ssml_is_stripped_before_the_request(self, captured, tmp_path):
        ssml = '<speak>Check the sender.<break time="1s"/>Then hover the link.</speak>'
        synth().synthesize(ssml, "aura-2-draco-en", tmp_path / "a.wav")
        assert captured[0]["text"] == "Check the sender. Then hover the link."

    def test_no_angle_bracket_ever_reaches_deepgram(self, captured, tmp_path):
        synth().synthesize(
            '<speak><prosody rate="slow">Verify the domain.</prosody></speak>',
            "aura-2-draco-en",
            tmp_path / "a.wav",
        )
        body = captured[0]["text"]
        assert "<" not in body and ">" not in body

    def test_plain_text_reaches_deepgram_untouched(self, captured, tmp_path):
        """The common path must be a byte-for-byte passthrough."""
        text = "Check the sender. Then hover the link."
        synth().synthesize(text, "aura-2-draco-en", tmp_path / "a.wav")
        assert captured[0]["text"] == text

    def test_stripping_is_logged_as_a_warning(self, captured, tmp_path, caplog):
        """Silent repair would hide the caller bug that produced the markup."""
        with caplog.at_level(logging.WARNING, logger="app.providers.deepgram_tts"):
            synth().synthesize(
                '<speak>Check the sender.</speak>', "aura-2-draco-en", tmp_path / "a.wav"
            )
        assert any("stripped markup" in r.message for r in caplog.records)

    def test_plain_text_logs_no_warning(self, captured, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="app.providers.deepgram_tts"):
            synth().synthesize("Check the sender.", "aura-2-draco-en", tmp_path / "a.wav")
        assert not caplog.records

    def test_markup_only_input_raises_instead_of_synthesizing_silence(self, captured, tmp_path):
        with pytest.raises(ValueError, match="only markup"):
            synth().synthesize("<speak><break time='1s'/></speak>", "v", tmp_path / "a.wav")
        assert captured == []

    def test_the_audio_is_still_written(self, captured, tmp_path):
        out = tmp_path / "a.wav"
        assert synth().synthesize("<speak>Go.</speak>", "v", out) == out
        assert out.read_bytes().startswith(b"RIFF")

    def test_chunking_still_applies_after_stripping(self, captured, tmp_path, monkeypatch):
        """Stripping happens first, so the chunk budget counts spoken words not tags."""
        monkeypatch.setattr("app.providers.deepgram_tts.MAX_CHARS", 40)
        monkeypatch.setattr(
            "app.providers.deepgram_tts._concat_wavs",
            lambda pieces, out: out.write_bytes(WAV),
        )
        sentence = "Check the sender carefully every time. "
        synth().synthesize(f"<speak>{sentence * 3}</speak>", "v", tmp_path / "a.wav")
        assert len(captured) > 1
        assert all("<" not in body["text"] for body in captured)
