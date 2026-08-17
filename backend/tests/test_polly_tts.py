"""Tests for the Amazon Polly synthesizer.

Two tiers, matching `tests/test_providers.py`:

  * Everything outside `TestLiveAws` runs offline. boto3 is stubbed at the client seam
    (`unittest.mock`, no moto), so request shaping, SSML adaptation, credential precedence
    and error translation are covered without spending a character.
  * `TestLiveAws` hits the real API and is skipped unless RUN_LIVE_AWS=1:

        uv run pytest tests/test_polly_tts.py                    # offline only
        RUN_LIVE_AWS=1 uv run pytest tests/test_polly_tts.py     # everything

    `-k "not live"` deselects it, so no offline test name here may contain the substring
    "live" — which rules out the word "delivery".

Artifacts go to pytest's tmp_path, never into the repo.
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.ports import SpeechSynthesizer
from app.providers import polly_tts
from app.providers.polly_tts import (
    DEFAULT_ENGINE,
    EMPHASIS_TO_PROSODY,
    ENGINE_PREFERENCE,
    MAX_BILLED_CHARS,
    PollyCredentialsError,
    PollyError,
    PollyRegionError,
    PollySynthesizer,
    PollyThrottledError,
    PollyVoiceError,
    SsmlError,
    TextTooLongError,
    adapt_ssml,
    best_engine,
    billed_chars,
    is_ssml,
    list_voices,
    pcm_to_wav,
    resolve_credentials,
    shape_voice,
    validate_ssml,
)

LIVE = os.environ.get("RUN_LIVE_AWS") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_AWS=1 to hit real AWS")

AWS_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
)


# ------------------------------------------------------------------ stub plumbing


def silence(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """Raw signed-16 mono pcm, the exact shape Polly's `pcm` OutputFormat returns."""
    return struct.pack(f"<{int(rate * seconds)}h", *([0] * int(rate * seconds)))


def fake_client(audio: bytes | None = None, *, voices: list[dict] | None = None) -> MagicMock:
    """A stand-in for `boto3.client("polly")` recording the request it was handed."""
    client = MagicMock()
    stream = MagicMock()
    stream.read.return_value = audio if audio is not None else silence()
    client.synthesize_speech.return_value = {"AudioStream": stream, "ContentType": "audio/pcm"}
    client.describe_voices.return_value = {"Voices": voices or []}
    return client


def client_error(code: str, message: str = "nope", status: int = 400) -> Exception:
    """A botocore ClientError with the real shape, without importing botocore in a fixture."""
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "SynthesizeSpeech",
    )


VOICE_ENTRIES = [
    {
        "Id": "Matthew",
        "Name": "Matthew",
        "Gender": "Male",
        "LanguageCode": "en-US",
        "LanguageName": "US English",
        "SupportedEngines": ["neural", "standard", "generative"],
    },
    {
        "Id": "Danielle",
        "Name": "Danielle",
        "Gender": "Female",
        "LanguageCode": "en-US",
        "LanguageName": "US English",
        "SupportedEngines": ["generative", "long-form", "neural"],
    },
    {
        "Id": "Gregory",
        "Name": "Gregory",
        "Gender": "Male",
        "LanguageCode": "en-US",
        "LanguageName": "US English",
        "SupportedEngines": ["long-form", "neural"],
    },
    {
        "Id": "Amy",
        "Name": "Amy",
        "Gender": "Female",
        "LanguageCode": "en-GB",
        "LanguageName": "British English",
        "SupportedEngines": ["neural", "standard"],
    },
]


class BlankSettings:
    """`Settings` with every credential empty, so an offline test cannot pass because the
    developer happens to have working keys in `.env`."""

    aws_access_key_id = ""
    aws_secret_access_key = ""
    aws_session_token = ""
    aws_region = ""
    video_default_polly_voice = "Matthew"
    video_polly_engine = "generative"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No ambient AWS config, no cached catalogue, and no real sleeping."""
    for key in AWS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Three separate leaks to plug: the shell, `Settings` (which now carries the aws_*
    # fields and reads the real .env), and this module's own .env fallback.
    monkeypatch.setattr(polly_tts, "get_settings", lambda: BlankSettings())
    monkeypatch.setattr(polly_tts, "REPO_ROOT", Path("/nonexistent-for-tests"))
    monkeypatch.setattr(polly_tts.time, "sleep", lambda _s: None)
    polly_tts.reset_voice_cache()
    yield
    polly_tts.reset_voice_cache()


def synth(client: MagicMock | None = None, **kwargs) -> PollySynthesizer:
    kwargs.setdefault("voice", "Matthew")
    return PollySynthesizer(client=client or fake_client(), **kwargs)


def sent(client: MagicMock) -> dict:
    return client.synthesize_speech.call_args.kwargs


# =========================================================== Protocol conformance


class TestProtocol:
    def test_it_satisfies_the_speech_synthesizer_port(self):
        assert isinstance(synth(), SpeechSynthesizer)

    def test_supports_ssml_is_true(self):
        """Polly is the reason the flag exists — Aura vocalises tags, Polly parses them."""
        assert PollySynthesizer.supports_ssml is True
        assert synth().supports_ssml is True

    def test_it_mirrors_the_deepgram_synthesizer_surface(self):
        """Interchangeable at every call site, so swapping engines is a config change."""
        provider = synth()
        for name in ("synthesize", "duration", "synthesize_with_duration"):
            assert callable(getattr(provider, name)), name

    def test_the_worker_factory_can_construct_it(self):
        """`factory._construct` passes only `settings`/`api_key`/`model`/`voice`, so every
        other constructor argument must have a default."""
        import inspect

        params = inspect.signature(PollySynthesizer).parameters
        required = [
            name
            for name, p in params.items()
            if p.default is inspect.Parameter.empty and name != "self"
        ]
        assert required == [], f"factory cannot supply {required}"
        assert "voice" in params and "settings" in params


# =========================================================== text vs SSML


class TestTextType:
    def test_plain_text_is_sent_as_text(self):
        client = fake_client()
        synth(client).synthesize("Check the sender address.", "Matthew", Path("/dev/null"))
        assert sent(client)["TextType"] == "text"

    def test_a_speak_root_is_sent_as_ssml(self, tmp_path):
        client = fake_client()
        synth(client).synthesize(
            '<speak>Check the sender.<break time="800ms"/>Then hover.</speak>',
            "Matthew",
            tmp_path / "a.wav",
        )
        assert sent(client)["TextType"] == "ssml"

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ("<speak>hi</speak>", True),
            ("  \n <speak>hi</speak>", True),
            ("<SPEAK>hi</SPEAK>", True),
            ("<speaker>hi</speaker>", False),
            ("Check the <b>sender</b>.", False),
            ("", False),
        ],
    )
    def test_ssml_detection(self, payload, expected):
        assert is_ssml(payload) is expected

    def test_empty_text_is_refused_before_any_call(self):
        client = fake_client()
        with pytest.raises(ValueError, match="empty"):
            synth(client).synthesize("   ", "Matthew", Path("/dev/null"))
        client.synthesize_speech.assert_not_called()

    def test_the_engine_and_voice_reach_the_request(self, tmp_path):
        client = fake_client()
        synth(client, engine="neural").synthesize("Hello.", "Danielle", tmp_path / "a.wav")
        assert sent(client)["Engine"] == "neural"
        assert sent(client)["VoiceId"] == "Danielle"


# =========================================================== SSML validation


class TestSsmlValidation:
    def test_malformed_ssml_is_rejected_before_the_api_call(self, tmp_path):
        client = fake_client()
        with pytest.raises(SsmlError, match="malformed SSML"):
            synth(client).synthesize("<speak>Tom & Jerry</speak>", "Matthew", tmp_path / "a.wav")
        client.synthesize_speech.assert_not_called()

    def test_an_unclosed_tag_is_rejected_before_the_api_call(self, tmp_path):
        client = fake_client()
        with pytest.raises(SsmlError, match="malformed SSML"):
            synth(client).synthesize(
                "<speak>Check the <prosody rate='90%'>sender.</speak>",
                "Matthew",
                tmp_path / "a.wav",
            )
        client.synthesize_speech.assert_not_called()

    def test_the_error_names_the_offending_markup(self):
        with pytest.raises(SsmlError) as exc:
            validate_ssml("<speak>Ben & Jerry are here</speak>")
        text = str(exc.value)
        assert "line 1" in text and "column" in text
        assert "&amp;" in text, "the message should say how to fix it"

    def test_the_amazon_prefix_does_not_read_as_malformed(self):
        """Polly SSML never declares xmlns:amazon, which ElementTree calls an unbound
        prefix. It must not be mistaken for the caller's mistake."""
        validate_ssml('<speak><amazon:effect name="drc">Hi.</amazon:effect></speak>', "neural")

    def test_a_non_speak_root_is_rejected(self):
        with pytest.raises(SsmlError, match="<speak> root"):
            validate_ssml("<prosody rate='90%'>hi</prosody>")

    def test_a_wrong_root_element_is_named(self):
        with pytest.raises(SsmlError, match="root must be <speak>"):
            validate_ssml("<speak-x>hi</speak-x>")

    @pytest.mark.parametrize("tag", ["audio", "lexicon", "lookup", "voice"])
    def test_tags_polly_never_supports_are_rejected(self, tag):
        with pytest.raises(SsmlError, match="not supported by Amazon Polly"):
            validate_ssml(f"<speak><{tag}>hi</{tag}></speak>", "standard")

    @pytest.mark.parametrize("engine", ["neural", "long-form", "generative"])
    def test_emphasis_is_rejected_on_the_engines_that_measured_as_unsupporting_it(self, engine):
        with pytest.raises(SsmlError, match="emphasis"):
            validate_ssml("<speak>Check the <emphasis>sender</emphasis>.</speak>", engine)

    def test_emphasis_is_allowed_on_standard(self):
        validate_ssml("<speak>Check the <emphasis>sender</emphasis>.</speak>", "standard")

    @pytest.mark.parametrize("engine", ["neural", "long-form", "generative"])
    def test_prosody_pitch_is_rejected_off_standard_but_rate_and_volume_are_not(self, engine):
        with pytest.raises(SsmlError, match="pitch"):
            validate_ssml('<speak><prosody pitch="+10%">hi</prosody></speak>', engine)
        validate_ssml('<speak><prosody rate="90%" volume="+2dB">hi</prosody></speak>', engine)

    def test_drc_is_rejected_on_generative_only(self):
        payload = '<speak><amazon:effect name="drc">hi</amazon:effect></speak>'
        with pytest.raises(SsmlError, match="drc"):
            validate_ssml(payload, "generative")
        for engine in ("standard", "neural", "long-form"):
            validate_ssml(payload, engine)

    def test_newscaster_is_neural_only(self):
        payload = '<speak><amazon:domain name="news">hi</amazon:domain></speak>'
        validate_ssml(payload, "neural")
        for engine in ("standard", "long-form", "generative"):
            with pytest.raises(SsmlError, match="neural engine"):
                validate_ssml(payload, engine)

    def test_whispering_requires_the_standard_engine(self):
        payload = '<speak><amazon:effect name="whispered">hi</amazon:effect></speak>'
        with pytest.raises(SsmlError, match="standard"):
            validate_ssml(payload, "generative")

    def test_auto_breaths_requires_the_standard_engine(self):
        payload = "<speak><amazon:auto-breaths>hi</amazon:auto-breaths></speak>"
        with pytest.raises(SsmlError, match="standard"):
            validate_ssml(payload, "neural")

    def test_an_attribute_free_prosody_is_rejected(self):
        with pytest.raises(SsmlError, match="at least one"):
            validate_ssml("<speak><prosody>hi</prosody></speak>", "standard")

    @pytest.mark.parametrize("value", ["800ms", "1s", "0.5s", "10s", "10000ms"])
    def test_valid_break_durations_pass(self, value):
        validate_ssml(f'<speak>a<break time="{value}"/>b</speak>')

    @pytest.mark.parametrize("value", ["11s", "20000ms"])
    def test_a_break_over_ten_seconds_is_rejected(self, value):
        with pytest.raises(SsmlError, match="10 second"):
            validate_ssml(f'<speak>a<break time="{value}"/>b</speak>')

    def test_a_nonsense_break_duration_is_rejected(self):
        with pytest.raises(SsmlError, match="not a duration"):
            validate_ssml('<speak>a<break time="soon"/>b</speak>')

    def test_the_tags_measured_as_working_everywhere_pass_on_every_engine(self):
        payload = (
            "<speak><p><s>Report it within "
            '<say-as interpret-as="cardinal">24</say-as> hours.</s>'
            '<s>Watch for <phoneme alphabet="ipa" ph="ˈfɪʃɪŋ">phishing</phoneme>, '
            '<sub alias="Information Technology">IT</sub> says.</s></p>'
            '<mark name="b1"/><break time="700ms"/>'
            '<lang xml:lang="fr-FR">Bonjour</lang> '
            '<w role="amazon:VB">read</w></speak>'
        )
        for engine in ENGINE_PREFERENCE:
            validate_ssml(payload, engine)


# =========================================================== SSML adaptation


class TestSsmlAdaptation:
    def test_emphasis_becomes_prosody_on_generative(self, tmp_path):
        client = fake_client()
        provider = synth(client, engine="generative")
        provider.synthesize(
            '<speak>Check the <emphasis level="strong">sender</emphasis> first.</speak>',
            "Matthew",
            tmp_path / "a.wav",
        )
        payload = sent(client)["Text"]
        assert "<emphasis" not in payload
        assert '<prosody rate="90%" volume="+4dB">' in payload
        assert "</prosody>" in payload
        assert "sender" in payload
        assert provider.last_adaptations, "the rewrite should be reported"

    @pytest.mark.parametrize("level", ["strong", "moderate", "reduced"])
    def test_each_emphasis_level_maps_to_its_own_prosody(self, level):
        out, notes = adapt_ssml(
            f'<speak><emphasis level="{level}">x</emphasis></speak>', "generative"
        )
        mapping = EMPHASIS_TO_PROSODY[level]
        assert mapping is not None
        for key, value in mapping.items():
            assert f'{key}="{value}"' in out
        assert "pitch" not in out, "pitch is rejected on exactly the same engines"
        assert len(notes) == 1

    def test_emphasis_without_a_level_uses_the_moderate_mapping(self):
        out, _ = adapt_ssml("<speak><emphasis>x</emphasis></speak>", "generative")
        assert '<prosody rate="95%" volume="+2dB">' in out

    def test_emphasis_level_none_is_unwrapped_without_a_stray_close_tag(self):
        out, notes = adapt_ssml(
            '<speak>a<emphasis level="none">b</emphasis>c</speak>', "generative"
        )
        assert out == "<speak>abc</speak>"
        assert "unwrapped" in notes[0]
        validate_ssml(out, "generative")

    def test_mixed_levels_close_the_right_tags(self):
        out, _ = adapt_ssml(
            '<speak><emphasis level="none">a</emphasis>'
            '<emphasis level="strong">b</emphasis></speak>',
            "generative",
        )
        assert out.count("<prosody") == 1 and out.count("</prosody>") == 1
        validate_ssml(out, "generative")

    def test_nested_emphasis_stays_balanced(self):
        out, _ = adapt_ssml(
            '<speak><emphasis level="strong">a'
            '<emphasis level="reduced">b</emphasis>c</emphasis></speak>',
            "generative",
        )
        assert out.count("<prosody") == 2 and out.count("</prosody>") == 2
        validate_ssml(out, "generative")

    def test_a_self_closing_emphasis_leaves_no_close_tag(self):
        out, _ = adapt_ssml('<speak>a<emphasis level="strong"/>b</speak>', "generative")
        assert "</prosody>" not in out
        validate_ssml(out, "generative")

    def test_prosody_pitch_is_dropped_and_rate_kept(self):
        out, notes = adapt_ssml(
            '<speak><prosody rate="90%" pitch="+10%">x</prosody></speak>', "generative"
        )
        assert 'rate="90%"' in out and "pitch" not in out
        assert notes and "pitch" in notes[0]
        validate_ssml(out, "generative")

    def test_a_pitch_only_prosody_keeps_a_valid_attribute(self):
        """Stripping the only attribute would leave <prosody> with none, which Polly
        rejects — so it becomes a no-op rate instead."""
        out, _ = adapt_ssml('<speak><prosody pitch="high">x</prosody></speak>', "neural")
        assert 'rate="100%"' in out
        validate_ssml(out, "neural")

    def test_nothing_is_rewritten_on_the_standard_engine(self):
        payload = (
            '<speak><emphasis level="strong">x</emphasis><prosody pitch="high">y</prosody></speak>'
        )
        out, notes = adapt_ssml(payload, "standard")
        assert out == payload and notes == []

    def test_plain_text_is_never_touched(self):
        out, notes = adapt_ssml("Check the sender address.", "generative")
        assert out == "Check the sender address." and notes == []

    def test_supported_ssml_passes_through_unchanged(self):
        payload = '<speak>a<break time="700ms"/>b</speak>'
        out, notes = adapt_ssml(payload, "generative")
        assert out == payload and notes == []

    def test_strict_mode_raises_instead_of_rewriting(self, tmp_path):
        client = fake_client()
        provider = synth(client, engine="generative", strict_ssml=True)
        with pytest.raises(SsmlError, match="emphasis"):
            provider.synthesize(
                '<speak><emphasis level="strong">x</emphasis></speak>',
                "Matthew",
                tmp_path / "a.wav",
            )
        client.synthesize_speech.assert_not_called()


# =========================================================== credential precedence


class TestCredentials:
    def test_env_secrets_beat_the_ambient_profile(self, monkeypatch):
        """AWS_PROFILE is set on dev boxes; boto3's chain would silently prefer that SSO
        role over the keys someone put in .env. Env secrets must win."""
        monkeypatch.setenv("AWS_PROFILE", "some-sso-profile")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

        creds = resolve_credentials()
        assert creds["aws_access_key_id"] == "AKIAENV"
        assert creds["aws_secret_access_key"] == "envsecret"
        assert creds["region_name"] == "us-west-2"

    def test_constructor_args_beat_the_environment(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
        creds = resolve_credentials(aws_access_key_id="AKIAARG", aws_secret_access_key="argsecret")
        assert creds["aws_access_key_id"] == "AKIAARG"
        assert creds["aws_secret_access_key"] == "argsecret"

    def test_settings_attributes_beat_the_environment(self, monkeypatch):
        """`Settings` is the canonical source — it has already merged .env and os.environ —
        so it must outrank a stale shell export."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")

        class Fake:
            aws_access_key_id = "AKIASETTINGS"
            aws_secret_access_key = "settingssecret"
            aws_region = "eu-west-1"

        monkeypatch.setattr(polly_tts, "get_settings", lambda: Fake())
        creds = resolve_credentials()
        assert creds["aws_access_key_id"] == "AKIASETTINGS"
        assert creds["region_name"] == "eu-west-1"

    def test_the_dotenv_file_is_read_directly_as_a_last_resort(self, monkeypatch, tmp_path):
        """Why this step exists: `Settings` sets extra="ignore", so before it declared the
        aws_* fields it read AWS_ACCESS_KEY_ID out of .env and discarded it without ever
        touching os.environ — keys in .env were invisible and boto3 silently fell through to
        AWS_PROFILE. The fields exist now, so this is a net rather than the main path."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "AWS_ACCESS_KEY_ID=ASIADOTENV\n"
            'AWS_SECRET_ACCESS_KEY="dotenvsecret"\n'
            "AWS_SESSION_TOKEN='dotenvtoken'\n"
            "AWS_REGION=us-east-1\n"
            "NOT_A_PAIR\n",
            encoding="utf-8",
        )
        creds = resolve_credentials(env_file=env_file)
        assert creds["aws_access_key_id"] == "ASIADOTENV"
        assert creds["aws_secret_access_key"] == "dotenvsecret"
        assert creds["aws_session_token"] == "dotenvtoken"
        assert creds["region_name"] == "us-east-1"

    def test_the_settings_field_names_this_module_reads_still_exist(self):
        """A contract test against config we do not own. `resolve_credentials` reads these
        by getattr, so a rename would silently demote us to the .env fallback (or to
        AWS_PROFILE) instead of failing — this catches it at test time."""
        from app.core.config import Settings

        for field in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region",
            "video_default_polly_voice",
            "video_polly_engine",
        ):
            assert field in Settings.model_fields, field

    def test_the_environment_beats_the_dotenv_file(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("AWS_ACCESS_KEY_ID=fromfile\nAWS_SECRET_ACCESS_KEY=fromfile\n")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fromenv")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fromenv")
        assert resolve_credentials(env_file=env_file)["aws_access_key_id"] == "fromenv"

    def test_aws_default_region_is_honoured_when_settings_has_no_region(self, monkeypatch):
        """pydantic maps AWS_REGION onto `aws_region` but not the AWS_DEFAULT_REGION
        spelling, so this step covers it. Note it only applies when `Settings.aws_region` is
        blank — its real default is a live region, which legitimately wins."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
        assert resolve_credentials()["region_name"] == "ap-south-1"

    def test_the_settings_region_outranks_the_default_region_spelling(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")

        class Fake(BlankSettings):
            aws_region = "eu-central-1"

        monkeypatch.setattr(polly_tts, "get_settings", lambda: Fake())
        assert resolve_credentials()["region_name"] == "eu-central-1"

    def test_no_credentials_anywhere_defers_to_the_boto3_chain(self):
        """Absent explicit keys we must NOT invent any — boto3's own chain (profile, SSO,
        instance role) is the documented fallback."""
        creds = resolve_credentials()
        assert creds["aws_access_key_id"] is None
        assert creds["aws_secret_access_key"] is None

    def test_a_session_token_is_passed_through_for_temporary_keys(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ASIATEMP")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "tempsecret")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "tempsessiontoken")
        captured = {}

        def spy(service, **kwargs):
            captured.update(kwargs)
            captured["service"] = service
            return MagicMock()

        import boto3

        monkeypatch.setattr(boto3, "client", spy)
        polly_tts.make_client()
        assert captured["aws_session_token"] == "tempsessiontoken", (
            "STS keys are useless without the token"
        )
        assert captured["aws_access_key_id"] == "ASIATEMP"
        assert captured["service"] == "polly"

    def test_half_a_credential_pair_is_a_clear_error(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAONLY")
        with pytest.raises(PollyCredentialsError) as exc:
            polly_tts.make_client()
        assert "AWS_SECRET_ACCESS_KEY" in str(exc.value)

    def test_a_missing_credential_error_names_what_to_set(self, monkeypatch, tmp_path):
        from botocore.exceptions import NoCredentialsError

        client = MagicMock()
        client.synthesize_speech.side_effect = NoCredentialsError()
        with pytest.raises(PollyCredentialsError) as exc:
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        message = str(exc.value)
        assert "AWS_ACCESS_KEY_ID" in message and "AWS_SECRET_ACCESS_KEY" in message

    def test_an_expired_token_says_so_rather_than_blaming_the_code(self, monkeypatch, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error("ExpiredTokenException", "expired")
        with pytest.raises(PollyCredentialsError) as exc:
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert "AWS_SESSION_TOKEN" in str(exc.value)

    def test_a_missing_region_is_reported_as_a_region_problem(self, tmp_path):
        from botocore.exceptions import NoRegionError

        client = MagicMock()
        client.synthesize_speech.side_effect = NoRegionError()
        with pytest.raises(PollyRegionError, match="AWS_REGION"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_an_unreachable_endpoint_is_reported_as_a_region_problem(self, tmp_path):
        from botocore.exceptions import EndpointConnectionError

        client = MagicMock()
        client.synthesize_speech.side_effect = EndpointConnectionError(
            endpoint_url="https://polly.nowhere.amazonaws.com"
        )
        with pytest.raises(PollyRegionError, match="AWS_REGION"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_an_unlogged_in_sso_profile_says_to_run_sso_login(self, tmp_path):
        from botocore.exceptions import UnauthorizedSSOTokenError

        client = MagicMock()
        client.synthesize_speech.side_effect = UnauthorizedSSOTokenError()
        with pytest.raises(PollyCredentialsError, match="aws sso login"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")


# =========================================================== other error mapping


class TestErrors:
    def test_a_deepgram_voice_id_is_rejected_with_a_useful_message(self):
        """`aura-2-draco-en` is the other engine's default and can arrive from a stored job
        row; the wire answer would be a ValidationException listing a hundred names."""
        with pytest.raises(PollyVoiceError) as exc:
            PollySynthesizer(voice="aura-2-draco-en", client=fake_client())
        message = str(exc.value)
        assert "aura-2-draco-en" in message
        assert "Matthew" in message, "the message should show the shape of a real id"
        assert "VIDEO_DEFAULT_POLLY_VOICE" in message, "and name the setting to fix"

    def test_the_default_voice_comes_from_the_polly_specific_setting(self):
        """Not VIDEO_DEFAULT_TTS_VOICE — that one holds a Deepgram model name."""
        assert PollySynthesizer(client=fake_client()).default_voice == "Matthew"

    def test_the_default_engine_comes_from_the_polly_engine_setting(self, monkeypatch):
        class Fake(BlankSettings):
            video_polly_engine = "neural"

        monkeypatch.setattr(polly_tts, "get_settings", lambda: Fake())
        assert PollySynthesizer(client=fake_client()).engine == "neural"

    def test_a_voice_not_offered_in_the_region_is_named(self, tmp_path):
        client = fake_client(voices=VOICE_ENTRIES)
        provider = PollySynthesizer(voice="Matthew", engine="auto", client=client)
        with pytest.raises(PollyVoiceError, match="Nonexistent"):
            provider.synthesize("Hi.", "Nonexistent", tmp_path / "a.wav")
        client.synthesize_speech.assert_not_called()

    def test_a_voice_engine_mismatch_is_translated(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error(
            "ValidationException", "This voice does not support one of the used SSML features"
        )
        with pytest.raises(PollyVoiceError, match="voice does not support"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_engine_not_supported_is_translated(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error("EngineNotSupportedException")
        with pytest.raises(PollyVoiceError, match="EngineNotSupportedException"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_a_wire_ssml_rejection_points_at_the_engine_matrix(self, tmp_path):
        """Polly says only 'Unsupported Generative feature' — the message has to explain."""
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error(
            "InvalidSsmlException", "Unsupported Generative feature"
        )
        with pytest.raises(SsmlError) as exc:
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert "engine capability" in str(exc.value)

    def test_an_unknown_engine_is_refused_at_construction(self):
        with pytest.raises(PollyVoiceError, match="unknown engine"):
            PollySynthesizer(voice="Matthew", engine="turbo", client=fake_client())

    def test_an_unknown_source_format_is_refused_at_construction(self):
        with pytest.raises(PollyError, match="source_format"):
            PollySynthesizer(voice="Matthew", source_format="flac", client=fake_client())

    def test_an_empty_audio_stream_is_an_error_not_an_empty_file(self, tmp_path):
        client = fake_client(audio=b"")
        with pytest.raises(PollyError, match="empty AudioStream"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_a_missing_audio_stream_is_an_error(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.return_value = {}
        with pytest.raises(PollyError, match="no AudioStream"):
            synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_a_server_error_is_not_dressed_up_as_the_callers_fault(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error(
            "ServiceFailureException", "boom", status=500
        )
        with pytest.raises(PollyError, match="server error"):
            synth(client, attempts=1).synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_no_error_leaks_a_raw_botocore_type(self, tmp_path):
        """Every failure must be a PollyError so callers can catch one family."""
        from botocore.exceptions import ClientError, NoCredentialsError

        for boom in (
            client_error("ThrottlingException"),
            client_error("InvalidSsmlException"),
            client_error("TextLengthExceededException"),
            client_error("AccessDeniedException"),
            client_error("SomethingBrandNew"),
            NoCredentialsError(),
        ):
            client = MagicMock()
            client.synthesize_speech.side_effect = boom
            provider = synth(client, attempts=1)
            with pytest.raises(PollyError):
                provider.synthesize("Hi.", "Matthew", tmp_path / "a.wav")
            assert not isinstance(boom, PollyError)
            assert isinstance(boom, ClientError | NoCredentialsError)


# =========================================================== throttling


class TestThrottling:
    def test_a_throttle_is_retried_and_then_succeeds(self, tmp_path):
        stream = MagicMock()
        stream.read.return_value = silence()
        client = MagicMock()
        client.synthesize_speech.side_effect = [
            client_error("ThrottlingException", "slow down"),
            client_error("ThrottlingException", "slow down"),
            {"AudioStream": stream},
        ]
        out = synth(client, attempts=4).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert out.exists()
        assert client.synthesize_speech.call_count == 3

    def test_persistent_throttling_raises_a_throttled_error_mentioning_the_tps_cap(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error("ThrottlingException", "slow down")
        with pytest.raises(PollyThrottledError, match="8 tps"):
            synth(client, attempts=3).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert client.synthesize_speech.call_count == 3

    def test_the_backoff_grows_and_is_jittered(self, tmp_path, monkeypatch):
        delays: list[float] = []
        monkeypatch.setattr(polly_tts.time, "sleep", delays.append)
        monkeypatch.setattr(polly_tts.random, "uniform", lambda _a, _b: 0.0)
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error("ThrottlingException")
        with pytest.raises(PollyThrottledError):
            synth(client, attempts=4, base_delay=0.5).synthesize(
                "Hi.", "Matthew", tmp_path / "a.wav"
            )
        assert delays == [0.5, 1.0, 2.0], "exponential, three sleeps for four attempts"

    def test_a_400_that_is_not_retryable_is_not_retried(self, tmp_path):
        client = MagicMock()
        client.synthesize_speech.side_effect = client_error("InvalidSsmlException")
        with pytest.raises(SsmlError):
            synth(client, attempts=5).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert client.synthesize_speech.call_count == 1, "retrying a 400 just wastes time"

    def test_a_transient_service_failure_is_retried(self, tmp_path):
        stream = MagicMock()
        stream.read.return_value = silence()
        client = MagicMock()
        client.synthesize_speech.side_effect = [
            client_error("ServiceFailureException", status=500),
            {"AudioStream": stream},
        ]
        assert synth(client, attempts=3).synthesize("Hi.", "Matthew", tmp_path / "a.wav").exists()

    def test_botocore_retries_are_disabled_so_the_backoff_is_not_doubled(self, monkeypatch):
        captured = {}

        def spy(service, **kwargs):
            captured["config"] = kwargs.get("config")
            return MagicMock()

        import boto3

        monkeypatch.setattr(boto3, "client", spy)
        polly_tts.make_client()
        assert captured["config"].retries["max_attempts"] == 1


# =========================================================== length guard


class TestLength:
    def test_normal_scene_narration_is_nowhere_near_the_cap(self, tmp_path):
        narration = " ".join(["word"] * 70)
        client = fake_client()
        synth(client).synthesize(narration, "Matthew", tmp_path / "a.wav")
        assert client.synthesize_speech.called

    def test_over_long_text_is_refused_rather_than_truncated(self, tmp_path):
        client = fake_client()
        with pytest.raises(TextTooLongError) as exc:
            synth(client).synthesize("x" * 7000, "Matthew", tmp_path / "a.wav")
        client.synthesize_speech.assert_not_called()
        assert "truncating" in str(exc.value)
        assert "7000" in str(exc.value)

    def test_the_billed_count_ignores_ssml_tags(self):
        """AWS does not bill tags, so a tag-heavy payload must not trip the billed cap."""
        text = "a" * 100
        assert billed_chars(text) == 100
        tagged = f'<speak><prosody rate="90%" volume="+2dB">{text}</prosody></speak>'
        assert billed_chars(tagged) == 100
        assert len(tagged) > 100

    def test_tag_heavy_ssml_under_the_billed_cap_is_allowed(self, tmp_path):
        body = "word " * 500  # 2500 billed
        tagged = (
            "<speak>"
            + "".join(f'<prosody rate="95%">{w} </prosody>' for w in ["word"] * 20)
            + body[:2000]
            + "</speak>"
        )
        client = fake_client()
        if billed_chars(tagged) <= MAX_BILLED_CHARS and len(tagged) <= 6000:
            synth(client).synthesize(tagged, "Matthew", tmp_path / "a.wav")
            assert client.synthesize_speech.called

    def test_billed_characters_over_the_cap_are_refused(self, tmp_path):
        client = fake_client()
        with pytest.raises(TextTooLongError, match="billed"):
            synth(client).synthesize(
                f"<speak>{'a' * (MAX_BILLED_CHARS + 1)}</speak>", "Matthew", tmp_path / "a.wav"
            )
        client.synthesize_speech.assert_not_called()


# =========================================================== output format


class TestOutput:
    def test_a_wav_target_requests_lossless_pcm_at_16k(self, tmp_path):
        client = fake_client()
        out = synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.wav")
        assert sent(client)["OutputFormat"] == "pcm"
        assert sent(client)["SampleRate"] == "16000"
        assert out == tmp_path / "a.wav"

    def test_the_written_wav_is_a_real_riff_container(self, tmp_path):
        out = synth(fake_client(audio=silence(0.5))).synthesize(
            "Hi.", "Matthew", tmp_path / "a.wav"
        )
        assert out.read_bytes()[:4] == b"RIFF"
        with wave.open(str(out)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16000
            assert handle.getnframes() == 8000

    def test_the_written_wav_is_probeable_and_reports_the_right_duration(self, tmp_path):
        """The pipeline's clock is ffprobe, so the file has to survive a real probe."""
        provider = synth(fake_client(audio=silence(0.75)))
        out, seconds = provider.synthesize_with_duration("Hi.", "Matthew", tmp_path / "a.wav")
        assert out.exists() and out.stat().st_size > 44
        assert seconds == pytest.approx(0.75, abs=0.02)
        assert provider.duration(out) == pytest.approx(0.75, abs=0.02)

    def test_an_mp3_target_is_written_straight_through_at_24k(self, tmp_path):
        client = fake_client(audio=b"\xff\xfb\x90\x00fake mp3 body")
        out = synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.mp3")
        assert sent(client)["OutputFormat"] == "mp3"
        assert sent(client)["SampleRate"] == "24000"
        assert out.read_bytes().startswith(b"\xff\xfb")

    def test_an_ogg_target_asks_for_ogg_vorbis(self, tmp_path):
        client = fake_client(audio=b"OggS-fake")
        synth(client).synthesize("Hi.", "Matthew", tmp_path / "a.ogg")
        assert sent(client)["OutputFormat"] == "ogg_vorbis"

    def test_asking_for_24k_wav_routes_via_mp3_because_pcm_caps_at_16k(self, tmp_path):
        """Measured: pcm + SampleRate=24000 returns InvalidSampleRateException. 24 kHz wav
        is what DeepgramSynthesizer emits, so the option has to exist."""
        client = fake_client()
        provider = PollySynthesizer(voice="Matthew", sample_rate=24000, client=client)
        fmt, rate = provider._format_for(tmp_path / "a.wav")
        assert (fmt, rate) == ("mp3", 24000)

    def test_an_impossible_pcm_sample_rate_is_refused_with_the_reason(self, tmp_path):
        provider = PollySynthesizer(
            voice="Matthew", sample_rate=24000, source_format="pcm", client=fake_client()
        )
        with pytest.raises(PollyError, match="InvalidSampleRateException"):
            provider.synthesize("Hi.", "Matthew", tmp_path / "a.wav")

    def test_an_impossible_mp3_sample_rate_is_refused(self, tmp_path):
        provider = PollySynthesizer(voice="Matthew", sample_rate=12345, client=fake_client())
        with pytest.raises(PollyError, match="supports only"):
            provider.synthesize("Hi.", "Matthew", tmp_path / "a.mp3")

    def test_pcm_to_wav_round_trips_the_samples_exactly(self, tmp_path):
        pcm = struct.pack("<4h", 0, 1000, -1000, 32767)
        out = pcm_to_wav(pcm, tmp_path / "x.wav", 16000)
        with wave.open(str(out)) as handle:
            assert handle.readframes(4) == pcm

    def test_the_parent_directory_is_created(self, tmp_path):
        out = synth(fake_client()).synthesize(
            "Hi.", "Matthew", tmp_path / "deep" / "nested" / "a.wav"
        )
        assert out.exists()


# =========================================================== engine selection


class TestEngineSelection:
    def test_the_default_engine_is_generative(self):
        """Voice quality dominates perceived production value; the one tag it costs us
        (<emphasis>) is rewritten as <prosody>, which measured as working mid-sentence."""
        assert DEFAULT_ENGINE == "generative"
        assert synth().engine == "generative"

    def test_auto_picks_the_best_tier_the_voice_supports(self, tmp_path):
        client = fake_client(voices=VOICE_ENTRIES)
        provider = PollySynthesizer(voice="Matthew", engine="auto", client=client)
        provider.synthesize("Hi.", "Gregory", tmp_path / "a.wav")
        assert sent(client)["Engine"] == "long-form", "Gregory has no generative tier"
        assert provider.last_engine == "long-form"

    def test_auto_prefers_generative_when_the_voice_has_it(self, tmp_path):
        client = fake_client(voices=VOICE_ENTRIES)
        provider = PollySynthesizer(voice="Matthew", engine="auto", client=client)
        provider.synthesize("Hi.", "Danielle", tmp_path / "a.wav")
        assert sent(client)["Engine"] == "generative"

    def test_an_explicit_engine_is_not_second_guessed(self, tmp_path):
        client = fake_client(voices=VOICE_ENTRIES)
        PollySynthesizer(voice="Matthew", engine="standard", client=client).synthesize(
            "Hi.", "Matthew", tmp_path / "a.wav"
        )
        assert sent(client)["Engine"] == "standard"
        client.describe_voices.assert_not_called(), "an explicit engine needs no lookup"

    @pytest.mark.parametrize(
        "engines,expected",
        [
            (["standard", "neural", "generative"], "generative"),
            (["long-form", "neural"], "long-form"),
            (["neural", "standard"], "neural"),
            (["standard"], "standard"),
            ([], None),
        ],
    )
    def test_best_engine_follows_the_documented_preference(self, engines, expected):
        assert best_engine(engines) == expected

    def test_the_preference_order_is_ssml_fidelity_at_high_quality(self):
        assert ENGINE_PREFERENCE == ("generative", "long-form", "neural", "standard")


# =========================================================== voice catalogue


class TestVoiceCatalogue:
    def test_it_serves_the_keys_the_voices_endpoint_declares(self):
        """`app/api/voices.py` exposes id/name/accent/tags/use_cases, so these dicts have
        to drop into that response with no adapter."""
        from app.api.voices import Voice

        voices = list_voices(client=fake_client(voices=VOICE_ENTRIES))
        assert voices
        for voice in voices:
            model = Voice(**{k: v for k, v in voice.items() if k in Voice.model_fields})
            assert model.id and model.name

    def test_each_voice_reports_language_gender_and_engines(self):
        voices = list_voices(client=fake_client(voices=VOICE_ENTRIES))
        matthew = next(v for v in voices if v["id"] == "Matthew")
        assert matthew["language_code"] == "en-US"
        assert matthew["gender"] == "Male"
        assert matthew["engines"] == ["generative", "neural", "standard"]
        assert matthew["best_engine"] == "generative"
        assert matthew["provider"] == "polly", "a unified list needs to say which engine"

    def test_the_accent_label_matches_the_deepgram_vocabulary(self):
        voices = list_voices(language_code=None, client=fake_client(voices=VOICE_ENTRIES))
        by_id = {v["id"]: v for v in voices}
        assert by_id["Matthew"]["accent"] == "American"
        assert by_id["Amy"]["accent"] == "British"

    def test_gender_and_engines_become_pickable_tags(self):
        shaped = shape_voice(VOICE_ENTRIES[1])
        assert "feminine" in shaped["tags"]
        assert "generative" in shaped["tags"] and "long-form" in shaped["tags"]

    def test_the_best_voices_sort_first(self):
        voices = list_voices(language_code=None, client=fake_client(voices=VOICE_ENTRIES))
        assert voices[0]["best_engine"] == "generative"
        assert voices[-1]["best_engine"] == "neural"

    def test_the_language_and_engine_filters_reach_describe_voices(self):
        client = fake_client(voices=VOICE_ENTRIES)
        list_voices(language_code="en-GB", engine="neural", client=client)
        params = client.describe_voices.call_args.kwargs
        assert params["LanguageCode"] == "en-GB"
        assert params["Engine"] == "neural"
        assert params["IncludeAdditionalLanguageCodes"] is True

    def test_an_unknown_engine_filter_is_refused(self):
        with pytest.raises(PollyVoiceError, match="unknown engine"):
            list_voices(engine="turbo", client=fake_client())

    def test_pagination_is_followed(self):
        client = MagicMock()
        client.describe_voices.side_effect = [
            {"Voices": VOICE_ENTRIES[:2], "NextToken": "more"},
            {"Voices": VOICE_ENTRIES[2:]},
        ]
        assert len(list_voices(language_code=None, client=client)) == len(VOICE_ENTRIES)
        assert client.describe_voices.call_count == 2

    def test_the_catalogue_is_fetched_once_per_process(self):
        client = fake_client(voices=VOICE_ENTRIES)
        list_voices(client=client)
        list_voices(client=client)
        assert client.describe_voices.call_count == 1

    def test_the_cache_can_be_bypassed_and_reset(self):
        client = fake_client(voices=VOICE_ENTRIES)
        list_voices(client=client)
        list_voices(client=client, use_cache=False)
        polly_tts.reset_voice_cache()
        list_voices(client=client)
        assert client.describe_voices.call_count == 3

    def test_a_catalogue_failure_is_translated_too(self):
        client = MagicMock()
        client.describe_voices.side_effect = client_error("AccessDeniedException", "no polly")
        with pytest.raises(PollyCredentialsError):
            list_voices(client=client)


# =========================================================== live AWS


@live_only
class TestLiveAws:
    """Real Polly. Requires AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION in .env
    (plus AWS_SESSION_TOKEN for temporary STS keys)."""

    @pytest.fixture(autouse=True)
    def _real_env(self, monkeypatch):
        """Undo the isolation fixture: this tier is meant to read the real .env."""
        import app.core.config as config

        monkeypatch.setattr(polly_tts, "REPO_ROOT", config.REPO_ROOT)
        creds = resolve_credentials()
        if not (creds["aws_access_key_id"] or os.environ.get("AWS_PROFILE")):
            pytest.skip("no AWS credentials in .env or the environment")

    def test_plain_text_synthesis_produces_probeable_audio(self, tmp_path):
        provider = PollySynthesizer(voice="Matthew")
        out, seconds = provider.synthesize_with_duration(
            "Check the sender address before you click anything in that message.",
            "Matthew",
            tmp_path / "plain.wav",
        )
        assert out.read_bytes()[:4] == b"RIFF"
        assert 2.0 < seconds < 12.0, f"implausible duration {seconds}"
        print(
            f"\nlive polly text: {seconds:.3f}s, {out.stat().st_size} bytes, {provider.last_engine}"
        )

    def test_ssml_breaks_lengthen_the_audio(self, tmp_path):
        """Proof that SSML is parsed, not vocalised: the same words plus a 1.5s pause must
        come back about 1.5s longer, and must not contain the words "break time"."""
        provider = PollySynthesizer(voice="Matthew")
        words = "Check the sender. Then hover the link."
        bare, plain_s = provider.synthesize_with_duration(words, "Matthew", tmp_path / "b.wav")
        _, ssml_s = provider.synthesize_with_duration(
            '<speak>Check the sender.<break time="1500ms"/>Then hover the link.</speak>',
            "Matthew",
            tmp_path / "c.wav",
        )
        assert bare.exists()
        print(f"\nlive polly ssml: plain {plain_s:.3f}s vs break {ssml_s:.3f}s")
        assert ssml_s > plain_s + 1.0, "the <break> was not honoured"

    def test_emphasis_is_adapted_and_accepted_by_the_generative_engine(self, tmp_path):
        provider = PollySynthesizer(voice="Matthew", engine="generative")
        out = provider.synthesize(
            '<speak>Check the <emphasis level="strong">sender address</emphasis> first.</speak>',
            "Matthew",
            tmp_path / "emph.wav",
        )
        assert out.exists()
        assert provider.last_adaptations, "the rewrite should have fired"
        print(f"\nlive polly emphasis adaptation: {provider.last_adaptations}")

    def test_strict_mode_reproduces_the_measured_engine_limit(self, tmp_path):
        """Locks the matrix in: <emphasis> really is rejected on generative."""
        provider = PollySynthesizer(voice="Matthew", engine="generative", strict_ssml=True)
        with pytest.raises(SsmlError, match="emphasis"):
            provider.synthesize(
                "<speak>Check the <emphasis>sender</emphasis>.</speak>",
                "Matthew",
                tmp_path / "x.wav",
            )

    def test_the_catalogue_reports_real_engine_support(self):
        voices = list_voices(language_code="en-US")
        by_id = {v["id"]: v for v in voices}
        assert "Matthew" in by_id and "Danielle" in by_id
        assert "generative" in by_id["Matthew"]["engines"]
        assert "long-form" in by_id["Danielle"]["engines"]
        assert "long-form" not in by_id["Matthew"]["engines"], (
            "Matthew is not a long-form voice — probing long-form with him measures a "
            "voice/engine mismatch, not an SSML limit"
        )
        print(
            "\nlive polly generative en-US: "
            f"{sorted(v['id'] for v in voices if 'generative' in v['engines'])}"
        )

    def test_a_long_form_voice_accepts_the_tags_matthew_could_not(self, tmp_path):
        """The corrected long-form row: with Danielle, break/prosody/say-as all work."""
        provider = PollySynthesizer(voice="Danielle", engine="long-form")
        out = provider.synthesize(
            '<speak>Report it within <say-as interpret-as="cardinal">24</say-as> hours.'
            '<break time="600ms"/><prosody rate="95%" volume="+2dB">Every time.</prosody>'
            "</speak>",
            "Danielle",
            tmp_path / "lf.wav",
        )
        assert out.read_bytes()[:4] == b"RIFF"

    def test_over_long_text_is_refused_locally_not_by_the_wire(self, tmp_path):
        provider = PollySynthesizer(voice="Matthew")
        with pytest.raises(TextTooLongError):
            provider.synthesize("word " * 1400, "Matthew", tmp_path / "long.wav")
