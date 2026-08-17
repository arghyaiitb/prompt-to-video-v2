"""Tests for `app.providers.ssml`.

The two that matter are `test_round_trip_preserves_every_token` and
`test_anchors_identical_before_and_after_markup`. Everything else is detail; those two are
the contract. If SSML ever changes the words, the aligner's reference text stops matching
the audio and every on-screen bullet drifts — silently, and only visibly in the finished
video. So the round trip is asserted over a corpus of hostile narrations, and the bullet
anchors are asserted to be byte-identical before and after markup.
"""

from __future__ import annotations

import pytest

from app.core.models import SceneRole, Word
from app.providers import ssml as mod
from app.providers.bullet_timing import find_anchors, time_bullets
from app.providers.deepgram_align import align_tokens, tokenize

# Realistic narration, plus the strings that break naive XML builders: bare ampersands,
# angle brackets, straight and curly quotes, acronyms, domains, URLs, numbers, phone
# numbers, abbreviations that are not sentence ends, and em dashes.
NARRATIONS = [
    "Phishing emails are the most common way attackers get in.",
    "Check the sender domain before you trust the message. Hover the link to see where it "
    "really goes. Report anything suspicious to IT.",
    "Attackers register lookalike domains like paypa1-secure.com and hope you skim.",
    "Always enable MFA on every account, and never approve a push you did not ask for.",
    "Tom & Jerry's invoice arrived from billing@example.com — it was fake.",
    'She said "don\'t click that link" & she was right.',
    "Is 5 < 10 > 3? The maths does not matter; the <script> tag in the attachment does.",
    "Q&A: how do we report a phish? Forward it to security@example.com within 24 hours.",
    "Call 555-0142 to verify a payment request, e.g. any wire over $10,000.",
    "The U.S. team saw 1,284 attempts last quarter — up 37% year over year.",
    "MFA, SSO and VPN access all rely on one thing: you not handing over your password.",
    "Don't reply. Don't forward. Just report it.",
    "One sentence with no terminal punctuation at all",
    "Short.",
    "URLs in an email can lie; the status bar tells you the truth about https://example.com/login.",
]

BULLETED = [
    (
        "Check the sender domain before you trust the message. Hover the link to see where "
        "it really goes. Report anything suspicious to IT.",
        ["Check the sender domain", "Hover the link", "Report anything suspicious"],
    ),
    (
        "Always enable MFA on every account, and never approve a push you did not ask for.",
        ["Enable MFA on every account", "Never approve an unexpected push"],
    ),
    (
        "Tom & Jerry's invoice arrived from billing@example.com — it was fake, and the "
        "finance team nearly paid it. Verify every payment change by phone.",
        ["Invoice was fake", "Verify payment changes by phone"],
    ),
    (
        "Attackers register lookalike domains like paypa1-secure.com and hope you skim. "
        "Read the domain character by character.",
        ["Lookalike domains", "Read the domain character by character"],
    ),
]

ENGINES = ["polly-standard", "polly-neural", "polly-long-form", "polly-generative"]
ROLES = list(SceneRole)


# --------------------------------------------------------------------------- invariant 1


@pytest.mark.parametrize("narration", NARRATIONS)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("role", ROLES)
def test_round_trip_preserves_every_token(narration: str, engine: str, role: SceneRole) -> None:
    """`strip_ssml(build_ssml(n))` tokenises identically to `n`, for every engine and role.

    Compared with `deepgram_align.tokenize` — the aligner's own tokenizer — so this asserts
    the property the pipeline actually depends on rather than a private notion of a word.
    """
    built = mod.build_ssml(narration, role=role, engine=engine)
    assert tokenize(mod.strip_ssml(built)) == tokenize(narration)


@pytest.mark.parametrize(("narration", "bullets"), BULLETED)
@pytest.mark.parametrize("engine", ENGINES)
def test_round_trip_preserves_tokens_with_bullets(
    narration: str, bullets: list[str], engine: str
) -> None:
    """Emphasis markup is the densest case: tags land mid-sentence, around token cores."""
    built = mod.build_ssml(
        narration, bullets=bullets, engine=engine, spell_out=["paypa1-secure.com"]
    )
    assert tokenize(mod.strip_ssml(built)) == tokenize(narration)


def test_round_trip_violation_is_a_hard_error() -> None:
    """A builder that dropped or rewrote a word must fail loudly, not warn."""
    with pytest.raises(mod.SsmlInvariantError, match="would change the narration"):
        mod._assert_round_trip(
            "Check the sender domain.", "<speak>Check the sender domains.</speak>"
        )
    with pytest.raises(mod.SsmlInvariantError):
        mod._assert_round_trip("Check the sender domain.", "<speak>Check the sender.</speak>")


def test_separating_tags_cannot_fuse_two_words() -> None:
    """The failure mode that would silently halve a token count."""
    assert (
        mod.strip_ssml('<speak>Check the sender.<break time="800ms"/>Then hover.</speak>')
        == "Check the sender. Then hover."
    )
    assert mod.strip_ssml("<speak><s>One.</s><s>Two.</s></speak>") == "One. Two."


def test_inline_tags_keep_punctuation_attached() -> None:
    """`<say-as>MFA</say-as>,` must come back as "MFA," — one token, not two."""
    doc = '<speak>Enable <say-as interpret-as="characters">MFA</say-as>, now.</speak>'
    stripped = mod.strip_ssml(doc)
    assert stripped == "Enable MFA, now."
    assert tokenize(stripped) == ["Enable", "MFA,", "now."]


# --------------------------------------------------------------------------- invariant 2


@pytest.mark.parametrize(("narration", "bullets"), BULLETED)
@pytest.mark.parametrize("engine", ENGINES)
def test_anchors_identical_before_and_after_markup(
    narration: str, bullets: list[str], engine: str
) -> None:
    """THE test. Bullet anchors must be unchanged by the presence of SSML.

    `find_anchors` is run on the original narration and on the stripped SSML with the same
    bullets; every field the reveal time is derived from — word index, match length, method
    and the matched words themselves — must be identical. If this passes, adding SSML
    cannot move a bullet.
    """
    stripped = mod.strip_ssml(mod.build_ssml(narration, bullets=bullets, engine=engine))

    before = find_anchors(bullets, _fake_words(narration))
    after = find_anchors(bullets, _fake_words(stripped))

    assert [(a.word_index, a.match_len, a.method, a.matched_words) for a in before] == [
        (a.word_index, a.match_len, a.method, a.matched_words) for a in after
    ]


@pytest.mark.parametrize(("narration", "bullets"), BULLETED)
def test_bullet_reveal_times_identical_before_and_after_markup(
    narration: str, bullets: list[str]
) -> None:
    """The same property expressed end to end: identical `appear_at` for every bullet."""
    stripped = mod.strip_ssml(mod.build_ssml(narration, bullets=bullets, engine="polly-neural"))
    duration = float(len(tokenize(narration))) + 2.0

    before = time_bullets(bullets, _fake_words(narration), 0.0, duration)
    after = time_bullets(bullets, _fake_words(stripped), 0.0, duration)

    assert [(b.text, b.appear_at, b.emphasis) for b in before] == [
        (b.text, b.appear_at, b.emphasis) for b in after
    ]


def test_emphasis_lands_on_the_anchor_phrase() -> None:
    """Stress must cover the words the bullet echoes — the whole point of the feature."""
    narration = "Check the sender domain before you trust the message. Then hover the link."
    built = mod.build_ssml(
        narration, bullets=["Check the sender domain"], engine="polly-standard"
    )
    inner = built.split('<emphasis level="moderate">')[1].split("</emphasis>")[0]
    assert tokenize(mod.strip_ssml(inner)) == ["Check", "the", "sender", "domain"]


def test_spelled_acronym_keeps_anchor_timing() -> None:
    """`say-as characters` makes the STT hear three tokens. The aligner must still cope.

    This is the one place SSML changes what comes back from the transcript, so it is
    asserted directly against `align_tokens`: the reference token "MFA" must inherit the
    span of the spoken letters, not collapse to zero or steal a neighbour's timing.
    """
    reference = tokenize("Always enable MFA on every account.")
    heard = [
        Word(word="always", start=0.00, end=0.40),
        Word(word="enable", start=0.40, end=0.80),
        Word(word="m", start=0.90, end=1.10),
        Word(word="f", start=1.10, end=1.30),
        Word(word="a", start=1.30, end=1.55),
        Word(word="on", start=1.60, end=1.75),
        Word(word="every", start=1.75, end=2.05),
        Word(word="account", start=2.05, end=2.60),
    ]
    aligned = align_tokens(reference, heard, 2.7)

    mfa = aligned[2]
    assert mfa.display == "MFA"
    assert mfa.start == pytest.approx(0.90, abs=0.01)
    assert mfa.end == pytest.approx(1.55, abs=0.01)
    assert aligned[3].display == "on"
    assert aligned[3].start == pytest.approx(1.60, abs=0.01)


# --------------------------------------------------------------------------- escaping


@pytest.mark.parametrize(
    "narration",
    [
        "Tom & Jerry",
        "5 < 10 and 11 > 3",
        "The <script> tag & the </script> tag",
        'He said "no" and she said \'yes\'',
        "R&D & Q&A & AT&T",
        "a && b &amp; c &#65; d",
        "‘curly’ “quotes” — and an em dash",
    ],
)
@pytest.mark.parametrize("engine", ENGINES)
def test_hostile_strings_produce_valid_xml(narration: str, engine: str) -> None:
    """A bare "&" invalidates the document and Polly 400s the whole request."""
    built = mod.build_ssml(narration, engine=engine)
    assert mod.validate_ssml(built, engine=engine) == []
    assert tokenize(mod.strip_ssml(built)) == tokenize(narration)


def test_ampersand_is_escaped_exactly_once() -> None:
    built = mod.build_ssml("Tom & Jerry", engine="polly-neural")
    assert "&amp;" in built
    assert "&amp;amp;" not in built
    assert mod.strip_ssml(built) == "Tom & Jerry"


def test_literal_markup_in_narration_survives() -> None:
    """Narration that *talks about* SSML must not be parsed as SSML."""
    narration = 'The tag <break time="800ms"/> inserts a pause.'
    built = mod.build_ssml(narration, engine="polly-neural")
    assert "&lt;break" in built
    assert mod.estimate_pause_seconds(built) == 0.0
    assert tokenize(mod.strip_ssml(built)) == tokenize(narration)


# --------------------------------------------------------------------------- the guard


def test_sanitise_removes_the_measured_deepgram_failure() -> None:
    """Aura vocalises tags. This is the exact payload that produced the bad audio."""
    hostile = '<speak>Check the sender.<break time="800ms"/>Then hover the link.</speak>'
    cleaned = mod.sanitise_for_plain_tts(hostile)
    assert cleaned == "Check the sender. Then hover the link."
    assert "<" not in cleaned and "break" not in cleaned.lower()


def test_sanitise_leaves_plain_text_alone() -> None:
    plain = "Check the sender domain before you trust the message."
    assert mod.sanitise_for_plain_tts(plain) == plain
    assert mod.sanitise_for_plain_tts("") == ""


def test_sanitise_is_token_preserving_against_narration() -> None:
    """The guard must not itself become a source of drift for the aligner."""
    for narration in NARRATIONS:
        built = mod.build_ssml(narration, engine="polly-neural")
        assert tokenize(mod.sanitise_for_plain_tts(built)) == tokenize(narration)


def test_build_ssml_refuses_a_non_ssml_engine() -> None:
    """Refusing beats returning markup a caller might forward to Aura."""
    with pytest.raises(ValueError, match="does not parse SSML"):
        mod.build_ssml("Check the sender.", engine="deepgram")


def test_text_for_synthesis_routes_by_capability() -> None:
    narration = "Check the sender."
    marked = mod.build_ssml(narration, engine="polly-neural")
    assert mod.text_for_synthesis(narration, marked, supports_ssml=True) == marked
    assert mod.text_for_synthesis(narration, marked, supports_ssml=False) == narration
    assert mod.text_for_synthesis(narration, None, supports_ssml=True) == narration


# --------------------------------------------------------------------------- validation


def test_emphasis_is_rejected_on_the_engines_that_reject_it() -> None:
    """Measured on a live account: neural and generative raise InvalidSsmlException."""
    doc = '<speak>Check the <emphasis level="moderate">sender domain</emphasis>.</speak>'
    assert mod.validate_ssml(doc, engine="polly-standard") == []
    for engine in ("polly-neural", "polly-long-form", "polly-generative"):
        problems = mod.validate_ssml(doc, engine=engine)
        assert any("emphasis" in problem for problem in problems), engine


def test_builder_never_emits_emphasis_where_it_fails() -> None:
    """The validator is the net; not emitting the tag is the fix."""
    narration = "Check the sender domain before you trust the message. Then hover the link."
    bullets = ["Check the sender domain"]
    assert "<emphasis" in mod.build_ssml(narration, bullets=bullets, engine="polly-standard")
    for engine in ("polly-neural", "polly-long-form", "polly-generative"):
        built = mod.build_ssml(narration, bullets=bullets, engine=engine)
        assert "<emphasis" not in built, engine
        assert "<prosody" in built, engine
        assert mod.validate_ssml(built, engine=engine) == [], engine


def test_generative_stresses_with_volume_only() -> None:
    """Measured on a live account: generative honours `volume` per phrase but not `rate`.

    `<prosody rate>` returns byte-identical audio for 88%-100% and snaps 85%/80%/slow to a
    single 1.25x step, so emitting a rate there ships a no-op or a drawl. `volume="loud"`
    moves the phrase-versus-context loudness gap from +3.1 dB to +4.8 dB on the anchored
    phrase — a real, located effect.
    """
    narration = "Check the sender domain before you trust it. Then hover the link."
    built = mod.build_ssml(
        narration, bullets=["Check the sender domain"], engine="polly-generative"
    )
    assert '<prosody volume="loud">' in built
    assert "rate=" not in built
    assert mod.validate_ssml(built, engine="polly-generative") == []


def test_prosody_rate_is_rejected_on_generative() -> None:
    """A rate on generative is not an error to Polly — it is worse, it is silent."""
    doc = '<speak>Check the <prosody rate="95%">sender domain</prosody> first.</speak>'
    assert mod.validate_ssml(doc, engine="polly-neural") == []
    assert mod.validate_ssml(doc, engine="polly-long-form") == []
    assert any("rate" in p for p in mod.validate_ssml(doc, engine="polly-generative"))


def test_stress_lands_on_the_phrase_on_every_prosody_tier() -> None:
    """Whatever the tier, the wrapper must cover the anchor phrase and nothing else."""
    narration = "Check the sender domain before you trust the message. Then hover."
    for engine in ("polly-neural", "polly-long-form", "polly-generative"):
        built = mod.build_ssml(
            narration, bullets=["Check the sender domain"], engine=engine
        )
        inner = built.split("<prosody", 1)[1].split(">", 1)[1].split("</prosody>")[0]
        assert tokenize(mod.strip_ssml(inner)) == ["Check", "the", "sender", "domain"], engine


def test_prosody_pitch_is_rejected_off_standard() -> None:
    doc = '<speak><prosody pitch="high">Check the sender.</prosody></speak>'
    assert mod.validate_ssml(doc, engine="polly-standard") == []
    for engine in ("polly-neural", "polly-long-form", "polly-generative"):
        assert any("pitch" in p for p in mod.validate_ssml(doc, engine=engine)), engine


def test_say_as_characters_is_withheld_from_neural() -> None:
    """Neural falls back to the STANDARD voice for that sentence — an audible voice swap."""
    doc = '<speak>Enable <say-as interpret-as="characters">MFA</say-as> today.</speak>'
    assert mod.validate_ssml(doc, engine="polly-generative") == []
    assert any("say-as" in p for p in mod.validate_ssml(doc, engine="polly-neural"))

    built = mod.build_ssml("Enable MFA today on every account.", engine="polly-neural")
    assert "say-as" not in built
    spelled = mod.build_ssml("Enable MFA today on every account.", engine="polly-generative")
    assert "say-as" in spelled


def test_sub_is_rejected_everywhere() -> None:
    """Polly supports `<sub>`; this pipeline cannot, because it rewrites the spoken words."""
    doc = '<speak>Read the <sub alias="world wide web">WWW</sub> page.</speak>'
    for engine in ENGINES:
        assert any("sub" in p for p in mod.validate_ssml(doc, engine=engine)), engine


def test_validation_catches_unescaped_ampersand() -> None:
    problems = mod.validate_ssml("<speak>Tom & Jerry</speak>", engine="polly-neural")
    assert any("well-formed" in p for p in problems)


def test_validation_catches_break_ceilings_and_junk() -> None:
    assert mod.validate_ssml('<speak>A.<break time="9s"/>B.</speak>', engine="polly-neural") == []
    assert any(
        "exceeds" in p
        for p in mod.validate_ssml('<speak>A.<break time="11s"/>B.</speak>', engine="polly-neural")
    )
    assert any(
        "exceeds" in p
        for p in mod.validate_ssml('<speak>A.<break time="4s"/>B.</speak>', engine="elevenlabs")
    )
    assert any(
        "not a duration" in p
        for p in mod.validate_ssml('<speak>A.<break time="soon"/>B.</speak>', engine="polly-neural")
    )


def test_validation_rejects_ssml_for_deepgram_and_unknown_engines() -> None:
    doc = "<speak>Check the sender.</speak>"
    assert mod.validate_ssml(doc, engine="deepgram")
    assert mod.validate_ssml(doc, engine="some-new-vendor")
    assert mod.validate_ssml("", engine="polly-neural") == ["empty document"]


def test_validation_accepts_everything_the_builder_produces() -> None:
    """No combination of role, engine and bullets may produce SSML its engine rejects."""
    for engine in ENGINES:
        for role in ROLES:
            for narration, bullets in BULLETED:
                built = mod.build_ssml(narration, role=role, bullets=bullets, engine=engine)
                assert mod.validate_ssml(built, engine=engine) == [], (engine, role)


def test_elevenlabs_accepts_breaks_and_nothing_else() -> None:
    assert mod.validate_ssml('<speak>A.<break time="1.5s"/>B.</speak>', engine="elevenlabs") == []
    problems = mod.validate_ssml(
        '<speak>A.<prosody rate="95%">B.</prosody></speak>', engine="elevenlabs"
    )
    assert any("prosody" in p for p in problems)


# --------------------------------------------------------------------------- markup shape


def test_breaks_go_between_sentences_only() -> None:
    """Nothing before the first sentence, nothing after the last — that beat is unheard."""
    narration = "One thing matters. Two things matter. Three things matter."
    built = mod.build_ssml(narration, role=SceneRole.CONTENT, engine="polly-neural")
    assert built.count("<break") == 2
    assert not built.startswith("<speak><break")
    assert not built.endswith('<break time="250ms"/></speak>')


def test_single_sentence_gets_no_breaks() -> None:
    built = mod.build_ssml("Phishing is the most common way in.", engine="polly-neural")
    assert "<break" not in built
    assert mod.estimate_pause_seconds(built) == 0.0


def test_instruction_gets_the_longer_beat_but_only_once() -> None:
    narration = (
        "Attackers are patient and they do their homework. Check the sender domain. "
        "Hover every link. Report what looks wrong."
    )
    built = mod.build_ssml(narration, role=SceneRole.CONTENT, engine="polly-neural")
    assert built.count(f'<break time="{mod.INSTRUCTION_BREAK_MS}ms"/>') == 1
    assert built.count(f'<break time="{mod.SENTENCE_BREAK_MS}ms"/>') == 2


def test_abbreviations_do_not_split_a_sentence() -> None:
    built = mod.build_ssml(
        "Verify any wire, e.g. anything over $10,000, by phone.", engine="polly-neural"
    )
    assert "<break" not in built


def test_acronyms_are_spelled_but_shouting_is_not() -> None:
    built = mod.build_ssml(
        "NEVER share your MFA code, not even with IT.", engine="polly-generative"
    )
    assert '<say-as interpret-as="characters">MFA</say-as>' in built
    assert '<say-as interpret-as="characters">IT</say-as>' in built
    assert "characters\">NEVER" not in built


def test_word_acronyms_are_left_alone() -> None:
    """Spelling out a pronounced acronym makes the audio worse, not better."""
    built = mod.build_ssml("Your PIN and your SIM are both targets.", engine="polly-generative")
    assert "say-as" not in built


def test_domains_are_slowed_not_spelled_by_default() -> None:
    """Spelling "e x a m p l e dot c o m" is worse than the engine's own reading.

    What a lookalike domain needs is time on the ear, so it gets a slower rate and a weak
    beat in front of it — except on generative, where sub-sentence prosody is illegal and
    the beat is all that is left.
    """
    narration = "Attackers register domains like paypa1-secure.com to fool you."
    phrase_scoped = mod.build_ssml(narration, engine="polly-neural")
    assert '<prosody rate="slow">paypa1-secure.com</prosody>' in phrase_scoped
    assert f'<break time="{mod.DOMAIN_BREAK_MS}ms"/>' in phrase_scoped
    assert "say-as" not in phrase_scoped

    sentence_scoped = mod.build_ssml(narration, engine="polly-generative")
    assert "<prosody" not in sentence_scoped
    assert f'<break time="{mod.DOMAIN_BREAK_MS}ms"/>' in sentence_scoped


def test_a_lookalike_domain_can_be_forced_to_spell_out() -> None:
    built = mod.build_ssml(
        "Attackers register domains like paypa1-secure.com to fool you.",
        engine="polly-generative",
        spell_out=["paypa1-secure.com"],
    )
    assert '<say-as interpret-as="characters">paypa1-secure.com</say-as>' in built


def test_phone_numbers_use_the_telephone_reading() -> None:
    built = mod.build_ssml("Call 555-0142 to verify.", engine="polly-generative")
    assert '<say-as interpret-as="telephone">555-0142</say-as>' in built


def test_role_rate_is_withheld_where_rate_does_nothing() -> None:
    """A title card cannot be slowed on generative, so no tag is emitted pretending to."""
    built = mod.build_ssml(
        "Spotting phishing emails.", role=SceneRole.TITLE, engine="polly-generative"
    )
    assert "<prosody" not in built
    assert mod.validate_ssml(built, engine="polly-generative") == []


def test_role_sets_the_speaking_rate() -> None:
    title = mod.build_ssml("Spotting phishing emails.", role=SceneRole.TITLE, engine="polly-neural")
    content = mod.build_ssml(
        "Spotting phishing emails.", role=SceneRole.CONTENT, engine="polly-neural"
    )
    assert f'<prosody rate="{mod.ROLE_RATE[SceneRole.TITLE]}">' in title
    assert "<prosody" not in content  # CONTENT is 100% — no tag rather than a no-op one


# --------------------------------------------------------------------------- duration


def test_estimate_pause_seconds_reads_both_forms() -> None:
    assert mod.estimate_pause_seconds('<speak>A.<break time="800ms"/>B.</speak>') == 0.8
    assert mod.estimate_pause_seconds('<speak>A.<break time="1.5s"/>B.</speak>') == 1.5
    assert mod.estimate_pause_seconds('<speak>A.<break strength="strong"/>B.</speak>') == 0.5
    assert mod.estimate_pause_seconds("<speak>A.<break/>B.</speak>") == 0.3
    assert mod.estimate_pause_seconds("<speak>No pauses here.</speak>") == 0.0


@pytest.mark.parametrize("role", ROLES)
def test_pauses_stay_inside_the_role_budget(role: SceneRole) -> None:
    """A title card cannot afford the silence a content scene can."""
    narration = (
        "Attackers are patient. Check the sender domain. Hover every link. "
        "Report what looks wrong. Then get on with your day. Nobody blames a careful "
        "person. Speed is the attacker's friend. Slow down."
    )
    built = mod.build_ssml(narration, role=role, engine="polly-neural")
    assert mod.estimate_pause_seconds(built) <= mod.pause_budget(role) + 1e-6
    assert mod.pause_budget(role) == pytest.approx(role.target_duration[0] * 0.10)


def test_budget_keeps_the_instruction_beat_when_it_has_to_drop_breaks() -> None:
    """Under pressure, the pause that carries meaning is the one that survives."""
    narration = (
        "Attackers are patient. They do their homework. They wait for a busy morning. "
        "Check the sender domain."
    )
    built = mod.build_ssml(narration, engine="polly-neural", max_pause_seconds=0.2)
    assert mod.estimate_pause_seconds(built) <= 0.2 + 1e-6
    assert built.count("<break") == 1
    assert tokenize(mod.strip_ssml(built)) == tokenize(narration)


def test_explicit_budget_of_zero_removes_pauses() -> None:
    built = mod.build_ssml("One. Two. Three.", engine="polly-neural", max_pause_seconds=0.0)
    assert mod.estimate_pause_seconds(built) == 0.0


# --------------------------------------------------------------------------- misc


def test_capability_aliases_and_unknown_engines() -> None:
    assert mod.capability("polly").engine == "polly-neural"
    assert mod.capability("PoLLy-Generative").engine == "polly-generative"
    assert mod.capability(None).engine == "polly-neural"
    assert mod.capability("brand-new-tts").supports_ssml is False
    assert mod.capability("deepgram").supports_ssml is False


def test_emphasis_style_per_engine() -> None:
    assert mod.capability("polly-standard").emphasis_style() == "tag"
    assert mod.capability("polly-neural").emphasis_style() == "prosody"
    assert mod.capability("polly-long-form").emphasis_style() == "prosody"
    assert mod.capability("polly-generative").emphasis_style() == "prosody"
    assert mod.capability("deepgram").emphasis_style() == "none"


def test_empty_narration_is_refused() -> None:
    for value in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="empty narration"):
            mod.build_ssml(value, engine="polly-neural")


def test_fuzzy_anchors_are_not_emphasised() -> None:
    """Stress near the phrase is worse than no stress — the words would not match."""
    narration = "Attackers spoof familiar domains to earn a moment of trust."
    assert mod.anchor_spans(tokenize(narration), ["Something entirely unrelated"]) == []


def test_capability_matrix_is_serialisable() -> None:
    matrix = mod.capability_matrix()
    assert set(matrix) == set(mod.CAPABILITIES)
    assert matrix["polly-generative"]["emphasis"] == "prosody"
    assert matrix["polly-generative"]["prosody_attrs"] == ["volume"]
    assert matrix["deepgram"]["supports_ssml"] is False


def _fake_words(narration: str) -> list[Word]:
    """One synthetic second per token — enough for the matcher, which only reads order."""
    return [
        Word(word=token, start=float(index), end=float(index) + 0.9)
        for index, token in enumerate(tokenize(narration))
    ]
