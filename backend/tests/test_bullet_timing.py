"""Tests for `app.providers.bullet_timing`.

Pure and offline: word timings are built by hand, so every assertion is about the
algorithm rather than about Deepgram. The live end-to-end check lives in
`tests/test_providers.py::TestLiveBulletTiming`.
"""

from __future__ import annotations

import pytest

from app.core.models import BulletPoint, Word
from app.providers.bullet_timing import (
    LEAD,
    TAIL_GUARD,
    anchor_position,
    find_anchors,
    time_bullets,
)

NARRATION = (
    "Phishing starts with a message that looks routine. Check the sender domain before you "
    "trust the name it displays. Hover over the link to reveal the real destination. "
    "Report anything suspicious to the security team right away."
)

BULLETS = [
    "Check The Sender Domain",
    "Hover Over The Link",
    "Reveal The Real Destination",
    "Report Anything Suspicious",
]


def words_from(text: str, *, start: float = 0.0, per_word: float = 0.4) -> list[Word]:
    """Evenly paced words on a chosen timebase — a stand-in for aligner output."""
    tokens = text.split()
    return [
        Word(word=token, start=start + i * per_word, end=start + (i + 1) * per_word * 0.9)
        for i, token in enumerate(tokens)
    ]


def index_of(text: str, phrase: str) -> int:
    """Word index where `phrase` begins in `text`, for asserting on real positions."""
    tokens = [t.strip(".,").lower() for t in text.split()]
    target = phrase.lower().split()
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i : i + len(target)] == target:
            return i
    raise AssertionError(f"{phrase!r} not in text")


SCENE_WORDS = words_from(NARRATION)
SCENE_DURATION = SCENE_WORDS[-1].end + 0.3


# =========================================================== anchoring


class TestAnchoring:
    def test_bullets_land_on_the_words_they_quote(self):
        anchors = find_anchors(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert [a.method for a in anchors] == ["ngram"] * 4
        assert anchors[0].matched_words[0].lower().startswith("check")
        assert anchors[1].matched_words[0].lower().startswith("hover")
        assert anchors[2].matched_words[0].lower().startswith("reveal")
        assert anchors[3].matched_words[0].lower().startswith("report")

    def test_anchor_index_is_the_real_narration_position(self):
        anchors = find_anchors(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert anchors[0].word_index == index_of(NARRATION, "check the sender")
        assert anchors[1].word_index == index_of(NARRATION, "hover over the link")

    def test_appear_at_is_one_lead_before_the_spoken_word(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        spoken = SCENE_WORDS[index_of(NARRATION, "check the sender")].start
        assert points[0].appear_at == pytest.approx(spoken - LEAD, abs=1e-3)
        assert points[0].appear_at < spoken

    def test_stopwords_between_content_words_do_not_break_the_match(self):
        # "sender domain" is contiguous in the bullet but padded in the narration.
        text = "You should check the sender domain of every message you receive today."
        anchors = find_anchors(["Check Sender Domain"], words_from(text), 0.0, 10.0)
        assert anchors[0].method == "ngram"
        assert anchors[0].word_index == index_of(text, "check")

    def test_reworded_bullet_falls_back_to_fuzzy(self):
        text = "Attackers spoof familiar domains so the message looks entirely legitimate."
        anchors = find_anchors(["Spoofed Domain"], words_from(text), 0.0, 8.0)
        assert anchors[0].method == "fuzzy"
        assert "spoof" in " ".join(anchors[0].matched_words).lower()

    def test_bullet_absent_from_narration_is_placed_proportionally_not_dropped(self):
        anchors = find_anchors(
            ["Check The Sender Domain", "Quarterly Budget Reconciliation Spreadsheet"],
            SCENE_WORDS,
            0.0,
            SCENE_DURATION,
        )
        assert anchors[1].method == "proportional"
        assert len(anchors) == 2

    def test_duplicate_anchor_phrases_resolve_left_to_right(self):
        text = (
            "Check the sender domain on the first message. Later, check the sender domain "
            "again on every reply."
        )
        words = words_from(text)
        anchors = find_anchors(["Check The Sender Domain", "Check The Sender Domain Again"],
                               words, 0.0, 20.0)
        first, second = anchors[0].word_index, anchors[1].word_index
        assert first < second, (first, second)
        assert second == index_of(text, "check the sender domain again")

    def test_anchor_position_matches_the_timers_rule(self):
        assert anchor_position("Hover Over The Link", NARRATION) == index_of(
            NARRATION, "hover over the link"
        )
        assert anchor_position("Quarterly Budget Spreadsheet", NARRATION) is None
        assert anchor_position("anything", "") is None


# =========================================================== timebase


class TestGlobalTimebase:
    """`Timeline.scenes[*].words` carry whole-video timings; `appear_at` does not."""

    def test_scene_start_is_subtracted(self):
        local = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        rebased = words_from(NARRATION, start=137.5)
        global_ = time_bullets(BULLETS, rebased, 137.5, SCENE_DURATION)
        assert [p.appear_at for p in global_] == [p.appear_at for p in local]

    def test_first_bullet_is_never_negative_even_when_it_opens_the_scene(self):
        words = words_from("Check the sender domain of every inbound message you receive.")
        points = time_bullets(["Check The Sender Domain"], words, 0.0, 6.0)
        assert points[0].appear_at >= 0.0

    def test_words_from_neighbouring_scenes_are_ignored(self):
        # A caller handing over the whole video's words must still time this scene.
        before = words_from("Some earlier scene said other things entirely.", start=0.0)
        scene = words_from(NARRATION, start=20.0)
        after = words_from("A later scene continues past the end.", start=60.0)
        points = time_bullets(BULLETS, before + scene + after, 20.0, 30.0)
        assert all(0.0 <= p.appear_at <= 30.0 - TAIL_GUARD for p in points)
        assert points[0].appear_at == pytest.approx(
            scene[index_of(NARRATION, "check the sender")].start - 20.0 - LEAD, abs=1e-3
        )


# =========================================================== ordering and spacing


class TestOrderingAndSpacing:
    def test_output_is_in_input_order(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert [p.text for p in points] == BULLETS

    def test_inverted_matches_are_pushed_forward_not_reordered(self):
        # Deliberately listed out of spoken order: text order is preserved, times are not
        # allowed to go backwards.
        reversed_bullets = list(reversed(BULLETS))
        points = time_bullets(reversed_bullets, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert [p.text for p in points] == reversed_bullets
        assert_monotonic(points, 0.6)

    def test_ties_are_separated_by_min_gap(self):
        text = "Verify the domain and verify the domain carefully every single time you read."
        words = words_from(text)
        points = time_bullets(["Verify The Domain", "Verify The Domain"], words, 0.0, 12.0)
        assert points[1].appear_at - points[0].appear_at >= 0.6 - 1e-6

    def test_custom_min_gap_is_honoured(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION, min_gap=1.5)
        assert_monotonic(points, 1.5)

    def test_nothing_appears_inside_the_tail_guard(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert max(p.appear_at for p in points) <= SCENE_DURATION - TAIL_GUARD + 1e-6

    def test_late_anchors_compress_earlier_gaps_instead_of_dropping_bullets(self):
        # Every bullet quotes the tail of the narration, so the naive times all cluster
        # near the end and cannot all fit before the tail guard.
        text = "A long quiet opening runs on for a while and then check the sender domain."
        words = words_from(text)
        duration = words[-1].end + 0.5
        bullets = ["Check The Sender", "Sender Domain", "The Domain", "Domain"]
        points = time_bullets(bullets, words, 0.0, duration)
        assert len(points) == len(bullets)
        assert_monotonic(points, 0.6)
        assert max(p.appear_at for p in points) <= duration - TAIL_GUARD + 1e-6

    def test_scene_too_short_for_the_bullet_track_still_returns_every_bullet(self):
        duration = 1.2  # far below len(BULLETS) * min_gap
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, duration)
        assert len(points) == len(BULLETS)
        times = [p.appear_at for p in points]
        assert times == sorted(times)
        assert max(times) <= duration - TAIL_GUARD + 1e-6

    def test_zero_duration_scene_collapses_to_zero_without_crashing(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, 0.0)
        assert [p.appear_at for p in points] == [0.0] * len(BULLETS)


# =========================================================== degenerate inputs


class TestEdgeCases:
    def test_no_bullets_returns_empty(self):
        assert time_bullets([], SCENE_WORDS, 0.0, SCENE_DURATION) == []
        assert find_anchors([], SCENE_WORDS, 0.0, SCENE_DURATION) == []

    def test_blank_bullets_are_dropped(self):
        points = time_bullets(["", "   ", "Check The Sender Domain"], SCENE_WORDS, 0.0, 20.0)
        assert [p.text for p in points] == ["Check The Sender Domain"]

    def test_no_words_falls_back_to_even_distribution(self):
        points = time_bullets(BULLETS, [], 0.0, 12.0)
        assert len(points) == len(BULLETS)
        gaps = [
            round(b.appear_at - a.appear_at, 3)
            for a, b in zip(points, points[1:], strict=False)
        ]
        assert len(set(gaps)) == 1, gaps
        assert points[0].appear_at == 0.0
        assert_monotonic(points, 0.6)

    def test_punctuation_only_words_are_ignored(self):
        words = [Word(word="--", start=0.0, end=0.1)] + words_from(NARRATION, start=0.2)
        points = time_bullets(BULLETS, words, 0.0, SCENE_DURATION + 1)
        assert [p.text for p in points] == BULLETS
        assert points[0].appear_at > 0.0

    def test_single_bullet_needs_no_spacing(self):
        points = time_bullets(["Check The Sender Domain"], SCENE_WORDS, 0.0, SCENE_DURATION)
        assert len(points) == 1

    def test_one_word_bullet_can_still_anchor(self):
        anchors = find_anchors(["Phishing"], SCENE_WORDS, 0.0, SCENE_DURATION)
        assert anchors[0].method == "ngram"
        assert anchors[0].word_index == 0

    def test_more_bullets_than_comfortably_fit_are_all_kept(self):
        many = BULLETS + ["Security Team", "Looks Routine", "Trust The Name"]
        points = time_bullets(many, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert len(points) == len(many)
        assert_monotonic(points, 0.0)


# =========================================================== emphasis


class TestEmphasis:
    def test_at_most_one_bullet_is_emphasised(self):
        points = time_bullets(BULLETS, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert sum(p.emphasis for p in points) == 1

    def test_imperative_bullet_wins_over_a_longer_observation(self):
        text = (
            "The message body contains an unusually long and elaborate description of the "
            "invoice. Check the sender domain."
        )
        words = words_from(text)
        bullets = [
            "Unusually Long Elaborate Description Of Invoice",
            "Check The Sender Domain",
        ]
        points = time_bullets(bullets, words, 0.0, words[-1].end + 1)
        assert [p.emphasis for p in points] == [False, True]

    def test_longest_anchor_wins_when_no_bullet_is_an_instruction(self):
        bullets = ["Looks Routine", "Reveal The Real Destination"]
        points = time_bullets(bullets, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert [p.emphasis for p in points] == [False, True]

    def test_an_anchored_bullet_beats_an_unanchored_one(self):
        bullets = ["Quarterly Budget Reconciliation Spreadsheet", "Looks Routine"]
        points = time_bullets(bullets, SCENE_WORDS, 0.0, SCENE_DURATION)
        assert [p.emphasis for p in points] == [False, True]

    def test_no_bullets_means_no_emphasis(self):
        assert time_bullets([], SCENE_WORDS, 0.0, SCENE_DURATION) == []


# =========================================================== property-style sweep


def assert_monotonic(points: list[BulletPoint], min_gap: float) -> None:
    times = [p.appear_at for p in points]
    assert times == sorted(times), times
    for earlier, later in zip(times, times[1:], strict=False):
        assert later - earlier >= min_gap - 1e-6, times


CASES = [
    BULLETS,
    list(reversed(BULLETS)),
    ["Check The Sender Domain"],
    ["Report Anything Suspicious", "Check The Sender Domain"],
    ["Nothing Matches Here", "Neither Does This One", "Nor This Third Thing"],
    BULLETS + ["Security Team", "Looks Routine"],
    ["Domain", "Domain", "Domain", "Domain"],
]


@pytest.mark.parametrize("bullets", CASES)
@pytest.mark.parametrize("duration", [3.0, 8.0, 21.0, 60.0])
@pytest.mark.parametrize("min_gap", [0.0, 0.6, 1.2])
@pytest.mark.parametrize("scene_start", [0.0, 91.25])
def test_appear_at_is_always_ordered_bounded_and_spaced(bullets, duration, min_gap, scene_start):
    """The three invariants the renderer relies on, over the whole input cross-product."""
    words = words_from(NARRATION, start=scene_start)
    points = time_bullets(bullets, words, scene_start, duration, min_gap=min_gap)

    assert [p.text for p in points] == bullets, "order must never change"
    assert sum(p.emphasis for p in points) <= 1

    times = [p.appear_at for p in points]
    assert times == sorted(times), times
    assert all(t >= 0.0 for t in times), times
    assert max(times, default=0.0) <= max(0.0, duration - TAIL_GUARD) + 1e-6, times

    # min_gap is honoured whenever the scene is long enough to honour it at all.
    if len(times) > 1 and (len(times) - 1) * min_gap <= duration - TAIL_GUARD:
        for earlier, later in zip(times, times[1:], strict=False):
            assert later - earlier >= min_gap - 1e-6, times


@pytest.mark.parametrize("bullets", CASES)
def test_alignment_failure_degrades_to_even_spacing(bullets):
    points = time_bullets(bullets, [], 0.0, 18.0)
    assert [p.text for p in points] == bullets
    assert_monotonic(points, 0.6 if len(bullets) <= 5 else 0.0)
