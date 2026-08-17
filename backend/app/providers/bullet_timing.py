"""Map on-screen bullets onto the real word timings of the narration that says them.

Bullets are written by the script provider to echo a distinctive phrase from their
scene's narration ("each bullet reuses 2+ consecutive content words verbatim"), which
gives every bullet a genuine lexical anchor in the aligned word stream. This module finds
that anchor and turns it into a reveal time, so a point appears exactly as the narrator
reaches it instead of on a guessed cadence.

TIMEBASE — verified against `app/worker/pipeline.py::_stage_align`
------------------------------------------------------------------
`_stage_align` rebases every scene's aligner output onto the whole-video clock
(`Word(start=w.start + cursor, ...)`) and sets `scene.start = cursor`, so
`Timeline.scenes[*].words` carry GLOBAL timings. `BulletPoint.appear_at`, by contrast, is
documented as scene-relative. `time_bullets` therefore takes `scene_start` and subtracts
it: pass `scene.words` and `scene.start` straight from the Timeline and the conversion is
handled here. Words falling outside the scene span are ignored, so handing over the whole
video's word list would still time the right slice — but only the scene's own words are
matched against.

Three properties hold for every returned list, because a bullet track that breaks them
looks broken on screen rather than merely mistimed:

    * bullets come back in the order they were given, never reordered;
    * `appear_at` is non-decreasing and, whenever the scene is long enough to allow it,
      at least `min_gap` apart — closer than that reads as one flash, not a sequence;
    * nothing appears after `scene_duration - TAIL_GUARD`, so a late bullet still gets
      screen time before the transition takes the slide away.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from app.core.models import BulletPoint, Word
from app.providers.deepgram_align import normalize

LEAD = 0.25
"""Seconds a bullet appears BEFORE its anchor word is spoken.

Text that lands a beat early reads as the narrator arriving at a point already on screen;
text that lands late reads as a lagging caption. The asymmetry is deliberate.
"""

TAIL_GUARD = 0.4
"""No bullet may appear later than `scene_duration - TAIL_GUARD`."""

FUZZY_THRESHOLD = 0.55
"""Minimum `SequenceMatcher` ratio for a fuzzy anchor to be trusted over proportional
placement.

Calibrated on real drift: inflection differences ("spoofed domain" against "spoof
familiar domains") score 0.57-0.70, while a bullet about something else entirely tops out
around 0.42. Below the threshold the "match" is coincidental shared letters.
"""

# Function words carry no anchoring signal: "of the" occurs everywhere in the narration,
# so matching on it would place a bullet at random.
_STOPWORDS = frozenset(
    """a an the and or but of to in on at for with from by as is are was were be been
    being it its this that these those they them their we our you your he she his her
    i me my not no so if then than there here what which who whom when where why how
    all any both each few more most other some such only own same too very
    just do does did done have has had also into over under about after before while
    because through during above below up down out off again once will can may might""".split()
)

# Leading verbs that make a bullet an instruction rather than an observation. The one
# emphasised bullet per scene should be the thing the viewer is asked to DO.
_IMPERATIVES = frozenset(
    """check verify confirm inspect hover report forward delete pause stop avoid never
    always call ask enable use type look read watch question double slow think trust
    dont don't do treat assume click open reply send scan compare examine""".split()
)

_TOKEN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class BulletAnchor:
    """Where in the narration a bullet was found, and how much to trust it."""

    text: str
    word_index: int
    """Index into the scene's own matchable words. -1 when nothing matched."""

    match_len: int
    """How many consecutive narration words the bullet echoed."""

    method: str
    """``"ngram"`` (verbatim run), ``"fuzzy"``, or ``"proportional"`` (no anchor found)."""

    matched_words: list[str] = field(default_factory=list)
    """The narration tokens the bullet landed on — the receipt for a match."""

    anchor_time: float = 0.0
    """Scene-relative start of the first matched word, before lead or spacing."""


def time_bullets(
    bullets: list[str],
    words: list[Word],
    scene_start: float,
    scene_duration: float,
    min_gap: float = 0.6,
) -> list[BulletPoint]:
    """Time `bullets` against the scene's aligned `words`.

    `words` carry global timings (see module docstring); `scene_start` rebases them.
    Returns one `BulletPoint` per non-blank bullet, in input order, with scene-relative
    `appear_at` and `emphasis` set on at most one of them.
    """
    texts = [text.strip() for text in bullets if text and text.strip()]
    if not texts:
        return []

    scene_duration = max(0.0, scene_duration)
    min_gap = max(0.0, min_gap)
    ceiling = max(0.0, scene_duration - TAIL_GUARD)

    anchors = find_anchors(texts, words, scene_start, scene_duration)
    # No anchor word to lead into (alignment produced nothing) means no lead to apply;
    # subtracting it there would just dent the first gap of an even distribution.
    raw = [
        max(0.0, anchor.anchor_time - (LEAD if anchor.word_index >= 0 else 0.0))
        for anchor in anchors
    ]
    times = _space_out(raw, ceiling, min_gap)
    emphasis_index = _emphasis_index(anchors)

    return [
        BulletPoint(text=anchor.text, appear_at=round(time, 3), emphasis=index == emphasis_index)
        for index, (anchor, time) in enumerate(zip(anchors, times, strict=True))
    ]


def find_anchors(
    bullets: list[str],
    words: list[Word],
    scene_start: float = 0.0,
    scene_duration: float = 0.0,
) -> list[BulletAnchor]:
    """Locate each bullet's anchor phrase in the scene's word sequence.

    Exposed for diagnostics: the caller can print which narration words a bullet actually
    landed on, which is the only way to tell a real anchor from a lucky number.
    """
    texts = [text.strip() for text in bullets if text and text.strip()]
    if not texts:
        return []

    scene_words = _scene_words(words, scene_start, scene_duration)
    keys = [normalize(word.word) for word in scene_words]

    if not scene_words:
        # Alignment failed or produced nothing usable: spread evenly and say so.
        span = max(0.0, scene_duration - TAIL_GUARD)
        step = span / len(texts) if len(texts) > 1 else 0.0
        return [
            BulletAnchor(
                text=text,
                word_index=-1,
                match_len=0,
                method="proportional",
                anchor_time=index * step,
            )
            for index, text in enumerate(texts)
        ]

    anchors: list[BulletAnchor] = []
    search_from = 0
    for index, text in enumerate(texts):
        content = _content_words(text)
        found = _best_ngram(content, keys, search_from) or _best_fuzzy(content, keys, search_from)
        if found is None:
            # No lexical anchor at all — place proportionally through the scene's words.
            position = min(len(scene_words) - 1, (index * len(scene_words)) // max(1, len(texts)))
            anchors.append(
                BulletAnchor(
                    text=text,
                    word_index=position,
                    match_len=0,
                    method="proportional",
                    matched_words=[scene_words[position].display],
                    anchor_time=scene_words[position].start - scene_start,
                )
            )
            continue

        position, length, span_end, method = found
        anchors.append(
            BulletAnchor(
                text=text,
                word_index=position,
                match_len=length,
                method=method,
                matched_words=[w.display for w in scene_words[position:span_end]],
                anchor_time=scene_words[position].start - scene_start,
            )
        )
        # Later bullets look after this one, so duplicate phrases resolve in order.
        search_from = position + 1

    return anchors


def anchor_position(bullet: str, narration: str) -> int | None:
    """Word index in `narration` where `bullet`'s anchor phrase starts, else None.

    The text-only half of the matcher, used before any audio exists — the script provider
    calls it to check a bullet is anchored at all and to order bullets as the narration
    says them, using exactly the rule the timer will later apply to real word timings.
    """
    keys = [key for key in (normalize(tok) for tok in _TOKEN.findall(narration)) if key]
    if not keys:
        return None
    hit = _best_ngram(_content_words(bullet), keys, 0)
    return None if hit is None else hit[0]


# --------------------------------------------------------------------------- matching


def _scene_words(words: list[Word], scene_start: float, scene_duration: float) -> list[Word]:
    """The scene's own matchable words, from a possibly global word list."""
    end = scene_start + scene_duration if scene_duration > 0 else None
    kept: list[Word] = []
    for word in words or []:
        if not normalize(word.word):
            continue
        if word.start < scene_start - 1e-6:
            continue
        if end is not None and word.start > end + 1e-6:
            continue
        kept.append(word)
    return kept


def _content_words(text: str) -> list[str]:
    """Normalised content words of a bullet, stopwords dropped.

    A bullet made entirely of function words keeps them — a weak anchor beats none.
    """
    tokens = [normalize(tok) for tok in _TOKEN.findall(text)]
    tokens = [tok for tok in tokens if tok]
    content = [tok for tok in tokens if tok not in _STOPWORDS]
    return content or tokens


def _best_ngram(
    content: list[str], keys: list[str], search_from: int
) -> tuple[int, int, int, str] | None:
    """Longest contiguous run of the bullet's content words present in the narration.

    Content words are matched as a contiguous run of the BULLET, but the narration is
    allowed to interleave stopwords between them ("check the sender domain" anchors
    "sender domain"), which is how a 2-6 word fragment echoes real prose.

    Ties prefer the first occurrence at or after `search_from`, so repeated phrases are
    consumed left to right and bullet order survives.

    Returns ``(start, matched_content_words, span_end, "ngram")`` — the span can be wider
    than the match count because of those interleaved stopwords.
    """
    # A one-word bullet may anchor on that single word; longer ones need 2+ to be
    # distinctive, matching the instruction given to the model.
    floor = 2 if len(content) > 1 else 1
    for length in range(len(content), floor - 1, -1):
        for offset in range(len(content) - length + 1):
            phrase = content[offset : offset + length]
            hit = _find_run(phrase, keys, search_from) or _find_run(phrase, keys, 0)
            if hit is not None:
                start, span_end = hit
                return start, length, span_end, "ngram"
    return None


def _find_run(phrase: list[str], keys: list[str], search_from: int) -> tuple[int, int] | None:
    """First `(start, end)` at or after `search_from` where `phrase` appears.

    Stopwords in the narration are skippable, so the returned span may be longer than the
    phrase.
    """
    limit = max(0, min(search_from, len(keys)))
    for start in range(limit, len(keys)):
        if keys[start] != phrase[0]:
            continue
        cursor = start + 1
        matched = 1
        while matched < len(phrase) and cursor < len(keys):
            if keys[cursor] == phrase[matched]:
                matched += 1
            elif keys[cursor] in _STOPWORDS:
                pass  # narration may pad the phrase with function words
            else:
                break
            cursor += 1
        if matched == len(phrase):
            return start, cursor
    return None


def _best_fuzzy(
    content: list[str], keys: list[str], search_from: int
) -> tuple[int, int, int, str] | None:
    """Best sliding-window match, for bullets that reword rather than quote.

    Character-level so inflection drift ("domains"/"domain", "spotting"/"spot") still
    lands. Windows starting at or after `search_from` get a small preference so ties do
    not walk backwards.
    """
    target = " ".join(content)
    if not target:
        return None
    best_score = 0.0
    best: tuple[int, int] | None = None
    # A reworded bullet is rarely the same length as the phrase it echoes, so the window
    # is allowed to breathe by one word either side.
    widths = {w for w in (len(content) - 1, len(content), len(content) + 1) if w >= 1}
    for width in sorted(w for w in widths if w <= len(keys)):
        for start in range(len(keys) - width + 1):
            window = " ".join(keys[start : start + width])
            ratio = difflib.SequenceMatcher(None, window, target).ratio()
            if start < search_from:
                ratio -= 0.05
            if ratio > best_score:
                best_score, best = ratio, (start, width)
    if best is None or best_score < FUZZY_THRESHOLD:
        return None
    start, width = best
    return start, width, start + width, "fuzzy"


# --------------------------------------------------------------------------- spacing


def _space_out(raw: list[float], ceiling: float, min_gap: float) -> list[float]:
    """Force `raw` into a non-decreasing sequence spaced by at least `min_gap`.

    Forward pass pushes inversions and ties later. If that runs past `ceiling`, a
    backward pass pulls the earlier bullets in — compressing the front of the scene
    rather than dropping the bullet that overflowed. When even the minimum spacing cannot
    fit (`ceiling < (n-1) * min_gap`, i.e. a scene shorter than its own bullet track), the
    gap shrinks uniformly to `ceiling / (n-1)`: still strictly ordered, just tighter than
    ideal, which is the least-bad option when the alternative is losing content.
    """
    count = len(raw)
    if count == 0:
        return []
    if count == 1:
        return [min(max(0.0, raw[0]), ceiling)]

    needed = (count - 1) * min_gap
    if needed > ceiling:
        step = ceiling / (count - 1)
        return [index * step for index in range(count)]

    times = [min(max(0.0, value), ceiling) for value in raw]
    for index in range(1, count):
        times[index] = max(times[index], times[index - 1] + min_gap)
    if times[-1] > ceiling:
        times[-1] = ceiling
        for index in range(count - 2, -1, -1):
            times[index] = min(times[index], times[index + 1] - min_gap)
        # The backward pass can only have moved times down; `needed <= ceiling` above
        # guarantees times[0] >= 0, so no second forward pass is required.
    return times


# --------------------------------------------------------------------------- emphasis


def _emphasis_index(anchors: list[BulletAnchor]) -> int:
    """Pick the single bullet to render in the accent colour.

    RULE, highest score wins, ties broken by earliest bullet:

        +3  the bullet opens with an imperative verb ("Check the sender domain") — an
            instruction is what a training video wants the viewer to remember
        +2  the anchor was a verbatim n-gram rather than a fuzzy or proportional guess,
            so the highlight is provably on the words being spoken
        +1  per anchored narration word, rewarding the longest, most distinctive echo

    A bullet with no anchor at all can still win only if nothing else anchored either;
    the +2/+1 terms make any real match beat it.
    """
    best_index = 0
    best_score = float("-inf")
    for index, anchor in enumerate(anchors):
        content = _content_words(anchor.text)
        score = float(anchor.match_len)
        if anchor.method == "ngram":
            score += 2.0
        if content and content[0] in _IMPERATIVES:
            score += 3.0
        if score > best_score:
            best_score, best_index = score, index
    return best_index
