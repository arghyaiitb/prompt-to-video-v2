"""Script generation. Two providers, one Protocol.

`GeminiScriptProvider` writes a script from a topic. `VerbatimScriptProvider` slices a
script the user already wrote. Callers depend only on `ScriptProvider`, so the choice
between "write it for me" and "say exactly this" is config, not a code path.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import get_settings
from app.core.models import Motion, SceneScript, Script
from app.providers._gemini import GeminiError, generate_content, text_from

# Google's schema dialect requires SCREAMING type names; lowercase "object" is a 400.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "narration": {"type": "STRING"},
                    "heading": {"type": "STRING"},
                    "image_prompt": {"type": "STRING"},
                    "motion": {
                        "type": "STRING",
                        "enum": ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"],
                    },
                },
                "required": ["id", "narration", "heading", "image_prompt", "motion"],
            },
        },
    },
    "required": ["title", "scenes"],
}

# Cycled to break up runs of identical camera moves.
MOTION_CYCLE: tuple[Motion, ...] = (
    Motion.ZOOM_IN,
    Motion.PAN_RIGHT,
    Motion.ZOOM_OUT,
    Motion.PAN_LEFT,
    Motion.STATIC,
)

PROMPT_TEMPLATE = """You are writing a short narrated explainer video about: {topic}

Produce EXACTLY {slide_count} scenes. Return JSON matching the provided schema.

For every scene:

narration — 2 to 3 complete spoken sentences, roughly 18-30 words total. This text is
fed straight to a text-to-speech engine, so it must sound natural read aloud by a
human presenter. Write flowing prose. Absolutely no bullet fragments, no headings, no
markdown, no asterisks, no emoji, no parentheticals, no abbreviations a narrator would
stumble over. Spell out numbers and units the way a person would say them. The
narration of scene {slide_count} must land as a conclusion, not trail off.

heading — a SHORT on-screen title of 3 to 7 words. Title Case. No terminal period, no
colon-subtitle construction, no emoji. It labels the scene for a viewer skimming it.

image_prompt — describe ONE photographic or cinematic BACKGROUND image for this scene.
Name the subject, the setting, the lens or framing, the lighting, and the mood. It must
be a real-looking photograph, not an illustration, diagram, chart, or infographic.
Compose it with generous open space — plain sky, empty wall, shallow-focus
foreground — in the lower third where a text caption will be overlaid. End every
image_prompt with this exact sentence: "No text, no letters, no words, no numbers, no
labels, no signage, no watermarks anywhere in the image." Image models render lettering
as garbled nonsense, so any request for readable text ruins the frame.

motion — the camera move over the still. Vary it across the video so that no two
consecutive scenes use the same value. Choose the move that suits the shot: zoom_in to
build toward a detail, zoom_out to reveal scale, pan_left or pan_right across a wide
scene, static for a portrait or a moment that should feel still.

Number the scene ids 1 through {slide_count} in order. The scenes must tell one
continuous story with no repeated facts between them."""


class GeminiScriptProvider:
    """Satisfies `ScriptProvider` by asking Gemini for structured JSON.

    Uses `responseMimeType=application/json` plus `responseSchema` rather than parsing
    free text: the model then cannot emit a code fence or a preamble, which is the
    usual source of flaky script generation.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.video_default_llm_model
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, topic: str, slide_count: int) -> Script:
        if slide_count < 1:
            raise ValueError("slide_count must be at least 1")

        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        }
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature

        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": PROMPT_TEMPLATE.format(
                                topic=topic.strip(), slide_count=slide_count
                            )
                        }
                    ]
                }
            ],
            "generationConfig": generation_config,
        }

        response = generate_content(self.model, body, self.api_key, timeout=self.timeout)
        raw = text_from(response)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeminiError(f"structured output was not valid JSON: {raw[:400]}") from exc

        scenes = [
            SceneScript(
                id=int(item.get("id") or index + 1),
                narration=_clean_narration(str(item.get("narration", ""))),
                heading=_clean_heading(str(item.get("heading", ""))),
                image_prompt=str(item.get("image_prompt", "")).strip(),
                motion=_coerce_motion(item.get("motion"), index),
            )
            for index, item in enumerate(payload.get("scenes") or [])
        ]
        if not scenes:
            raise GeminiError("model returned zero scenes")

        scenes = _fit_scene_count(scenes, slide_count)
        scenes = _renumber(scenes)
        scenes = _vary_motion(scenes)

        title = _clean_heading(str(payload.get("title") or topic)) or topic.strip()
        return Script(topic=topic.strip(), title=title, scenes=scenes)


class VerbatimScriptProvider:
    """Satisfies `ScriptProvider` without a network call — the script is already written.

    `topic` is carried through for provenance and `slide_count` decides how many pieces
    the text is cut into. Splits on sentence boundaries and balances words per scene, so
    narration timing stays even; falls back to clause and word chunking when there are
    fewer sentences than requested scenes.
    """

    def __init__(self, script_text: str, *, title: str | None = None) -> None:
        if not script_text or not script_text.strip():
            raise ValueError("script_text is empty — nothing to narrate")
        self.script_text = script_text.strip()
        self.title = title

    def generate(self, topic: str, slide_count: int) -> Script:
        if slide_count < 1:
            raise ValueError("slide_count must be at least 1")

        segments = _split_into_segments(self.script_text, slide_count)
        if len(segments) < slide_count:
            raise ValueError(
                f"script has only {len(self.script_text.split())} words — cannot be split "
                f"into {slide_count} scenes (got {len(segments)})"
            )
        scenes = [
            SceneScript(
                id=index + 1,
                narration=segment,
                heading=_heading_from(segment, fallback=f"Part {index + 1}"),
                image_prompt=_image_prompt_from(segment, topic),
                motion=MOTION_CYCLE[index % len(MOTION_CYCLE)],
            )
            for index, segment in enumerate(segments)
        ]
        title = (self.title or _heading_from(self.script_text, fallback=topic)).strip()
        return Script(topic=topic.strip(), title=title or topic.strip(), scenes=scenes)


# --------------------------------------------------------------------------- helpers

_MARKDOWN = re.compile(r"[*_`#>\[\]]|~~")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️]"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")

_STOPWORDS = frozenset(
    """a an the and or but of to in on at for with from by as is are was were be been
    being it its this that these those they them their we our you your he she his her
    i me my not no so if then than there here what which who whom when where why how
    all any both each few more most other some such only own same too very can will
    just do does did done have has had also into over under about after before while
    because through during above below up down out off again once""".split()
)


def _clean_narration(text: str) -> str:
    """Strip anything TTS would read aloud as noise."""
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    return re.sub(r"\s+", " ", text).strip()


def _clean_heading(text: str) -> str:
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".:;,- ").strip()


def _coerce_motion(value: Any, index: int) -> Motion:
    """Trust the enum but never crash on it — an off-schema value falls back to the cycle."""
    if isinstance(value, Motion):
        return value
    if isinstance(value, str):
        try:
            return Motion(value.strip().lower())
        except ValueError:
            pass
    return MOTION_CYCLE[index % len(MOTION_CYCLE)]


def _fit_scene_count(scenes: list[SceneScript], slide_count: int) -> list[SceneScript]:
    """Guarantee the caller's slide_count even if the model miscounted.

    Over-delivery is truncated. Under-delivery is filled by re-splitting the narration
    of the longest scenes, which keeps the words the model chose rather than inventing
    filler.
    """
    if len(scenes) > slide_count:
        return scenes[:slide_count]

    while len(scenes) < slide_count:
        target = max(range(len(scenes)), key=lambda i: len(scenes[i].narration.split()))
        victim = scenes[target]
        halves = _split_into_segments(victim.narration, 2)
        if len(halves) < 2 or min(len(h.split()) for h in halves) < 3:
            # Nothing left to divide — duplicate the last scene's visual with its text.
            scenes.append(victim.model_copy(update={"id": len(scenes) + 1}))
            continue
        scenes[target] = victim.model_copy(update={"narration": halves[0]})
        scenes.insert(
            target + 1,
            victim.model_copy(
                update={
                    "narration": halves[1],
                    "heading": _heading_from(halves[1], fallback=victim.heading),
                }
            ),
        )
    return scenes


def _renumber(scenes: list[SceneScript]) -> list[SceneScript]:
    return [scene.model_copy(update={"id": index + 1}) for index, scene in enumerate(scenes)]


def _vary_motion(scenes: list[SceneScript]) -> list[SceneScript]:
    """Break adjacent duplicates. Prompting asks for variety; this enforces it."""
    result: list[SceneScript] = []
    previous: Motion | None = None
    for index, scene in enumerate(scenes):
        motion = scene.motion
        if motion == previous:
            for candidate in MOTION_CYCLE[index % len(MOTION_CYCLE) :] + MOTION_CYCLE:
                if candidate != previous:
                    motion = candidate
                    break
        result.append(scene.model_copy(update={"motion": motion}))
        previous = motion
    return result


def _split_into_segments(text: str, count: int) -> list[str]:
    """Cut `text` into `count` chunks of roughly equal word count on clean boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if count <= 1:
        return [text]

    units = [u.strip() for u in _SENTENCE_END.split(text) if u.strip()]
    units = _ensure_units(units, count)
    if len(units) <= count:
        # One unit per scene; pad by repeating nothing — caller handles the shortfall.
        return units

    total_words = sum(len(u.split()) for u in units)
    per_scene = total_words / count

    segments: list[list[str]] = [[] for _ in range(count)]
    words_so_far = 0
    slot = 0
    for position, unit in enumerate(units):
        unit_words = len(unit.split())
        units_left = len(units) - position  # counting this one
        slots_left = count - slot  # counting the current slot

        # Close the slot when doing so lands closer to the running word target than
        # cramming one more sentence in would.
        target = per_scene * (slot + 1)
        closing_is_closer = abs(words_so_far - target) <= abs(
            words_so_far + unit_words - target
        )
        # Every remaining slot still needs at least one unit, in both directions:
        # advance early rather than starve the tail, and never advance if doing so
        # would leave a later slot with nothing.
        must_advance = units_left <= slots_left - 1
        would_starve = (units_left - 1) < (slots_left - 2)

        advance = must_advance or (closing_is_closer and not would_starve)
        if slot < count - 1 and segments[slot] and advance:
            slot += 1
        segments[slot].append(unit)
        words_so_far += unit_words

    joined = [" ".join(s).strip() for s in segments]
    return [s for s in joined if s]


def _ensure_units(units: list[str], count: int) -> list[str]:
    """Subdivide until there are at least `count` units, or nothing left to cut."""
    while len(units) < count:
        target = max(range(len(units)), key=lambda i: len(units[i].split()))
        pieces = [p.strip() for p in _CLAUSE.split(units[target]) if p.strip()]
        if len(pieces) < 2:
            words = units[target].split()
            if len(words) < 2:
                break
            middle = len(words) // 2
            pieces = [" ".join(words[:middle]), " ".join(words[middle:])]
        units[target : target + 1] = pieces
    return units


def _heading_from(text: str, *, fallback: str) -> str:
    """Deterministic 3-7 word title from the segment's own opening words."""
    words = re.findall(r"[A-Za-z0-9'-]+", text)
    if not words:
        return _clean_heading(fallback)
    keywords = [w for w in words if w.lower() not in _STOPWORDS]
    chosen = (keywords or words)[:6]
    if len(chosen) < 3:
        chosen = words[:5]
    heading = " ".join(w if w.isupper() else w.capitalize() for w in chosen)
    return _clean_heading(heading) or _clean_heading(fallback)


def _image_prompt_from(segment: str, topic: str) -> str:
    """Build a background-image prompt from the segment without calling an LLM."""
    words = re.findall(r"[A-Za-z0-9'-]+", segment)
    keywords = [w.lower() for w in words if w.lower() not in _STOPWORDS][:10]
    subject = ", ".join(dict.fromkeys(keywords)) or topic.strip()
    return (
        f"A cinematic documentary photograph illustrating {topic.strip()}: {subject}. "
        "Real photography, natural light, shallow depth of field, wide landscape framing, "
        "muted cinematic color grade, generous empty space across the lower third for a "
        "caption overlay. No text, no letters, no words, no numbers, no labels, no "
        "signage, no watermarks anywhere in the image."
    )
