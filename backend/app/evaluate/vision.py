"""LLM judgement of the rendered frames and the script. The half arithmetic cannot do.

:mod:`app.evaluate.metrics` can prove a heading measures 4.4:1 against its background. It
cannot notice that a slide titled "Staying Safe with Smart Habits" is illustrated with a
sunlit office atrium full of houseplants and has nothing to do with phishing. That is what
this module is for, and topical relevance is the single most valuable thing it reports.

Two passes:

*Per scene.* A representative frame — the real rendered frame, text and scrim included, not
the source image — plus the video's topic and that scene's narration. Judging the composite
rather than the raw asset means the model sees what a viewer sees, so its legibility and
composition verdicts are about the output rather than the input.

*Once per video.* The script alone: headings, narration and bullets as text. Flow and
actionability are properties of the sequence, so they cannot be assessed slide by slide.

Both use ``responseMimeType: application/json`` with an explicit ``responseSchema``
(uppercase type names — the REST API rejects lowercase), and both retry once on malformed
JSON before giving up. A failed judgement returns ``None`` and the scorer drops the
dimension rather than substituting a neutral score: an unmeasured dimension must never look
like a passing one.
"""

from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.models import Timeline
from app.evaluate import metrics as M

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-flash-latest"
"""Alias, not a pinned id, so the evaluator follows the current flash generation.

``gemini-pro-latest`` scores a little more harshly and costs a lot more; flash already
separates an off-topic image from an on-topic one by 6+ points, which is all the signal the
scorer needs.
"""

FRAME_WIDTH = 1280
"""Frames are sent at 720p. The judgements asked for here — is this the right subject, can
you read the heading — do not improve at 1080p, and the payload halves."""

MAX_WORKERS = 4
"""Scene judgements are independent; four at a time keeps a five-slide video near the cost
of a single call without tripping rate limits."""


class VisionUnavailable(RuntimeError):
    """No API key, or the transport is unusable. Callers degrade to metrics-only."""


# --------------------------------------------------------------------------- schemas


class SceneVerdict(BaseModel):
    """One scene's judgement. Scores are 1-10 as the model returns them."""

    scene_id: int
    topical_relevance: int = Field(ge=1, le=10)
    text_legibility: int = Field(ge=1, le=10)
    composition: int = Field(ge=1, le=10)
    professionalism: int = Field(ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    relevance_reason: str = ""
    suggested_image_prompt: str | None = None


class ScriptVerdict(BaseModel):
    narrative_flow: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    actionability: int = Field(ge=1, le=10)
    bullets_echo_narration: str = "no_bullets"
    """``yes`` | ``partly`` | ``no`` | ``no_bullets``. The fourth value exists because
    older jobs predate ``Scene.bullets``, and "no bullets to check" is not a failure."""

    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class VisionReport(BaseModel):
    model: str = DEFAULT_MODEL
    scenes: dict[int, SceneVerdict] = Field(default_factory=dict)
    script: ScriptVerdict | None = None
    errors: list[str] = Field(default_factory=list)


_SCENE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "topical_relevance": {"type": "INTEGER"},
        "relevance_reason": {"type": "STRING"},
        "text_legibility": {"type": "INTEGER"},
        "composition": {"type": "INTEGER"},
        "professionalism": {"type": "INTEGER"},
        "issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        "suggested_image_prompt": {"type": "STRING"},
    },
    "required": [
        "topical_relevance",
        "relevance_reason",
        "text_legibility",
        "composition",
        "professionalism",
        "issues",
        "suggested_image_prompt",
    ],
    "propertyOrdering": [
        "topical_relevance",
        "relevance_reason",
        "text_legibility",
        "composition",
        "professionalism",
        "issues",
        "suggested_image_prompt",
    ],
}

_SCRIPT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "narrative_flow": {"type": "INTEGER"},
        "clarity": {"type": "INTEGER"},
        "actionability": {"type": "INTEGER"},
        "bullets_echo_narration": {
            "type": "STRING",
            "enum": ["yes", "partly", "no", "no_bullets"],
        },
        "issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        "suggestions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "narrative_flow",
        "clarity",
        "actionability",
        "bullets_echo_narration",
        "issues",
        "suggestions",
    ],
}

_SCENE_INSTRUCTIONS = """\
You are a demanding reviewer of corporate training video slides. You are shown one real \
rendered frame from a slide. Score it strictly on a 1-10 scale; 7 means "acceptable to \
publish", 10 is reserved for genuinely excellent work, and do not inflate.

topical_relevance — judge ONLY what the image's subject IS, against the video's topic and \
this slide's narration. Ignore how attractive or well-shot it is. Use these anchors \
literally; do not compress everything into the middle or the bottom:

  1-2  The subject has no connection to the topic at all — architecture, landscaping, \
plants, scenery, an empty lobby or atrium, an abstract texture. It could be swapped into a \
video on any subject whatsoever without anyone noticing.
  3-4  Connected only by generic "business" context: a meeting, a handshake, people \
walking, an office interior with no relevant activity.
  5-6  The right general domain — someone using a computer, a phone, a keyboard, a screen \
— but not the specific situation this slide describes.
  7-8  Depicts the situation the narration describes: the right person doing the right \
thing in the right setting.
  9-10 Depicts the specific mechanism or detail the narration names.

relevance_reason — one sentence naming what the image literally shows, then which anchor \
band that puts it in.

text_legibility — can the overlaid heading be read effortlessly at a glance? Consider the \
background directly behind the glyphs, not the frame overall. White text over a bright, \
sunlit or high-detail area scores 1-4 even when an outline makes it technically decipherable.

composition — is there clean, uncluttered space where the text sits, and is the subject \
placed so the text does not cover it?

professionalism — does this look like material a company would show employees?

issues — short, specific, concrete problems. Empty array if genuinely none.

suggested_image_prompt — if topical_relevance is below 7, write a complete replacement \
image-generation prompt: photographic, specific to this narration, describing the subject, \
framing and lighting, leaving clean space where the heading sits, and ending with an \
instruction that the image contain no text, letters, numbers or logos. If relevance is 7 \
or above, return an empty string."""

_SCRIPT_INSTRUCTIONS = """\
You are reviewing the script of a corporate training video. Score strictly 1-10; 7 means \
"acceptable to publish".

narrative_flow — do the slides build in a sensible order, each following from the last, \
with no repetition and no gap?
clarity — is the language concrete and easy to follow when heard once, aloud?
actionability — does a viewer come away knowing what to actually DO?
bullets_echo_narration — do the on-screen bullets repeat phrases the narration says at \
roughly that moment? Answer "no_bullets" if no slide has bullets.
issues — specific problems, naming the slide number.
suggestions — concrete rewrites or additions, most valuable first."""


# ------------------------------------------------------------------------ transport


def _generate(model: str, body: dict[str, Any], api_key: str) -> str:
    """One structured call, returning the concatenated text parts.

    Delegates to the verified transport in :mod:`app.providers._gemini`, which already
    handles the retry statuses and the ``parts`` quirk where a single part carries
    ``inlineData`` and ``thoughtSignature`` but no ``text`` key at all.
    """
    from app.providers._gemini import generate_content, text_from

    return text_from(generate_content(model, body, api_key, timeout=180.0))


def _json_call(
    model: str, body: dict[str, Any], api_key: str, *, what: str
) -> dict[str, Any] | None:
    """Structured call plus one retry when the model returns unparseable JSON.

    ``responseSchema`` makes malformed output rare but not impossible — truncation at the
    token limit produces a valid prefix of invalid JSON. The retry appends an explicit
    instruction rather than repeating the identical request, because an identical request
    tends to fail identically.
    """
    attempts = [body, _with_retry_nudge(body)]
    for index, payload in enumerate(attempts, start=1):
        try:
            raw = _generate(model, payload, api_key)
        except Exception as exc:  # noqa: BLE001 - transport already retried internally
            logger.warning("%s: gemini call failed (attempt %d): %s", what, index, exc)
            continue
        try:
            parsed = json.loads(_strip_fence(raw))
        except (json.JSONDecodeError, TypeError):
            logger.warning("%s: malformed JSON (attempt %d): %s", what, index, raw[:400])
            continue
        if isinstance(parsed, dict):
            return parsed
        logger.warning("%s: expected an object, got %s", what, type(parsed).__name__)
    return None


def _strip_fence(text: str) -> str:
    """Tolerate a ```json fence even though the schema should prevent one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _with_retry_nudge(body: dict[str, Any]) -> dict[str, Any]:
    retry = json.loads(json.dumps(body))
    retry["contents"][0]["parts"].append(
        {"text": "Return ONLY the JSON object required by the schema. No prose, no fences."}
    )
    return retry


# -------------------------------------------------------------------- frame capture


def scene_frame_jpeg(timeline: Timeline, scene_index: int, video_path: Path) -> bytes | None:
    """A representative frame for one scene, JPEG-encoded, ready to inline.

    Sampled from the scene's own clip when it exists, otherwise from the final video with
    the accumulated xfade overlap subtracted — without that correction a late scene's
    "representative frame" is its predecessor's.
    """
    scene = timeline.scenes[scene_index]
    source = M.frame_source(scene, timeline, video_path)
    if not source.path.exists():
        return None
    at = M.sample_timestamps(source, 1)[0]
    argv = [
        M.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{max(0.0, at):.3f}", "-i", str(source.path), "-frames:v", "1",
        "-vf", f"scale={FRAME_WIDTH}:-2:flags=lanczos",
        "-q:v", "3", "-f", "mjpeg", "-",
    ]  # fmt: skip
    proc = M._run(argv, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        logger.warning("could not extract a frame for scene %s", scene.id)
        return None
    return proc.stdout


# --------------------------------------------------------------------- scene verdict


def judge_scene(
    *,
    frame_jpeg: bytes,
    topic: str,
    heading: str,
    narration: str,
    scene_id: int,
    scene_number: int,
    scene_count: int,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> SceneVerdict | None:
    """Score one rendered frame. ``None`` when the call or the parse failed."""
    prompt = (
        f"{_SCENE_INSTRUCTIONS}\n\n"
        f"VIDEO TOPIC: {topic}\n"
        f"SLIDE {scene_number} OF {scene_count}\n"
        f"ON-SCREEN HEADING: {heading}\n"
        f"NARRATION SPOKEN OVER THIS SLIDE: {narration}\n"
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(frame_jpeg).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _SCENE_SCHEMA,
            # Judgements should be stable enough that a re-score after a fix is
            # attributable to the fix rather than to sampling noise.
            "temperature": 0.0,
        },
    }
    parsed = _json_call(model, body, api_key, what=f"scene {scene_id}")
    if parsed is None:
        return None
    suggestion = (parsed.get("suggested_image_prompt") or "").strip()
    try:
        return SceneVerdict(
            scene_id=scene_id,
            topical_relevance=_clamp(parsed.get("topical_relevance")),
            text_legibility=_clamp(parsed.get("text_legibility")),
            composition=_clamp(parsed.get("composition")),
            professionalism=_clamp(parsed.get("professionalism")),
            issues=[str(i) for i in (parsed.get("issues") or []) if str(i).strip()],
            relevance_reason=str(parsed.get("relevance_reason") or "").strip(),
            suggested_image_prompt=suggestion or None,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("scene %s: verdict failed validation: %s", scene_id, exc)
        return None


def _clamp(value: Any) -> int:
    """Coerce a model-supplied score into 1-10.

    Models occasionally answer ``0``, ``"8"`` or ``8.5`` despite an INTEGER schema, and a
    ValidationError on the whole verdict would throw away four good scores over one.
    """
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, number))


# -------------------------------------------------------------------- script verdict


def script_text(timeline: Timeline) -> str:
    """The script as the model should see it: headings, narration, bullets, timings."""
    lines = [f"TOPIC: {timeline.topic}", f"TITLE: {timeline.title}", ""]
    for index, scene in enumerate(timeline.scenes, start=1):
        lines.append(f"SLIDE {index} — {scene.heading}  ({scene.duration:.1f}s)")
        lines.append(f"  narration: {scene.narration}")
        if scene.bullets:
            for bullet in scene.bullets:
                mark = " *" if bullet.emphasis else ""
                lines.append(f"  bullet @{bullet.appear_at:.1f}s{mark}: {bullet.text}")
        else:
            lines.append("  bullets: (none on this slide)")
        lines.append("")
    return "\n".join(lines)


def judge_script(
    timeline: Timeline, *, api_key: str, model: str = DEFAULT_MODEL
) -> ScriptVerdict | None:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{_SCRIPT_INSTRUCTIONS}\n\n{script_text(timeline)}"}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _SCRIPT_SCHEMA,
            "temperature": 0.0,
        },
    }
    parsed = _json_call(model, body, api_key, what="script")
    if parsed is None:
        return None
    echo = str(parsed.get("bullets_echo_narration") or "no_bullets").lower()
    if echo not in {"yes", "partly", "no", "no_bullets"}:
        echo = "no_bullets"
    if not any(s.bullets for s in timeline.scenes):
        # Ground truth beats the model here: there is nothing to echo.
        echo = "no_bullets"
    return ScriptVerdict(
        narrative_flow=_clamp(parsed.get("narrative_flow")),
        clarity=_clamp(parsed.get("clarity")),
        actionability=_clamp(parsed.get("actionability")),
        bullets_echo_narration=echo,
        issues=[str(i) for i in (parsed.get("issues") or []) if str(i).strip()],
        suggestions=[str(s) for s in (parsed.get("suggestions") or []) if str(s).strip()],
    )


# ------------------------------------------------------------------------- top level


def judge_timeline(
    timeline: Timeline,
    video_path: Path,
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    max_workers: int = MAX_WORKERS,
) -> VisionReport:
    """Every vision judgement for one video. Partial failure degrades, never raises."""
    if not api_key:
        from app.core.config import get_settings

        api_key = get_settings().gemini_api_key
    if not api_key:
        raise VisionUnavailable("GEMINI_API_KEY is not set; run with --no-vision")

    report = VisionReport(model=model)
    count = len(timeline.scenes)

    def one(index: int) -> SceneVerdict | None:
        scene = timeline.scenes[index]
        frame = scene_frame_jpeg(timeline, index, video_path)
        if frame is None:
            report.errors.append(f"scene {scene.id}: no frame could be extracted")
            return None
        return judge_scene(
            frame_jpeg=frame,
            topic=timeline.topic,
            heading=scene.heading,
            narration=scene.narration,
            scene_id=scene.id,
            scene_number=index + 1,
            scene_count=count,
            api_key=api_key,
            model=model,
        )

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, count or 1))) as pool:
        verdicts = list(pool.map(one, range(count)))
    for scene, verdict in zip(timeline.scenes, verdicts, strict=True):
        if verdict is None:
            report.errors.append(f"scene {scene.id}: vision judgement unavailable")
        else:
            report.scenes[scene.id] = verdict

    report.script = judge_script(timeline, api_key=api_key, model=model)
    if report.script is None:
        report.errors.append("script judgement unavailable")
    return report
