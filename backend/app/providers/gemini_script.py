"""Script generation. Two providers, one Protocol.

`GeminiScriptProvider` writes a script from a topic. `VerbatimScriptProvider` slices a
script the user already wrote. Callers depend only on `ScriptProvider`, so the choice
between "write it for me" and "say exactly this" is config, not a code path.

STRUCTURE
---------
A generated script is not a queue of interchangeable slides. Every scene carries a
`SceneRole` (see `app/core/models.py`) and the role decides three things at once: how many
words the narration gets, how many bullets the slide may show, and — because audio is the
clock — how long the scene lasts. `role_plan` lays the shape out deterministically:

    title -> content ... -> [summary] -> closing

The summary only appears from `SUMMARY_FROM_SLIDES` scenes up: below that the closing
already restates the key point, so a recap is the redundancy principle violated and it
costs a teaching slide to buy. The plan is both sent to the model — scene by scene, with
each scene's word and bullet budget spelled out — and re-imposed on the way back by
position, so a model that argues with the instruction still yields a shaped script.

Every number here is `docs/DIRECTION.md`: §1.1 for the sequence per slide count, §1.2 for
the word and bullet budgets, §2.1/§2.2 for the one-line copy caps, §5 for the 135 wpm pace,
§9 for the corrections to the committed `SceneRole` values.

LANGUAGE
--------
`generate(..., language=...)` writes the script NATIVELY in the target language — narration,
heading and bullets all in-language, never English-then-translated. Translation is the wrong
tool twice over: it produces narration that scans as translated when read aloud, and it
breaks bullet anchoring, because a bullet translated independently of its narration no
longer quotes it.

Two things do NOT translate, and the split is deliberate:

  * `image_prompt` and `clip_prompt` stay ENGLISH. They are conditioning text for the
    image and video models, which are English-trained; a Hindi prompt degrades the frame
    for no benefit, since nothing in a visual prompt is ever seen or heard by a viewer.
  * the word budgets are not the English ones. A word budget is a DURATION (audio is the
    clock), so the budget must be scaled by the language's measured speaking rate or the
    scene overruns its role's window. See `LANGUAGE_WPM`.

ANCHORING IS NOT UNIFORM ACROSS SCRIPTS — the load-bearing caveat. `bullet_timing` matches
bullets to narration through `deepgram_align.normalize`, which reduces a token to `[a-z0-9]`
only. Latin scripts survive that (Spanish accents are stripped on BOTH sides, so they still
compare equal); Devanagari does not — every Hindi token normalises to the empty string, so
`anchor_position` returns None for every Hindi bullet and `find_anchors` reports
`proportional` for all of them. `anchoring_supported` PROBES that rather than hardcoding it —
so this file starts reporting Hindi as anchorable the day `normalize` is widened, instead of
carrying a stale assumption — and `_clean_bullets` stops second-guessing the model's bullets
in a script where the check cannot run at all. This is a real product constraint, not a bug
in this file: fixing it means widening `normalize`, which lives in `deepgram_align`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from app.core.config import get_settings
from app.core.models import Language, Motion, SceneRole, SceneScript, Script
from app.providers._gemini import GeminiError, generate_content, text_from
from app.providers.bullet_timing import anchor_position

logger = logging.getLogger(__name__)

# Google's schema dialect requires SCREAMING type names; lowercase "object" is a 400.
# Enums are honoured by this API (verified against the `motion` field), so `role` is
# declared as one rather than as a free string that would come back as prose.
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
                    "role": {
                        "type": "STRING",
                        "enum": ["title", "content", "summary", "closing"],
                    },
                    "narration": {"type": "STRING"},
                    "heading": {"type": "STRING"},
                    "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "image_prompt": {"type": "STRING"},
                    "clip_prompt": {"type": "STRING"},
                    "motion": {
                        "type": "STRING",
                        "enum": ["zoom_in", "zoom_out", "pan_left", "pan_right", "static"],
                    },
                },
                "required": [
                    "id",
                    "role",
                    "narration",
                    "heading",
                    "bullets",
                    "image_prompt",
                    "clip_prompt",
                    "motion",
                ],
            },
        },
    },
    "required": ["title", "scenes"],
}


class StructuredSceneScript(SceneScript):
    """`SceneScript` plus the two fields the structured script carries.

    A subclass rather than an edit to `app.core.models`: `SceneScript` is the shape of the
    Gemini `responseSchema` and is shared with the verbatim path, while `role` and
    `clip_prompt` are only meaningful for a script that was *planned*. Pydantic keeps
    subclass instances intact inside `Script.scenes`, so callers that only know about
    `SceneScript` are unaffected — they use `scene_role`/`scene_clip_prompt` to ask.
    """

    role: SceneRole = SceneRole.CONTENT
    clip_prompt: str | None = None
    """Motion description for a video model. See `Scene.clip_prompt`."""


def scene_role(scene: SceneScript) -> SceneRole:
    """The scene's role, defaulting to CONTENT for a plain `SceneScript`."""
    role = getattr(scene, "role", None)
    return role if isinstance(role, SceneRole) else SceneRole.CONTENT


def scene_clip_prompt(scene: SceneScript) -> str | None:
    """The scene's motion prompt, or None when the provider does not emit one."""
    value = getattr(scene, "clip_prompt", None)
    text = str(value).strip() if value else ""
    return text or None

# On-screen bullet budget. Fewer than three looks empty in the left panel; more than five
# cannot be read before the scene ends, so a caller's `bullets_per_slide` is clamped here.
BULLET_MIN = 3
BULLET_MAX = 5
BULLET_DEFAULT = 4
BULLET_WORD_MIN = 2
BULLET_WORD_MAX = 6

# Narration length is derived from the scene's ROLE, because audio is the clock: a scene
# lasts exactly as long as its narration takes to speak, so a word budget IS a duration.
#
# 135 wpm is `docs/DIRECTION.md` §5. Measured narration on real scripts ran 120.8-150.2
# wpm; the defect that rate spec fixes is not the average but the 24% swing WITHIN one
# video, and the only honest lever on it is the word count per scene — changing the TTS
# rate changes the timbre.
WORDS_PER_MINUTE = 135.0

# (min, target, max) narration words per role. Straight from `docs/DIRECTION.md` §1.2,
# which is the authority on pacing; `test_word_budgets_match_the_role_durations` checks they
# stay consistent with `SceneRole.target_duration` at `WORDS_PER_MINUTE`.
#
# `target` is the number the prompt leads with, because a model given a range writes to its
# floor. The range exists to bound the damage, not to be aimed at.
ROLE_NARRATION_WORDS: dict[SceneRole, tuple[int, int, int]] = {
    SceneRole.TITLE: (9, 10, 14),
    SceneRole.CONTENT: (25, 34, 43),
    SceneRole.SUMMARY: (20, 27, 31),
    SceneRole.CLOSING: (13, 17, 20),
}

# Effective words per minute per language — `docs/LANGUAGES.md` §6.2, which is the authority
# on every number in this block. English stays at DIRECTION §5's 135.
#
# These are NOT independent measurements of "how fast is Spanish". LANGUAGES.md §6.3 is
# emphatic that the dominant variable is not the language at all: measured words-per-second
# ranged 2.40-3.80 across its experiments, a 58% spread, "driven almost entirely by how many
# sentence-final pauses the model wrote". A first pass at this file measured en at 114 wpm and
# hi at 184 wpm and derived factors of 1.41/1.61 from them — numbers that were really
# measuring sentence length, because each language's sample had a different sentence count.
# §6.2 holds sentence count and structure FIXED across the three translations, so only its
# RATIOS are trustworthy, and they are far smaller: es 1.04, hi 1.15.
LANGUAGE_WPM: dict[Language, float] = {
    Language.EN: WORDS_PER_MINUTE,  # 135, DIRECTION §5
    Language.ES: 140.0,
    Language.HI: 155.0,
}

# Words-per-second ratio to English, `docs/LANGUAGES.md` §6.2. Kept for provenance and for
# callers that want to explain a budget; the budgets themselves are the explicit table below.
LANGUAGE_WORD_FACTOR: dict[Language, float] = {
    Language.EN: 1.0,
    Language.ES: 1.04,
    Language.HI: 1.15,
}

# (min, target, max) narration words per role PER LANGUAGE — `docs/LANGUAGES.md` §6.2,
# transcribed rather than recomputed from `LANGUAGE_WORD_FACTOR`. Deliberately: the doc's
# table is the authority, and multiplying-then-rounding does not reproduce it exactly (43 x
# 1.15 = 49.45, which rounds to 49 where the doc says 50). Encoding the arithmetic instead of
# the answer would silently disagree with the document this is supposed to implement.
#
# A budget is a DURATION, which is why it scales at all — and the scaling is much gentler than
# the translation expansion. §6.3: Spanish needs 1.12x the words to say the same thing but
# only gets 1.04x the budget, so a Spanish slide carries about 93% of the English slide's
# information and a Hindi one about 90%. That is a real teaching-content loss with no lever on
# it that does not break the duration clamps; the doc's advice is to accept it or add a scene.
ROLE_NARRATION_WORDS_BY_LANGUAGE: dict[Language, dict[SceneRole, tuple[int, int, int]]] = {
    Language.EN: ROLE_NARRATION_WORDS,
    Language.ES: {
        SceneRole.TITLE: (9, 10, 15),
        SceneRole.CONTENT: (26, 35, 45),
        SceneRole.SUMMARY: (21, 28, 32),
        SceneRole.CLOSING: (14, 18, 21),
    },
    Language.HI: {
        SceneRole.TITLE: (10, 12, 16),
        SceneRole.CONTENT: (29, 39, 50),
        SceneRole.SUMMARY: (23, 31, 36),
        SceneRole.CLOSING: (15, 20, 23),
    },
}

# Sentences per scene, by role. `docs/LANGUAGES.md` §6.3: "The prompt must constrain sentences
# per scene, or the word budget means nothing" — the same 34-word budget produced 20.2-23.5s
# of staccato English but 12.8-13.0s of Hindi, purely on sentence count. A word budget with no
# sentence budget is not a duration.
#
# Applied to the NON-ENGLISH prompts only. Not timidity: §6.3 finds English is "the language
# currently out of spec" (34-word content scenes running 20-23s against a 19.0s max), so
# English needs this too — but that is a change to English pacing, which is DIRECTION §5's
# territory and is measured by other tests and other agents. Adding it here would silently
# retune English inside a language change. Flagged for whoever owns English pacing instead.
ROLE_SENTENCES: dict[SceneRole, str] = {
    SceneRole.TITLE: "exactly 1 sentence",
    SceneRole.CONTENT: "3 sentences, not four short ones and not one long one",
    SceneRole.SUMMARY: "2 or 3 sentences",
    SceneRole.CLOSING: "2 sentences",
}

# Fewest scenes that earn a recap — `docs/DIRECTION.md` §1.1, which is explicit that a
# six-slide video does not get one either: below ~100s there is nothing to recap, the
# closing already restates the key point, and a summary costs a teaching slide to buy.
SUMMARY_FROM_SLIDES = 7

# Fewest scenes that can carry a shape at all. DIRECTION §1.1 sets the real floor at four
# ("below that there is no room for a shape"), but the API accepts two, so a request below
# the floor degrades instead of failing a job that was legal to submit: the ending is kept
# and the opener is what gives. A video that stops dead on its last content slide is the
# defect being fixed here; a missing title card is only a missed opportunity.
STRUCTURE_FROM_SLIDES = 3

# On-screen copy caps — `docs/DIRECTION.md` §2.1/§2.2, sized from the 1080p grid so that a
# heading and a bullet each fit on ONE line. A wrapped heading moves the first bullet's
# baseline, and a bullet stack that starts at a different height on every slide is a large
# part of what "all over the place" means.
HEADING_CHAR_MAX = 22
BULLET_CHAR_MAX = 34
TITLE_CHAR_MAX = 50

# Audience register. Each clause changes WHAT the script says, not just its adjectives:
# what may be assumed known, what gets defined, and which consequence is named. Injected
# as its own paragraph; an unknown or absent tone injects nothing at all, leaving the
# prompt byte-identical to the untoned version.
TONE_CLAUSES: dict[str, str] = {
    "new_hires": """AUDIENCE — brand-new hires in their first days. Assume NO prior
context: no familiarity with internal tools, team names, systems or jargon. The first time
any term of art appears, define it in plain words in that same sentence. Say why the thing
matters before saying what to do about it, and name who to ask when unsure. The voice is
warm and encouraging — not knowing this yet is normal, and asking is the right move.
Never write "as you know", "remember to", or "simply".""",
    "all_staff": """AUDIENCE — the entire company, every role and every level of technical
comfort. Use plain everyday language and zero jargon: no acronyms, no product names, no
security or engineering vocabulary. Every scene must centre on what the viewer should DO:
a concrete observable action they can take today at their own desk, phrased as an
instruction. Skip mechanism and background; if a sentence explains how something works
rather than what to do about it, cut it and spend the words on another action.""",
    "technical": """AUDIENCE — technical staff who already know the fundamentals. Use
precise correct terminology and do not define it. Explain the MECHANISM: how the thing
actually works, the specific protocol, header, field, control or failure mode involved,
and what an attacker or system does step by step. Prefer specifics — exact names, values,
sequences — over analogies, and never substitute a metaphor for the real detail. Skip
motivational framing; assume the audience is already bought in.""",
    "executives": """AUDIENCE — senior executives with minutes, not hours. Lead with
business impact: financial exposure, operational and regulatory risk, cost of an incident
against cost of the control. Quantify wherever the topic allows — losses, time, volume,
likelihood — and name the decision or resource commitment being asked for. Skip
step-by-step procedure and tooling detail; that is someone else's job. Be brief and
declarative, with no hedging and no preamble.""",
}

# A sentence is only worth cutting in two if each half still holds a real fragment plus
# the function words around it. Below this, splitting produces scraps like "Check The".
_SPLITTABLE_WORDS = 10

# What the model should call the language it is writing. Named in the language's own terms as
# well as in English: "write in Spanish" is followed more reliably when the target name is
# also present in the target language.
LANGUAGE_NAMES: dict[Language, str] = {
    Language.EN: "English",
    Language.ES: "Spanish (español), neutral Latin American",
    Language.HI: "Hindi (हिन्दी), in the Devanagari script",
}

# Per-language copy conventions. ONLY the conventions that actually differ — the shared rules
# (sentence case over Title Case, no terminal punctuation on a heading) stay in the main
# template, so a language clause is never a restatement of the whole brief.
#
# These are typography rules, and imposing English typography on another script is a visible
# defect: a Devanagari sentence ending in "." reads as a missing danda to anyone who reads
# Hindi, and Spanish with no opening ¿ reads as machine output.
LANGUAGE_CLAUSES: dict[Language, str] = {
    Language.EN: "",
    Language.ES: """SPANISH CONVENTIONS — write correct, fully accented Spanish
orthography. Every diacritic that belongs on a word must be there: acentos (á é í ó ú),
la eñe (ñ), la diéresis (ü). Unaccented Spanish reads as machine output, and dropping an
accent changes the word — "está" is not "esta", "sí" is not "si". Any question or
exclamation is opened AND closed with its inverted mark: "¿Reconoce el remitente?",
"¡No haga clic!" — never only the closing one. Use "usted" throughout and keep it
consistent; never mix it with "tú" in the same video. Sentence case means Spanish sentence
case, so do NOT capitalise nouns, adjectives of nationality, days, months or languages the
way English does. Prefer the plain Spanish term over an English loanword when one exists
("correo electrónico", "contraseña", "enlace"), but keep an established technical term in
English where that is what people actually say ("phishing", "malware").""",
    Language.HI: """HINDI CONVENTIONS — write natural spoken Hindi in the Devanagari
script, the register of a professional presenter: everyday Hindi, NOT literary or heavily
Sanskritised vocabulary. Where a technical term is normally said in English, write it in
Devanagari transliteration as people actually say it (फ़िशिंग, ईमेल, पासवर्ड, लिंक, डोमेन)
rather than coining an obscure Hindi equivalent nobody would use out loud. End every
sentence with the danda "।" — NOT a full stop "." — and use "?" and "!" only where the
sentence really is a question or an exclamation. Devanagari has no upper and lower case, so
there is nothing to capitalise: ignore every instruction about sentence case and Title
Case, and do not romanise anything to work around it. Use the respectful "आप" form
throughout. Spell out digits as Devanagari words, never as numerals. Take care with nukta
and matra: फ़िशिंग not फिशिग.""",
}

# How the anchor rule is qualified for the language's grammar. English barely needs one;
# an inflected language needs telling that the ECHO must be the inflected form the narration
# itself used, because agreement changes silently break a verbatim match.
LANGUAGE_ANCHOR_NOTES: dict[Language, str] = {
    Language.EN: "",
    Language.ES: """
    Spanish inflects, and this rule is about SURFACE FORM, not meaning. Copy the phrase
    across in the EXACT form the narration used it in — same gender, same number, same
    person, same tense, same article, same enclitic pronoun. If the narration says
    "verifique el dominio del remitente", the bullet is "El dominio del remitente" or
    "Verifique el dominio" — NOT "verificar los dominios", which is the same idea in a form
    that no longer matches a single word of what is spoken. Do not switch an infinitive for
    an imperative, a singular for a plural, or "el" for "los" on the way to the slide.""",
    Language.HI: """
    Hindi inflects heavily and this rule is about SURFACE FORM, not meaning. Copy the run of
    words across EXACTLY as the narration has them — same postposition (का, के, की, को, से,
    में), same gender and number agreement, same verb form. Do not convert जांचें to जांचना,
    do not drop or change a postposition, and do not switch the oblique form for the direct
    one. If the narration says "भेजने वाले का डोमेन जांचें", the bullet is
    "भेजने वाले का डोमेन" — the same words, untouched.""",
}

# Cycled to break up runs of identical camera moves.
MOTION_CYCLE: tuple[Motion, ...] = (
    Motion.ZOOM_IN,
    Motion.PAN_RIGHT,
    Motion.ZOOM_OUT,
    Motion.PAN_LEFT,
    Motion.STATIC,
)

# One paragraph per role, injected only for the roles the plan actually uses. A description
# of the summary role in a five-slide prompt is an invitation to write one.
ROLE_CLAUSES: dict[SceneRole, str] = {
    SceneRole.TITLE: (
        "title — the OPENING CARD, and the shortest scene in the video. It renders the\n"
        "video's title at the largest type size used anywhere, so its heading is the title\n"
        "itself: repeat the top-level title verbatim, at most {title_chars} characters. Its\n"
        "narration is ONE short sentence naming the subject and the payoff — nothing else.\n"
        "No teaching, no advice, no statistics, no \"in this video we will explore the many\n"
        "ways in which\", no welcome speech. ZERO bullets: return an empty bullets array.\n"
        "This card is on screen for about four and a half seconds, so a narration longer\n"
        "than its word budget simply cannot be spoken over it."
    ),
    SceneRole.CONTENT: (
        "content — the TEACHING BODY, and the only scenes allowed to introduce new\n"
        "material. Each makes ONE point and makes it concretely: name the thing, name the\n"
        "action, name the consequence. EXACTLY {bullet_count} short on-screen points each —\n"
        "count them before you answer: {fewer} is too few and {more} is too many."
    ),
    SceneRole.SUMMARY: (
        "summary — the RECAP. It reviews points the content scenes have ALREADY made and\n"
        "introduces NOTHING new: no new advice, no new term, no new example, no new number.\n"
        "Name those points again in the order they were taught and, crucially, in the SAME\n"
        "WORDS the content scenes used — this scene's bullets have to quote this scene's own\n"
        "narration, so reuse the original phrasing rather than inventing a synonym for it.\n"
        "At most {summary_bullets} on-screen points."
    ),
    SceneRole.CLOSING: (
        "closing — the ENDING. It says what the viewer should DO next: the one action to\n"
        "take, where to go, or who to ask. Then it lands — the final sentence must sound\n"
        "like a conclusion, not a thought that trails off, and must not open a new topic.\n"
        "Its heading is an instruction, not a topic: \"If in doubt, report it\", never\n"
        "\"Summary and next steps\". Every one of its {closing_bullets} points is an\n"
        "imperative, and it teaches nothing new."
    ),
}

PROMPT_TEMPLATE = """You are writing a short narrated corporate-training video about: {topic}

Produce EXACTLY {slide_count} scenes. Return JSON matching the provided schema.
{language_block}
The register throughout is corporate training: concrete, actionable, specific. Say "check
the sender domain", never "be careful". Name the thing, name the action.

title — the whole video's title, at most {title_chars} characters. It is set in the largest
type in the video and must fit two lines, so a long subtitle after a colon will not fit.
{tone_clause}
STRUCTURE — a training video has a shape, and each scene has a different job. Every scene
below is listed with its role, its narration word budget and its bullet count. These are
not suggestions. The narration is read aloud at about {wpm} words per minute and each scene
stays on screen for exactly as long as its own narration takes to speak, so the word budget
IS the scene length: a title card written like a content scene runs for twenty seconds and
reads as a stall. Write each narration to the TARGET word count — the bracketed range is
the tolerance, not the goal.

{structure_block}

{role_clauses}
For every scene:

role — copy the role this scene number is given in the STRUCTURE list above, exactly.

heading — at most {heading_chars} characters, INCLUDING spaces. It is rendered on one line
and is not allowed to wrap: a heading that wraps moves the first bullet down the slide, so
the bullets sit at a different height on every scene. Count the characters. No terminal
punctuation.

narration — complete spoken sentences, at THIS scene's target word count from the STRUCTURE
list. Count the words before you answer. This text is fed straight to a text-to-speech
engine, so it must sound natural read aloud by a human presenter. Write flowing prose.
Absolutely no bullet fragments, no headings, no markdown, no asterisks, no emoji, no
parentheticals, no abbreviations a narrator would stumble over, and never read a list
aloud. Spell out numbers and units the way a person would say them.

bullets — exactly as many short on-screen points as the STRUCTURE list gives this scene,
and zero when it says zero. Each is a FRAGMENT of {word_min} to {word_max} words,
at most {bullet_chars} characters, on one line, with NO terminal period, no markdown, no
emoji, and no leading dash, asterisk, number or bullet glyph — the renderer draws the
glyph itself, so any character you add appears twice on screen.

    Sentence case, NOT Title Case: "Check the sender domain", never "Check The Sender
    Domain". Capitalised articles read as a template artefact. Keep one grammatical form
    per scene — either every point on that slide is an imperative, or every point is a
    noun phrase, never a mix. And never lift a fragment that inverts its meaning out of
    context: a phrase from a sentence about what an ATTACKER does reads on screen as an
    instruction to the viewer.

    CRITICAL: each bullet must reuse 2 or more CONSECUTIVE CONTENT WORDS from that same
    scene's narration, verbatim. Each bullet is revealed on screen at the exact moment
    the narrator speaks its words, and the reveal time is computed by searching the
    narration for the bullet's wording. A bullet that paraphrases instead of quoting has
    nothing to match and will be mistimed. So if the narration says "hover over the link
    to reveal the real destination", the bullet is "Hover Over The Link" or "Reveal The
    Real Destination" — not "Link Safety".{anchor_note}

    List the bullets in the SAME ORDER the narration mentions them, first to last. Write
    each narration so it naturally contains that scene's number of quotable phrases, spread
    evenly across its whole length rather than clustered in the first sentence: every point
    is revealed as its own phrase is spoken, so the narration needs room for that many
    separate moments. This applies to the recap and the ending as much as to the body.

image_prompt —{visual_en} describe ONE photographic or cinematic BACKGROUND image for this scene.
Name the subject, the setting, the lens or framing, the lighting, and the mood. It must
be a real-looking photograph, not an illustration, diagram, chart, or infographic.

    The image must stay visually anchored to the VIDEO'S TOPIC — "{topic}" — and not
    merely to this scene's heading. Every image_prompt must name a concrete subject from
    that topic's own domain: the actual objects, tools, places, materials or people the
    topic is about, doing the thing the scene describes. A scene about habits or takeaways
    is still a scene about "{topic}", so it still shows that topic's subject matter.
    NEVER produce a generic office, meeting room, lobby, atrium, handshake, sunset,
    landscape, plant, nature or abstract-texture scene; those read as unrelated stock
    filler and break the video. If a scene feels abstract, pick a physical detail from the
    topic and shoot that close up.

Compose it with generous open space — plain sky, empty wall, shallow-focus
foreground — in the lower third where a text caption will be overlaid. End every
image_prompt with this exact sentence: "No text, no letters, no words, no numbers, no
labels, no signage, no watermarks anywhere in the image." Image models render lettering
as garbled nonsense, so any request for readable text ruins the frame.

clip_prompt —{visual_en} the SAME shot as image_prompt, described as MOTION instead of composition,
for a video model that will animate this scene. One or two sentences, present tense. Name
the camera move (slow push in, handheld follow, tilt up, orbit left, locked-off static) and
name what physically happens in frame (a hand reaching for the handset, a cursor sliding
over a link, steam lifting off the weld, a queue shuffling forward). It must show the same
subject as image_prompt — this is that photograph moving, not a different shot. No cuts, no
camera flashes, no dialogue, no on-screen speaker, and no lettering or captions in frame.

motion — the camera move over the still. Vary it across the video so that no two
consecutive scenes use the same value. Choose the move that suits the shot: zoom_in to
build toward a detail, zoom_out to reveal scale, pan_left or pan_right across a wide
scene, static for a portrait or a moment that should feel still.

Number the scene ids 1 through {slide_count} in order. The scenes must tell one
continuous story with no repeated facts between them.

Before you answer, check every scene against the STRUCTURE list one more time: its role,
its bullet count, and above all its narration word count. A scene that overruns its word
budget makes the finished video the wrong shape, and that is the single most common way
this task is failed.{language_reminder}"""


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

    def generate(
        self,
        topic: str,
        slide_count: int,
        *,
        bullets_per_slide: int = BULLET_DEFAULT,
        tone: str | None = None,
        language: Language = Language.EN,
    ) -> Script:
        """See `ScriptProvider.generate`.

        `bullets_per_slide` sets the on-screen point budget for CONTENT scenes; the title,
        summary and closing take theirs from `SceneRole.bullet_budget` instead. Both the
        per-role point counts and the per-role narration word budgets reach the model in
        the prompt and are re-imposed on the way back, so the returned scenes are correctly
        shaped even when the model miscounts. `tone` selects an audience clause; an
        unrecognised value is ignored.

        `language` decides what the VIEWER reads and hears — narration, heading and bullets
        are written natively in it, while `image_prompt` and `clip_prompt` stay English for
        the English-conditioned image and video models. It also rescales every word budget by
        the language's measured speaking rate, so the scene still lands inside its role's
        duration window instead of inside English's word count. Keyword-only with a default,
        so `ScriptProvider` conformance and every existing caller are unaffected; an
        unrecognised value degrades to English rather than failing the job.
        """
        if slide_count < 1:
            raise ValueError("slide_count must be at least 1")

        language = coerce_language(language)
        bullet_count = _bullet_target(bullets_per_slide)

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
                            "text": _build_prompt(
                                topic=topic.strip(),
                                slide_count=slide_count,
                                bullet_count=bullet_count,
                                tone=tone,
                                language=language,
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

        scenes: list[SceneScript] = []
        for index, item in enumerate(payload.get("scenes") or []):
            narration = _clean_narration(str(item.get("narration", "")))
            scenes.append(
                StructuredSceneScript(
                    id=int(item.get("id") or index + 1),
                    role=_coerce_role(item.get("role")),
                    narration=narration,
                    heading=_clean_heading(str(item.get("heading", "")), language),
                    # Cleaned against the CONTENT budget here and re-cut to the role's own
                    # budget in `_apply_roles`: the role a scene ends up with depends on its
                    # final position, which `_fit_scene_count` may still change.
                    bullets=_clean_bullets(
                        item.get("bullets"), narration, bullet_count, language=language
                    ),
                    image_prompt=str(item.get("image_prompt", "")).strip(),
                    clip_prompt=_clean_narration(str(item.get("clip_prompt", ""))) or None,
                    motion=_coerce_motion(item.get("motion"), index),
                )
            )
        if not scenes:
            raise GeminiError("model returned zero scenes")

        scenes = _fit_scene_count(scenes, slide_count, bullet_count, language)
        scenes = _renumber(scenes)
        scenes = _vary_motion(scenes)
        scenes = _apply_roles(scenes, bullet_count, language)

        title = _clean_heading(str(payload.get("title") or topic), language) or topic.strip()
        return Script(topic=topic.strip(), title=title, scenes=scenes)


class VerbatimScriptProvider:
    """Satisfies `ScriptProvider` without a network call — the script is already written.

    `topic` is carried through for provenance and `slide_count` decides how many pieces
    the text is cut into. Splits on sentence boundaries and balances words per scene, so
    narration timing stays even; falls back to clause and word chunking when there are
    fewer sentences than requested scenes.

    ASYMMETRY, deliberate: `bullets_per_slide` is honoured — it is how many fragments each
    segment is cut into — but `tone` is IGNORED, and that is not an oversight. Tone can
    only be expressed by choosing different words, and this provider writes none: every
    narration word is the user's own and every bullet is a verbatim run of it. Rewriting
    the text to sound executive would break the one promise the verbatim path makes. A
    caller who wants a register change wants `GeminiScriptProvider`.

    The same promise is why every scene here is `SceneRole.CONTENT`. Roles carry duration
    targets — a title card is 3-6s, a closing 5-10s — and this provider cannot resize a
    segment to fit one without cutting or padding the user's own words. Labelling a
    forty-word slice "title" would tell the renderer to treat a twenty-second scene as a
    four-second card. Structure is something the writer put in the text, or did not.
    """

    def __init__(self, script_text: str, *, title: str | None = None) -> None:
        if not script_text or not script_text.strip():
            raise ValueError("script_text is empty — nothing to narrate")
        self.script_text = script_text.strip()
        self.title = title

    def generate(
        self,
        topic: str,
        slide_count: int,
        *,
        bullets_per_slide: int = BULLET_DEFAULT,
        tone: str | None = None,
        language: Language = Language.EN,
    ) -> Script:
        """See `ScriptProvider.generate`. `tone` is accepted and ignored — see the class
        docstring for why; `bullets_per_slide` sets how many fragments each segment yields,
        subject to there being enough source text to cut without producing scraps.

        `language` is accepted and used, but ONLY for typography and tokenisation — it names
        what script the user's own text is already in, and never causes a single word of it to
        change. So a pasted Hindi script gets its sentences split on the danda instead of on a
        full stop that is not there, its headings picked using Hindi function words rather
        than English ones, and its bullets stripped of a trailing danda; the words themselves
        stay exactly as written. Passing `Language.HI` here does NOT translate English text —
        that would break the one promise this provider makes.
        """
        if slide_count < 1:
            raise ValueError("slide_count must be at least 1")

        language = coerce_language(language)
        bullet_count = _bullet_target(bullets_per_slide)
        segments = _split_into_segments(self.script_text, slide_count)
        if len(segments) < slide_count:
            raise ValueError(
                f"script has only {len(self.script_text.split())} words — cannot be split "
                f"into {slide_count} scenes (got {len(segments)})"
            )
        scenes: list[SceneScript] = [
            StructuredSceneScript(
                id=index + 1,
                role=SceneRole.CONTENT,
                narration=segment,
                heading=_heading_from(
                    segment, fallback=f"Part {index + 1}", language=language
                ),
                bullets=_bullets_from(segment, bullet_count, language),
                # English regardless of `language`: these condition the image and video
                # models, and are never read or heard by a viewer.
                image_prompt=_image_prompt_from(segment, topic, language),
                clip_prompt=_clip_prompt_from(segment, topic, language),
                motion=MOTION_CYCLE[index % len(MOTION_CYCLE)],
            )
            for index, segment in enumerate(segments)
        ]
        title = (
            self.title or _heading_from(self.script_text, fallback=topic, language=language)
        ).strip()
        return Script(topic=topic.strip(), title=title or topic.strip(), scenes=scenes)


# --------------------------------------------------------------------------- helpers

_MARKDOWN = re.compile(r"[*_`#>\[\]]|~~")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️]"
)
# The danda is in the class alongside ".!?" because it IS the Hindi full stop: without it a
# whole Devanagari narration is one unsplittable sentence, and every path here that works by
# cutting on sentence boundaries — segment splitting, fallback bullets — silently gives up.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])[\"'’)\]]*\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
# Leading list decoration the renderer draws itself: "- ", "1. ", "•", "a)".
_BULLET_GLYPH = re.compile(r"^(?:[-•·–—+]+|\d+[.)]|[A-Za-z][.)])\s*")

_STOPWORDS = frozenset(
    """a an the and or but of to in on at for with from by as is are was were be been
    being it its this that these those they them their we our you your he she his her
    i me my not no so if then than there here what which who whom when where why how
    all any both each few more most other some such only own same too very can will
    just do does did done have has had also into over under about after before while
    because through during above below up down out off again once""".split()
)

# Function words per language, for the deterministic fallbacks (`_fragment_from`,
# `_heading_from`) that pick a scene's own content words when the model under-delivers.
# English stopwords applied to Spanish would treat "el dominio del remitente" as four content
# words and open a bullet on "el", which reads as a template artefact in any language.
_STOPWORDS_BY_LANGUAGE: dict[Language, frozenset[str]] = {
    Language.EN: _STOPWORDS,
    Language.ES: frozenset(
        """el la los las un una unos unas lo al del de a en y o u pero que se su sus
        es son era eran ser sido estar esta este esto estos estas ese esa eso esos esas
        aquel aquella con por para sin sobre entre desde hasta como cuando donde porque
        si no ni ya muy mas más menos todo toda todos todas otro otra cada mismo misma
        yo tu tú él ella usted ustedes nosotros nosotras ellos ellas me te le les nos
        mi mis nuestro nuestra vuestro su hay ha han he has hemos habia había puede
        pueden debe deben tiene tienen tener hacer haga hagan""".split()
    ),
    Language.HI: frozenset(
        """का के की को से में पर और या तो ही भी है हैं था थे थी हूँ हो होता होती होते
        होना करना करें कर किया किए गई गए गया जाता जाती जाते यह वह ये वे इस उस इन उन
        जो कि अगर तब जब कहाँ क्या कौन कैसे क्यों नहीं ना न एक कोई सब सभी अपना अपनी अपने
        मैं मुझे मेरा हम हमें हमारा आप आपका आपको आपके आपकी वो उसका उसके उसकी लिए साथ
        तक बाद पहले फिर ओर वाला वाले वाली रहा रहे रही सकता सकते सकती चाहिए""".split()
    ),
}

# Characters that hold a word together: letters, digits, COMBINING MARKS, and the internal
# apostrophe or hyphen the old ASCII pattern allowed.
#
# The mark category is the whole point and the easy thing to get wrong. Python's `\w` covers
# Unicode categories L* and N* but NOT Mn/Mc — and Devanagari vowel signs, the anusvara and
# the nukta are all marks. So a `[^\W_]+` "Unicode-aware" pattern silently DROPS them:
# "जांचें" comes back as "जच", which is the same corruption `Language.needs_shaping` warns
# about, just at the tokeniser instead of the renderer. Categories are tested directly
# instead, which needs no per-script character ranges and cannot miss one.
_WORD_MARKS = frozenset("'’-")


def _words(text: str) -> list[str]:
    """Word tokens of `text`, Unicode-aware and mark-preserving.

    Replaces `re.findall(r"[A-Za-z0-9'-]+", ...)`, which was ASCII-only: it cut "protección"
    into "protecci" + "n" and matched NOTHING at all in Devanagari, so every deterministic
    fallback in this file returned an empty list for Hindi. Byte-identical to the old pattern
    on ASCII input — verified by differential test — while admitting accents, eñes, and
    Devanagari consonants with their matras and nuktas attached.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char in _WORD_MARKS or unicodedata.category(char)[0] in {"L", "N", "M"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens

# Devanagari sentence terminator. A Hindi heading must not end in one, and a Hindi sentence
# must not end in a full stop — imposing "." on Devanagari is a visible defect, not a nit.
DANDA = "।"
DOUBLE_DANDA = "॥"


def _bullet_target(bullets_per_slide: Any) -> int:
    """The point budget actually used, clamped into the legible range.

    Defensive on purpose: the API validates 3-5, but this provider is also driven from
    scripts and tests, and a bullet count of 0 or 40 would silently produce an empty or
    unreadable panel rather than an error anyone would notice.
    """
    try:
        value = int(bullets_per_slide)
    except (TypeError, ValueError):
        return BULLET_DEFAULT
    return max(BULLET_MIN, min(BULLET_MAX, value))


def _exact_bullet_target(value: Any) -> int:
    """A role-derived point count: taken as given, only capped at what fits on screen.

    The difference from `_bullet_target` is the floor. A caller's `bullets_per_slide` is
    clamped up to `BULLET_MIN` because three points is where a content panel stops looking
    empty; a role's budget of zero or two is the design, not an under-supplied request.
    """
    try:
        return max(0, min(BULLET_MAX, int(value)))
    except (TypeError, ValueError):
        return BULLET_DEFAULT


def role_bullet_target(role: SceneRole, bullets_per_slide: Any = BULLET_DEFAULT) -> int:
    """Points this scene may show: the caller's budget, capped by the role's.

    The caller sizes the CONTENT slides; the title, summary and closing are sized by what
    they are for. `SceneRole.bullet_budget` is a ceiling, so asking for five points still
    leaves the title with none and the closing with two.
    """
    return max(0, min(_bullet_target(bullets_per_slide), role.bullet_budget))


def coerce_language(value: Any) -> Language:
    """Read a language from an enum, a code, or junk. Never raises — English is the floor.

    The API layer will hand this whatever arrives in a request body, and a bad language code
    is not worth failing a job over when the alternative is a perfectly good English script.
    """
    if isinstance(value, Language):
        return value
    if isinstance(value, str):
        try:
            return Language(value.strip().lower())
        except ValueError:
            logger.info("unknown language %r; writing in English", value)
    return Language.EN


def language_word_factor(language: Language = Language.EN) -> float:
    """Multiplier on the English word budgets for `language`. See `LANGUAGE_WORD_FACTOR`."""
    return LANGUAGE_WORD_FACTOR.get(language, 1.0)


def language_wpm(language: Language = Language.EN) -> float:
    """Planning pace for `language`: the rate at which its scaled budget hits the role window.

    Anchored on the English spec rate rather than on `MEASURED_WPM[EN]`, so English stays
    exactly 135 wpm and every other language is expressed relative to it.
    """
    return WORDS_PER_MINUTE * language_word_factor(language)


def narration_words(role: SceneRole, language: Language = Language.EN) -> tuple[int, int, int]:
    """(min, target, max) narration words for `role`. Audio is the clock, so this is the
    only lever the script has on how long a scene lasts.

    Scaled by `language`, because the budget is a duration: the same 11-19s content window
    buys 25-43 English words but 35-61 Spanish ones. `language` is positional-with-default so
    every existing caller — and every published number in `docs/DIRECTION.md` §1.2 — is
    untouched.

    Notably it does NOT vary with `bullets_per_slide`. A content scene is 11-19s whatever it
    carries, so its word budget is fixed; asking for one fewer point buys a slower reveal,
    not a shorter scene.
    """
    low, target, high = ROLE_NARRATION_WORDS[role]
    factor = language_word_factor(language)
    if factor == 1.0:
        return low, target, high
    # Rounded independently then re-ordered: rounding three numbers can in principle
    # collapse or invert the range, and a min above its max would be sent to the model.
    scaled = sorted(max(1, round(value * factor)) for value in (low, target, high))
    return scaled[0], scaled[1], scaled[2]


def words_spoken_in(seconds: float, language: Language = Language.EN) -> float:
    """Words a narrator gets through in `seconds`, at `language`'s planning pace."""
    return seconds * language_wpm(language) / 60.0


def anchoring_supported(language: Language) -> bool:
    """Whether the verbatim-anchor invariant can hold for `language` AT ALL.

    PROBED, not declared. `bullet_timing` reaches a bullet's anchor through
    `deepgram_align.normalize`, which keeps only `[a-z0-9]`; a script whose characters all
    fall outside that range normalises to nothing, so no bullet in it can ever match its own
    narration. Rather than hardcode "Hindi cannot anchor" — which would be a lie the day
    `normalize` is widened to Unicode — this asks the real matcher a question it cannot get
    wrong: does a phrase lifted verbatim out of a sample find itself?

    False means bullet reveals fall back to proportional placement for that language. It is a
    product constraint to be reported, not a condition this file can repair: the fix lives in
    `deepgram_align.normalize`, which this provider only consumes.
    """
    sample = _ANCHOR_PROBE.get(language)
    if sample is None:
        return True
    narration, bullet = sample
    return anchor_position(bullet, narration) is not None


# One (narration, bullet) pair per language where the bullet is a verbatim run of the
# narration. If the matcher cannot find it, nothing in that script will ever anchor.
_ANCHOR_PROBE: dict[Language, tuple[str, str]] = {
    Language.EN: ("Hover over the link to reveal the real destination.", "reveal the real"),
    Language.ES: ("Verifique el dominio del remitente con atención.", "dominio del remitente"),
    Language.HI: ("भेजने वाले का डोमेन ध्यान से जांचें।", "वाले का डोमेन"),
}


def role_plan(slide_count: int) -> list[SceneRole]:
    """The role of each scene, by position. This is the video's shape.

    title, then the content body, then the ending — with a recap inserted before the ending
    once the body is long enough to be worth recapping (`SUMMARY_FROM_SLIDES`).

    Below `STRUCTURE_FROM_SLIDES` there is no room for all three, so the opener is what
    gives: a two-scene video that teaches and then closes says more than one that announces
    itself and stops. A single scene is not a video with a shape at all.
    """
    count = max(1, slide_count)
    if count == 1:
        return [SceneRole.CONTENT]
    if count < STRUCTURE_FROM_SLIDES:
        return [SceneRole.CONTENT] * (count - 1) + [SceneRole.CLOSING]

    tail = [SceneRole.CLOSING]
    if count >= SUMMARY_FROM_SLIDES:
        tail.insert(0, SceneRole.SUMMARY)
    body = [SceneRole.CONTENT] * (count - 1 - len(tail))
    return [SceneRole.TITLE, *body, *tail]


def _coerce_role(value: Any) -> SceneRole:
    """Read the model's own role label. Position is authoritative, so this never raises."""
    if isinstance(value, SceneRole):
        return value
    if isinstance(value, str):
        try:
            return SceneRole(value.strip().lower())
        except ValueError:
            pass
    return SceneRole.CONTENT


def _apply_roles(
    scenes: list[SceneScript], bullet_count: int, language: Language = Language.EN
) -> list[SceneScript]:
    """Impose `role_plan` by position and re-cut each scene's bullets to its role's budget.

    Runs last, after `_fit_scene_count` has settled how many scenes there are: a scene's
    role is a function of where it ends up, not of what the model called it. The model is
    told the plan and usually complies, so the override is normally a no-op — a mismatch is
    logged because it means the prompt is losing an argument.
    """
    plan = role_plan(len(scenes))
    result: list[SceneScript] = []
    for scene, role in zip(scenes, plan, strict=True):
        if scene_role(scene) is not role:
            logger.info(
                "scene %d came back as %s; the plan puts %s there",
                scene.id,
                scene_role(scene).value,
                role.value,
            )
        target = role_bullet_target(role, bullet_count)
        bullets = list(scene.bullets)
        if len(bullets) > target:
            # TRIM, not re-derive. The parse pass already topped this scene up and may have
            # deliberately kept an unanchored point to avoid an empty-looking panel; a
            # second full clean would discard exactly that point and end up one short.
            bullets = _clean_bullets(bullets, scene.narration, target, language=language)
        result.append(
            _as_structured(scene).model_copy(update={"role": role, "bullets": bullets})
        )
    return result


def _as_structured(scene: SceneScript) -> StructuredSceneScript:
    """Widen a plain `SceneScript` without losing anything it already carries."""
    if isinstance(scene, StructuredSceneScript):
        return scene
    return StructuredSceneScript(**scene.model_dump())


def _tone_clause(tone: str | None) -> str:
    """The audience paragraph for `tone`, or "" for unknown/absent.

    Returned with its own surrounding blank lines so that an empty clause leaves the
    prompt exactly as it was before tones existed — no stray whitespace to change the
    model's behaviour on the untoned path.
    """
    if not tone:
        return ""
    clause = TONE_CLAUSES.get(str(tone).strip().lower())
    return f"\n{clause}\n" if clause else ""


def _language_block(language: Language) -> str:
    """The LANGUAGE paragraph, or "" for English.

    English injects NOTHING — the prompt for an English script is byte-identical to the one
    that existed before languages did. That is not tidiness, it is how the measured English
    result (22/22 bullets anchored, 0 fallbacks) is guaranteed not to move: there is no new
    text in the English prompt to move it.
    """
    if language is Language.EN:
        return ""
    name = LANGUAGE_NAMES[language]
    conventions = LANGUAGE_CLAUSES.get(language, "").strip()
    block = f"""
LANGUAGE — write this video in {name}.

Everything the VIEWER reads or hears is in {name}: the video's title, and every scene's
heading, every bullet and every word of narration. Compose it NATIVELY — think in the
language and write in it directly. Do NOT draft in English and translate: a translated
script scans as translated the moment it is read aloud, and its bullets stop quoting the
narration they are supposed to be lifted from, which breaks the on-screen timing.

ONE EXCEPTION, and only this one: image_prompt and clip_prompt stay in ENGLISH. No viewer
ever sees them — they are conditioning text for image and video models trained on English
captions, so writing them in {name} produces a worse frame and buys nothing at all. Every
other field is {name}; those two are English.
"""
    return f"{block}\n{conventions}\n" if conventions else block


def _language_reminder(language: Language) -> str:
    """Final-position language check. Empty for English, which needs no reminder."""
    if language is Language.EN:
        return ""
    name = LANGUAGE_NAMES[language]
    return (
        f"\n\nCheck the language too: narration, heading and bullets in {name} on every "
        f"single scene, with image_prompt and clip_prompt in English on every single scene. "
        f"A script half in English is worse than either language alone."
    )


def _anchor_note(language: Language) -> str:
    """The inflection qualifier on the anchor rule. Empty for English."""
    note = LANGUAGE_ANCHOR_NOTES.get(language, "").strip("\n").rstrip()
    return f"\n\n{note}" if note else ""


def _visual_english(language: Language) -> str:
    """The "this field stays English" flag, injected AT the two visual-prompt fields.

    Empty for English — where saying "in English" would be noise — and stated at the field
    rather than only in the LANGUAGE block up top, because that is where the model is
    actually deciding what to write and a rule six paragraphs away loses to the surrounding
    Spanish or Hindi.
    """
    if language is Language.EN:
        return ""
    return " IN ENGLISH, not in the language of the narration —"


def _structure_block(
    plan: list[SceneRole], bullet_count: int, language: Language = Language.EN
) -> str:
    """The scene-by-scene contract sent to the model: role, word budget, point count.

    Spelled out per scene rather than described in general terms. A model given "the first
    scene should be short" writes a nineteen-second opener; a model given "scene 1 — role:
    title — narration 7-14 words — 0 bullets" writes a title card.

    The word budgets are `language`'s, not English's — this is the one place the per-language
    factor actually reaches the model, so a Spanish scene is asked for Spanish-many words.
    """
    lines = []
    for index, role in enumerate(plan, start=1):
        bullets = role_bullet_target(role, bullet_count)
        low, target, high = narration_words(role, language)
        plural = "" if bullets == 1 else "s"
        lines.append(
            f"  scene {index} — role: {role.value} — narration {target} words "
            f"({low}-{high}) — {bullets} bullet{plural}"
        )
    return "\n".join(lines)


def _role_clauses(plan: list[SceneRole], bullet_count: int) -> str:
    """Descriptions of only the roles this video uses, in the order they first appear."""
    ordered = list(dict.fromkeys(plan))
    clauses = [
        ROLE_CLAUSES[role].format(
            bullet_count=role_bullet_target(SceneRole.CONTENT, bullet_count),
            fewer=role_bullet_target(SceneRole.CONTENT, bullet_count) - 1,
            more=role_bullet_target(SceneRole.CONTENT, bullet_count) + 1,
            summary_bullets=role_bullet_target(SceneRole.SUMMARY, bullet_count),
            closing_bullets=role_bullet_target(SceneRole.CLOSING, bullet_count),
            title_chars=TITLE_CHAR_MAX,
        )
        for role in ordered
    ]
    return "\n\n".join(clauses) + "\n"


def _build_prompt(
    *,
    topic: str,
    slide_count: int,
    bullet_count: int,
    tone: str | None,
    language: Language = Language.EN,
) -> str:
    """Fill the template. The per-scene structure block carries the pacing contract.

    For `Language.EN` every language-derived placeholder is empty and `wpm` is the spec's
    135, so the English prompt is byte-identical to the pre-language one.
    """
    plan = role_plan(slide_count)
    return PROMPT_TEMPLATE.format(
        topic=topic,
        slide_count=slide_count,
        word_min=BULLET_WORD_MIN,
        word_max=BULLET_WORD_MAX,
        heading_chars=HEADING_CHAR_MAX,
        bullet_chars=BULLET_CHAR_MAX,
        title_chars=TITLE_CHAR_MAX,
        wpm=int(language_wpm(language)),
        structure_block=_structure_block(plan, bullet_count, language),
        role_clauses=_role_clauses(plan, bullet_count),
        tone_clause=_tone_clause(tone),
        language_block=_language_block(language),
        language_reminder=_language_reminder(language),
        anchor_note=_anchor_note(language),
        visual_en=_visual_english(language),
    )


def _clean_narration(text: str) -> str:
    """Strip anything TTS would read aloud as noise."""
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    return re.sub(r"\s+", " ", text).strip()


def _clean_heading(text: str, language: Language = Language.EN) -> str:
    """One-line on-screen title: no markup, no terminal punctuation.

    The strip set is the language's own. A Devanagari heading ending in a danda needs the
    danda removed for exactly the reason an English one needs the full stop removed, and a
    Spanish heading keeps a leading "¿" or "¡" because only the TRAILING end is stripped —
    an opened question that never closes would be worse than the terminal mark.
    """
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    strip = f".:;,-{DANDA}{DOUBLE_DANDA} " if language is Language.HI else ".:;,- "
    return text.rstrip(strip).strip()


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


def _fit_scene_count(
    scenes: list[SceneScript],
    slide_count: int,
    bullet_count: int = BULLET_DEFAULT,
    language: Language = Language.EN,
) -> list[SceneScript]:
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
        # A bullet must stay with the half of the narration that actually says it,
        # otherwise its anchor phrase is in the other scene and it cannot be timed.
        head, tail = _partition_bullets(
            victim.bullets, halves[0], halves[1], bullet_count, language
        )
        scenes[target] = victim.model_copy(update={"narration": halves[0], "bullets": head})
        scenes.insert(
            target + 1,
            victim.model_copy(
                update={
                    "narration": halves[1],
                    "heading": _heading_from(
                        halves[1], fallback=victim.heading, language=language
                    ),
                    "bullets": tail,
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


def _clean_bullets(
    raw: Any,
    narration: str,
    target: int = BULLET_DEFAULT,
    *,
    language: Language = Language.EN,
) -> list[str]:
    """Normalise the model's bullets, then guarantee `target` of them where possible.

    `target` is the caller's `bullets_per_slide`, already clamped. It is enforced on the
    way back, not merely requested in the prompt: a model that returns six points for a
    three-point request would otherwise overflow the panel the user sized deliberately.

    Four defences, all of which fire in practice:
      * cosmetic — a leading dash or a trailing period would be drawn on top of the
        renderer's own glyph, so both are stripped;
      * anchoring — a bullet whose wording appears nowhere in its own narration cannot be
        timed against it, so it yields to a phrase taken from the narration itself. It is
        kept only when dropping it would leave fewer than `target` points;
      * shortfall — under-delivery is topped up from the narration for the same reason;
      * order — bullets are sorted by where their anchor phrase falls in the narration, so
        a model that lists them out of order still reveals them in spoken order.

    The top-up can fall short of `target` when the narration is too thin to yield another
    distinct phrase; showing fewer points beats showing scraps like "Check The".

    `target` is honoured EXACTLY rather than clamped into the 3-5 legible range, because it
    is now role-derived: a title card's zero and a closing's two are deliberate, and
    rounding them up to three is the "every scene looks the same" defect.

    LANGUAGE, and this is the subtle part: the anchoring and ordering defences BOTH depend on
    `anchor_position`, which cannot see a script `deepgram_align.normalize` reduces to nothing
    (Devanagari, today). In such a language every bullet looks unanchored, so applying the
    defences unchanged would demote every bullet the model wrote in favour of a mechanically
    sliced fragment — on the strength of a test that returned "no" to every question it was
    asked. When the check cannot run, it is not consulted: the model's own bullets are kept,
    in the model's own order, which is the best available guess at spoken order. That is a
    degradation to be reported, but a much smaller one than shredding good copy.
    """
    target = _exact_bullet_target(target)
    if target == 0:
        return []
    values = [raw] if isinstance(raw, str) else list(raw or [])
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_bullet(str(value), language)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)

    if not anchoring_supported(language):
        bullets = cleaned[:target]
        for candidate in _bullets_from(narration, target, language):
            if len(bullets) >= target:
                break
            if _is_redundant(candidate, bullets):
                continue
            bullets.append(candidate)
        return bullets

    bullets = [b for b in cleaned if anchor_position(b, narration) is not None][:target]
    spare = [b for b in cleaned if b not in bullets]

    for source in (_bullets_from(narration, target, language), spare):
        for candidate in source:
            if len(bullets) >= target:
                break
            if _is_redundant(candidate, bullets):
                continue
            bullets.append(candidate)

    return _order_by_narration(bullets, narration)


def _clean_bullet(text: str, language: Language = Language.EN) -> str:
    """One on-screen fragment: no markdown, no glyph, no terminal punctuation.

    Only the trailing end is stripped, which is what lets a Spanish fragment keep an opening
    "¿"/"¡" while still losing a trailing "." — and lets a Hindi fragment lose its danda.
    """
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    text = _BULLET_GLYPH.sub("", text).strip()
    strip = f".,;:!?-– {DANDA}{DOUBLE_DANDA}" if language is Language.HI else ".,;:!?-– "
    return text.rstrip(strip).strip()


def _order_by_narration(bullets: list[str], narration: str) -> list[str]:
    """Sort by first anchor occurrence; unanchored bullets keep their given slot."""
    if len(bullets) < 2:
        return bullets
    ranked: list[tuple[float, int]] = []
    previous = -1.0
    for index, bullet in enumerate(bullets):
        position = anchor_position(bullet, narration)
        key = float(position) if position is not None else previous + 0.5
        previous = key
        ranked.append((key, index))
    return [bullets[index] for _, index in sorted(ranked)]


def _bullets_from(
    text: str, target: int = BULLET_DEFAULT, language: Language = Language.EN
) -> list[str]:
    """Derive up to `target` on-screen points from the text that will be narrated.

    Every fragment is a contiguous run of the source's own words, so the anchor the timer
    looks for is present by construction. Used by the verbatim provider — which has no
    model to ask for a different number of points, so `bullets_per_slide` lands here as
    how finely the source text is cut — and as the fallback whenever the model
    under-delivers.
    """
    target = _bullet_target(target)
    units = [u.strip() for u in _SENTENCE_END.split(text) if u.strip()]
    if not units:
        return []
    # Subdividing a short sentence yields two-word scraps that read worse on screen than
    # simply showing fewer points, so only units with room for two real fragments are cut.
    # This is why a `target` of five over short source text can still yield three: the
    # count is a budget, not a quota to be met with fragments of two words.
    while len(units) < target and max(len(u.split()) for u in units) >= _SPLITTABLE_WORDS:
        before = len(units)
        units = _ensure_units(units, len(units) + 1)
        if len(units) == before:
            break
    if len(units) > target:
        # Spread the picks across the whole segment rather than taking the first few.
        step = len(units) / target
        units = [units[min(len(units) - 1, int(i * step))] for i in range(target)]

    bullets: list[str] = []
    for unit in units:
        fragment = _fragment_from(unit, language)
        if not fragment or len(fragment.split()) < BULLET_WORD_MIN:
            continue
        if _is_redundant(fragment, bullets):
            continue
        bullets.append(fragment)
    return bullets


def _is_redundant(candidate: str, bullets: list[str]) -> bool:
    """True when `candidate` repeats a point already on screen.

    Substring either way, not equality: "Check The Sender Domain" and "Check The Sender
    Domain First" are one point shown twice.
    """
    key = candidate.lower()
    return any(key in b.lower() or b.lower() in key for b in bullets)


def _fragment_from(unit: str, language: Language = Language.EN) -> str:
    """A 2-6 word verbatim run of `unit`, starting at its first content word.

    Sentence case, per `docs/DIRECTION.md` §2.2: a mechanically Title-Cased bullet reads as
    a template artefact, and the capitalised articles are the giveaway. Acronyms keep their
    own casing, and the run is trimmed to `BULLET_CHAR_MAX` words-first so it still fits on
    one line.

    `language` picks the stopword set, so a Spanish fragment opens on "dominio" rather than
    on "el", and Devanagari — which has no case at all — is left uncased rather than being
    put through a capitalisation rule borrowed from English.
    """
    words = _words(unit)
    if not words:
        return ""
    stopwords = _STOPWORDS_BY_LANGUAGE.get(language, _STOPWORDS)
    start = next((i for i, w in enumerate(words) if w.lower() not in stopwords), 0)
    run = words[start : start + BULLET_WORD_MAX]
    while len(run) > BULLET_WORD_MIN and run[-1].lower() in stopwords:
        run.pop()
    if len(run) < BULLET_WORD_MIN:
        run = words[-BULLET_WORD_MIN:]
    while len(run) > BULLET_WORD_MIN and len(" ".join(run)) > BULLET_CHAR_MAX:
        run.pop()
    if language.script != "latin":
        return " ".join(run)
    return _sentence_case(run)


def _sentence_case(words: list[str]) -> str:
    """First word capitalised; the rest keep the narration's own casing.

    Leaving the tail alone is the point: proper nouns and acronyms survive, and ordinary
    words stay lowercase instead of being promoted into Title Case.
    """
    if not words:
        return ""
    head = words[0] if words[0].isupper() else words[0].capitalize()
    return " ".join([head, *words[1:]])


def _partition_bullets(
    bullets: list[str],
    first: str,
    second: str,
    target: int = BULLET_DEFAULT,
    language: Language = Language.EN,
) -> tuple[list[str], list[str]]:
    """Split one scene's bullets across the two halves of its narration.

    Normally each bullet goes to the half that actually says it. Where the anchor check
    cannot run at all, asking "which half says this?" returns "neither" for every bullet and
    the whole set would be discarded, so the split falls back to position: the bullets were
    written in spoken order, so the front ones belong to the front half.
    """
    if not anchoring_supported(language):
        middle = len(bullets) // 2
        return (
            _clean_bullets(bullets[:middle], first, target, language=language),
            _clean_bullets(bullets[middle:], second, target, language=language),
        )

    head: list[str] = []
    tail: list[str] = []
    for bullet in bullets:
        if anchor_position(bullet, first) is not None:
            head.append(bullet)
        elif anchor_position(bullet, second) is not None:
            tail.append(bullet)
        # Anchored in neither half: dropped, and both halves are topped up below.
    return (
        _clean_bullets(head, first, target, language=language),
        _clean_bullets(tail, second, target, language=language),
    )


def _heading_from(text: str, *, fallback: str, language: Language = Language.EN) -> str:
    """Deterministic 3-7 word title from the segment's own opening words."""
    words = _words(text)
    if not words:
        return _clean_heading(fallback, language)
    stopwords = _STOPWORDS_BY_LANGUAGE.get(language, _STOPWORDS)
    keywords = [w for w in words if w.lower() not in stopwords]
    chosen = (keywords or words)[:6]
    if len(chosen) < 3:
        chosen = words[:5]
    if language.script == "latin":
        heading = " ".join(w if w.isupper() else w.capitalize() for w in chosen)
    else:
        # Devanagari is unicameral: `.capitalize()` is a no-op on it, so calling it would
        # only imply a casing rule that does not exist for this script.
        heading = " ".join(chosen)
    return _clean_heading(heading, language) or _clean_heading(fallback, language)


def _clip_prompt_from(
    segment: str, topic: str, language: Language = Language.EN
) -> str:
    """Motion counterpart to `_image_prompt_from`, for a video model.

    Deliberately generic about the *action* — there is no LLM here to invent one, and a
    made-up action would contradict the still. What it does commit to is the camera move,
    which is the part `image_prompt` cannot express at all.
    """
    return (
        "Slow steady push in on a real scene showing "
        f"{_clip_subject(segment, topic, language)}. "
        "Live-action documentary footage, natural light, shallow depth of field, subject "
        "moving gently within the frame, handheld micro-motion, no cuts, no camera flash, "
        "no on-screen speaker, no dialogue. No text, no letters, no words, no numbers, no "
        "labels, no signage, no watermarks anywhere in the frame."
    )


def _clip_subject(segment: str, topic: str, language: Language = Language.EN) -> str:
    """The segment's own content words, as a subject phrase for either visual prompt.

    Both visual prompts are English, and there is no translator on this path — so for a
    non-Latin script the segment's words cannot go into one. Devanagari keywords in an
    English image prompt are worse than no keywords: the model is English-conditioned, so
    they contribute noise instead of subject matter, and the topic alone at least names the
    right thing. Spanish keywords are kept — Latin script, and the image models read them.
    """
    stopwords = _STOPWORDS_BY_LANGUAGE.get(language, _STOPWORDS)
    if language.script != "latin":
        return topic.strip()
    words = _words(segment)
    keywords = [w.lower() for w in words if w.lower() not in stopwords][:10]
    return ", ".join(dict.fromkeys(keywords)) or topic.strip()


def _image_prompt_from(segment: str, topic: str, language: Language = Language.EN) -> str:
    """Build a background-image prompt from the segment without calling an LLM.

    Always English boilerplate — see `_clip_subject` for what happens to a non-Latin segment.
    """
    subject = _clip_subject(segment, topic, language)
    return (
        f"A cinematic documentary photograph illustrating {topic.strip()}: {subject}. "
        "Real photography, natural light, shallow depth of field, wide landscape framing, "
        "muted cinematic color grade, generous empty space across the lower third for a "
        "caption overlay. No text, no letters, no words, no numbers, no labels, no "
        "signage, no watermarks anywhere in the image."
    )
