"""Optional karaoke captions as an ASS subtitle file.

Off by default: the burned-in heading is the primary on-screen text and a second
block of moving text competes with it. Enable per job when the narration itself is
the point (tutorials, language content).

ASS timing is in *centiseconds* — ``0:00:01.50`` is one and a half seconds, and
``\\k`` durations are centiseconds too. Getting that wrong by 10x is the classic bug.
"""

from __future__ import annotations

from pathlib import Path

from app.core.models import RenderProfile, Timeline, Word

MAX_WORDS_PER_LINE = 7
MAX_LINE_SECONDS = 3.5


def format_ass_time(seconds: float) -> str:
    """``H:MM:SS.cc`` — ASS uses hundredths, not milliseconds."""
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    """``{`` and ``}`` delimit override blocks; ``\\`` starts a tag."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def group_words(words: list[Word]) -> list[list[Word]]:
    """Break a word stream into caption-sized chunks."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and (
            len(current) >= MAX_WORDS_PER_LINE
            or word.end - current[0].start > MAX_LINE_SECONDS
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def karaoke_line(group: list[Word]) -> str:
    """``{\\k<cs>}word`` per word, so libass highlights them one at a time."""
    parts: list[str] = []
    for index, word in enumerate(group):
        # Include the inter-word gap in the previous word's dwell time, otherwise
        # the highlight runs ahead of the voice.
        end = group[index + 1].start if index + 1 < len(group) else word.end
        centiseconds = max(1, int(round((end - word.start) * 100)))
        parts.append(f"{{\\k{centiseconds}}}{escape_ass_text(word.display)}")
    return " ".join(parts)


def build_ass(timeline: Timeline, profile: RenderProfile | None = None) -> str:
    """Full ASS document for every aligned word in the timeline.

    Timings are on the *narration* clock. Because xfade shrinks the assembled video,
    pass the caption file through :func:`shift_for_transitions` first if you burn it
    onto the assembled output rather than onto individual scene clips.
    """
    profile = profile or timeline.profile
    font_size = max(16, round(profile.height * 0.042))
    margin_v = round(profile.height * 0.14)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {profile.width}
PlayResY: {profile.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, \
BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,{font_size},&H00FFFFFF,&H00A0FFFF,&H00000000,&H64000000,\
-1,0,0,0,100,100,0,0,1,3,1,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    for scene in timeline.scenes:
        for group in group_words(scene.words):
            start = format_ass_time(group[0].start)
            end = format_ass_time(group[-1].end)
            lines.append(
                f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{karaoke_line(group)}"
            )
    return header + "\n".join(lines) + "\n"


def shift_for_transitions(timeline: Timeline) -> dict[int, float]:
    """Per-scene shift from the narration clock to the assembled-video clock.

    xfade consumes overlap, so scene *i* appears earlier in the final video than its
    narration timestamps say. Returns ``{scene_id: negative_shift_seconds}``.
    """
    shifts: dict[int, float] = {}
    overlap = 0.0
    for index, scene in enumerate(timeline.scenes):
        if index > 0 and scene.plan is not None:
            from app.core.models import Transition

            if scene.plan.transition_in is not Transition.CUT:
                overlap += scene.plan.transition_duration
        shifts[scene.id] = -overlap
    return shifts


def write_ass(timeline: Timeline, out_path: Path, profile: RenderProfile | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_ass(timeline, profile), encoding="utf-8")
    return out_path


def burn_filter(ass_path: Path) -> str:
    """``subtitles`` filter fragment. Requires an ffmpeg built with libass."""
    escaped = str(ass_path).replace("\\", "\\\\").replace("'", "'\\''")
    return f"subtitles=filename='{escaped}'"
