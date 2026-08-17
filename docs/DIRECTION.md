# DIRECTION — structure, type and motion spec for corporate training video

Status: normative. Every number here is implementable without asking a question.
Reference frame is **1920×1080 @ 30 fps**; all pixel values scale linearly by
`width / 1920`, so a 960×540 draft is a pure 0.5 scale.

Grounding used, and where it comes from:

- **Text height / line length / dwell time** — [legibility.info, "Rules for text in
  videos"](https://legibility.info/rules-for-text-in-videos): body text 40–60 px at
  1080p, titles ≥50 % larger than body, max 30 characters per line, max 3 lines at
  once, and a minimum dwell of **1 second per 13 characters** for motionless text.
- **Text height as a fraction of frame** — BBC/EBU subtitle practice: text height
  1/20–1/10 of frame height (54–108 px at 1080p); BBC accessibility floor 44 px at HD.
  ([BBC subtitle guidelines summary](https://www.clevercast.com/bbc-subtitling-guidelines/))
- **Segmentation** — [NN/g, *Videos as Instructional Content*](https://www.nngroup.com/articles/instructional-video-guidelines/):
  prefer several low-granularity segments, each with a clear beginning and end, over one
  long run; viewers scan and need to know immediately whether the video answers their
  question. This is the argument for a real title card and a real end card, not for
  ornament.
- **Broadcast lower-third convention** — a caption is always fully off screen before the
  shot changes. Nobody dissolves type across type. See §4.3.
- **Multimedia-learning coherence/redundancy principles (Mayer)** — exclude material that
  doesn't serve the objective; don't say the same thing twice in two channels. This is
  the argument for §7 (footage only where there is nothing to read) and for dropping
  bullet emphasis (§3.4).

---

## 0. The one-line diagnosis

We built a **variety engine** (rotating layouts, rotating transitions, rotating
entrances, rotating camera moves, two marker shapes) and shipped it at a topic where
variety is the enemy. "All over the place" is literally correct: in
`out/b85ee53b.../video.mp4` four scenes use four different layouts, three different
transitions, four different heading entrances and two marker shapes. Nothing repeats,
so nothing reads as a system.

**The governing rule for everything below: repetition is the design.** The viewer should
be able to predict where the next heading will appear before it appears. Only the
title card and the end card are allowed to break the grid, and they break it the same
way every time.

---

## 1. Structure

### 1.1 Roles and counts

`slide_count` (`N`) is the **total** scene count, including title and closing.

| N | TITLE | CONTENT | SUMMARY | CLOSING | sequence | ≈ runtime |
|---|---|---|---|---|---|---|
| 4 | 1 | 2 | 0 | 1 | T C C X | 46 s |
| 5 | 1 | 3 | 0 | 1 | T C C C X | 62 s |
| 6 | 1 | 4 | 0 | 1 | T C C C C X | 78 s |
| 7 | 1 | 4 | 1 | 1 | T C C C C S X | 91 s |
| 8 | 1 | 5 | 1 | 1 | T C C C C C S X | 107 s |
| 9 | 1 | 6 | 1 | 1 | T C×6 S X | 123 s |
| 10 | 1 | 7 | 1 | 1 | T C×7 S X | 139 s |

Rules:

- `N ≥ 4`. Below that there is no room for a shape; reject or pad the request.
- Exactly one TITLE, always first. Exactly one CLOSING, always last.
- **SUMMARY exists iff `N ≥ 7`** (i.e. iff CONTENT count would otherwise be ≥ 5), and it
  always sits at position `N-1`, immediately before CLOSING. Never mid-video.
- **Answer to "should summary appear in a 4-slide video": no, and not in a 6-slide one
  either.** With ≤4 content scenes the CLOSING already restates the key point; a summary
  on top of it is the redundancy principle violated, and it costs a teaching slide.
  Below ~100 s there is nothing to recap.
- Accept the trade-off explicitly: a 4-slide video now has **2 teaching slides**. That is
  honest. Structure costs slides. Document it in the UI ("4 slides ≈ 45 s, 2 teaching
  points") rather than smuggling teaching content into the title card, which is what
  produced the rejected output.

### 1.2 Durations

Narrated duration per scene, in seconds. `target` is what the script provider must aim
for; `min`/`max` are hard clamps the pipeline asserts.

| role | min | target | max | bullets | narration words @135 wpm |
|---|---|---|---|---|---|
| TITLE | 4.0 | 4.5 | 6.5 | 0 | 10 (9–14) |
| CONTENT | 11.0 | 15.0 | 19.0 | **exactly 4** | 34 (25–43) |
| SUMMARY | 9.0 | 12.0 | 14.0 | 4 | 27 (20–31) |
| CLOSING | 6.0 | 7.5 | 9.0 | 2 | 17 (13–20) |

Plus, appended to the CLOSING clip and **not narrated**: a 2.0 s END CARD (§1.5).

### 1.3 What the TITLE card says

Three elements, nothing else. It must answer "is this video about my problem?" in one
glance (NN/g scanning behaviour).

| element | content | source |
|---|---|---|
| kicker | `TRAINING MODULE` (uppercase, tracked) | constant, or `tone` mapped to a label |
| title | `Script.title`, ≤ 50 chars, ≤ 2 lines | the script's own title, verbatim |
| rule | 4 px × 120 px accent, centred | — |

Hard rule: **the title card never renders a scene heading and never renders bullets.**
The rejected video's scene 1 is a `title_card` layout carrying the content heading
"Inspect the Sender" and four bullets. That is the single worst defect in the output.

Narration for the title: **yes, one sentence, ≤14 words**, naming the subject and the
payoff ("This module shows you how a phishing email is built, and the four things that
give it away."). A silent 4.5 s card over music reads as a stalled player. The type
lands first: title fades in at t=0.70 s, the voice starts at t=1.10 s.

### 1.4 What the CLOSING says

Imperatives only — what to do on Monday. Heading is an instruction, not a topic
("If in doubt, report it"). Exactly 2 bullets, both imperative, both ≤34 chars.
The CLOSING uses the **same `hero_right` grid as every content scene** (§6). It is
distinguished by being short, having 2 bullets, and (optionally, §7) being the one
scene whose hero region moves.

### 1.5 END CARD

The end card is what "the video just stops" is missing. It is a sub-segment of the
CLOSING clip, not a separate scene, so it costs no extra narration or alignment.

| property | value |
|---|---|
| duration | 2.0 s (0.4 s in, 1.2 s hold, 0.4 s out) |
| background | `theme.bg`, solid. No image, no footage. |
| logo | centred, height `0.11 × frame height` (119 px), opacity 1.0 |
| line 1 | `Script.title`, 44 px, `theme.muted`, centred, ≤2 lines, 90 px below logo |
| watermark | suppressed (the centred logo replaces it) |
| out | fade to `theme.bg`, **not to black** |

The whole video then ends on the theme background, held for 0.3 s, then the file ends.
Fading to black from a dark-navy theme is invisible; fading to black from a light theme
is a jarring flash.

---

## 2. Type scale

Modular scale, ratio **1.333** (perfect fourth), base = bullet. Few levels, big jumps —
correct for a surface read at 2 m.

| token | px @1080p | / frame height | ratio to base | line height | max chars/line | max lines |
|---|---|---|---|---|---|---|
| `kicker` | 32 | 1/34 | 0.727 | 1.30 (42 px) | 28 | 1 |
| `bullet` | **44** | 1/24.5 | 1.000 | 1.22 (54 px) | **34** | **1** |
| `heading` | 78 | 1/13.8 | 1.773 | 1.22 (95 px) | **22** | **1** |
| `title` | 105 | 1/10.3 | 2.386 | 1.22 (128 px) | 25 | 2 |

Derivation and checks:

- `bullet` 44 px clears the 40 px floor for 1080p body text and the BBC 44 px HD
  accessibility floor. **Our current 36 px is below both** — and `SHRINK_STEPS` can take
  it to 21 px before it gives up, which is how a bullet ends up unreadable.
- `heading / bullet` = 1.77 ≥ 1.5, satisfying "titles at least 50 % larger than body".
- `title` 105 px sits inside the EBU 1/20–1/10 band.
- Char limits use `AVG_GLYPH_RATIO = 0.52`: `chars = column_width / (size × 0.52)`.
- Uppercase tracking on `kicker` only: `+0.12em`. Nothing else is tracked. The rejected
  output tracks emphasised bullets (`t=8.0 s`, "Inspect The Full Sender Address" is
  visibly wider-spaced than its neighbours) which is one of the two signals reading as
  a font bug.

### 2.1 Headings are one line — enforced upstream

Answer to "ours wraps a heading to 2 lines at ~24 chars — is that right?": **no, and the
fix is not a wider column.** Set `MAX_LINES = 1` for headings and cap generated headings
at 22 characters. A 2-line heading is a script defect, not a render problem.

Why it matters more than it looks: with a variable heading line count, the first bullet
starts at a different y on every scene. In the rejected video scenes 2 and 3 have 2-line
headings and scene 4 has 1 line, so the bullet stack sits at three different heights
across four slides. **That is a large part of what "all over the place" means.** See §6.2
for the fixed-baseline rule that makes it impossible.

Bullets are also one line: cap generation at 34 chars. `MAX_BULLET_LINES` drops 4 → 2 as
a safety net only; a 2-line bullet must never be produced deliberately.

### 2.2 Bullet copy rules (script provider)

- **Sentence case.** Not Title Case. Every bullet in the rejected video is mechanically
  title-cased — "Hit The Report Button", "Look For Subtle Spelling Errors" — and the
  capitalised articles are read as a template artefact.
- One grammatical form per scene: all imperatives, or all noun phrases. Never mixed.
- Never lift a narration fragment verbatim if it inverts the meaning. `b85` scene 2
  bullet 4 is "Leave Unexpected File Attachments" — lifted from a sentence about what
  attackers do, and on screen it reads as an instruction to the viewer.
- No terminal punctuation on bullets.

---

## 3. Bullet system

### 3.1 One marker: `dash`

Recommendation: **change `Theme.marker` default from `"disc"` to `"dash"`.**

| shape | verdict |
|---|---|
| `dash` | **Chosen.** A 20×2 px accent rule. It has no ambiguous variant — there is no "hollow dash" — so the failure we shipped is structurally impossible. It shares a graphic language with the heading rule (§6.2): the deck is built from horizontal accent strokes. Neutral, non-sequential, mirror-safe. This is the McKinsey/Deloitte house-deck marker for exactly these reasons. |
| `disc` | Supported, second choice. At `MARKER_RATIO 0.30 × 36 px` it is an 11 px dot — sub-1 mm on a laptop. Small enough to read as dirt, large enough to be noticed. If kept, raise the diameter (§3.2). |
| `chevron` | Rejected. Implies ordered progression; our bullets are unordered. Directional, so it fights any mirrored layout. |
| `ring` | Rejected outright and **deleted**, see §3.4. |
| `none` | Available for the summary if we ever want a bare list. Not the default. |

### 3.2 Marker geometry

| property | `dash` | `disc` |
|---|---|---|
| gutter width (fixed) | 44 px (1.0 em) | 44 px |
| ink width | 20 px | 16 px diameter |
| ink height | 2 px (min 2 px at any scale) | 16 px |
| vertical alignment | centred on x-height: `cap_top + 0.36 em` = +16 px | same |
| colour | `theme.accent` | `theme.accent` |
| text start x | `column_x + 44` — fixed, independent of marker | same |

The gutter is **fixed width**, not "marker width + gap". That guarantees the text edge
is at the same x on every bullet of every scene, and a wrapped line hangs to that same
edge.

### 3.3 Vertical rhythm

| measure | value |
|---|---|
| bullet pitch (single-line → single-line) | **84 px** (1.91 em) |
| extra per wrapped line | 54 px (1.22 em) |
| heading cap-bottom → rule | 34 px |
| rule → first bullet cap-top | 52 px |

84 px pitch = 54 px line + 30 px gap. Constant, so a wrapped bullet pushes the stack
down rather than colliding with it (this part of the current implementation is right;
only the numbers change).

### 3.4 Emphasis: drop it entirely

Recommendation: **remove bullet emphasis from the renderer.**

Reasoning, in order of weight:

1. **The narration already carries the emphasis.** Each bullet is revealed as it is
   spoken. The point being spoken is the point being emphasised, and the reveal marks
   it. A permanent visual emphasis on bullet 2 actively contradicts the temporal
   emphasis on bullets 1, 3 and 4 when *they* are the live point.
2. **The remaining channels can't carry it cleanly.** Colour is off the table
   (`uniform_text`), marker shape is off the table (one shape, video-wide). That leaves
   weight and size. The rejected output uses both at once — `EMPHASIS_SIZE_RATIO` 1.06
   *and* a faux-bold stroke *and* an `EMPHASIS_STROKE_BOOST` of 1.35 — and the result
   (`t=8.0 s`, `t=33 s`, `t=65 s`) is a bullet in what looks like a different font at a
   different size. The user read it as sloppiness because it *is* indistinguishable from
   a rendering fault.
3. **If a point matters more, express it structurally**, not typographically: give it its
   own CONTENT scene, or make it the CLOSING's first bullet. That is what the role system
   is for.

Implementation: keep `BulletPoint.emphasis` in the model (old timelines deserialise) and
have the renderer ignore it. **Delete** `MARKER_RING_RATIO`, `EMPHASIS_SIZE_RATIO`,
`EMPHASIS_SIZE_RATIO_NO_WEIGHT`, `EMPHASIS_STROKE_BOOST`, `EMPHASIS_FAUX_BOLD_RATIO` and
every code path that reads them. Deleting, not merely not-calling: a dormant path that
swaps marker shape will come back.

---

## 4. Motion

### 4.1 One entrance for the whole video

Retire `HEADING_ANIMATION_ROTATION` and `BULLET_ANIMATION_ROTATION`. The planner emits
the same values on every scene.

| element | animation | duration | travel | easing |
|---|---|---|---|---|
| kicker | fade | 0.35 s | 0 | `ease_out_cubic` |
| heading | fade + rise | 0.40 s | 12 px up | `ease_out_cubic` |
| rule | width wipe L→R | 0.30 s | — | `ease_out_cubic` |
| bullet | fade + rise | **0.28 s** | **8 px up** | `ease_out_cubic` |
| hero image | fade | 0.45 s | 0 | `linear` |
| end card | fade | 0.40 s | 0 | `ease_out_cubic` |

Changes from current, with reasons:

- **`anim_duration` 0.45 s → 0.40 s heading / 0.28 s bullet.** One duration for two very
  different-sized elements is wrong: a 44 px bullet moving for 0.45 s is slower per pixel
  than a 78 px heading. 0.28 s is the shortest duration that still eases perceptibly at
  30 fps (8 frames).
- **`ease_in_out` → `ease_out`.** Ease-in-out on an *entrance* means the element
  accelerates from rest, which reads as sluggish. Entrances ease out (fast start,
  settle); exits ease in. Standard motion practice (Material, Apple HIG).
- **`SLIDE_LEFT` → drop; `SLIDE_DISTANCE_RATIO` 60 px → 8 px.** A horizontal slide on a
  left-aligned list sweeps the text *through* the marker gutter. 60 px is a swipe, not an
  entrance. 8 px vertical is a settle — present, not noticed. For comprehension-first
  material this is the right trade.
- `POP` and `TYPEWRITER` are never emitted. `POP` overshoot on a training slide is flair
  with no informational content.

### 4.2 Stagger

| knob | current | spec | reason |
|---|---|---|---|
| `bullet_min_gap` | 0.6 s | **1.6 s** | 0.6 s lets two bullets land inside one spoken clause; the eye is still on bullet *N* when *N+1* arrives. 1.6 s ≈ one short clause at 135 wpm (3.6 words). |
| `ANIM_TAIL_MARGIN` | 0.35 s | **2.60 s** | Dwell rule: 1 s per 13 characters. A 34-char bullet needs 2.6 s motionless before the scene leaves. 0.35 s means the last point is legally on screen and practically unread. **This is the highest-impact single change in this document.** |
| first reveal earliest | — | 1.15 s | after heading entrance completes (0.45 + 0.40) + 0.30 s |
| `ANIM_MAX_GAP_FRACTION` | 0.8 | 0.6 | at 0.8 the previous bullet is still settling |

Feasibility check (why CONTENT is exactly 4 bullets): at the 11 s floor, the usable
reveal window is `11 − 1.15 − 2.60 − 0.28 − 0.70 = 6.27 s`. Four bullets need 3 gaps
× 1.6 = 4.8 s ✓. Five bullets need 4 × 1.6 = 6.4 s ✗. Five only fits from 13 s up.
Rather than make bullet count duration-dependent, **fix it at 4 for every content
scene** — every content slide having the same number of points is itself a uniformity
win, and it makes the whole deck previewable.

### 4.3 Transitions — the burned-in-text rule

**Kill `SLIDE_LEFT` and `WIPE_RIGHT` from `TRANSITION_ROTATION` immediately.** They are
not a style choice, they are a defect when text is burned into the scene clip:

- `b85 t=38.0 s` (slideleft): the incoming heading is caught half off frame, reading
  "Recogn… / Pressur…". Two slides visible, one of them cropped mid-word.
- `b85 t=57.0 s` (wiperight): the wipe edge cuts through both text stacks. On screen
  simultaneously: two headings and eight bullet fragments — "ize / re Tactics",
  "e Artificial Panic", "ur Password". This is the ugliest frame in the video.

And crossfade alone is not enough — `b85 t=18.2 s` shows a clean 0.5 s crossfade
producing a double exposure of *eight lines of type in two different layouts*.

The rule, from broadcast lower-third practice: **type is fully off screen before the
shot changes, and enters after it has changed.** Never dissolve type across type.

| stage | timing (relative to scene clip of duration `d`) |
|---|---|
| text exit begins | `d − 0.70` |
| text exit (fade + 8 px down, `ease_in`) | 0.30 s |
| text fully gone | `d − 0.40` |
| image crossfade | `[d − 0.35, d]` |
| next scene's kicker/heading enter | `+0.45` into the incoming clip |

| knob | current | spec |
|---|---|---|
| transition type | rotating fade/slideleft/wiperight | **`fade` (crossfade), always** |
| `transition_duration` | 0.5 s | **0.35 s** |
| `MIN_TRANSITION_DURATION` | 0.10 s | 0.20 s (below that, demote to `CUT`) |
| first scene | fade from black | fade from `theme.bg`, 0.5 s |

0.35 s rather than 0.5 s because the two frames now share a grid (§6) — the only thing
actually cross-dissolving is the hero photograph, and 0.35 s is enough for that while
costing 0.15 s less dead air per boundary.

### 4.4 Camera move on stills

| knob | current | spec |
|---|---|---|
| move | rotating zoom_in / pan_right / zoom_out / pan_left | **`zoom_in`, every scene** |
| span | 0.08–0.15 rotating | **0.06**, constant |
| easing | `ease_in_out` | **`linear`** |
| `upscale_factor` | 4 | **8** |

- Alternating direction between adjacent scenes is *visible* and reads as indecision.
  One direction, one span, forever. A Ken Burns move exists to keep the frame from
  feeling dead, not to be seen.
- `ease_in_out` stalls the move at both ends, which is where duplicate frames come from.
  `linear` at 6 % over 15 s = 0.4 %/s: never static, never noticed.
- Arithmetic for `upscale_factor 8`: 6 % of 1920 = 115 px over 450 frames = 0.26
  px/frame. `zoompan` truncates x/y to integers, so the frame holds for ~4 frames at a
  time — the mechanism behind the measured `duplicate_frame_ratio` of 0.37 on `b85`
  scene 2 (noise floor 0.12). At 8× the source is 15360 px wide and the per-frame step is
  2.05 source px, sub-pixel after downscale.
- **Role-aware motion scoring:** the TITLE and the END CARD are solid-colour and
  *correctly* static. `b85` scene 1 scored 0.7/10 on motion with a 0.77 duplicate ratio;
  that is the scorer being wrong, not the render. Exempt `title_card` scenes from the
  duplicate-frame check.

---

## 5. Pacing

| knob | current | spec | reason |
|---|---|---|---|
| narration rate | measured **120.8–150.2 wpm** across four scenes of one video | **135 wpm ± 8** (127–143) | The problem is not the absolute rate, it is a **24 % swing inside one video** — heard as inconsistent energy. Enforce by adjusting *word count* per scene, never by changing the TTS rate (which changes timbre). Reject `b85` scene 4 at 150 wpm: too fast when four bullets must be read at the same time. |
| inter-scene silence | 1.0 s everywhere | **0.6 s**, except **1.0 s after the TITLE** | 1.0 s × 5 boundaries = 5 s of dead air in a 75 s video (6.7 %), and it lands on a static slide where the viewer has finished reading. The beat after the title is deliberate ("here we go"); the rest is just latency. |
| title narration | n/a (no title card existed) | **yes**, ≤14 words, voice in at 1.10 s | A silent card over music reads as a stalled player. |
| end card narration | — | **none** | Music only. It is a lockup, not a message. |
| final tail | — | 1.2 s of held end card after fade-in, then 0.4 s out, then 0.3 s hold |

Word budgets follow directly from `135 wpm = 2.25 words/s` — see the table in §1.2.
Assert them in the script provider's schema, not after generation.

---

## 6. Layout discipline

### 6.1 Two layouts. Total.

**Answer to "is our alternation variety or noise": it is noise, and it also breaks the
crossfades.** Evidence: `b85 t=21 s` (`hero_right`) → `t=45 s` (`hero_left`) moves the
text block **~940 px horizontally** between consecutive scenes. The viewer re-hunts for
the text every scene. And `t=65 s` (`full_bleed`) puts the heading directly over the
subject's forearm and keyboard at a measured contrast of **10.45** against 18.8 on the
solid slides — the one layout that varies is also the one that fails legibility.

| role | layout | image |
|---|---|---|
| TITLE | `title_card` | none |
| CONTENT | `hero_right` | still |
| SUMMARY | `hero_right` | still |
| CLOSING | `hero_right` | still, or clip (§7) |
| END CARD | `title_card` | none |

`hero_left`, `image_band` and `full_bleed` — **retire them.** Keep the enum members so
old timelines deserialise; the planner must never emit them. Delete
`LAYOUT_ROTATION`, `FULL_BLEED_MIN_SCENES` and the three-in-a-row de-duplication loop.

`alternate_text_position` and the `text_position` rotation go with them. `TextPosition`
is `CENTER` for `title_card` and `LEFT_PANEL` for `hero_right`. Nothing else.

### 6.2 The grid (1920×1080)

Every value is fixed on every `hero_right` scene. Nothing depends on bullet count or
heading length.

| element | x | y | w | h |
|---|---|---|---|---|
| text column | 104 | 90 | 904 | 900 |
| hero region | 1096 | 90 | 720 | 900 |
| kicker cap-top | 104 | 262 | — | 32 |
| heading cap-top | 104 | **330** | ≤904 | 78 |
| accent rule | 104 | **442** | **88** | 4 |
| bullet 1 cap-top | 148 (text), 104 (marker) | **494** | ≤860 | 44 |
| bullet 2/3/4 cap-top | " | 578 / 662 / 746 | " | " |
| watermark | 54 | 977 | — | 49 |

- **Fixed first-bullet baseline (`y = 494`) on every scene.** This is the rule that makes
  the deck read as one deck. It is only possible because headings are one line (§2.1).
- Hero region is **4:5 (720×900)**. Generated stills must be requested at 4:5, not 16:9 —
  our images come back 2752×1536 (1.79) and get cropped to 0.80, discarding 55 % of the
  width. That is why the framing in `b85 t=45 s` is a tight head-and-shoulders when the
  prompt asked for a desk scene. Corner radius 20 px, `theme.surface` frame 5 px.
- Hero region shares the text column's top and bottom (90 / 990), so the image is the
  tallest element and the grid is legible. In `db2aa068 t=30 s` the image spans a
  different vertical extent from the text block and the two elements look unrelated.
- **Accent rule is a fixed 88 px.** Currently `RULE_WIDTH_RATIO 0.18 × column`, so it is
  266 px on the title card and 137 px on a hero slide — the same graphic element at two
  widths in one video.
- **Alignment is left, always.** `b85 t=2.0 s` centres the heading and the rule while
  left-aligning the bullets from a centred block; the bullet stack's left edge lands at
  no meaningful x. One alignment per scene.
- 4 bullets gives a text block spanning y 262–790, optical centre 526 against a frame
  centre of 540. Deliberate: visual centre sits slightly above geometric centre.

### 6.3 The title-card grid

| element | value |
|---|---|
| block | centred horizontally; block optical centre at `y = 0.48 × H` = 518 |
| column | x 221, w 1478 (`TITLE_MARGIN_X_RATIO` 0.115 — keep) |
| kicker | 32 px, `theme.muted`, uppercase, `+0.12em`, centred |
| title | 105 px, `theme.text`, centred, ≤2 lines @ 25 chars |
| rule | 4 px × 120 px, `theme.accent`, centred, 48 px below the title |

The block is centred per-slide here (unlike §6.2) because there is exactly one title
card — there is no second one to be inconsistent with. `b85 t=3.0 s` top-anchors the
block and leaves the bottom **55 % of the frame empty**, which is what makes it read as
a slide that failed to load rather than a title.

---

## 7. When to use generated video instead of a still

### 7.1 The constraint arithmetic

Veo 3.1 returns a **fixed 8.0 s, 1280×720, 24 fps** clip with an audio track to strip.
Our timeline is 1920×1080 @ 30 fps.

| placement | required upscale | verdict |
|---|---|---|
| `full_bleed` 1920×1080 | 1.50× on the largest area on screen | **Never.** |
| `hero_right` region 720×900 | 1.25× (scale-to-cover 900 from 720) | Acceptable |
| title card | n/a — no image region | Never |

Frame rate: 24 → 30 always duplicates or interpolates. We take plain
`fps=30` duplication (48 of 240 frames, 20 %) and require the `clip_prompt` to specify
*slow* motion, where a repeat every 5th frame on a slow drift is below perception.
`minterpolate` is not worth the ghosting; a global move to a 24 fps `RenderProfile` is
the clean long-term fix and should be evaluated separately.

### 7.2 Recommendation

**Footage goes where there is nothing to read. Stills go where there is.**

That is the coherence principle applied literally: a content scene asks the viewer to
read four points in 15 s, and putting moving footage beside them is split attention. A
still with a slow 6 % zoom is not a compromise there — it is the correct choice.

| role | visual | why |
|---|---|---|
| TITLE | solid colour, no image | Softness on the first thing the viewer sees is the worst place for it. Pure type on solid colour is what makes a title read as a title, and it is the cheapest thing in the deck to get right. |
| CONTENT | **still**, always | 4 bullets to read. Never footage — not even on the shortest ones. |
| SUMMARY | **still** | Same. |
| CLOSING | **clip** (opt-in), in the `hero_right` hero region | 2 bullets, 6–9 s, and 8.0 s of footage covers it natively. |
| END CARD | solid colour | — |

**At most one clip per video.** The closing is where it lands: the one thing that changes
at the end is that the image comes alive, and the grid never moves. That gives the video
a lift at the end without a layout change — which is precisely the trade the rejected
output got backwards (it changed layout for lift and lost the grid).

### 7.3 Covering the seconds — the ladder

Never freeze the last frame. A held frame after moving footage reads as a decoder stall,
and it is the failure mode most likely to look worse than a still would have.

| closing duration `d` | action |
|---|---|
| `d ≤ 8.0 s` | **Preferred.** Play the first `d` seconds; **trim the tail**, not the head — Veo's opening frames are the most faithful to the prompt and late frames drift. |
| `8.0 < d ≤ 8.6 s` | `setpts` retime by up to 1.075×. ≤8 % is imperceptible on slow motion. |
| `8.6 < d ≤ 15.5 s` | Ping-pong: forward then reversed, giving a mathematically seamless loop point. **Only** if the `clip_prompt` was authored as reversible — no directional cause and effect, no hand completing an action, no readable UI or text. The planner must not request footage it cannot author this way. |
| `d > 15.5 s` | **No footage. Use the still.** |

Because CLOSING is clamped to 6.0–9.0 s (§1.2), rows 1–2 cover it in practice; the
ping-pong row exists so the rule is total rather than because we intend to use it.

**Not on the ladder: multi-clip coverage.** `veo_video.clips_needed()` computes
`ceil(d / 8.0)`, which would cover a 16 s scene with two generations. Do not use it.
Two independent generations from one prompt do not match on lighting, subject, lens or
colour, so the mid-scene join is a jump cut between two different people at two
different desks — worse than any still, at twice the cost. Keep the helper for cost
estimation; never let the renderer act on a value above 1.

### 7.4 Shipping gates

1. Ship with clips **off** by default (`video_enable_clips = False`).
2. `-an` on the clip input. Narration is authoritative.
3. Raise the `duplicate_frame_ratio` threshold to 0.25 for clip-backed scenes only.
4. Require the scored `motion` dimension for the clip-backed scene to **beat** the same
   scene rendered as a still before the flag flips on. This is the highest-risk feature
   in the roadmap; a wrong call here looks worse than a still, so make it prove itself.
5. `clip_prompt` must describe *movement*, never composition, and must specify slow
   ambient motion with no on-screen text or UI.

---

## 8. What we are doing wrong today

Each item cites a frame from a real render.

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | No title card exists. Scene 1 is a **content** scene wearing `title_card` clothing: heading "Inspect the Sender", 4 bullets, 18.53 s. | `b85 t=0.3 → 18.5 s` | §1.3 |
| 2 | Bottom **55 %** of the title card is empty; the block is top-anchored, not centred. Reads as a slide that failed to load. | `b85 t=3.0 s`; `db2 t=12.0 s` | §6.3 |
| 3 | 12 seconds in, only 3 of 4 bullets have appeared. Comprehension is not the bottleneck — the reveal schedule is. | `b85 t=12.0 s` | §4.2 |
| 4 | **Two marker shapes in one stack**: filled disc on the emphasised bullet, hollow ring on the rest. Both ~11 px — too small to distinguish, big enough to notice. | `b85 t=8.0 s`, `t=33 s`, `t=65 s` | §3.1, §3.4 |
| 5 | Emphasis is signalled **three ways at once** (size +6 %, faux-bold stroke, outline ×1.35) and the result looks like a font substitution bug. | `b85 t=33 s` ("Never Click Links" vs its neighbours) | §3.4 |
| 6 | Centred heading + centred rule + left-aligned bullets on the same slide. The bullet stack's left edge is at no meaningful x. | `b85 t=2.0 s`, `t=17.5 s` | §6.2 |
| 7 | **`wiperight` shreds burned-in text.** Two headings and eight bullet fragments visible at once — "ize / re Tactics", "e Artificial Panic". | `b85 t=57.0 s` | §4.3 |
| 8 | **`slideleft`** catches the incoming heading cropped mid-word: "Recogn… / Pressur…". | `b85 t=38.0 s` | §4.3 |
| 9 | Even a clean crossfade double-exposes 8 lines of type across 2 different layouts. | `b85 t=18.2 s` | §4.3 |
| 10 | Layout changes on **every** scene: `title_card → hero_right → hero_left → full_bleed`. The text block jumps ~940 px horizontally between scenes 2 and 3. | `b85 t=21 s` vs `t=45 s` | §6.1 |
| 11 | `full_bleed` heading sits on the subject's forearm; contrast **10.45** vs 18.8 on solid slides, and the scorer's alternative position measured **4.86**. | `b85 t=65 s`, `score.json` scene 4 | §6.1 |
| 12 | Bullets at 36 px are below both the 40 px 1080p body floor and the BBC 44 px HD floor — before `SHRINK_STEPS` gets involved. | all frames | §2 |
| 13 | Heading wraps to 2 lines, so the first bullet baseline differs across scenes 2, 3 and 4. | `b85 t=21 s` vs `t=65 s` | §2.1, §6.2 |
| 14 | Accent rule is 266 px on the title card and 137 px on a hero slide — one element, two widths. | `b85 t=3.0 s` vs `t=21 s` | §6.2 |
| 15 | Every bullet is mechanically Title Cased ("Hit The Report Button"); scene 2 bullet 4 reads as an instruction to do the wrong thing ("Leave Unexpected File Attachments"). | `b85 t=33 s`, `t=70 s` | §2.2 |
| 16 | 16:9 stills (2752×1536) cropped to a 1.05 hero region discard 55 % of the width, so the framing never matches the prompt. Scored relevance 3/10 and 5/10. | `score.json` scenes 1, 2, 4 | §6.2 |
| 17 | 37 % of frames repeat on scene 2 — eased zoom over a 4× upscale steps. | `score.json` scene 2 `duplicate_frame_ratio 0.3667` | §4.4 |
| 18 | Narration swings 120.8 → 150.2 wpm inside one video. | `score.json`, all scenes | §5 |
| 19 | **No end scene.** The last frame is a content slide fading out; no lockup, no CTA, no branding moment. | `b85 t=74.4 s`; `e2932f37 t=78.0 s` | §1.5 |
| 20 | 1.0 s of silence × every boundary = 5 s of dead air on a static, fully-revealed slide. | `config.video_scene_pause_s` | §5 |

---

## 9. Corrections to the `SceneRole` numbers as committed

| field | committed | spec | reasoning |
|---|---|---|---|
| `TITLE.target_duration` | (3.0, 6.0) | **(4.0, 6.5)** | The card is narrated (§5). A ≤14-word sentence at 135 wpm is 6.2 s including a 0.4 s lead-in, so 6.0 is a hair tight and 3.0 s is below the floor for reading a 2-line 105 px title (25 chars/line at 13 chars/s = 1.9 s per line + entrance). |
| `CONTENT.target_duration` | (14.0, 24.0) | **(11.0, 19.0)** | 24 s on one static slide with every bullet already revealed is the stall the user described — and every measured content scene (17.9–20.1 s) already sits in the top half of the committed range and was still rejected. 15 s target. |
| `SUMMARY.target_duration` | (8.0, 14.0) | **(9.0, 14.0)** | 4 bullets × 1.6 s stagger + 1.15 s lead + 2.6 s dwell + 0.7 s exit = 9.25 s. 8.0 is not achievable. |
| `CLOSING.target_duration` | (5.0, 10.0) | **(6.0, 9.0)** | 5.0 s cannot fit 2 bullets with a 2.6 s dwell (1.15 + 1.6 + 2.6 + 0.7 = 6.05). The 9.0 ceiling also keeps it inside the 8.6 s Veo retime window (§7.3). |
| `CONTENT.bullet_budget` | 5 | **4** | At the 11 s floor the reveal window is 6.27 s; 5 bullets need 6.4 s at a 1.6 s stagger. And fixing it at 4 on *every* content scene is itself a uniformity win — the viewer can predict the slide. |
| `SUMMARY.bullet_budget` | 4 | 4 | Correct. Tie it to "one line per preceding content scene, capped at 4". |
| `CLOSING.bullet_budget` | 2 | 2 | Correct. |
| `TITLE.bullet_budget` | 0 | 0 | Correct, and the most important number in the enum. |
| `TITLE.heading_scale` | 1.8 | **1.35** | Arithmetic: the base heading must rise 63 → 78 px to clear the video legibility floors (§2). 1.8 × 78 = 140 px, which in the 1478 px title column fits 20 chars/line — a real title like "How Phishing Attacks Work and How to Spot Them" (46 chars) needs 3 lines or a shrink. 1.35 × 78 = 105 px fits 25 chars/line × 2 = 50 chars. |
| `SUMMARY.heading_scale` | 1.1 | **1.0** | A 1.1× heading is a 7.8 px difference — undetectable in isolation, but it breaks the fixed `y = 494` bullet baseline, which *is* detectable. Four heading sizes in one video is the opposite of the uniformity that was asked for. |
| `CLOSING.heading_scale` | 1.3 | **1.0** | Same. The closing earns its weight from having 2 bullets, an imperative heading, a moving hero and an end card — not from 23 px of extra type that costs the grid. |

Net: **exactly two heading sizes in the video** — `title` (105 px, on the title card
only) and `heading` (78 px, everywhere else).

One collision to resolve while implementing: `SceneRole.heading_scale` and
`text_overlay.TITLE_HEADING_W_RATIO` (0.0448 → 86 px) are two sources of truth for the
same number. Delete `TITLE_HEADING_W_RATIO` and derive from `heading_scale`.

---

## 10. Implementation checklist

Ordered by impact per line of code.

1. `ANIM_TAIL_MARGIN` 0.35 → 2.60; `bullet_min_gap` 0.6 → 1.6.
2. `TRANSITION_ROTATION` → `(Transition.FADE,)`; `transition_duration` 0.5 → 0.35.
3. Text exit/entry windows around the crossfade (§4.3 table).
4. `LAYOUT_ROTATION` → `hero_right` only; `_layouts()` returns `title_card` for scene 0
   and `hero_right` for the rest.
5. Synthesize the TITLE scene from `Script.title` with `bullet_budget 0`.
6. Append the END CARD sub-segment to the CLOSING clip.
7. `Theme.marker` default `"disc"` → `"dash"`; delete every ring/emphasis constant.
8. Type scale: 32 / 44 / 78 / 105; `MAX_LINES` 2 → 1 for headings; char caps 22 / 34
   asserted in the script provider's schema.
9. Fixed grid constants from §6.2; fixed 88 px accent rule.
10. Single heading/bullet animation; `ease_out`; 8 px travel.
11. `zoom_in` 1.00→1.06 linear on every scene; `upscale_factor` 4 → 8.
12. Request 4:5 stills.
13. `video_scene_pause_s` 1.0 → 0.6, with 1.0 after the title.
14. 135 wpm word budgets in the script schema.
15. Role-aware `duplicate_frame_ratio` (exempt `title_card`).
16. Veo, behind a flag, closing scene only, hero region only. Last.
