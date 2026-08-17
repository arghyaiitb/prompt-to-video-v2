# prompt-to-video

Type a topic. Get a narrated, branded, 1920×1080 training video.

You give it a subject, a slide count, a theme and a voice. It writes the script, generates
the background imagery, synthesizes the narration, measures the real word timings, reveals
each bullet at the moment the narrator says it, lays a music bed underneath and renders the
whole thing to an MP4 you can put in an LMS.

**Concrete example.** Topic `How phishing attacks work and how to spot them`, 4 slides,
`midnight` theme → a **40.1 s** MP4, **1920×1080 @ 30 fps**, **h264 (CRF 18, ~1.0 Mb/s)** +
**AAC stereo 48 kHz (~193 kb/s)**, 6.06 MB. Four scenes: a title card, two teaching slides
of four bullets each, and a closing slide of two. Those are the ffprobed numbers from
`out/25960d08-.../video.mp4`, not a target.

![A content slide: heading "Inspect Every Email", four bullets in the left column, generated photo panel on the right](docs/images/frame-content-slide.jpg)

*Frame at t=17 s of that video, extracted with ffmpeg. Left text column, hero image right,
one accent rule, one bullet marker shape, brand mark bottom-left.*

---

## The output structure

The product is not "a video of the slides" — it is a video with a **shape**. Every scene
has a role, and the role decides its length, its type scale and how many bullets it may
show ([`SceneRole`](backend/app/core/models.py)).

```
title  ──▶  content …  ──▶  [summary]  ──▶  closing
4–6.5s      11–19s each      9–14s          6–9s
0 bullets   4 bullets        4 bullets      2 bullets
```

The summary only appears from 7 slides up — below that the closing already restates the key
point, and a recap would cost a teaching slide. A 4-slide video is therefore
`title → content → content → closing`: **two teaching slides**. That trade-off is
deliberate; see [`docs/DIRECTION.md`](docs/DIRECTION.md) §1.1.

The title card is the one scene that renders no bullets and no scene heading — it exists to
answer "is this video about my problem?" in one glance, so it gets a kicker, the script's own
title at 1.35× the base heading size, and an accent rule. Nothing else.

![The title card: kicker "TRAINING MODULE" above the large title "Spot and Stop Phishing", with a short amber rule beneath, on a solid navy ground](docs/images/frame-title-card.jpg)

Here is the real timeline of the video pictured above, straight out of
`GET /api/jobs/{id}/timeline`. Spans are on the narration clock; the finished file is 1.03 s
shorter than the 41.16 s total, because each of the three crossfades consumes its own
overlap:

| # | role | span | heading | bullets (scene-relative reveal) |
|---|---|---|---|---|
| 1 | `title` | 0.00 – 5.32 s | **Spot and Stop Phishing** | *(none — a title card carries no bullets)* |
| 2 | `content` | 5.32 – 19.37 s | Inspect Every Email | `+0.07` Check the sender address · `+3.43` Spot unexpected urgency · `+6.63` Look for generic greetings · `+9.51` Review odd payment requests |
| 3 | `content` | 19.37 – 34.52 s | Handle Links Safely | `+0.39` Hover over every link · `+1.99` Inspect destination web links · `+4.63` Avoid unknown attachments · `+8.47` Verify through known phone numbers |
| 4 | `closing` | 34.52 – 41.16 s | Report Suspicious Mail | `+0.00` Immediately click the report · `+1.60` Click the report button |

### The bit that matters: bullets are anchored to spoken words

Those reveal times are not a stagger. The script provider is instructed to make every
bullet reuse a distinctive phrase from its own scene's narration verbatim, so each bullet
has a genuine lexical anchor in the audio. The pipeline then:

1. synthesizes the narration **first**,
2. sends it to a word-level aligner and gets back `(word, start, end)` for every word,
3. locates each bullet's anchor phrase in that word stream
   ([`bullet_timing.py`](backend/app/providers/bullet_timing.py), exact match → fuzzy match
   at a calibrated 0.55 similarity floor → proportional fallback),
4. reveals the bullet **0.25 s before** its anchor word is spoken.

Scene 2 above is narrated *"Always **check the sender address** for subtle spelling mistakes.
Next, **spot unexpected urgency** demanding instant action. Then **look for generic
greetings** without your real name, and carefully **review odd payment requests** before
replying."* — which is exactly why its four bullets land at `+0.07`, `+3.43`, `+6.63` and
`+9.51` rather than on an even 3.5 s beat. The text arrives a fraction ahead of the voice, so
it reads as a narrator reaching a point already on screen rather than as a lagging caption.
That asymmetry is what makes the output feel authored instead of generated.

Everything downstream follows from the same principle — **audio is the clock**. Scene
boundaries, clip lengths, xfade offsets and the music bed's length are all derived from
measured durations. Word counts are never used to guess timing.

```
                    ┌─ images ─────────────────────────────┐
scripting ──┬───────┤                                      ├── plan ── render ── assemble
            └── narrate ── align ── music (sized from it) ──┘
```

---

## Prerequisites

Honest list. All six are required; ImageMagick is not optional.

| tool | why | verified version here |
|---|---|---|
| `ffmpeg` + `ffprobe` | render, mix, probe | 8.1.1 |
| **`imagemagick`** | **all** on-screen text | 7.1.2-3 |
| `uv` | Python 3.12+ env and runner | 0.7.2 |
| `pnpm` | frontend deps | 10.10.0 |
| `node` | Vite | v20.18.3 (Vite warns; 20.19+ / 22.12+ preferred) |
| `mprocs` | runs backend + frontend in one terminal | 0.8.3 |

**Why ImageMagick is required:** this ffmpeg build has no `drawtext` filter (no
libfreetype), so every heading and bullet is rasterised to a PNG by ImageMagick and
composited as an overlay. Without `magick`, no text reaches the frame.
`scripts/doctor.sh` detects which path is live and says so.

```bash
brew install ffmpeg imagemagick uv pnpm node mprocs
```

Then check the lot in one shot:

```bash
./scripts/doctor.sh
```

It verifies the toolchain, the text-render path, `.env`, **live 200s from Deepgram and
Gemini**, the frontend typecheck gate, that artifacts live in a single `out/` tree, and
that both dependency trees are installed. It currently reports **17 passed, 0 failed**.

---

## Setup

```bash
git clone <this repo> && cd prompt-to-video-v2

cp .env.example .env      # then fill in the keys below

cd backend  && uv sync --extra dev  && cd ..
cd frontend && pnpm install --force && cd ..

./scripts/doctor.sh
```

### API keys

| key | unlocks | required? |
|---|---|---|
| `GEMINI_API_KEY` | script generation, background images, Lyria music bed, Veo clips, and the evaluator's vision pass | **yes** |
| `DEEPGRAM_API_KEY` | TTS **and** the word-level alignment that all bullet timing depends on | **yes** |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | AWS Polly as a second, SSML-capable TTS engine | optional |
| `AWS_SESSION_TOKEN` | only for temporary STS credentials (`ASIA…` key ids) — **these expire within hours** | optional |
| `ANTHROPIC_API_KEY` | declared in `.env.example`, **not wired to anything yet** | no |
| `ELEVENLABS_API_KEY` | declared, and ElevenLabs appears in the SSML capability table, but there is **no provider implementation** | no |

There is only one aligner. Deepgram is not swappable today: without it, bullets fall back
to proportional placement and stop matching the narration, which removes the feature above.

Non-secret defaults (LLM/TTS/aligner model ids, output and cache dirs) also live in `.env`
and are safe to override per project; `VIDEO_OUTPUT_DIR` and `VIDEO_CACHE_DIR` are anchored
to the repo root regardless of where you launch from.

---

## Running it

### The whole stack, one command

```bash
mprocs                 # from the repo root
./scripts/dev.sh       # or this, from anywhere — it cd's for you and preflights deps
```

That starts the backend on **http://127.0.0.1:8000** and the frontend on
**http://127.0.0.1:5173**. Open the UI at <http://127.0.0.1:5173> and fill in the form.

`mprocs` keys: `<tab>`/arrows switch panes, `s` start, `x` stop, `r` restart, `q` quit.
There are three extra panes you start on demand — `tests`, `lint`, `doctor`.

> `mprocs` resolves each pane's `cwd` against the directory it was **launched** from, not
> the config's location (0.8.3 has no `<CONFIG_DIR>` placeholder). Run it from the repo
> root or use `scripts/dev.sh`.

### The CLI path

Render through the real HTTP API and ffprobe the result — no browser involved:

```bash
./scripts/e2e.py "How phishing attacks work and how to spot them" 4
./scripts/e2e.py "Locking your screen" 2 aura-2-thalia-en     # topic, slide count, voice
```

It needs the backend already running on `:8000` (`mprocs`, or just
`cd backend && uv run uvicorn app.main:app --port 8000`).

It polls until the job is `done` or `failed`, prints each stage as it changes, then ffprobes
the output and reconciles the final duration against the timeline's narration total minus
the xfade overlap. Exits non-zero on failure or stall. Measured here: a 2-slide job
completes in **78 s**; expect a few minutes for 4–10 slides.

Score a finished video:

```bash
cd backend
uv run python scripts/evaluate_job.py <job_id>              # includes a Gemini vision pass
uv run python scripts/evaluate_job.py <job_id> --no-vision  # deterministic metrics only, offline, free
uv run python scripts/evaluate_job.py <job_id> --fail-under 80 --fail-on-blocker
```

It writes `out/<job_id>/score.json` and prints a per-scene scorecard with actionable fixes.
Real output from the smoke-test render above:

```
  OVERALL 90.6/100   GRADE A
    scenes 9.0 (60%)   audio 8.8 (15%)   script n/a (15%)   technical 9.8 (10%)
  scene 1: contrast 4.55:1, dup 5.8%, 145 wpm, 14.86s, 4 bullets
  video: 23.27s (expected 23.25s, 0.5 frames drift), 1920x1080 @30fps, h264+aac
  audio: -16.5 LUFS (target -16), true peak -1.0 dBFS, LRA 3.0 LU
  [minor] [AUTO] Narration sits only 8.7 dB above the music bed → duck a further 1.3 dB
```

### Artifacts

Each job writes everything it produced to `out/<job_id>/`, which is what makes a bad render
diagnosable: `scene_NN.png` (generated image), `scene_NN.mp3` (narration take),
`scene_NNN.text/` (rasterised text layers), `scene_NNN.mp4` (per-scene clip), `music.mp3`,
`video.mp4`, and `score.json` if you evaluated it. `out/` and `cache/` are gitignored.

### HTTP API

| endpoint | does |
|---|---|
| `POST /api/jobs` | queue a render; returns `202` with a `job_id` in milliseconds |
| `GET /api/jobs` | 20 most recent jobs |
| `GET /api/jobs/{id}` | status, `progress`, `current_stage`, `error`, `video_url` |
| `GET /api/jobs/{id}/timeline` | the persisted `Timeline` — real word timings, plans, paths |
| `GET /api/jobs/{id}/video` | the MP4 |
| `DELETE /api/jobs/{id}` | delete the row and its `out/` directory |
| `GET /api/themes` | 8 presets with swatches and **measured** contrast ratios |
| `GET /api/engines` | TTS engines, each with a verified `available` flag and a reason if not |
| `GET /api/voices?engine=…` | voices for that engine (53 Deepgram, 7 Polly here) |
| `GET /api/health` | `{"status":"ok"}` |

Stages and their progress values are contractual — the UI stepper renders them:
`scripting 10 → imaging 30 → narrating 50 → aligning 60 → scoring 70 → rendering 90 →
assembling 95 → done 100`.

---

## What you can configure

| option | values | notes |
|---|---|---|
| Slide count | **2 – 10** (UI default 4, API default 5) | total scenes, including title and closing. Below 3 there is no room for a title card, so the opener is what gives |
| Bullets per slide | **3 – 4** (default 4) | 5 is accepted by the API but capped at the role's budget of 4 — at the 11 s scene floor, five bullets need 6.4 s of reveal window and only 6.27 s exists |
| Audience tone | `new_hires`, `all_staff` (default), `technical`, `executives` | a closed set: the script prompt says something concrete for each |
| Theme | 8 presets — dark: `midnight` (default), `graphite`, `halo`, `forest`; light: `daylight`, `boardroom`, `paper`, `lilac` | every preset clears WCAG **AAA** (7.0:1) for text on background; measured 13.3–17.9:1 |
| Custom palette | 5 colours (`bg`, `surface`, `text`, `muted`, `accent`) | validated live in the UI and again server-side. Below WCAG AA it is **rejected** with a `422` carrying a suggested fix; between AA and 7.0 it warns |
| TTS engine | `deepgram` (Aura 2, default) or `polly` (AWS) | Polly parses SSML; Deepgram Aura **vocalises the tags**, so marked-up narration only goes to engines that declare support |
| Voice | per engine, from `GET /api/voices` | a voice belonging to the other engine is a `422`, not a silent substitution |
| Music | on / off | Lyria bed, ducked under the narration; best-effort — a failure leaves narration-only rather than losing the job |

Text is burned into pixels, so an unreadable palette cannot be fixed after the render. That
is why contrast is a gate at submit time and not a warning afterwards.

A few more knobs are environment-only — not in `.env.example`, but read from it if present
(see [`core/config.py`](backend/app/core/config.py)). `VIDEO_ENABLE_VEO=true` opts into
generated video clips (the most expensive call in the pipeline, off by default),
`VIDEO_SCENE_PAUSE_S` sets the audible gap between scenes, `VIDEO_MUSIC_DUCK_DB` the music
level under narration, and `VIDEO_API_CONCURRENCY` the per-provider fan-out (lower it if a
provider starts returning 429).

---

## Repo map

| path | what's in it |
|---|---|
| `backend/` | FastAPI app, the pipeline, providers, renderer, evaluator, ~1860 tests |
| `frontend/` | React 19 + Vite + Tailwind 4 + shadcn UI |
| `docs/` | normative design spec, language research, architecture notes, `images/` (frames used above) |
| `scripts/` | `dev.sh` (launch), `doctor.sh` (preflight), `e2e.py` (real-API smoke test) |
| `out/` | one directory per job — images, takes, clips, music, final MP4 *(gitignored)* |
| `cache/` | provider caches *(gitignored)* |
| `mprocs.yaml` | the one-command dev stack |
| `.env.example` | every key and default, commented |

Inside `backend/app/`: `core/` (models, config, theme registry), `api/` (routers),
`worker/` (pipeline + provider factory), `providers/` (Gemini, Deepgram, Polly, Lyria, Veo,
SSML, bullet timing), `render/` (planner, ffmpeg backend, text rasteriser), `evaluate/`
(scorer, metrics, vision), `db/`.

### Read these first, in this order

1. [`backend/app/core/models.py`](backend/app/core/models.py) — the canonical data model.
   `Timeline` is the load-bearing artifact; `SceneRole` is the video's shape. Start here.
2. [`backend/app/worker/pipeline.py`](backend/app/worker/pipeline.py) — the orchestrator.
   One job, eight stages, the audio-is-the-clock rule, and the two parallel branches.
3. [`backend/app/providers/bullet_timing.py`](backend/app/providers/bullet_timing.py) — how
   a bullet finds its word.
4. [`backend/app/api/jobs.py`](backend/app/api/jobs.py) — the request contract and every
   validation decision, with the reasoning attached.
5. [`docs/DIRECTION.md`](docs/DIRECTION.md) — the normative structure/type/motion spec.
   Every number in the code traces back to a rule in here.
6. [`frontend/src/components/CreateForm.tsx`](frontend/src/components/CreateForm.tsx) — the
   whole product surface in one file.

The comments in this codebase record *why*, usually with the measurement that settled it.
They are worth reading rather than skimming.

---

## Verifying a change

```bash
cd backend
uv run pytest -q                 # ~110s
uv run ruff check app/
uv run python -m app.core.themes # prints the contrast table for all 8 presets

cd ../frontend
pnpm typecheck                   # tsc -b --force  ← THE typecheck gate
pnpm lint                        # oxlint — warnings only today, no errors
pnpm build                       # tsc -b && vite build
pnpm test:contrast               # palette-validation unit test: 164 passed

cd .. && ./scripts/doctor.sh
```

> ### ⚠️ `pnpm exec tsc --noEmit` passes vacuously here. Never use it as a gate.
>
> `frontend/tsconfig.json` is a **solution file** — `"files": []` plus project references.
> `--noEmit` ignores the references, compiles the resulting empty program and exits 0 even
> on a blatant type error. Verified: `tsc --noEmit --listFiles` lists **zero** files.
>
> Use **`pnpm typecheck`** (`tsc -b --force`), which builds the referenced projects.
> `doctor.sh` asserts that the `typecheck` script still exists, to guard the regression.

---

## Current status

Read this before assuming something works.

**Working end to end.** A job submitted through the UI or `scripts/e2e.py` renders to a
finished, narrated, branded 1920×1080 MP4 with word-anchored bullets, a ducked music bed and
a logo. Verified in this session: a 2-slide job completed in 78 s and scored 90.6/100. Both
TTS engines report `available: true`; both Gemini and Deepgram return live 200s.

**Tests.** `1837 passed, 4 failed, 20 skipped` (`uv run pytest -q`, ~110 s). All four
failures are in `tests/test_render.py`, and `app/render/*` is mid-refactor and under active
edit — one of them is a malformed ffmpeg filter chain (`fps_mode`, an output option, ending
up inside `-vf`). Treat the render suite as red until that work lands; the rest of the suite
is green.

**The quality evaluator works, but nothing calls it.** `app/evaluate/` scores a rendered
video across seven per-scene dimensions plus audio, script and technical, with deterministic
ffmpeg metrics and an optional Gemini vision pass. It is reachable only via
`backend/scripts/evaluate_job.py`. There is no import of `app.evaluate` anywhere in the
pipeline or the API, so nothing is scored automatically and no score reaches the UI. The
`scoring` pipeline stage is the **music** stage, despite the name.

**Spanish and Hindi are in progress, not shipped.** `Language` exists in the model and both
languages have been researched end to end
([`docs/LANGUAGES.md`](docs/LANGUAGES.md)), but there is no language selector in the UI and
English is the only path exercised. Spanish is close: 17 Deepgram voices plus 10 Polly
voices, and rendering needs no change — the aligner call needs an explicit `language`.
Hindi is further out and has a trap worth knowing about: **it renders a completely blank
text panel today and nothing detects it**, because the font resolver returns a face with no
Devanagari glyphs, ImageMagick exits 0, and the layout code measures widths from missing
glyphs. Hindi also has zero Deepgram voices — Polly is the only option, with one usable
voice.

**Specified but not built.** `docs/DIRECTION.md` is normative and ahead of the code in
places. The 2 s end card (§1.5), for instance, is fully specified and not implemented — the
video currently ends on the last narrated frame. Don't read the spec as a description of
shipped behaviour.

---

## Further reading

- [`CLAUDE.md`](CLAUDE.md) — conventions and working agreements for anyone (or anything)
  editing this repo. Read it before your first commit.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — **as built**: module boundaries, the
  port/provider seam, and how the render backend stays swappable.
- [`docs/DIRECTION.md`](docs/DIRECTION.md) — **normative**: the structure, type-scale,
  bullet, motion, pacing and layout spec. Every magic number in the code cites a section
  here. It leads the implementation in places — see "Specified but not built" above.
- [`docs/LANGUAGES.md`](docs/LANGUAGES.md) — **measured**: a command-by-command account of
  what Spanish and Hindi actually cost, including the corrections it makes to earlier
  assumptions still recorded in `Language`'s docstring.
