# CLAUDE.md — prompt-to-video-v2

Topic + slide count in, narrated corporate-training MP4 out. FastAPI + SQLite backend
(`backend/`, uv, **Python 3.14**), Vite/React/Tailwind v4/shadcn frontend (`frontend/`, pnpm).
Providers are swapped behind `typing.Protocol` ports, selected by config, imported lazily.

**The code is the source of truth and it is heavily commented — read it before theorising.**
~1860 tests. Companion docs: [`README.md`](README.md) ·
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/DIRECTION.md`](docs/DIRECTION.md) (authoritative on design) ·
[`docs/LANGUAGES.md`](docs/LANGUAGES.md) (authoritative on es/hi).

---

## Run it

| What | Command | Where |
|---|---|---|
| Whole stack (backend :8000 + frontend :5173) | `./scripts/dev.sh` (or `mprocs` from repo root) | repo root |
| Preflight every external dep | `./scripts/doctor.sh` | repo root |
| Backend only | `uv run uvicorn app.main:app --reload --port 8000` | `backend/` |
| Frontend only | `pnpm dev --port 5173 --strictPort` | `frontend/` |
| Real end-to-end render | `./scripts/e2e.py "phishing" 4 [voice]` | repo root |
| Score a finished job (free, offline) | `uv run python scripts/evaluate_job.py <job_id> --no-vision` | `backend/` |
| Score with the vision judge (**costs credits**) | `uv run python scripts/evaluate_job.py <job_id>` | `backend/` |

`mprocs.yaml` resolves `cwd` against the **launch** directory, not the config's location
(mprocs 0.8.3 has no `<CONFIG_DIR>`). Launch from the repo root or use `scripts/dev.sh`.

Deps: `cd backend && uv sync --extra dev` · `cd frontend && pnpm install --force`.
Config is `.env` at the **repo root** (see `.env.example`); `backend/app/core/config.py` reads it.

---

## Verification gates — and the one that lies

| Gate | Command | Cost |
|---|---|---|
| Backend tests | `uv run pytest -q` (in `backend/`) | ~110s, offline |
| Backend lint | `uv run ruff check app/` (in `backend/`) | instant |
| **Frontend types** | **`pnpm typecheck`** or `pnpm build` (in `frontend/`) | ~10s |
| Frontend lint | `pnpm lint` (oxlint) | instant |
| Everything external | `./scripts/doctor.sh` | ~3s, hits free list endpoints |

### `pnpm exec tsc --noEmit` PASSES VACUOUSLY. Never use it as a gate.

`frontend/tsconfig.json` is a **solution file** (`files: []` + project references). Without
`-b`, tsc compiles an *empty program* and exits 0 on any type error. Proof in one command:

```bash
cd frontend
pnpm exec tsc --noEmit    --listFiles | grep -c '/frontend/src/'   # -> 0   (compiles NOTHING)
pnpm exec tsc -b --force  --listFiles | grep -c '/frontend/src/'   # -> 44  (the real gate)
```

`package.json` defines `typecheck: tsc -b --force`. `doctor.sh` fails if that script disappears.

---

## Things that will surprise you

Every row was measured on this machine. Re-verify with the command, don't re-litigate.

### Environment

| Fact | Re-verify |
|---|---|
| **`ffmpeg` 8.1.1 has NO `drawtext`** — built without libfreetype. ALL text is rasterised to RGBA PNGs by ImageMagick and composited with `overlay`. | `ffmpeg -hide_banner -filters \| grep -c drawtext` → `0`; `ffmpeg -version \| grep -c freetype` → `0` |
| **No `subtitles`/libass either.** Burned captions are opt-in and need the filter to exist; `app/render/captions.py` writes `.ass` but it cannot be burned here. | `ffmpeg -hide_banner -filters \| grep -cE '^ .. (subtitles\|ass) '` → `0` |
| `xfade`, `zoompan`, `overlay`, `geq` **are** present. `app/render/ffmpeg.py::has_filter` caches the probe. | `ffmpeg -hide_banner -filters \| grep -c ' xfade '` → `1` |
| **ImageMagick's PANGO coder is unusable in this build** (`---` = no read, no write). So the Devanagari fix cannot go through `magick pango:`. `pango-view`, `hb-shape` and `fc-list` *are* installed as standalone binaries, and 33 Devanagari faces exist. See `docs/LANGUAGES.md` §3 for the sanctioned route. | `magick -list format \| grep -E '^\s+PANGO'` → `PANGO ---` |
| Devanagari through plain freetype **drops the nukta and anusvara** (फ़िशिंग → फिशिग). Needs real shaping. | `docs/LANGUAGES.md` §3 |
| **`rsvg-convert` is missing**, so ImageMagick falls back to its own MSVG renderer, which implements neither `<mask>` nor `<filter>`. `text_overlay.flatten_svg_paths` keeps only the root's **direct `<path>` children** and writes `<stem>.flat.svg`. | `command -v rsvg-convert` → missing; `magick -list delegate \| grep svg` |
| Text is never interpolated into argv — it goes via `@file` (`write_text_file` / `imagemagick_text_arg`). Keep it that way. | `app/render/text_overlay.py` |
| Fonts are addressed by **absolute path**, not fontconfig family (`FONT_CANDIDATES`, override with `VIDEO_FONT_FILE`). | `app/render/text_overlay.py:84` |

### Provider APIs — these contradict the vendors' own docs

| Fact | Re-verify |
|---|---|
| **Deepgram Aura does not support SSML and there is no flag.** 15 request shapes tested: `{"ssml":…}` → 400, `application/ssml+xml` → 415, and `?ssml=true` / `?enable_ssml=true` / `?input_type=ssml` / `?text_type=ssml` / `X-Deepgram-SSML` all return **200 while vocalising the tags** ("break time equals eight hundred milliseconds"). aura-1 silently **corrupts the adjacent word**. Deepgram staff confirmed on record it is not on the roadmap. `supports_ssml = False`. | `uv run pytest tests/test_deepgram_ssml.py -q`; evidence in the `app/providers/deepgram_tts.py` module docstring |
| **A 200 is not evidence a flag exists.** Unknown query params are accepted-and-ignored by many of these APIs. Always round-trip the *result* (e.g. TTS → STT) before believing a capability. | — |
| **Polly SSML support varies by engine tier** (measured on this account): `<emphasis>` works **only** on `standard`; `<prosody pitch>` fails on neural/long-form/generative; generative `<prosody rate>` is **quantized to a no-op**; `<prosody volume>` **does** work phrase-scoped on generative — contradicting AWS's docs, which claim sentence scope only. **Use volume for stress.** Encoded in `providers/ssml.py::CAPABILITIES` and `polly_tts.py::adapt_ssml`. | `uv run pytest tests/test_polly_tts.py tests/test_ssml.py -q` |
| **Polly emits 16 kHz, Deepgram 24 kHz.** `deepgram_tts` joins its own chunks with the concat demuxer + `-c copy`, which requires identical formats. **Do not mix engines inside one video.** | `ffprobe -v error -show_entries stream=sample_rate out/*/scene_01.mp3` → `24000` |
| Gemini image responses are **always `image/jpeg`**, and `inlineData` shares **one part** with `thoughtSignature` — a filter that skips parts containing `thoughtSignature` drops the image. Scan all parts for `inlineData`; never index positionally. Default is 1376×768 (below 1080p) unless `imageConfig` asks for more (`imageSize` `1K`/`2K`). | `app/providers/gemini_image.py`, `_gemini.inline_data_from` |
| Lyria returns a **variable ~30 s** clip (measured 29.57 s and 30.77 s — **never hardcode**) and `parts[0]` is **text**, not audio. `lyria_music.py` measures with ffprobe then loops/crossfades to the exact target. | `app/providers/lyria_music.py` |
| Veo 3.1: **8.000 s fixed**, 1280×720, 24 fps, ships an audio track that must be stripped. Download needs `x-goog-api-key` as a **header** plus `follow_redirects=True`. Gated off by default (`VIDEO_ENABLE_VEO=false`) — it is the most expensive call in the pipeline. | `uv run pytest tests/test_veo_video.py -q` |
| Gemini `responseSchema` works with UPPERCASE type names and honours enums. | `app/providers/gemini_script.py` |

---

## Invariants — breaking any of these ships a subtly wrong video

1. **Audio is the clock.** Narration is synthesised first, aligned second. Scene `start`/`end`
   come from measured audio (ffprobe, then word timings) — **never** from word counts.
   Break it → every bullet and transition drifts. `app/worker/pipeline.py::_measured_duration`.
2. **`xfade` consumes its overlap.** `final = sum(scene durations) − sum(transition durations)`.
   `Timeline.final_duration()` computes it; `assemble()` re-probes the output and raises
   `DurationMismatchError` when `|drift| > max(0.1s, (n_scenes+2)/fps)` (`strict_duration=True`
   is the default — do not turn it off to make a test pass). Break it → ~0.3 s desync per scene.
3. **The aligner must receive PLAIN `narration`, never SSML.** Every on-screen bullet is anchored
   to a verbatim n-gram in the narration (`app/providers/bullet_timing.py`); tags in the
   reference text corrupt every anchor. SSML may add pauses/prosody but **must never change the
   words** — enforced by a round-trip test (`ssml.build_ssml` → `strip_ssml` → `tokenize`, raising
   `SsmlInvariantError`). `Scene.narration` stays the plain source of truth; `Scene.ssml` is
   additive and only reaches an engine whose `supports_ssml` is True.
4. **`zoompan` truncates x/y to integers**, so slow moves visibly step. Mitigated by pre-upscaling.
   `RenderProfile.upscale_factor` is **not** a direct multiplier — it scales an *area budget*
   (`CANVAS_PIXEL_BUDGET * factor / UPSCALE_BUDGET_BASIS`, basis 4) in
   `ffmpeg_backend.upscale_factors()`. **Ken Burns now runs inside a smaller image REGION, so the
   factor needs recalibrating to region size** — the evaluator still reports ~30% duplicate frames
   at `upscale_factor=4` and recommends 8. Verify with the evaluator, not by eye.
5. **`create_all` will NOT add a column** to the existing `backend/videos.db`. There is an
   idempotent `ALTER TABLE` migration in `app/db/models.py::migrate` (`_ADDED_COLUMNS`) run from
   `init_db()` on every startup — **extend that tuple**, never hand-edit the DB.
6. **SQLite drops tzinfo.** A naive timestamp is read as *local* time by ECMAScript, so clients
   misreport age by their whole UTC offset. Re-attach UTC when serialising
   (`app/api/jobs.py::_as_utc`).
7. **Per-scene clips render on up to 4 concurrent threads** (`RenderProfile.resolve_concurrency`,
   `ThreadPoolExecutor` in `render_all`). Everything in the render path must be thread-safe:
   content-addressed cache keys, per-path locks, and **atomic writes** (`mkstemp` in the target
   dir → `os.replace`, see `text_overlay.render_cached_png`). **No fixed temp filenames.**
   `assemble()` is inherently serial. Known latent exception: `<stem>.flat.svg` is a fixed name
   written unlocked — safe today only because `assemble()` is single-threaded.
8. **`VIDEO_OUTPUT_DIR=./out` is CWD-relative** and is anchored to the repo root by a validator in
   `config.py::_anchor_to_repo`. Without it you get two artifact trees and finished videos vanish
   from the API. `doctor.sh` fails if `backend/out/` ever appears.
9. Providers are resolved **lazily, one import per function** (`app/worker/factory.py`). A provider
   mid-edit must surface as a failed job with a readable message, not a dead web server. Do not
   hoist those imports to module scope. (This is not hypothetical — `app/render/ffmpeg_backend.py`
   had a transient `SyntaxError` while this file was being written.)
10. Provider failure is per-stage: images/narration/alignment fan out with
    `return_exceptions=True` and report **every** failing scene. Music is best-effort — losing the
    bed must never lose a completed script, images and narration.

---

## Design rules — `docs/DIRECTION.md` is authoritative

> The governing rule: **repetition is the design.** We shipped a "variety engine" at a topic
> where variety is the enemy. One layout, one transition, one camera move, one entrance.

| Rule | Value |
|---|---|
| Bullet marker | **ONE shape per video** (`dash`). Never varies within a video. |
| Bullet emphasis | **OFF by default.** Weight-only emphasis shifts the ink baseline 8 px and reads as a rendering fault. |
| Heading sizes | **Exactly two.** `SceneRole.heading_scale` is 1.35 for TITLE, 1.0 for everything else. |
| Transition | `fade`, 0.35 s. One per video. |
| Layouts | TITLE/END → `title_card`; everything else → `hero_right`. `hero_left`, `image_band`, `full_bleed` are **retired** — enum members stay so old timelines deserialise, but the planner must never emit them. |
| Text colour | `uniform_text=True`. Emphasis is non-chromatic; `accent` colours graphics only. |
| Contrast | Every palette must clear WCAG AA 4.5:1. A custom palette that fails is a **422**, not a style choice — text is burned into pixels and cannot be fixed after the render. |

**`hero_right`'s image region is 4:5 — exactly 720×900 at x=1096, y=90** (verified: `1624` inner
width × `904/1624` text share → 720 wide, `1080 − 2×90` → 900 tall, ratio 0.800).
**Stills must be REQUESTED at 4:5.** Images come back 2752×1536 (1.79) and get cropped to 0.80,
discarding 55% of the width — *this* was the real cause of poor image relevance, not the prompts.

```bash
# canonical check (needs ffmpeg_backend.py importable)
uv run python -c "from app.core.models import RenderProfile,VisualPlan,SlideLayout as L; \
from app.render.ffmpeg_backend import layout_region as r; \
print(r(VisualPlan(layout=L.HERO_RIGHT), RenderProfile()))"
```

⚠️ Doc drift: `ffmpeg_backend.py`'s module docstring still says `hero_right` is 856×816. The real
region is **720×900**. Trust `layout_region` / `text_overlay.slide_geometry`.

---

## Ownership map

| Path | Owns |
|---|---|
| `backend/app/core/models.py` | **Canonical data model.** `Timeline`, `Scene`, `VisualPlan`, `Theme`, `RenderProfile`, all enums. Every module reads and writes these. |
| `backend/app/core/ports.py` | **The swappable seams** — `Protocol`s: `ScriptProvider`, `ImageProvider`, `VideoClipProvider`, `SpeechSynthesizer`, `Aligner`, `MusicProvider`, `VisualPlanner`, `VideoBackend`. |
| `backend/app/render/contracts.py` | **Text-rasterisation ↔ filtergraph seam** — `TextLayer`, `SceneText`. Render-time artifacts, deliberately not in the persisted Timeline. |
| `backend/app/core/config.py` | `Settings` from root `.env`. Provider selection is data, not code. |
| `backend/app/core/themes.py` | Palette presets. |
| `backend/app/providers/*` | One adapter per vendor + `ssml.py` (SSML build/strip/validate) + `bullet_timing.py` (anchoring). |
| `backend/app/render/*` | `planner.py` (pure, deterministic, no I/O) → `text_overlay.py` (rasterise) → `ffmpeg_backend.py` (filtergraph, render, assemble) → `ffmpeg.py` (subprocess + probes) → `captions.py`. |
| `backend/app/worker/factory.py` | Lazy provider resolution + `SPEECH_ENGINES` catalogue. |
| `backend/app/worker/pipeline.py` | The orchestrator. Stage order and progress values are **contractual** — the frontend stepper renders them. |
| `backend/app/db/*` | `Job` row, additive `migrate()`, WAL SQLite session. |
| `backend/app/api/*` | `jobs`, `engines`, `themes`, `voices`, `logos` (uploaded brand marks). |
| `backend/app/evaluate/*` | `metrics.py` (deterministic ffmpeg/ffprobe/mpdecimate, free) · `vision.py` (Gemini, **costs credits**) · `scorer.py` (grades, auto-fixable recommendations). |
| `frontend/src/lib/types.ts` | TS mirror of the API contract. |
| `out/`, `cache/`, `*.db` | Generated, gitignored. Never commit. |

Pipeline stages (`app/worker/pipeline.py::STAGE_PROGRESS`):
`scripting 10 → imaging 30 → narrating 50 → aligning 60 → scoring 70 → rendering 90 → assembling 95 → done 100`.
The visual branch (images) and audio branch (narrate → align → music) run concurrently.

API: `GET /api/health` · `POST|GET /api/jobs` · `GET|DELETE /api/jobs/{id}` ·
`GET /api/jobs/{id}/timeline` (**the debug trail** — real word timings, plans, paths) ·
`GET /api/jobs/{id}/video` · `GET /api/engines|themes|voices` · `POST|GET /api/logos`.

---

## Process

| Rule | Why |
|---|---|
| **Never run `git stash`** here. | This is a shared working tree with several agents editing concurrently. An agent ran it and stashed five other agents' uncommitted work. |
| Check before killing :8000. | `lsof -nP -iTCP:8000 -sTCP:LISTEN` — it is usually someone's `mprocs`. Use another port for throwaway servers. |
| Expect a dirty tree and transiently broken modules. | `git status` will show other agents' work in progress. Do not "fix" or revert files you were not asked to own. A `SyntaxError` in a sibling module is probably someone mid-edit, not a bug for you. |
| Don't spend credits casually. | `./scripts/e2e.py` and `evaluate_job.py` without `--no-vision` call paid APIs. Veo is gated off. Use `--no-vision` and the existing jobs in `out/` for iteration. |
| Measure, don't eyeball. | `ffprobe`, `mpdecimate` and `evaluate_job.py --no-vision` will tell you what actually rendered. "Looks fine" has been wrong every time. |
| AWS creds on this account are **temporary STS** and expire within hours. | `aws_configured` means "configured", not "unexpired". A Polly job can fail mid-render. |
