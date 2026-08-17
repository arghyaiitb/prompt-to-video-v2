# ARCHITECTURE — as built

> This document describes what the code **does today**, not what it should do.
> Normative design intent lives in `docs/DIRECTION.md`; language constraints in `docs/LANGUAGES.md`.
> Every path, class and function named here exists. Verified against the tree on 2026-08-17.
>
> Two areas are **in flux as of writing**: `backend/app/render/*` and
> `backend/app/providers/gemini_script.py` are being actively edited. Where a detail depends on
> them it is flagged `[in-flux]`.

---

## 1. Thesis

**This is a topic-to-video pipeline whose entire design rests on two ideas.** First, every
external service — the LLM that writes the script, the image model, the TTS engine, the forced
aligner, the music model, the renderer itself — sits behind a `typing.Protocol` port in
`backend/app/core/ports.py`, so swapping a vendor is one new class plus one config value and
never a change to the orchestrator. Second, the `Timeline` in `backend/app/core/models.py` is the
single load-bearing artifact: the script provider creates it, narration and alignment stamp real
measured times onto it, the planner decorates it with `VisualPlan`s, the bullet timer writes
reveal moments into it, the renderer consumes it, the database persists it as
`Job.timeline_json`, and the frontend's scene inspector reads it back over HTTP. Nothing in the
system passes ad-hoc tuples between stages; every stage reads a `Timeline` and returns a
`Timeline`. The corollary that organises all timing decisions is stated in
`app/worker/pipeline.py`'s own module docstring: **audio is the clock.**

---

## 2. Repository map

| Path | Role |
|---|---|
| `backend/app/core/models.py` | The canonical data model. `Timeline`, `Scene`, `SceneRole`, `SlideLayout`, `Theme`, `Language`, `VisualPlan`, `RenderProfile`, `BulletPoint`, `Word`, `JobStatus` |
| `backend/app/core/ports.py` | 8 `Protocol` ports — the swappable seams |
| `backend/app/core/config.py` | `Settings` (pydantic-settings) read from `<repo>/.env`; `get_settings()` is `lru_cache`d |
| `backend/app/core/themes.py` | 8 validated palettes + the WCAG gate (`validate_theme`, `review_theme`, `suggest_fix`) |
| `backend/app/worker/pipeline.py` | The orchestrator. `run_job(job_id)` and the 8 stages |
| `backend/app/worker/factory.py` | Provider resolution. Every import is lazy, inside a function |
| `backend/app/api/` | FastAPI routers: `jobs.py`, `themes.py`, `engines.py`, `voices.py`, `logos.py` |
| `backend/app/db/` | `Job` row (SQLModel), SQLite engine, WAL pragmas, additive migration |
| `backend/app/providers/` | Concrete adapters: Gemini, Deepgram, Polly, Lyria, Veo + `bullet_timing.py`, `ssml.py` |
| `backend/app/render/` | `planner.py` (pure), `ffmpeg_backend.py` (execution), `text_overlay.py` (rasterisation), `contracts.py` (the seam), `ffmpeg.py` (subprocess), `captions.py` |
| `backend/app/evaluate/` | Offline scorer. **Not wired into the pipeline** — see §11 |
| `backend/scripts/evaluate_job.py` | The only caller of the evaluator |
| `frontend/src/` | React 19 + Vite SPA |
| `mprocs.yaml` | `mprocs` from the repo root runs backend + frontend; `tests`/`lint`/`doctor` panes on demand |

Line counts, largest first: `text_overlay.py` 3414, `ffmpeg_backend.py` 1903, `gemini_script.py`
1566, `scorer.py` 1085, `ssml.py` 1083, `polly_tts.py` 1075, `metrics.py` 930, `veo_video.py` 752.

---

## 3. The `Timeline` contract

```
Timeline
├── job_id, topic, title, voice
├── language: Language = EN          # defined, NOT plumbed — see §13
├── music_path: str | None
├── logo_path: str | None            # per-job brand mark, overrides settings
├── profile: RenderProfile           # 1920x1080@30, crf 18, upscale_factor 4
├── theme: Theme                     # persisted so a re-render reproduces branding
└── scenes: list[Scene]
        ├── id, role: SceneRole, narration, heading, image_prompt
        ├── clip_prompt, ssml         # optional alternates
        ├── image_path / video_path / audio_path / clip_path   # artifacts on disk
        ├── start, end                # GLOBAL seconds — only real after alignment
        ├── words: list[Word]         # GLOBAL timings (rebased in _stage_align)
        ├── bullets: list[BulletPoint]  # appear_at is SCENE-RELATIVE
        └── plan: VisualPlan | None   # written by the planner
```

Two timebases coexist and this is the single most common source of bugs:
`Scene.words[*].start` is **global**, `BulletPoint.appear_at` is **scene-relative**. The
conversion happens in exactly one place, `bullet_timing.time_bullets`, which takes `scene_start`
and subtracts it.

`Timeline.final_duration()` is the arithmetic that keeps picture and voice together:
`narration_duration - sum(transition_duration for every non-CUT scene after the first)`, because
`xfade` **consumes** its overlap. `FFmpegBackend.assemble` checks its own ffprobe'd output
against this number and raises `DurationMismatchError` rather than shipping a file whose length
it cannot explain.

`SceneRole` is where video *structure* lives. It is not decoration — each role carries three
derived properties read across the codebase:

| Role | `target_duration` (s) | `bullet_budget` | `heading_scale` |
|---|---|---|---|
| `TITLE` | 4.0 – 6.5 | 0 | 1.35 |
| `CONTENT` | 11.0 – 19.0 | 4 | 1.0 |
| `SUMMARY` | 9.0 – 14.0 | 4 | 1.0 |
| `CLOSING` | 6.0 – 9.0 | 2 | 1.0 |

---

## 4. The pipeline as a DAG

`app/worker/pipeline.py:run_job` drives one job. Stage names and progress values are contractual
— `frontend/src/components/StageStepper.tsx` renders them.

```
                          ┌──────────────────────────────────────────────┐
   POST /api/jobs         │  asyncio.gather(visual_branch, audio_branch)  │
        │                 │  ── the two branches are INDEPENDENT ──       │
        ▼                 │                                              │
  ┌───────────┐           │   VISUAL                                     │
  │ scripting │──────────▶│   ┌──────────────────────┐                   │
  │    10%    │  Timeline │   │ imaging  (30%)       │                   │
  └───────────┘           │   │ fan-out: N images    │                   │
                          │   │ semaphore = 4        │                   │
                          │   └──────────────────────┘                   │
                          │                                              │
                          │   AUDIO  (strictly ordered)                  │
                          │   ┌──────────┐  ┌──────────┐  ┌───────────┐  │
                          │   │ narrate  │─▶│  align   │─▶│  music    │  │
                          │   │  (50%)   │  │  (60%)   │  │ ("scoring"│  │
                          │   │ fan-out N│  │ fan-out N│  │   70%)    │  │
                          │   └──────────┘  └──────────┘  └───────────┘  │
                          │        ▲              │            ▲         │
                          │        │        scene start/end     │        │
                          │        │        become REAL here    │        │
                          │        │              └────────────┘         │
                          │        │        music is sized from the      │
                          │        │        MEASURED narration length    │
                          └────────┼─────────────────────────────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │ plan  (90%) │  RuleBasedPlanner.plan — pure
                            └──────┬──────┘
                                   ▼
                            ┌─────────────┐
                            │  bullets    │  time_bullets — pure, no I/O
                            └──────┬──────┘
                                   ▼
                            ┌────────────────────────────────┐
                            │ render                         │
                            │ FFmpegBackend.render_all       │
                            │ ThreadPoolExecutor, 4 workers  │  ← FANS OUT
                            │ one .mp4 per scene, crf 12     │
                            └──────┬─────────────────────────┘
                                   ▼
                            ┌────────────────────────────────┐
                            │ assemble (95%)                 │
                            │ ONE ffmpeg process             │  ← INHERENTLY SERIAL
                            │ xfade chains clip N into N+1   │
                            │ + narration + ducked music     │
                            │ + logo + loudnorm              │
                            └──────┬─────────────────────────┘
                                   ▼
                              done (100%)
```

### 4.1 Why the branches run concurrently

`pipeline.py:475-489`. Images depend on nothing but their own prompt; the soundtrack depends on
nothing from the pictures. Only the render needs both. Running them in series wasted the whole of
whichever branch finished first. The audio branch is *internally* serial by necessity: alignment
needs the audio files, and the music bed is sized from `timeline.narration_duration`, which does
not exist until alignment has run.

### 4.2 Why work fans out *within* a stage

`_stage_images`, `_stage_narrate` and `_stage_align` each build one coroutine per scene and pass
them to `_gather_scenes`, which is `asyncio.gather(..., return_exceptions=True)` so one bad scene
does not cancel its siblings mid-flight and leave half-written files behind. Every call is bounded
by `asyncio.Semaphore(settings.video_api_concurrency)` — default **4** — so a 12-scene job cannot
trip a provider's rate limit.

`_stage_align` is the subtle one: the *network* calls fan out, but the *rebasing* cannot, because
each scene's offset is the sum of all previous durations. So the code gathers first, then walks
the scenes in order assigning a cursor (`pipeline.py:311-325`).

### 4.3 Why assembly is serial

`FFmpegBackend.assemble` builds one filtergraph in which `xfade` chains clip *N* into clip *N+1*:
`[c0][c1]xfade:offset=…[x1]`, `[x1][c2]xfade:offset=…[x2]`, and so on. Each stage's input is the
previous stage's *output*, and the offsets are cumulative over already-shortened material. There
is no parallelism to extract from a chain where every link needs its predecessor's frames.

### 4.4 Measured timings

These come from the project's own benchmarking notes; they are **not** reproducible from any file
in the repo, so treat them as recorded measurements rather than a test you can re-run.

| Phase | Before | After | Speed-up | Note |
|---|---|---|---|---|
| Pre-render (script → images ∥ audio) | 159 s | 33 s | **4.8×** | from branch concurrency + per-stage fan-out |
| Render (scene clips) | 310 s | 87 s | **3.56×** | byte-identical output |
| Total job | 321 s | ~60–186 s | — | depends on scene count |

The render concurrency shape **is** verifiable and I checked it:
`RenderProfile.resolve_concurrency` computes
`workers = render_concurrency or min(4, max(1, cpu // 3))` then
`threads = encoder_threads or max(1, cpu // workers)`. On this 12-core machine that is
**4 workers × 3 encoder threads**. The `//3` and the thread cap exist because libx264 scales
sublinearly with threads — several narrower processes beat one wide one for independent clips —
and because four x264 instances each grabbing every core just contend
(`FFmpegBackend._thread_args`).

### 4.5 Stage/progress quirks worth knowing before you debug a status bar

* `JobStatus.SCORING` (70%) means **musical scoring**, i.e. `_stage_music`. It has nothing to do
  with `app/evaluate/`. This trips people up constantly.
* Progress is **not monotonic**. `run_job` sets `NARRATING` (50%) *before* `asyncio.gather`, and
  then `visual_branch` sets `IMAGING` (30%) inside the gather (`pipeline.py:476, 488`). A poll
  landing in between reports 50 then 30 then 60.
* `_stage_plan` and `_stage_bullets` execute *after* `_set_stage(RENDERING)`, so the row says
  "rendering" while the planner is still running. The `Timeline` is deliberately persisted
  between planning and rendering so a render crash is diagnosable.
* Every blocking call is wrapped: `asyncio.to_thread(...)` for provider HTTP, ffmpeg subprocesses,
  and every DB write. Nothing in `pipeline.py` may stall the event loop, because the API is
  answering status polls the whole time.
* `run_job` never raises. Failure lands in the row (`status=failed`, `error` truncated to 2000
  chars). `asyncio.CancelledError` is caught, recorded as
  `"cancelled: server shut down before the render finished"`, and re-raised.
  `_discard_empty_job_dir` removes the husk directory a job that failed early left behind.

---

## 5. "Audio is the clock"

This is the central design decision, and it is why the DAG has the shape it does.

Word counts are a lie. A voice model's real pace varies with punctuation, numbers, and mood —
`docs/DIRECTION.md` §5 records **120.8–150.2 wpm measured across four scenes of one video**, a
24% swing inside a single render. So narration is synthesised first, aligned second, and only
then do scene boundaries exist.

```
scene.narration  ──TTS──▶  scene_NN.mp3  ──ffprobe──▶  real duration
                                │
                                └──aligner──▶  list[Word] with real start/end
                                                     │
        cursor = 0                                   │
        for scene:                                   ▼
            scene.start = cursor            _measured_duration(audio, words)
            scene.end   = cursor + duration    1. ffprobe container duration  (trusted)
            words rebased by + cursor          2. max(w.end for w in words)   (fallback)
            cursor = scene.end                 3. _MIN_SCENE_DURATION = 1.0   (last resort)
                                                     + _scene_tail_pad()
```

`_measured_duration` (`pipeline.py:145`) prefers ffprobe over the aligner, and both over any
estimate. `_scene_tail_pad()` = `video_scene_pause_s` (default **1.0 s**) + `_XFADE_BUDGET`
(**0.5 s**) — the crossfade budget is added on top because `xfade` consumes overlap and padding
by only the desired pause would leave roughly half of it audible.

Everything downstream is a function of those measured numbers:

| Consumer | What it derives |
|---|---|
| `RuleBasedPlanner._transition_duration` | transition clamped to 40% of the *shorter* adjacent scene |
| `RuleBasedPlanner._anim_duration` | entrance length, capped by scene length, reveal gaps, and a 2.6 s tail margin |
| `bullet_timing.time_bullets` | every bullet reveal time |
| `_stage_music` | `target_duration = timeline.narration_duration` |
| `FFmpegBackend.render_all` | frame counts, rounded **cumulatively**: `round(end*fps) - round(start*fps)` |
| `FFmpegBackend._video_chain` | every `xfade` offset |
| `FFmpegBackend._audio_chain` | every `adelay` that glues a voice segment to its picture |

Cumulative frame rounding is not pedantry: rounding each scene independently leaks up to half a
frame per scene, and the error accumulates across the video.

---

## 6. The bullet-anchoring chain

This is the most subtle mechanism in the codebase and the easiest thing to break. Four components
have to agree.

```
 ┌─ 1 ─────────────────────────────────────────────────────────────────────┐
 │ app/providers/gemini_script.py  [in-flux]                               │
 │ The prompt REQUIRES verbatim overlap:                                   │
 │                                                                         │
 │   "CRITICAL: each bullet must reuse 2 or more CONSECUTIVE CONTENT WORDS  │
 │    from that same scene's narration, verbatim. […] So if the narration   │
 │    says 'hover over the link to reveal the real destination', the bullet │
 │    is 'Hover Over The Link' or 'Reveal The Real Destination' — not       │
 │    'Link Safety'."                                                      │
 │                                                                         │
 │ Bullets must also be listed in narration order. `anchoring_supported()`  │
 │ PROBES `bullet_timing.anchor_position` on a per-language sample rather   │
 │ than hardcoding which languages can anchor.                             │
 └────────────────────────────────┬────────────────────────────────────────┘
                                  ▼
 ┌─ 2 ─────────────────────────────────────────────────────────────────────┐
 │ app/providers/deepgram_align.py :: DeepgramAligner.align(audio, ref)    │
 │ POST /v1/listen (nova-3, smart_format, punctuate), raw wav body.        │
 │ The REFERENCE TEXT IS NEVER SENT — no keyterm, keywords, paragraphs or  │
 │ search param. Deepgram supplies WHEN; the reference supplies WHAT.      │
 │ `align_tokens(tokenize(ref), stt_words, total)` then pairs them LOCALLY │
 │ with difflib.SequenceMatcher(autojunk=False) — autojunk must stay off,  │
 │ it would classify common words as noise past 200 tokens. `equal` and    │
 │ equal-length `replace` borrow the transcript's clock; other opcodes     │
 │ become gaps interpolated by token character length and marked           │
 │ INTERPOLATED_CONFIDENCE = 0.0. `_enforce_monotonic` clamps the result.  │
 │ `Scene.narration` is the reference. `Scene.ssml` MUST NOT be used here. │
 └────────────────────────────────┬────────────────────────────────────────┘
                                  ▼
 ┌─ 3 ─────────────────────────────────────────────────────────────────────┐
 │ app/providers/bullet_timing.py :: find_anchors(...)                     │
 │  a. `_scene_words` filters the (global) word list to this scene's span   │
 │  b. `_content_words` normalises and drops ~110 stopwords                │
 │  c. `_best_ngram` → longest contiguous run of the BULLET's content words │
 │     present in the narration; the narration may interleave stopwords     │
 │     ("check the sender domain" anchors "sender domain"). `_find_run`     │
 │     searches from `search_from`, so repeated phrases resolve left-to-    │
 │     right and bullet ORDER survives.                                    │
 │  d. else `_best_fuzzy` → sliding SequenceMatcher window ±1 word,         │
 │     FUZZY_THRESHOLD = 0.55 (calibrated: inflection drift scores          │
 │     0.57–0.70, an unrelated bullet tops out around 0.42)                │
 │  e. else method = "proportional" — spread evenly and SAY SO             │
 │ Returns BulletAnchor(word_index, match_len, method, matched_words,       │
 │ anchor_time) — `matched_words` is the receipt that proves a real match.  │
 └────────────────────────────────┬────────────────────────────────────────┘
                                  ▼
 ┌─ 4 ─────────────────────────────────────────────────────────────────────┐
 │ app/providers/bullet_timing.py :: time_bullets(...)                     │
 │  · scene-relative: anchor_time − scene_start                            │
 │  · LEAD = 0.25 s — text lands a beat BEFORE its word is spoken          │
 │    (deliberately asymmetric: early reads as arrival, late as a caption) │
 │  · `_space_out` enforces monotonicity and `min_gap`:                    │
 │      forward pass pushes inversions later;                              │
 │      if that overruns `ceiling`, a backward pass pulls earlier bullets   │
 │      IN rather than dropping the overflowing one;                        │
 │      if even min spacing cannot fit, the gap shrinks uniformly to        │
 │      ceiling/(n−1) — tighter than ideal, but never lost content         │
 │  · ceiling = scene_duration − TAIL_GUARD (0.4 s)                        │
 │  · `_emphasis_index` picks ONE bullet: +3 imperative opener,             │
 │    +2 verbatim n-gram, +1 per anchored word                             │
 └─────────────────────────────────────────────────────────────────────────┘
```

**The consequence of feeding the aligner anything but plain text.** `align()` tokenises the
reference and pairs those tokens against the transcript. Hand it SSML and every fragment of markup
becomes a reference token: the alignment diff goes off the rails, the returned word list no longer
corresponds to the narration the bullets quote, so `_best_ngram` fails everywhere, `find_anchors`
falls back to `method="proportional"` for every bullet, and the bullets still appear — evenly
spaced, silently wrong, with no error anywhere. `Scene.ssml` exists as a *separate field* precisely
so `Scene.narration` can stay the plain-text source of truth for both display and alignment
(`models.py:352-358`).

**`normalize` is the reason this only works in Latin script.** Both the matcher and the aligner go
through `deepgram_align.normalize`, which is `re.sub(r"[^a-z0-9]+", "", token.lower())` — lowercase
**ASCII alphanumerics only**. It makes `don't`/`dont` and `world-class`/`worldclass` compare equal,
and it makes every Devanagari token normalise to `""`. So no Hindi bullet can ever anchor.
`gemini_script.anchoring_supported(language)` *probes* this with `_ANCHOR_PROBE` — one
`(narration, bullet)` pair per language where the bullet is a verbatim run — rather than hardcoding
"Hindi cannot anchor", so the file will start reporting Hindi as anchorable the day `normalize` is
widened. When the probe fails, `_clean_bullets` **skips the anchoring and ordering defences
entirely** and keeps the model's bullets in the model's order, because otherwise every bullet would
look unanchored and good copy would be shredded in favour of mechanically sliced fragments.

**`Scene.ssml` is never written.** It is *read* at `pipeline.py:279` and it is always `None`:
nothing under `app/` calls `ssml.build_ssml`, and `app/providers/ssml.py` (1083 lines, with a
measured per-tier capability matrix) is imported only by `tests/test_ssml.py`. So Polly always
receives plain narration with `TextType="text"`, and the whole `supports_ssml` routing — correct as
it is — is currently unexercised in production. See §13.

The same hazard exists one step earlier. `_stage_narrate` reads
`send_ssml = getattr(synth, "supports_ssml", False)` and passes `scene.ssml` **only** when the
engine declares support. It uses `getattr` with a `False` default rather than a direct attribute
read, because absent must mean plain text, never "assume it copes".

A second, independent reveal-time floor lives in the renderer:
`text_overlay.bullet_times(bullets, plan, first_at=FIRST_REVEAL_EARLIEST)` re-applies
monotonicity and `plan.bullet_min_gap`, and no bullet may appear before `FIRST_REVEAL_EARLIEST`
(1.15 s). So reveal times are clamped twice, in two modules, by design.

---

## 7. The ports, and what actually implements them

`app/core/ports.py` — structural typing, `@runtime_checkable`, no inheritance, no registration.
`app/worker/factory.py` resolves each one; **every import is inside a function** so a provider
module that is mid-edit surfaces as a failed job with a readable message rather than a dead web
server, and `_construct` passes only the kwargs a constructor actually declares.

| Port | Method | Implementations | What varies |
|---|---|---|---|
| `ScriptProvider` | `generate(topic, slide_count, *, bullets_per_slide=4, tone=None) -> Script` | `GeminiScriptProvider`, `VerbatimScriptProvider` | LLM generation vs verbatim passthrough of user text. Gated on `VIDEO_DEFAULT_LLM_PROVIDER == "gemini"` |
| `ImageProvider` | `generate(prompt, out_path, width, height) -> Path` | `GeminiImageProvider`, `PlaceholderImageProvider` | Model + aspect snapping. Gemini's "16:9" is really 1.792:1 (2752×1536); the renderer fits, never stretches |
| `VideoClipProvider` | `generate(prompt, target_duration, out_path) -> Path` | `VeoVideoProvider`, `PlaceholderVideoProvider` | `target_duration` is a **request, not a guarantee**: Veo returns a fixed ~8 s at 1280×720/24 fps with an unwanted audio track. Gated behind `VIDEO_ENABLE_VEO` (default **false**) because constructing it *is* the decision to spend |
| `SpeechSynthesizer` | `supports_ssml: bool`; `synthesize(text, voice, out_path) -> Path` | `DeepgramSynthesizer` (`supports_ssml=False`), `PollySynthesizer` (`supports_ssml=True`) | See below |
| `Aligner` | `align(audio_path, reference_text) -> list[Word]` | `DeepgramAligner` (`nova-3`) | Deliberately separate from TTS: some vendors return timings with synthesis, some need a second STT pass. Fusing them would mean swapping TTS silently breaks captions. The reference text is matched **locally**, never uploaded |
| `MusicProvider` | `generate(mood, target_duration, out_path) -> Path` | `LyriaMusicProvider` | Lyria returns ~29.6–30.8 s (**not** a fixed 30 — measured), so the real length is ffprobe'd every call and the clip is looped with `acrossfade` or trimmed with a fade to hit the exact target |
| `VisualPlanner` | `plan(timeline) -> Timeline` | `RuleBasedPlanner` | Pure and deterministic: no network, no ffmpeg, no filesystem. Never mutates its input |
| `VideoBackend` | `render_scene(...)`, `render_all(timeline, clip_dir)`, `assemble(timeline, out)` | `FFmpegBackend` | `render_all` is batch, not per-scene, so each backend owns its own parallelism and cross-scene frame accounting |

### `supports_ssml` is a correctness switch, not a preference

Deepgram Aura does not parse SSML — it **vocalises the tags**. Measured on the live key and
round-tripped through `/v1/listen?model=nova-3` (`app/providers/deepgram_tts.py` module
docstring):

* `aura-2`: `<speak>X<break time="1s"/>Y</speak>` transcribes as *"Speak. X. Break time equals
  once. Y."* — words **inserted**.
* `aura-1`: tags are not spoken but mangle the adjacent word (*"carefully"* → *"CarefulLab"*) and
  the break is not honoured — words **corrupted**.
* With no `<speak>` wrapper, everything after the first tag can be **dropped**.

There is no flag, header or content type that turns it on: `{"ssml": …}` returns 400,
`Content-Type: application/ssml+xml` returns 415, and `?ssml=true` / `?input_type=ssml` /
`X-Deepgram-SSML` are silently **ignored** — which is exactly what makes this easy to get wrong.

`supports_ssml` is declared twice on purpose and the two must agree: on the provider instance
(authoritative at synthesis time) and on `factory.SpeechEngine` (so `GET /api/engines` can answer
without credentials, boto3, or a provider module another branch is still writing).
`factory.speech_engine_status` then does a *real* check — API key present, boto3 importable,
module importable — because telling a user an engine works and failing six minutes into a render
is worse than not offering it.

Polly's own quirks, measured on this account: `emphasis` and `prosody pitch` are **standard-only**;
`break`, `prosody rate|volume`, `say-as`, `phoneme`, `sub`, `mark`, `lang` and `p`/`s` work on all
four tiers; `amazon:effect drc` fails on generative and `amazon:domain news` works only on neural.
The failure is `InvalidSsmlException: Unsupported <Engine> feature` — the **tier** decides, not the
voice. Hence `polly_tts.adapt_ssml(ssml, engine)` rewrites `<emphasis>` into equivalent `<prosody>`
and strips `prosody pitch`, unless `strict_ssml=True`. `VIDEO_POLLY_ENGINE` also *filters the voice
catalogue*, because Polly rejects `Engine=generative` for a neural-only voice at synthesis time.

### Transport, retries and caching — per provider

| Provider | Transport | Retry policy |
|---|---|---|
| `GeminiScriptProvider`, `GeminiImageProvider`, `LyriaMusicProvider` | `httpx.post` to `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, via the shared `providers/_gemini.generate_content`. **No Google SDK anywhere in the package** | 3 attempts on `{408,429,500,502,503,504}` + transport errors, **linear** `sleep(2.0 * attempt)` |
| `LyriaMusicProvider` (extra layer) | — | A *second* loop, `EMPTY_PART_ATTEMPTS = 3`, for a 200 with no audio part (observed `finishReason='OTHER'`) — invisible to a status-code retry. It sleeps `1.5 * attempt` and **varies the prompt** each attempt, because an empty candidate can be deterministic per phrasing |
| `VeoVideoProvider` | `httpx`, but `:predictLongRunning` → poll the operation → download with an `x-goog-api-key` **header** (not a query param) and `follow_redirects=True`. The only async-job provider | 3 attempts, same statuses/backoff. Poll interval eases `min(interval*1.25, 30.0)` up to `max_wait=300s`. A timeout deliberately does **not** resubmit (already billed) and exposes `.operation` for `fetch_completed` |
| `DeepgramSynthesizer` | `httpx` `POST /v1/speak`, `Authorization: Token`, body `{"text": …}`, **voice id sent as `model`**. Response is raw wav, validated with `data.startswith(b"RIFF")` | 3 attempts, **no sleep between them** |
| `DeepgramAligner` | `httpx` `POST /v1/listen`, `Content-Type: audio/wav`, raw bytes | 3 attempts, **no sleep between them** |
| `PollySynthesizer` | **boto3** `polly.synthesize_speech`, lazily imported. botocore's own retries are **disabled** (`Config(retries={"max_attempts": 1})`) so backoff is owned here | 4 attempts, `base_delay * 2**(n-1) + uniform(0, base_delay)` — exponential with jitter |

**There is no on-disk cache in `app/providers/`.** The only cache is `polly_tts._voice_cache`, an
in-process dict keyed `"{language}:{engine}"`, cleared by `reset_voice_cache()`.
`settings.video_cache_dir` is created by the lifespan but is used only for logo rasterisation.
Every script, image, narration take and music bed is re-fetched on a re-run.

Two safety behaviours worth knowing: `DeepgramSynthesizer.synthesize` calls `strip_markup`
**unconditionally** and logs a warning if it fires — a belt-and-braces second line of defence
behind `supports_ssml`, and it is word-preserving (each tag becomes a *space*, because deleting it
would weld `carefully<break/>before` into one non-word). And `PollySynthesizer` **refuses**
over-length input (`TextTooLongError`, `MAX_BILLED_CHARS = 3000`) rather than truncating.

---

## 8. The render architecture

### 8.1 One clip per scene, then one assemble pass

```
 for each scene (4 concurrent ffmpeg processes, ThreadPoolExecutor):

   color=c=<theme.bg>   ─────────────────────────┐   solid brand canvas, ALWAYS
                                                 │
   image ──▶ _fit_chain ──▶ [scale to canvas] ──▶ │   cover-crop, or blurred fill
             (lanczos)      zoompan (Ken Burns)   │   past ASPECT_TOLERANCE=0.25
                            ──▶ _corner_chain     │   rounded-rect alphamerge mask
                                                 ├──▶ overlay at region.x/y ──▶ [base]
   OR clip ─▶ fps=30, setpts, _fit_chain,        │   no zoompan on moving footage
             split/xfade loop, tpad ─────────────┘   (a camera move on footage is seasick)

   [base] ──▶ one prep filter + one overlay stage per TextLayer ──▶ [vout]
              format=rgba, fade(alpha=1), optional geq wipe
              overlay x/y is an expression in t with smoothstep / back-out easing

   encode: libx264 crf 12 (INTERMEDIATE_CRF), preset veryfast, -frames:v exact
   verify: ffprobe duration within 1.5/fps of frames/fps, else DurationMismatchError

 then ONCE:

   [c0][c1]xfade ─▶ [x1][c2]xfade ─▶ … ─▶ fade in / fade out / captions ─▶ [faded]
                                                                            │
   [logo]format=rgba ────────────────────────────────────────── overlay ────┘ ──▶ [vout]

   narration_i: aresample, atrim to clip length, apad, afade, adelay(starts[i])
                ──▶ amix(normalize=0) ──▶ [narr]
   music:       atrim, apad, volume(-18 dB), afade in/out ──▶ [music]
                [music][narr_key] sidechaincompress ──▶ [music_ducked]
   [narr_out][music_ducked] amix ──▶ _master_chain: loudnorm(I=-16,TP=-1,LRA=11),
                                     alimiter, asetpts=N/SR/TB, atrim, apad
```

**Why not a monolithic filtergraph?** `ffmpeg_backend.py`'s module docstring is explicit: a
single `filter_complex` for a 12-scene video is unreadable, impossible to debug from an error
message, and forces a full re-render when one image changes. Per-scene clips are near-lossless
(`INTERMEDIATE_CRF = 12`) so chaining them costs nothing visible, and each clip is an independent
process writing its own file — which is exactly what makes the concurrency safe (no shared state,
every temp path derived from `out_path`'s unique stem).

### 8.2 The `SceneText` / `TextLayer` seam

`app/render/contracts.py` is deliberately narrow. `text_overlay` decides **what the text looks
like and where it sits**, and rasterises it. `ffmpeg_backend` decides **how it enters the frame**.
The only shared vocabulary is: a PNG path, a rectangle in output-frame pixels, a time, and an
animation name.

```python
@dataclass(frozen=True)
class TextLayer:
    png_path: Path;  x: int; y: int; width: int; height: int
    appear_at: float = 0.0;  disappear_at: float | None = None
    animation: TextAnimation = FADE_IN;  anim_duration: float = 0.45
    slide_distance: int = 60
    kind: str = "bullet"        # "scrim" | "heading" | "bullet" | "kicker"
```

`SceneText.sorted_layers()` fixes z-order by `kind` — scrim (0), heading (1), bullet (2) — because
the scrim must land *under* the type it exists to make legible. `x`/`y` are the layer's **final
resting place**; entry animations move toward it and never change where it ends up.

It is kept out of `core/models.py` because these are render-time artifacts (files on disk), not
part of the persisted `Timeline`. `ports.py` imports `SceneText` under `TYPE_CHECKING` only, so
`core` stays independent of `app.render`.

`ffmpeg_backend` reaches `text_overlay` through `getattr` probes —
`getattr(tx, "slide_geometry", None)`, `getattr(tx, "build_scene_text", None)` — with documented
fallbacks (`fallback_region`, the legacy single-PNG heading). That is the seam being kept loose
while `app/render/*` is refactored. `[in-flux]`

### 8.3 Why every glyph is an ImageMagick PNG

`text_overlay.resolve_text_mode("auto")` returns one of `drawtext` | `png` | `scrim`. On this
machine it returns **`png`**, and the warning is in the test output verbatim:

```
ffmpeg at /opt/homebrew/bin/ffmpeg has no drawtext filter (built without libfreetype);
rendering headings via an ImageMagick PNG overlay instead
```

The Homebrew formula dropped libfreetype, so `drawtext` does not exist. Every heading, bullet,
kicker, rule and marker is rendered by ImageMagick (`-annotate`, `-kerning`, `-draw`), cached in
the job directory keyed on everything that affects the pixels (`cache_path`, `render_cached_png`),
and composited with `overlay`. The `drawtext` path is retained for portability but nothing in the
slide layout depends on it, and `MAX_TEXT_LAYERS = 8` caps the graph (1 scrim + 1 heading + up to
5 bullets, one spare).

Animation without `drawtext` is done in the overlay stage:

* `fade=t=in:st=<appear_at>:d=…:alpha=1` — with `t=in` and `st>0` the alpha channel is held at
  **zero** for every frame before `st`, which is what guarantees the layer genuinely does not
  exist yet. (The classic bug is the PNG being visible from frame 0.)
* a time-varying `overlay` x/y expression, clamped by `_progress_expr` so the layer pins to its
  final position and stays there. `_smoothstep` for slides, `_back_out` (overshoot ~10%, settle)
  for `POP`.
* `TYPEWRITER` is a **wipe, not a true typewriter** — `_wipe_filter` uncovers the rasterised line
  with a moving `geq` alpha edge, because without `drawtext` there is no glyph-level clock.
* `enable='gte(t,appear_at)'` skips the overlay stage entirely while a layer is invisible. This is
  an optimisation, not the animation — `enable` cannot interpolate.

### 8.4 Why the logo is composited in `assemble()`, not per scene

`ffmpeg_backend.py:1403-1411` states the reason as built:

> The watermark is composited **once, over the finished chain** — never per scene. A logo burnt
> into each scene clip is an input to every `xfade`, so at each boundary the outgoing copy fades
> out while the incoming one fades in; because the two are pixel-identical and `xfade` is a linear
> blend of *frames*, not of layers, the result still dips wherever the crossfade curve does not
> sum to one. On screen that is a logo that pulses at every cut.

`tests/test_render.py:1787` adds the harsher case, and it is the one the brief describes: with a
`slideleft` transition **a per-scene logo literally slides off the left edge with the outgoing
frame**. The test therefore uses `SLIDE_LEFT` deliberately. Two caveats, both worth knowing:

* I found no "100% loss" measurement recorded anywhere in the repo — the documented rationale is
  the pulse, and the slide-off is asserted by that test rather than quantified in a comment.
* The test's claim that `slideleft` "is what the planner's rotation actually picks for the second
  boundary" is **stale**: `planner.TRANSITION_ROTATION` is now `(Transition.FADE,)` and
  `HELD_TRANSITION = FADE`, so `SLIDE_LEFT` and `WIPE_RIGHT` are removed, not deprioritised.

Placement is after the fades, not before — before them the mark would dip with the opening
fade-up and the closing fade-out, and "constant" is the requirement. The logo is a **single-frame
input** relying on `overlay`'s default `eof_action=repeat`, rather than a `-loop 1` stream in a
pass with no `-frames:v` to stop it. `resolve_logo_source` is three-way (`AUTO_LOGO` → settings →
`frontend/public/favicon.svg`; `None` or an empty-ish path → no branding; a real path → use it or
warn and skip) and a missing logo is **never** an error — the same three states
`Timeline.logo_path` and `Job.logo_id` carry, which is what lets a per-job upload flow straight
through. `pipeline._stage_render` passes it by inspecting the factory's signature and falls back to
assigning `backend.logo_source` directly, warning if the backend takes no mark at all.
`logo_conflicts` reports scenes whose
*ink* (not layer canvas) would overlap the mark, and reports rather than resolves — nudging the
logo per scene would make it move, which is exactly what a persistent brand mark must not do.

### 8.5 Layout and the 4:5 hero region

`text_overlay.slide_geometry(plan, profile, theme=, role=)` is the **single source of truth** for
geometry. It is pure arithmetic — no fonts, no subprocesses — so `ffmpeg_backend.layout_region`
can ask it for the image rectangle without paying for a rasterisation. Asking it, rather than
recomputing, is what stops type and picture overlapping.

```
SlideGeometry: layout, frame, text_column: Rect, image_region: Rect | None,
               align, vertical_anchor, heading_size, bullet_size, scale,
               over_image, image_radius, role
```

At 1920×1080, `HERO_RIGHT` resolves to:

```
 ┌──────────────────────────────────────────────────────────────┐
 │  margin 104px                                                │
 │        ┌───────────────────────┐   gutter   ┌─────────────┐  │
 │  y=226 │  heading (78px base)  │    88px    │             │  │
 │        │  ────  accent rule    │            │   IMAGE     │  │
 │        │  ·  bullet  (44px)    │            │   720×900   │  │
 │        │  ·  bullet            │            │    = 4:5    │  │
 │        │  ·  bullet            │            │             │  │
 │  y=990 │  ·  bullet            │            │  radius 24  │  │
 │        └───────────────────────┘            └─────────────┘  │
 │        text column 904px                     image 720px     │
 └──────────────────────────────────────────────────────────────┘
```

`TEXT_COLUMN_SHARE = 904/1624` splits what is left after margins and gutter. The image gets the
other 720px at the full column height (900px), making the hero region **4:5** — and the source
comment is explicit that this is *"a requirement on the image provider, not a crop preference:
stills generated at 16:9 lose 55% of their width to this region, which is why framing never
matches the prompt."* See §13; that requirement is not yet met.

`HERO_TEXT_TOP_RATIO = 226/1080` is fixed and the stack is **top-anchored**, not centred: a
centred stack shifts every time the bullet count or a wrap changes, which put the text block at
three different heights across four slides in the rejected reference video. A two-line heading
grows *upward* into the air above, leaving the rule and the whole bullet stack where they were.

Every region edge is forced **even** (`_even`) because `yuv420p` subsamples chroma 2×2 and an odd
overlay offset lands the layer on a half-pixel and visibly softens the panel edge. The logo's
*offsets* are evened for the same reason, but its *size* is left exactly as rasterised — that is
a measurement of an existing file, and rounding it down would under-report the box the collision
check needs.

What the planner actually emits today (`planner.py`): **two layouts total** —
`TITLE_CARD` for the opener, `HERO_RIGHT` for everything else. `HERO_LEFT`, `IMAGE_BAND` and
`FULL_BLEED` are retired; the enum members stay so old timelines deserialise. `LAYOUT_ROTATION`,
`MOTION_ROTATION`, `HEADING_ANIMATION_ROTATION` and the `hold_*` flags still exist to restore the
old per-scene rotation, and are exported for tests.

### 8.6 Ken Burns: two independent causes of stepping

Read `ffmpeg_backend.py:77-208` for the full derivation — every constant there carries its own
measured justification. The two-sentence version, because the shape of the fix is what matters:

1. **Quantisation.** `zoompan` truncates `x`/`y` and its crop size to whole **input canvas**
   pixels, so its finest move lands on screen as `zoom / U` output pixels. `motion_canvas()`
   derives the canvas from the region, the travel distance and the frame count rather than using a
   fixed 4 (which was calibrated when zoompan filled a 1920-wide frame, not a 720-wide panel).
   `CANVAS_STEP_TARGET = 1.0` is the physics, not a knob: below one canvas pixel per frame the
   render *will* emit byte-identical frames. `CANVAS_PIXEL_BUDGET = 24M` is where the honest
   compromise lives, and `slowest_step()` is the number to argue with.
2. **Easing.** A pure smoothstep has velocity `6u(1-u)` — **exactly zero** at both ends, so the
   opening of every move was genuinely stationary and no finite upscale can fix a zero.
   `eased_progress` blends in a linear ramp with `EASE_VELOCITY_FLOOR = 0.20`, swept over seven
   real layout/motion pairs; it is the larger of the two effects and it costs nothing.

`motion_canvas(plan, region, frames, profile, src_size) -> MotionCanvas` balances three
requirements that pull against each other, and it is worth reading its docstring because an earlier
draft got one of them wrong. `MotionCanvas` carries `fit` (the isotropic `region * detail` target
where lanczos runs), `canvas` (what zoompan actually reads, after a cheap anamorphic stretch) and
`detail`:

1. **No resolution loss.** zoompan crops `canvas/zoom` and scales it to the region, so
   `canvas >= region * zoom` on **both** axes or the crop is itself an upscale and the panel is
   measurably softer than the still it came from. The earlier draft worked in integer multiples of
   the region, so the only options were 1× (which upscales, since `zoom > 1`) and 2× (twice the
   area). Sizing in *pixels* makes 1.08× reachable. `plan_zoom_ceiling(plan)` supplies the zoom.
2. **Positional precision** on the travel axis — `canvas/region` sub-steps per output pixel.
3. **Cost**, budgeted as **area**, not factor.

So the cross axis takes the least it can get away with and the travel axis spends the remainder.
Nothing is distorted: the still is cover-fitted to the region's aspect first, so its net scale
through the stretch and back out of zoompan is `region * zoom / fit` on each axis, and `fit` has
the region's aspect, so the two are equal by construction.

Measured on `hero_right`/`pan_right` at 1080p over 604 frames, in the evaluator's own 4 s window:

| canvas | duplicate ratio | render |
|---|---|---|
| 2880×3600 (old, fixed 4×4) | 51.67% | 5.70 s |
| 11520×14400 (isotropic 16×) | ~13% | ~64 s |
| **13332×1800 (this)** | **4.17%** | **9.02 s** |

The isotropic canvas that reaches the same smoothness is seven times the pixels and seven times the
time. And because the budget is an area, the derivation makes `full_bleed` **cheaper** than the
fixed 4× it replaces (24 vs 33 Mpixels) while taking it from 30.83% duplicates to under 2%. A pan's
cross factor is exactly `detail_upscale` and not a pixel more — measured on
`full_bleed`/`pan_left`, a cross factor of 2 gave 9600×2160 and 19.17% duplicates, while spending
the same ~23M pixels on the axis that moves (21120×1080) gave **3.33%**. A zoom cannot do that: its
binding constraint is whichever axis steps last, so `ZOOM_CROSS_UPSCALE = 4` is the measured knee.

Note the planner currently defaults to `easing="linear"` (per DIRECTION §4.4) with
`HELD_ZOOM_SPAN = 0.06` — 6% over 15 s, i.e. 0.4%/s: never static, never noticed.

### 8.7 Audio in `assemble`

Each narration segment is trimmed and silence-padded to its scene's **clip** length and delayed
by `starts[i]` — the same value used as that scene's `xfade` offset. Segments therefore overlap by
exactly the crossfade duration, which is the point: the voice stays glued to the picture instead
of drifting later with every transition.

Music is ducked twice — a static `volume=-18dB` (`VIDEO_MUSIC_DUCK_DB`) plus
`sidechaincompress` keyed off the narration, because static ducking alone still fights the voice
on louder passages. Every path then goes through the same master bus:
`loudnorm(I=-16, TP=-1, LRA=11)` → `alimiter` → `asetpts=N/SR/TB` → `atrim` → `apad`.

The order there is load-bearing and the bug it fixes is documented with numbers: `loudnorm` runs a
lookahead buffer and re-bases timestamps off its own block clock, and `atrim` selects on
*timestamps*, so the clamp leaked. Measured on the reference render, `loudnorm` alone put
**+0.066992 s (+2.01 frames at 30 fps)** past an `atrim=0:74.633008`; since a container is as long
as its longest stream, that landed as the whole file's drift. `asetpts=N/SR/TB` regenerates
timestamps from the running sample count and the same graph lands on exactly 74.633000 s.
`alimiter` was checked separately and is duration-neutral; the logo overlay was innocent.

---

## 9. Data model and persistence

One table, `job`, one row per render. `backend/videos.db`, SQLite.

| Column | Type | Default |
|---|---|---|
| `id` | `str` | `str(uuid.uuid4())`, **primary key** |
| `topic`, `slide_count`, `voice` | `str`, `int`, `str` | required |
| `music` | `bool` | `False` |
| `tts_engine` | `str` | `settings.video_default_tts_engine` |
| `theme` | `str` | `"midnight"` (or `"custom"`) |
| `theme_custom` | `str \| None` | `None` — JSON palette blob |
| `bullets_per_slide` | `int` | `4` |
| `tone` | `str \| None` | `None` |
| `logo_id` | `str \| None` | `None` — an id from `POST /api/logos`, or `"none"` |
| `status` | `str` | `"queued"`, **indexed** |
| `progress`, `current_stage`, `error` | `int`, `str?`, `str?` | `0`, `None`, `None` |
| `timeline_json` | `str \| None` | `None` — **the debug trail** |
| `video_path` | `str \| None` | `None` |
| `created_at` (**indexed**), `updated_at` | `datetime` | `datetime.now(UTC)` |

**`timeline_json` is the debug trail.** `pipeline._save_timeline` rewrites it at every meaningful
boundary — after scripting, after the concurrent branches, after planning + bullet timing, after
rendering, and on failure. So a job that dies in `assemble` still has the full word timings,
plans, bullet reveal times and asset paths on the row. `GET /api/jobs/{id}/timeline` serves it via
`pipeline.timeline_from_job`, which tolerates a half-written row by returning `None` rather than
raising — a debug aid must not break a status response.

**The migration.** `db/models.migrate(engine)` exists because `SQLModel.metadata.create_all`
creates missing *tables* only and **never alters one that already exists**. A dev database from a
previous release would keep its old table and every INSERT would fail with *"table job has no
column named theme"*. So `_ADDED_COLUMNS` lists the post-release columns with their DDL — `theme`,
`theme_custom`, `bullets_per_slide`, `tone`, `tts_engine`, `logo_id` — and
`migrate` reads `PRAGMA table_info('job')`, skips whatever is present, and `ALTER TABLE job ADD
COLUMN`s the rest inside one transaction. It is idempotent and purely additive, returns the names
it added, and is run on **every** startup from `init_db()`. No rows exist yet → it returns `[]`
and lets `create_all` own that case. There is no Alembic; this is the whole migration story.

**WAL.** `db/session._apply_pragmas` is registered with `event.listen(engine, "connect", …)`, so
it runs on every new DBAPI connection: `journal_mode=WAL`, `synchronous=NORMAL`,
`busy_timeout=5000`, `foreign_keys=ON`. Two writer classes touch this file — request handlers on
the event loop's threadpool and the pipeline worker — hence also
`connect_args={"check_same_thread": False}`. WAL means the frontend's ~1.5 s status poll never
blocks on the writer, and `busy_timeout` means a poll landing mid-write waits instead of raising
*"database is locked"*.

**Why the API stays responsive during a render.** `POST /api/jobs` returns **202** immediately;
`jobs._spawn` starts the work with `asyncio.get_running_loop().create_task(...)` and holds a
strong reference in a module-level `_tasks` set (asyncio only keeps weak ones), discarding via
`add_done_callback`. There is a documented fallback: no running loop → a daemon thread running
`asyncio.run(...)`. Then *every* blocking operation inside `pipeline.py` goes through
`asyncio.to_thread` — provider HTTP, ffprobe, ffmpeg, and each `_patch_job` write, which is
deliberately a single-statement update to keep the write window tiny under WAL. Nothing on the
event loop ever waits on a subprocess.

Note there is **no** shutdown drain: the lifespan in `app/main.py` does nothing after `yield`, so
Ctrl-C mid-render relies on the pipeline's `CancelledError` handler to leave a truthful `failed`
row. A job on the daemon-thread fallback path gets no cancellation at all.

**API surface** (all routers carry their own `/api` prefix; CORS allows only the two spellings of
the Vite dev server; there is no static file mounting and no custom exception handler):

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/jobs` | **202**. `topic` (≤500), `slide_count` 2–10, `voice`, `music`, `tts_engine`, `theme`, `theme_custom`, `bullets_per_slide` 3–5, `tone` ∈ {new_hires, all_staff, technical, executives}, `logo_id` |
| `GET` | `/api/jobs` | `ORDER BY created_at DESC, id DESC LIMIT 20` |
| `GET` | `/api/jobs/{id}` | `JobStatusOut` |
| `GET` | `/api/jobs/{id}/timeline` | raw `Timeline` dump, 404 if none recorded yet |
| `GET` | `/api/jobs/{id}/video` | `FileResponse`, `video/mp4` |
| `DELETE` | `/api/jobs/{id}` | **204**, then `shutil.rmtree(out/<id>)` |
| `GET` | `/api/engines` | id, name, `supports_ssml`, `available`, `default`, `default_voice`, `reason` |
| `GET` | `/api/themes` | id, name, description, `is_light`, `is_default`, swatches, **computed** contrast |
| `GET` | `/api/voices?engine=` | id, name, accent, tags, use_cases. Unknown engine → **422** |
| `POST` | `/api/logos` | **201**. Multipart upload of a brand mark. See below |
| `GET` | `/api/logos` | stored marks |
| `GET` | `/api/logos/{id}`, `/{id}/render`, `/{id}/meta` | original file, rasterised render asset, metadata |
| `DELETE` | `/api/logos/{id}` | **204**, or **409** while a job still names it |
| `GET` | `/api/health` | `{"status":"ok"}` |

Three resolution rules are worth internalising because they differ deliberately. An unknown
**theme** or **engine** id is *normalised* to the default and the *resolved* value is what gets
stored, so a row never claims a palette or engine that was not used. But a **voice** that
demonstrably belongs to a different engine is **rejected with 422**, not substituted — it is always
a client bug (the voice list is served per engine by this same API), and silently substituting one
ships a six-minute video in a voice nobody chose. Likewise an unknown **`logo_id`** is a 422
(`_resolve_logo_id`), because a missing logo would degrade to a video branded with somebody else's
mark — the one outcome an upload feature exists to prevent. A custom palette failing WCAG **AA** is
also a 422, and the body carries `suggested_fix` + `suggested_contrast` so the UI can offer
one-click correction instead of a dead end.

`logo_id` is three-state throughout, mirroring `resolve_logo_source`'s own three states: omitted →
the bundled default mark; `"none"` → no branding; an id → that upload. `pipeline.resolve_job_logo`
turns the id into `Timeline.logo_path`, and an id whose file has since been deleted falls back to
the default with a warning rather than failing the render.

**The logo store** (`app/api/logos.py`) is the only endpoint that takes an untrusted *file* and
hands it to image tooling, and its validation is correspondingly paranoid — each rule states its
threat. The size cap (`settings.video_logo_max_bytes`, 4 MiB) is enforced **chunk by chunk while
the body arrives**, because a cap applied after buffering only limits what you keep. Format comes
from PNG magic bytes and from whether the document actually parses as XML with an `<svg>` root; the
filename and `Content-Type` choose only which diagnostic to return. Dimensions are read from the
PNG IHDR and from the SVG's own `width`/`height`/`viewBox` — **parsed, not decoded** — so a 40 KB
file that expands to 99999×99999 is rejected having never allocated a bitmap. The stored name is a
content hash, so `../../etc/x.png` is never a path that could reach the filesystem, and lookups
re-validate against `LOGO_ID_RE`. SVGs are rasterised **at upload time**, because ImageMagick on
this box has no `rsvg-convert` delegate and its built-in MSVG renderer implements neither `<mask>`
nor `<filter>` — turning such groups into black blobs, which is exactly the shape of the app's own
favicon. Anything the renderer cannot be faithful to comes back as a `warnings` entry on the upload
response rather than being discovered in a finished video. Metadata lives in a JSON sidecar next to
the file rather than a table, so `cache/logos` is self-describing and can be copied or wiped as a
unit; the only relational fact is `Job.logo_id`, which is what makes DELETE a **409** while a job
still names it.

`GET /api/voices` caches per engine in a process-lifetime dict, and **only on success** — a
degraded fallback list is never cached, so a later request retries the network.
`engine_for_voice` attributes a voice by *ownership* (`aura*` prefix, or membership of the 15
measured Polly names), not by catalogue membership, so a bad-DNS day cannot make the API reject
50 valid voices.

---

## 10. Frontend

React 19 + Vite 8 + TypeScript, Tailwind v4 (CSS-first), shadcn/ui on Radix, `sonner` toasts,
`lucide-react` icons, `oxlint`. `vite.config.ts` proxies `/api` → `http://127.0.0.1:8000`. No
router, no context providers, no state library, no react-query. The whole app is force-dark.

```
App                                   owns activeJobId (mirrored to ?job= via replaceState),
│                                     isSubmitting, contrastFailure
├── useEngines()   ← owns engine selection, refuses available===false
├── useVoices(engineId)                ← refetches per engine
├── useThemes()
├── useLogos()     ← owns logo selection + upload progress
├── useJobPolling(activeJobId)         ← 1500 ms setTimeout chain
├── useJobHistory()
├── useTimeline(activeJobId, status)   ← 2500 ms setTimeout chain
│
├── CreateForm ─ EngineSelector, VoicePicker, ThemePicker ─ PresetCard ─ SlidePreview
│                                                        └ PaletteEditor ─ ContrastRow
│              ├ LogoPicker, LogoUploader
│              └ SlidePreview (live)
├── ProgressView ─ StageStepper, SceneInspector ─ SceneCard ─ BulletTrack
├── ResultView   ─ ThemeBadge, EngineBadge, SceneSeekList, SceneInspector
└── JobHistory
```

View switching is a nested ternary in `App.tsx:184-252`, not a `view` enum: no job → form; job
loading → skeletons; `status === 'done'` → `ResultView`; otherwise `ProgressView` (which also
renders the `failed` state).

**Polling is a chained `setTimeout`, not `setInterval`** (`hooks/useJobPolling.ts`). The reschedule
happens at the *end* of `tick()`, so the 1500 ms is the gap *between completions* and two requests
can never overlap — which `setInterval` cannot guarantee against a slow response. `isTerminal`
(`done`|`failed`) stops the chain; a 404 gives up immediately with *"That job no longer exists."*;
otherwise five consecutive failures (`MAX_CONSECUTIVE_ERRORS`) stop it. Cleanup sets a `cancelled`
closure flag and clears the timeout. There is **no `AbortController` anywhere in `src/`** — an
in-flight response is simply discarded via the flag. `refresh()` bumps a `nonce` in the effect's
dep array to force an immediate refetch. `useTimeline` is the same pattern at 2500 ms / 4 errors,
gated by `timelineCouldExist(status)` and written so a populated timeline never regresses to null.

**Selection flow.** Engine state lives in `useEngines`, not in the form, because
`useVoices(engineId)` must see the change in the same render. Voice is stored as a *pair*
`{engine, id}` and the effective voice is **derived in a `useMemo`** — used only if
`voiceChoice.engine === engineId` and the id is present in the fetched list, else
`preferredVoiceId(...)`. That is deliberately not an effect, so no render can ever hold a
Deepgram id under Polly. `useVoices` reinforces it by returning `fallbackVoicesFor(engineId)`
synchronously during render whenever its cached engine does not match.

**The logo uploader validates before you spend three minutes.** `src/lib/logo.ts` ports the
renderer's geometry — `LOGO_HEIGHT_FRACTION = 0.045`, `LOGO_MARGIN_FRACTION = 0.028`,
`LOGO_OPACITY_DARK = 0.85` — with a citation on every constant back to `Theme` in
`core/models.py` and `logo_height`/`logo_rect` in `text_overlay.py`, so it can tell the user the
truth about a 49-pixel mark on their chosen palette *before* a render. Its WCAG arithmetic is
imported from `lib/contrast.ts` and never re-implemented, because that module is the one covered by
`pnpm run test:contrast`. `useLogos` owns the selection (`BUILT_IN_LOGO_ID` | `"none"` | an id),
upload progress and errors, and prepends a successful upload to the list rather than refetching —
a refetch would drop the caller into a loading state immediately after success.

**Client-side WCAG mirrors the backend.** `src/lib/contrast.ts` is dependency-free and is a
deliberate port: `relativeLuminance`, `contrastRatio`, `evaluatePalette`, `contrastReport` (same
four keys as `Theme.contrast_report()`), and `suggestFix` — an HLS lightness-only repair with the
same 24-iteration binary search, `roundHalfEven` to match Python's banker's rounding, and the same
"move foregrounds first, yield `bg` only as a last resort" order. `contrast.test.ts` asserts
against numbers taken from `python -c "from app.core.themes import contrast_table"`.
`CreateForm` blocks submit only when `isCustomTheme && !report.isValid`; presets are never
blocked. `PaletteEditor` prefers the server's `suggested_fix` over the local `suggestFix` when a
422 has come back.

**Scene inspector.** `SceneInspector`, `SceneSeekList` and `BulletTrack` all read the same object
— the `Timeline` from `GET /api/jobs/{id}/timeline`, passed down as props. None of them fetch.
`lib/timeline.analyzeBullets(scene)` computes each bullet's `ratio` along the scene, flags
`isOverflow` (`appear_at >= duration`) and `isCrowded` (`gap < plan.bullet_min_gap`), and is
guarded by `scene.duration > 0` so nothing is flagged before the aligner has run. `BulletTrack`
renders those as absolutely-positioned ticks with an enumerating `aria-label`, and draws evenly
spaced *ghost* ticks while duration is still 0. `SceneSeekList` is the only place that converts
back to global time: a bullet chip seeks `scene.start + bullet.appear_at`. `SlidePreview` reads no
endpoint at all — it is a pure CSS mock of `hero_right` with the geometry ratios hand-ported from
`text_overlay.py` and expressed in `cqw` units.

`lib/api.ts` is written to survive a backend that is a version ahead or behind: one `request()`
helper, `ApiError(message, status, detail)` with `detail` kept structurally (two 422 bodies are
objects that must be inspected), every catalogue fetch falling back to a bundled list rather than
rejecting, `isPlausibleVoiceForEngine` guarding against a backend that ignored `?engine=`,
`fetchTimeline` returning `null` for 404/409/425 ("not written yet" is not an error), and
`createJob` retrying once with `OPTIONAL_JOB_FIELDS` stripped when a 422 is neither structured
error. That tolerance is also why the missing `title`/`slide_count` fields (§13) go unnoticed.

---

## 11. The evaluator — and the fact that it is not wired in

`app/evaluate/` is three layers, separated so the cheap ones can run without the expensive one.

| Layer | File | What it does |
|---|---|---|
| `metrics` | `metrics.py` (930 ln) | Deterministic measurement via **ffmpeg/ffprobe only** — no PIL, no numpy. Pixel work is `bytes` from `format=gray` rawvideo plus stdlib `statistics` |
| `vision` | `vision.py` (477 ln) | Gemini judgement on **one frame per scene** plus one text pass over the script |
| `scorer` | `scorer.py` (1085 ln) | Folds both into `VideoScore`: per-dimension 0–10, weighted 0–100, a letter grade, and `Recommendation` objects |

Deterministic metrics include: WCAG contrast of the heading against the **brighter quartile** of
its background (not the mean or median — illegibility is caused by the bright patches the glyphs
cross), the same measurement at the *opposite* `TextPosition` for comparison,
`duplicate_frame_ratio` via a calibrated `mpdecimate=hi=128:lo=64:frac=0.05` against a plain
pass, `ebur128` loudness, speech-vs-bed separation over `speech_windows`/`narration_gap_windows`,
`silencedetect` on the narration asset, words-per-minute over the *spoken span*, bullet-timing
sanity, per-scene duration deviation from the median sibling, and container conformance against
`Timeline.final_duration()` and `Timeline.profile`.

Weights live in `scorer.py:74-99`:

```python
SCENE_WEIGHTS = {LEGIBILITY: .26, RELEVANCE: .24, COMPOSITION: .12,
                 PACING: .10, TIMING: .10, PROFESSIONALISM: .10, MOTION: .08}   # = 1.00
VIDEO_WEIGHTS = {"scenes": .60, "audio": .15, "script": .15, "technical": .10}
METRIC_LEGIBILITY_SHARE = 0.6      # WCAG measurement leads; vision gets 0.4
GRADE_CEILING = {BLOCKER: Grade.C, MAJOR: Grade.B}
```

Raw values become 0–10 through `_piecewise` over anchor tables (`CONTRAST_ANCHORS`,
`MOTION_ANCHORS`, `PACING_ANCHORS`, `LOUDNESS_ANCHORS`, `BALANCE_ANCHORS`, `DRIFT_ANCHORS`,
`PEAK_ANCHORS`). `overall` is left uncapped; only the letter `grade` is capped, with
`grade_capped=True` recorded.

`auto_fixable` (`models.py:129`) is true only when the fix is mechanical, and is always paired
with a stable `action` id and a `params` dict of already-computed values:

| `action` | Trigger |
|---|---|
| `raise_scrim_opacity` | contrast < 4.5 and the solved opacity exceeds the current one |
| `move_text_position` | the opposite band measures ≥ `max(4.5, ratio × 1.5)` |
| `regenerate_scene_image` | relevance < 7 **and** vision returned a `suggested_image_prompt` |
| `raise_upscale_factor` | `duplicate_frame_ratio ≥ 0.12` |
| `respace_bullets` | one per `bullet_issues` string |
| `renormalize_loudness` | `abs(LUFS − (−16)) ≥ 1.5` |
| `limit_true_peak` | true peak > −1.0 dBFS |
| `duck_music` | measured speech/bed separation < 10 dB |

Anything needing taste — wpm out of band, composition ≤5, professionalism ≤5, all `SCRIPT`
findings — is deliberately **not** auto-fixable.

### It is not wired into the pipeline. This is a known gap, not a feature.

Verified by grep: the **only** references to `app.evaluate` outside the package itself are
`backend/tests/test_evaluate.py` and `backend/scripts/evaluate_job.py`. `app/worker/pipeline.py`
never imports it — its final stage is `_stage_assemble` followed by `_patch_job(status=DONE)`.
There is no `GET /api/jobs/{id}/score`. `write_score` has exactly two callers, the CLI and a
test. So `out/<job_id>/score.json` appears **only** when someone runs
`uv run python scripts/evaluate_job.py <job_id>` by hand — which is why 2 of the 12 directories
under `out/` have one. `scripts/e2e.py` does not invoke it either.

There is also no auto-fix executor: nothing consumes `VideoScore.auto_fixable()` except
`render_report()`, which prints an `AUTO` tag. The scorer's own docstring names the intended loop
— *"score, fix the top recommendation, re-score, prove the number moved"* — and today only the
first and last steps exist, run by a human.

The evaluator's only coupling to the running app is **inbound**: `scorer.load_timeline` opens its
own read-only stdlib `sqlite3` connection (`file:{db}?mode=ro`) or falls back to
`GET /api/jobs/{id}/timeline`, deliberately bypassing the app's session layer.

---

## 12. Extension guide

This is the payoff of the port design. In every case the orchestrator is untouched.

### Add a TTS engine

1. Write `app/providers/<vendor>_tts.py` with a class exposing `supports_ssml: bool` and
   `synthesize(self, text: str, voice: str, out_path: Path) -> Path`. Its constructor must either
   take no required arguments or accept a subset of `settings` / `api_key` / `voice` /
   `default_voice` / `region` / `engine` / `tier` — `factory._construct` passes only the kwargs
   your signature declares and raises `ProviderUnavailableError` naming anything it cannot supply.
2. Add a `SpeechEngine(...)` entry to `factory.SPEECH_ENGINES` — `id`, display `name`, `module`,
   `class_name`, `supports_ssml`, and `default_voice_setting` (the name of a `Settings` field).
   Order in the tuple is the order the picker shows.
3. Add that default-voice field to `core/config.Settings`.
4. Add a branch to `factory.speech_engine_status` that verifies something **concrete** (a key, an
   importable SDK). Returning optimistic availability buys a job that fails minutes later.
5. If the engine parses SSML, extend `app/providers/ssml.py`; if it does not, set
   `supports_ssml=False` and the pipeline will send plain text automatically.

`GET /api/engines`, the picker, per-job selection, and `Job.tts_engine` all follow with no further
work. `app/api/voices.py` needs a catalogue branch only if you want a voice list.

### Add an image provider

Write a class with `generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path`
and point `factory.image_provider` at it. Note the pipeline asks for
`profile.width * profile.upscale_factor` × the same for height (7680×4320 by default), because
`zoompan` needs canvas headroom — see §8.6. Snap the requested aspect to whatever the API
actually accepts (`gemini_image.SUPPORTED_RATIOS`, `nearest_ratio`) and **never** stretch; the
renderer's `_fit_chain` will cover-crop or blur-fill.

### Add a theme

Add a `Theme(...)` to `core/themes.PRESETS` and a matching entry to `THEME_META`. The
parametrised test over every preset is the gate: `text_on_bg` and `text_on_surface` must clear
**AAA (7.0)**, `muted_on_bg` AA (4.5), `accent_on_bg` 3.0. Run
`python -m app.core.themes` to print `contrast_table()`. Nothing else changes —
`GET /api/themes` computes the ratios rather than transcribing them, and `frontend`'s
`FALLBACK_THEMES` is only a degraded-mode copy.

### Add a language

`Language` (`core/models.py:53`) already has `EN`, `ES`, `HI` with `script` and `needs_shaping`
properties, and `gemini_script` already accepts `language=` and carries measured wpm ratios per
language. What is **missing** is the plumbing (§13): add a `language` column to `Job`
(+ `_ADDED_COLUMNS` entry), a field on `JobCreate`, pass it through `pipeline._script_kwargs`,
set `Timeline.language` in `_stage_script`, and make voice selection language-aware
(`api/voices.py` currently hardcodes `POLLY_LANGUAGE = "en-US"`). Then verify four things per
language, because they are the ones that actually break:

1. **A TTS voice exists**, and — for a voice that carries the language in
   `AdditionalLanguageCodes` rather than in `LanguageCode` — that `PollySynthesizer.synthesize`
   passes `LanguageCode`. It does not today.
2. **The aligner returns word timings.** Without them bullet anchoring silently degrades to
   `method="proportional"` and stops matching the narration, with no error.
3. **`deepgram_align.normalize` keeps the script's characters.** Its `[^a-z0-9]+` regex erases any
   non-ASCII token, which is a hard stop on anchoring. Widening it is what
   `gemini_script.anchoring_supported` is waiting for.
4. **`text_overlay.FONT_CANDIDATES` contains a face with the right glyphs**, plus a shaping engine
   (Pango/HarfBuzz) when `Language.needs_shaping` is true.

### Add a whole alternative video backend

Implement `VideoBackend`'s three methods and change one line in `factory.video_backend`:

```python
def video_backend(settings=None, theme=None) -> VideoBackend:
    cls = _load("app.render.html_backend", "HtmlBackend", "render")   # was ffmpeg_backend
    return _construct(cls, "render", theme=theme)
```

You inherit the whole `VisualPlan` for free — layout, motion, zoom range, easing, transition,
text position, scrim opacity, both animations, `anim_duration`, `bullet_min_gap` — and the whole
`SceneText`/`TextLayer` seam if you want the rasterised assets, or you can ignore it and lay out
text yourself. The contracts you must honour: `render_all` populates `Scene.clip_path` for every
scene; the assembled duration must match `Timeline.final_duration()`; and frame counts should be
rounded cumulatively so clip boundaries land on the narration's own frame grid. One caveat to
check: `pipeline._stage_render` imports `app.render.ffmpeg_backend.layout_region` directly for its
pre-flight "does this scene need an image?" check, so that helper is currently a hard dependency
on the ffmpeg module even for a different backend.

---

## 13. Current state and known gaps

### Test suite, as of writing

```
$ uv run pytest tests/ -q -k "not live"
1 failed, 1893 passed, 27 deselected, 1 warning in 107.58s
```

The single failure is in `tests/test_render.py`, i.e. in the area that is mid-refactor:

| Test | Cause |
|---|---|
| `test_branding_is_duration_neutral_and_survives_every_transition` | The mark **does** pulse. Measured lifts across the video `[164.1, 96.2, 176.5, 188.6, 181.9]` — spread 92.4 against a tolerance of `0.35 × max = 66`. Duration-neutrality and presence both pass; *constancy* does not |

This looks like a real regression in output rather than a stale assertion, and it is worth noting
that the scene it dips on is a `slideleft` boundary — which the assemble-time compositing in §8.4
is specifically supposed to make impossible.

Two caveats on this number. `app/render/*` moved **during the writing of this document**: an
earlier run in the same session reported `4 failed, 1830 passed`, and three of those four
(a stale `hero_right` upscale-warning expectation, a `clip_dir.mkdir` ordering assumption, and a
harness bug passing `fps_mode=passthrough` inside `-vf`) were fixed upstream while this was being
written. Re-run the suite rather than trusting this line. Also note that
`ffmpeg_backend.upscale_factors` was replaced by `motion_canvas`/`MotionCanvas`/`plan_zoom_ceiling`
mid-session, and a whole logo-upload feature (`app/api/logos.py`, `Job.logo_id`,
`Timeline.logo_path`, `frontend/src/lib/logo.ts`) landed — §8.4/§8.6/§9/§10 describe the state
*after* those changes.

### Known gaps, specifically

1. **`app/render/*` is mid-refactor.** Treat `ffmpeg_backend.py`, `text_overlay.py` and
   `planner.py` as moving. The `getattr(tx, "slide_geometry", None)` /
   `getattr(tx, "build_scene_text", None)` probes with documented fallbacks are there precisely
   because the two modules are being changed independently. `gemini_script.py` is likewise being
   edited concurrently.
2. **The evaluator is not wired in.** No automatic `score.json` per job, no
   `GET /api/jobs/{id}/score`, no consumer of `auto_fixable()`. See §11.
3. **4:5 image prompts are not implemented, so image relevance suffers.** The hero region is
   720×900 (4:5), and `text_overlay.py:222` is explicit that this is *a requirement on the image
   provider*: a 16:9 still loses 55% of its width to the region. But the prompt in
   `gemini_script.py` still asks for the opposite — *"Compose it with generous open space — plain
   sky, empty wall, shallow-focus foreground — in the lower third where a text caption will be
   overlaid"* — which is 16:9 lower-third framing, and the pipeline requests
   `profile.width × upscale` by `profile.height × upscale`, i.e. a 16:9 canvas. So the composed
   subject is routinely cropped out of the panel. `[in-flux]`
4. **Hindi has no Deepgram voice and Devanagari needs shaping — and neither is worked around
   yet.** `Language.HI` records the constraints: *no Deepgram TTS voice exists for Hindi (0 of
   102)*; the only path is Polly `Aditi`/`Kajal`, which are `en-IN` voices carrying `hi-IN` in
   `AdditionalLanguageCodes`, so the request **must set `LanguageCode` explicitly**. It does not:
   `PollySynthesizer.synthesize` sends only `Text`, `TextType`, `VoiceId`, `Engine`,
   `OutputFormat`, `SampleRate` — there is no `LanguageCode` on the synthesis path at all. The
   only use of `LanguageCode` in that module is as a *filter* on `describe_voices`, and
   `api/voices.py` pins that filter to `POLLY_LANGUAGE = "en-US"`, so `Aditi`/`Kajal` never even
   surface in the picker. Separately, Devanagari needs a shaping engine — plain freetype drops the
   nukta and anusvara, rendering फ़िशिंग as फिशिग — and `text_overlay.FONT_CANDIDATES` lists only
   Arial / SF / Helvetica / DejaVu, none of which has Devanagari glyphs. And per §6,
   `normalize`'s ASCII-only regex means bullet anchoring degrades to `proportional` for Hindi by
   construction.
5. **`Language` is not plumbed at all.** `Timeline.language` exists and defaults to `EN`, and
   `gemini_script` supports it *thoroughly* — `_language_block`, `_anchor_note`, per-language
   stopwords, the danda `।` as a sentence terminator, capitalisation skipped for non-Latin scripts,
   and per-language word budgets. None of it is reachable: there is no `language` column on `Job`,
   no field on `JobCreate`, `pipeline._script_kwargs` only ever forwards `bullets_per_slide` and
   `tone`, and `_stage_script` builds the `Timeline` without `language=`. Today every job is
   English end to end.

   Worth knowing before you touch the pacing numbers: `LANGUAGE_WPM` is en 135 / es 140 / hi 155
   and `LANGUAGE_WORD_FACTOR` is 1.0 / 1.04 / 1.15, with `docs/LANGUAGES.md` §6.2 as the stated
   authority, and `ROLE_NARRATION_WORDS_BY_LANGUAGE` is **transcribed** from that document rather
   than recomputed — because multiplying and rounding does not reproduce it exactly (43 × 1.15 =
   49.45 → 49, where the doc says 50). An earlier pass at this file measured en 114 / hi 184 wpm
   and derived factors of 1.41/1.61; those were discarded as really measuring *sentence length*,
   since each language's sample had a different sentence count. LANGUAGES.md §6.3 records that
   words-per-second ranged 2.40–3.80 across experiments — a 58% spread driven almost entirely by
   how many sentence-final pauses the model wrote, not by the language. The consequence is stated
   plainly in the source: a Spanish slide carries ~93% and a Hindi slide ~90% of the English
   slide's information, which is a real teaching-content loss.
6. **`bullets_per_slide=5` is silently capped to 4.** `JobCreate` accepts `ge=3, le=5`, but
   `SceneRole.CONTENT.bullet_budget` is 4 and the budget is enforced three times —
   `pipeline._within_bullet_budget` at scripting, `RuleBasedPlanner.plan` when
   `enforce_bullet_budget`, and `_stage_bullets` again. The reason is arithmetic, recorded on the
   property: at the 11 s floor the usable reveal window is 6.27 s and five bullets at a 1.6 s
   stagger need 6.4 s. The API just doesn't say so.
7. **Progress can go backwards** (50 → 30 → 60) because `NARRATING` is set before the gather and
   `IMAGING` inside it. Cosmetic, but it will confuse anyone debugging the stepper.
8. **`JobStatus.SCORING` means music, not evaluation.** A permanent trap for new readers.
9. **`JobStatusOut` omits `title` and `slide_count`**, but `frontend/src/lib/api.ts:parseJob`
   reads both. They come back `undefined`; the inspector gets the title from the timeline endpoint
   instead. Harmless today, easy to trip over.
10. **The client's contrast gate is stricter than the server's on one pair.** Frontend
    `REQUIRED_THRESHOLDS` blocks on `muted_on_bg: 4.5`; backend `REQUIRED_THRESHOLDS`
    (`core/themes.py:298`) deliberately omits it, because `Theme.uniform_text` means `muted` never
    reaches a rendered frame. So a palette the server would accept can be blocked in the UI.
11. **`_XFADE_BUDGET = 0.5` in the pipeline vs `DEFAULT_TRANSITION_DURATION = 0.35` in the
    planner.** Each scene is padded for a 0.5 s crossfade it will not get, so every scene carries
    ~0.15 s of extra breath. The code calls this out as harmless; it is also a real 0.15 s ×
    boundaries of dead air.
12. **Inter-scene silence is 1.0 s everywhere.** `DIRECTION.md` §5 specifies **0.6 s**, except
    1.0 s after the title card — noting that 1.0 s × 5 boundaries is 5 s of dead air in a 75 s
    video (6.7%). `VIDEO_SCENE_PAUSE_S` is a single global.
13. **There is no shutdown drain and no cancel endpoint.** See §9.
14. **The frontend has no test runner.** The single test, `src/lib/contrast.test.ts`, is compiled
    by a bespoke `tsc` invocation and run under bare `node` (`pnpm run test:contrast`), which is
    why `contrast.ts` is deliberately dependency-free. No component or integration tests exist.
15. **`app/providers/ssml.py` is unused in production.** 1083 lines with a measured per-tier
    capability matrix (`CAPABILITIES` for `polly-standard|neural|long-form|generative`,
    `deepgram`, `elevenlabs`), calibrated break costs (`BREAK_COST_FACTOR`), an
    `SsmlInvariantError` that refuses to emit markup which would not strip back to the same
    tokens, and an outright ban on `<sub>` because it breaks the aligner's reference-text
    invariant. Nothing under `app/` imports it — only `tests/test_ssml.py`. `Scene.ssml` is read
    at `pipeline.py:279` but never written, so `supports_ssml=True` on Polly changes nothing today.
16. **No provider caches to disk.** A re-run of the same topic re-pays for the script, every
    image, every narration take and the music bed. The only cache in the package is
    `polly_tts._voice_cache`, in-process. `settings.video_cache_dir` is created at startup and
    used only for logo rasterisation.
17. **`text_overlay.EMPHASIS_MODE` defaults to `"off"`** via a raw environment read
    (`os.environ.get("VIDEO_EMPHASIS", "off")`) rather than through `Settings`, so it is invisible
    to `.env.example` and to `GET /api/*`. Meanwhile `bullet_timing._emphasis_index` still
    computes an emphasised bullet and `BulletPoint.emphasis` is still persisted and shown by the
    frontend's `BulletRow`/`BulletTrack`. The UI advertises an emphasis the renderer does not draw.
