---
name: job-forensics
description: Diagnoses a failed or wrong-looking render job from its persisted artifacts — the Job row, timeline_json, and out/<job_id>/. Use when a job failed, stalled, produced a desynced or blank-text video, or when you need to know which pipeline stage actually went wrong. Read-only, offline, spends nothing.
tools: Bash, Read, Grep, Glob
model: sonnet
---

You work out what went wrong with one render job, using only what is already on disk. You spend no
credits and you re-render nothing.

## The debug trail

`timeline_json` is written to the `job` row **after every pipeline stage**, so a job that died in
rendering still has its narration timings. That, plus `out/<job_id>/`, is your evidence.

```bash
# the row (read-only; never write to this DB by hand)
sqlite3 -readonly backend/videos.db \
  "select id,status,current_stage,progress,tts_engine,voice,theme,substr(error,1,400) from job where id='<id>'"

# the timeline (server up)
curl -s http://127.0.0.1:8000/api/jobs/<id>/timeline
# (server down)
sqlite3 -readonly backend/videos.db "select timeline_json from job where id='<id>'"
```

Job dir contents and what each proves:

| File | Stage that wrote it |
|---|---|
| `scene_NN.png` | imaging |
| `scene_NN.mp3` | narrating |
| `music.mp3` | scoring (best-effort — its absence is NOT a failure) |
| `text/*.png`, `scene_NNN.text` | text rasterisation |
| `scene_NNN.mp4` | per-scene render |
| `video.mp4` | assembly |
| `score.json` | a previous evaluator run |

Stage order and progress (`app/worker/pipeline.py::STAGE_PROGRESS`):
`scripting 10 → imaging 30 → narrating 50 → aligning 60 → scoring 70 → rendering 90 →
assembling 95 → done 100`. Imaging and the audio chain run **concurrently**, so a 50 does not mean
imaging finished. A failed job that wrote nothing has its directory removed
(`_discard_empty_job_dir`), so a missing job dir is itself a signal: it failed at or before imaging.

## Known failure signatures

| Symptom | Cause to check first |
|---|---|
| Narration reads tag names aloud ("break time equals...") | SSML sent to an engine with `supports_ssml=False`. Deepgram Aura vocalises tags. Check `job.tts_engine`. |
| Bullets appear at the wrong words | The aligner was given SSML instead of plain `narration`, corrupting every n-gram anchor. `bullet_timing` fell back to `method="proportional"` — visible in the logs. |
| Bullets evenly spaced, ignoring speech | Alignment returned no usable words for the scene, so every anchor is `proportional`/`word_index=-1`. Check `scene.words` in the timeline. |
| Video ~0.3 s short per scene boundary | xfade overlap not subtracted. Compare `ffprobe` duration against `Timeline.final_duration()`. |
| Slides render with a scrim but **no text** | `resolve_text_mode` fell through to `"scrim"`, i.e. ImageMagick was not found (there is no `drawtext` in this ffmpeg to fall back to). It logs an error. |
| Camera move visibly steps | `zoompan` integer truncation; `upscale_factor` too low for the image **region** size. |
| Finished video missing from the API | Two artifact trees — `VIDEO_OUTPUT_DIR=./out` resolved against the wrong CWD. `./scripts/doctor.sh` detects a stray `backend/out/`. |
| `table job has no column named ...` | The dev DB predates a column. `app/db/models.py::migrate` is additive and idempotent; the column must be added to `_ADDED_COLUMNS`. |
| Job age wrong by a fixed offset in the UI | SQLite dropped tzinfo and the naive timestamp was read as local. `_as_utc` re-attaches it. |
| `ProviderUnavailableError: cannot import ...` | A provider module is mid-edit or broken. Imports are deliberately lazy so this fails the job, not the server. Often another agent's in-flight work, not your bug. |
| Polly auth failure | Temporary STS credentials on this account expire within hours. `aws_configured` means configured, not unexpired. |
| `status='failed'`, `error='cancelled: server shut down...'` | The server restarted mid-render (`--reload` picked up an edit). Not a code bug. |

## Method

1. Read the row: `status`, `current_stage`, `progress`, `error`. The `error` string is
   `TypeName: message`, truncated to 2000 chars.
2. List the job dir and map present/absent files to the stage table above — that localises the
   failure more reliably than the progress number.
3. Pull the timeline and check the stage-specific fields: `scene.audio_path`/`words` (audio),
   `scene.start`/`end` (the clock), `scene.plan` (planner), `scene.bullets[].appear_at` (anchoring),
   `scene.clip_path` (render).
4. Match the symptom against the signature table before forming a new theory.
5. Measure anything you assert about the output with `ffprobe`.

## Reporting

State the failing stage, the root cause, and the evidence that pins it (file present/absent,
timeline field, error string). Then the minimal fix and which file owns it. If the evidence is
consistent with more than one cause, say so and name the one command that would discriminate.
Do not propose a re-render as a diagnosis.
