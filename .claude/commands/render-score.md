---
description: Render a video end-to-end through the real API, then measure and score it
argument-hint: "<topic>" [slide_count] [voice]
allowed-tools: Bash, Read, Grep, Glob
---

Run the full render-and-score loop. **This spends provider credits** (script, images, narration,
alignment, music) and takes several minutes. Confirm with the user before starting if they have not
just asked for a render.

Arguments: $ARGUMENTS — topic (quoted), then optional slide count (default 4) and voice
(default `aura-2-draco-en`).

## Steps

1. **Preflight.** `./scripts/doctor.sh` from the repo root. All 17 checks must pass. If the toolchain
   or a key is broken, stop — do not burn a render on a known-bad environment.

2. **Confirm the backend is up** on :8000: `curl -s http://127.0.0.1:8000/api/health`.
   If it is not, ask before starting anything — someone's `mprocs` may be mid-restart. Check with
   `lsof -nP -iTCP:8000 -sTCP:LISTEN` and **never kill a process you did not start**. If you need a
   server of your own, use a different port.

3. **Render.** `./scripts/e2e.py $ARGUMENTS` from the repo root. It polls to completion, prints the
   `job_id`, ffprobes the output, and prints the expected-vs-actual xfade duration check. Report the
   `job_id` as soon as it appears so the work is recoverable if anything later fails.

4. **Score.** From `backend/`:
   `uv run python scripts/evaluate_job.py <job_id> --no-vision`
   Keep `--no-vision` unless the user explicitly asked for the vision judge (one Gemini call per
   scene plus one for the script). Offline mode leaves relevance, composition, professionalism and
   script unassessed — say so rather than implying a clean bill of health.

5. **Verify the invariants that the scorer does not own.** Delegate to the `render-verifier`
   subagent, or check directly:
   - final duration == `sum(scene durations) − sum(transition durations)` (xfade eats overlap)
   - duplicate-frame ratio under 12% per clip
     (`mpdecimate=hi=128:lo=64:frac=0.05` — the ffmpeg defaults are inverted on this output)
   - 1920×1080 @ 30 fps, h264 + aac 48 kHz stereo
   - one marker shape, one transition, one layout, two heading sizes (`docs/DIRECTION.md`)

## Report

Give the `job_id`, the output path and size, the overall score and grade, then every BLOCKER and
MAJOR with the measured number and the file that owns the fix. List the auto-fixable
recommendations separately — those have an `action` and `params` and can be applied mechanically.
Finish with the scorer's own notes about what was **not** assessed.
