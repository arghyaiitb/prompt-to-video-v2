#!/usr/bin/env python3
"""End-to-end smoke test through the real HTTP API.

    ./scripts/e2e.py "phishing" 4 [voice]

Exits non-zero if the job fails or stalls. Python rather than shell because the
JSON payload braces get brace-expanded by the shell.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000/api"
MAX_WAIT = 1200
REPO = Path(__file__).resolve().parent.parent


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.load(r)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> int:
    topic = sys.argv[1] if len(sys.argv) > 1 else "How phishing attacks work"
    slides = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    voice = sys.argv[3] if len(sys.argv) > 3 else "aura-2-draco-en"

    print(f"topic  : {topic}")
    print(f"slides : {slides}")
    print(f"voice  : {voice}\n", flush=True)

    try:
        get("/health")
    except (urllib.error.URLError, OSError) as e:
        print(f"backend unreachable at {API}: {e}")
        return 1

    job = post("/jobs", {"topic": topic, "slide_count": slides, "voice": voice, "music": True})
    job_id = job["job_id"]
    print(f"job_id : {job_id}\n", flush=True)

    start = time.time()
    last = ""
    while True:
        s = get(f"/jobs/{job_id}")
        el = int(time.time() - start)
        line = f"{s['status']:11s} {(s.get('current_stage') or '-'):12s} {s['progress']:3d}%"
        if line != last:
            print(f"[{el:4d}s] {line}", flush=True)
            last = line
        if s["status"] == "done":
            print(f"\nSUCCESS in {el}s")
            break
        if s["status"] == "failed":
            print(f"\nFAILED after {el}s: {s.get('error')}")
            return 1
        if el > MAX_WAIT:
            print(f"\nTIMED OUT after {MAX_WAIT}s (last: {line})")
            return 1
        time.sleep(3)

    print(f"video_url : {s.get('video_url')}")
    job_dir = REPO / "out" / job_id
    mp4s = sorted(job_dir.glob("*.mp4")) if job_dir.exists() else []
    if not mp4s:
        print(f"WARNING: no mp4 under {job_dir}")
        if job_dir.exists():
            for p in sorted(job_dir.rglob("*")):
                print("   ", p.relative_to(job_dir), p.stat().st_size if p.is_file() else "")
        return 1

    out = mp4s[-1]
    print(f"file      : {out}  ({out.stat().st_size / 1e6:.2f} MB)\n")
    subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size,bit_rate",
            "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate,channels,sample_rate",
            "-of", "default=noprint_wrappers=1",
            str(out),
        ],
        check=False,
    )

    # Expected-vs-actual duration: the xfade-overlap check.
    try:
        tl = get(f"/jobs/{job_id}/timeline")
        scenes = tl.get("scenes", [])
        narration = max((sc.get("end", 0) for sc in scenes), default=0)
        overlap = sum(
            (sc.get("plan") or {}).get("transition_duration", 0)
            for sc in scenes[1:]
            if (sc.get("plan") or {}).get("transition_in") != "cut"
        )
        print(f"\nscenes            : {len(scenes)}")
        print(f"narration total   : {narration:.3f}s")
        print(f"xfade overlap     : {overlap:.3f}s")
        print(f"expected final    : {narration - overlap:.3f}s")
    except Exception as e:  # noqa: BLE001 - diagnostic only
        print(f"\n(timeline check unavailable: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
