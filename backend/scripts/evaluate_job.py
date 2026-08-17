#!/usr/bin/env python
"""Rate a rendered video and print what to change.

    uv run python scripts/evaluate_job.py <job_id>
    uv run python scripts/evaluate_job.py <job_id> --no-vision
    uv run python scripts/evaluate_job.py --video out/x/video.mp4 --timeline out/x/timeline.json

Writes ``out/<job_id>/score.json`` and prints the scorecard. Exit status is the point of
the ``--fail-under`` flag: wire it into CI and a regression in image relevance or heading
contrast fails the build instead of shipping.

Exit codes: 0 scored (and above the threshold), 1 could not evaluate, 2 scored below
``--fail-under`` or blocked by a BLOCKER finding when ``--fail-on-blocker`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluate.models import Severity  # noqa: E402
from app.evaluate.scorer import (  # noqa: E402
    load_timeline,
    render_report,
    resolve_video,
    score_timeline,
    write_score,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_job",
        description="Score a rendered video per scene and emit actionable fixes.",
    )
    parser.add_argument("job_id", nargs="?", help="job id to score")
    parser.add_argument("--video", type=Path, help="path to the rendered mp4")
    parser.add_argument(
        "--timeline", type=Path, help="path to a Timeline JSON file (skips the DB and API)"
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="deterministic metrics only — free, offline, no Gemini call",
    )
    parser.add_argument("--model", help="Gemini model for the vision pass")
    parser.add_argument("--out", type=Path, help="where to write score.json")
    parser.add_argument("--json", action="store_true", help="print the JSON, not the report")
    parser.add_argument(
        "--fail-under", type=float, default=None, metavar="SCORE", help="exit 2 below this"
    )
    parser.add_argument(
        "--fail-on-blocker", action="store_true", help="exit 2 if any BLOCKER is found"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.job_id and not args.timeline:
        print("error: give a job_id, or --timeline with --video", file=sys.stderr)
        return 1

    try:
        timeline = load_timeline(args.job_id or "", timeline_path=args.timeline)
    except Exception as exc:  # noqa: BLE001 - the message is the whole diagnostic
        print(f"error: {exc}", file=sys.stderr)
        return 1

    job_id = args.job_id or timeline.job_id
    video = resolve_video(job_id, timeline, args.video)
    if not video.exists():
        print(f"error: no rendered video at {video}", file=sys.stderr)
        return 1

    try:
        score = score_timeline(
            timeline, video, vision=not args.no_vision, vision_model=args.model
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: scoring failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    destination = args.out or (video.parent / "score.json")
    write_score(score, destination)

    if args.json:
        print(json.dumps(score.model_dump(mode="json"), indent=2))
    else:
        print(render_report(score))
        print(f"  wrote {destination}")

    blocked = args.fail_on_blocker and any(
        r.severity == Severity.BLOCKER for r in score.recommendations
    )
    if blocked or (args.fail_under is not None and score.overall < args.fail_under):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
