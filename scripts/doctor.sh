#!/usr/bin/env bash
# Preflight: verify every external dependency before blaming the code.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ok=0; fail=0
pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; ok=$((ok+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail+1)); }

echo "── toolchain ──"
for t in ffmpeg ffprobe uv pnpm node mprocs; do
  if command -v "$t" >/dev/null 2>&1; then pass "$t $($t --version 2>&1 | head -1 | cut -c1-40)"; else bad "$t missing"; fi
done

# This ffmpeg build lacks drawtext (no libfreetype), so headings are composited
# from an ImageMagick-rendered PNG. If drawtext ever appears, ImageMagick is optional.
if ffmpeg -hide_banner -filters 2>/dev/null | grep -q drawtext; then
  pass "ffmpeg drawtext (native text path)"
elif command -v magick >/dev/null 2>&1; then
  pass "imagemagick $(magick --version 2>&1 | head -1 | cut -c1-30) (drawtext absent, using overlay path)"
else
  bad "no text renderer: ffmpeg lacks drawtext AND magick missing (brew install imagemagick)"
fi

echo "── env ──"
if [ -f .env ]; then pass ".env present"; else bad ".env missing"; fi
for k in GEMINI_API_KEY DEEPGRAM_API_KEY; do
  v=$(grep "^${k}=" .env 2>/dev/null | cut -d= -f2)
  if [ -n "$v" ]; then pass "$k set"; else bad "$k empty"; fi
done

echo "── live APIs ──"
DG=$(grep '^DEEPGRAM_API_KEY=' .env 2>/dev/null | cut -d= -f2)
GK=$(grep '^GEMINI_API_KEY=' .env 2>/dev/null | cut -d= -f2)
c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Token $DG" https://api.deepgram.com/v1/models)
[ "$c" = "200" ] && pass "deepgram $c" || bad "deepgram $c"
c=$(curl -s -o /dev/null -w '%{http_code}' "https://generativelanguage.googleapis.com/v1beta/models?key=$GK")
[ "$c" = "200" ] && pass "gemini $c" || bad "gemini $c"

echo "── gates ──"
# Guard against a regression to `tsc --noEmit`, which passes vacuously here.
if grep -q '"typecheck"' frontend/package.json 2>/dev/null; then
  pass "frontend typecheck script present (tsc -b)"
else
  bad "frontend has no 'typecheck' script — 'tsc --noEmit' is vacuous with a solution tsconfig"
fi

echo "── artifacts ──"
# VIDEO_OUTPUT_DIR is `./out`, which once resolved against the launch CWD and produced a
# second tree under backend/. config.py anchors it to the repo root now; this catches a
# regression, since a split tree silently hides finished videos from the API.
if [ -d backend/out ]; then
  bad "stray backend/out/ ($(du -sh backend/out | cut -f1)) — VIDEO_OUTPUT_DIR resolved against the wrong CWD"
else
  pass "single artifact tree (out/)"
fi
if [ -d out ]; then
  empties=$(find out -mindepth 1 -maxdepth 1 -type d -empty 2>/dev/null | wc -l | tr -d ' ')
  [ "$empties" = "0" ] && pass "no empty job dirs" || bad "$empties empty job dir(s) in out/ (failed jobs)"
fi

echo "── deps ──"
[ -d backend/.venv ] && pass "backend venv" || bad "backend venv (run: cd backend && uv sync --extra dev)"
[ -d frontend/node_modules ] && pass "frontend node_modules" || bad "frontend node_modules (run: cd frontend && pnpm install --force)"

echo
printf '%d passed, %d failed\n' "$ok" "$fail"
[ "$fail" -eq 0 ] || exit 1
