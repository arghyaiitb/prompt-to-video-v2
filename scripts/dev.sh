#!/usr/bin/env bash
# Launch the full stack regardless of where this is called from.
# mprocs resolves `cwd` against the launch directory, so anchor it here.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v mprocs >/dev/null 2>&1 || { echo "mprocs not found: brew install mprocs"; exit 1; }
[ -d backend/.venv ]        || { echo "backend deps missing: cd backend && uv sync --extra dev"; exit 1; }
[ -d frontend/node_modules ] || { echo "frontend deps missing: cd frontend && pnpm install --force"; exit 1; }

exec mprocs -c mprocs.yaml
