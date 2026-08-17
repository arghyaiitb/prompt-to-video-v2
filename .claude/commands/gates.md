---
description: Run every verification gate for this repo (backend tests, ruff, real frontend typecheck, doctor)
allowed-tools: Bash, Read, Grep, Glob
---

Run all four gates and report a pass/fail table. Free and offline apart from `doctor.sh`, which
only hits free provider *list* endpoints. Run them all even if an early one fails — the user wants
the whole picture, not the first error.

```bash
cd backend  && uv run pytest -q                 # ~110s
cd backend  && uv run ruff check app/
cd frontend && pnpm typecheck                   # tsc -b --force
cd frontend && pnpm lint                        # oxlint
./scripts/doctor.sh                             # from the repo root
```

## Rules

- **Never substitute `pnpm exec tsc --noEmit`.** `frontend/tsconfig.json` is a solution file
  (`files: []` + project references), so without `-b` tsc compiles an *empty program* and exits 0
  on any type error. Proof: `pnpm exec tsc --noEmit --listFiles | grep -c '/frontend/src/'` → `0`
  versus `pnpm exec tsc -b --force --listFiles | grep -c '/frontend/src/'` → `44`.
  `pnpm typecheck` and `pnpm build` are the only trustworthy type gates.
- This is a **shared working tree with several agents editing concurrently**. Before reporting a
  failure as a regression, check `git status` and `git diff` to see whether the failing file is
  someone else's in-flight work. A `SyntaxError` or a handful of failures confined to one module
  you did not touch is almost certainly not yours.
- **Never run `git stash`**, `git reset --hard`, `git checkout .` or `git clean` to get a clean
  run. An agent did that once and stashed five other agents' uncommitted work.
- Do not "fix" failures in files outside your assignment. Report them and name the owner file.

## Report

A four-row table: gate, result, and the headline number (tests passed/failed, ruff findings, tsc
errors, doctor's `N passed, M failed`). For each failure give the shortest reproducing command and
say whether it looks like your change or another agent's.
