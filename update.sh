#!/usr/bin/env bash
# Re-snapshot the LIVAGX analysis folders, regenerate every overall figure,
# rebuild the docs site, and record the result as one git commit + tag.
#
#   ./update.sh                      # message defaults to "update"
#   ./update.sh "added livagr halves"
#   ./update.sh --push "message"     # also push commit + tag if 'origin' exists
#
# Only overall/ is written. The livag*/ source folders are read, never changed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY=/hopper/home/jzii2024/miniconda3/envs/pmf/bin/python
PUSH=0
if [[ "${1:-}" == "--push" ]]; then PUSH=1; shift; fi
MSG="${1:-update}"
STAMP=$(date +%Y-%m-%d_%H%M)
export SNAPSHOT_STAMP="$STAMP"

step() { printf '\n== %s\n' "$*"; }

step "sync data snapshot"
$PY scripts/sync_data.py

for s in fig_overview_grid fig_cross_system fig_halved_compare fig_qc export_explorer_data; do
    step "$s"
    $PY "scripts/$s.py"
done

step "build docs"
$PY scripts/build_docs.py

step "git snapshot"
git add -A
if git diff --cached --quiet; then
    echo "Nothing changed since the last snapshot; no commit made."
    exit 0
fi
SUMMARY=$(git diff --cached --shortstat)
git commit -q -m "Snapshot $STAMP: $MSG" -m "$SUMMARY"
git tag "snapshot-$STAMP"
git log -1 --format='committed %h  %s'

if [[ $PUSH == 1 ]]; then
    if git remote get-url origin >/dev/null 2>&1; then
        git push origin HEAD --follow-tags
    else
        echo "No 'origin' remote set; skipped push. Add one with: git remote add origin <url>"
    fi
fi
