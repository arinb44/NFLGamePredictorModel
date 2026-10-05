#!/bin/zsh
# Pre-kickoff capture of prime-time betting splits (see scripts/com.nflpredictor.splits.plist):
# snapshot DraftKings handle/bets % for TNF, SNF and MNF games that haven't kicked off, and push.
set -u
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT" || exit 1
PY="$PROJECT/.venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "===== splits $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

"$PY" -m src.splits || { echo "splits capture failed"; exit 1; }

git add reports/prime_time_splits.csv reports/prime_time_games.csv
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -q -m "Prime-time betting splits $(date '+%Y-%m-%d %H:%M %Z')"
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -q origin main \
    && echo "Pushed $(git log --oneline -1)" || echo "Push failed; commit kept locally."
fi
