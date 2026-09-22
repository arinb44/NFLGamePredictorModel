#!/bin/zsh
# Scheduled pregame run (see scripts/com.nflpredictor.weekly.plist):
# refresh data, predict upcoming games, record the live tests, and push results.
set -u
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT" || exit 1
PY="$PROJECT/.venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

# Only run while the season has games left to play.
if ! "$PY" -c "
import pandas as pd
from src.config import CURRENT_SEASON
from src.data_loader import load_schedules
s = load_schedules(refresh=True)
raise SystemExit(0 if s[(s.season == CURRENT_SEASON) & s.home_score.isna()].shape[0] else 1)
"; then
  echo "No upcoming games in the current season; nothing to do."
  exit 0
fi

"$PY" -m src.predict --refresh --top 3 || { echo "predict failed"; exit 1; }
"$PY" -m src.track | tail -8

# Update the hosted dashboard's data (GitHub Release "dashboard-data").
"$PY" -m src.publish 2>&1 | tail -1 || echo "Publishing dashboard data failed."

# Commit and push the pregame record (predictions + live-test tracking).
git add reports/predictions reports/live_tracking.csv
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -q -m "Pregame predictions $(date '+%Y-%m-%d %H:%M %Z')"
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -q origin main \
    && echo "Pushed $(git log --oneline -1)" || echo "Push failed; commit kept locally."
fi
