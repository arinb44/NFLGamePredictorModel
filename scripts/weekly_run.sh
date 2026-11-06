#!/bin/zsh
# Scheduled pregame run (see scripts/com.nflpredictor.weekly.plist):
# refresh data, predict upcoming games, record the live tests, and push results.
set -u
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT" || exit 1
PY="$PROJECT/.venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
step() { echo "--- $(date '+%H:%M:%S') $1"; }

# Right after the Mac wakes, the network may not be up yet: wait up to 10 minutes.
for i in {1..20}; do curl -sfI --max-time 10 https://github.com >/dev/null && break; sleep 30; done

# Only run while the season has games left to play (exit 3 = none left; anything else nonzero is an error).
"$PY" -c "
import pandas as pd
from src.config import CURRENT_SEASON
from src.data_loader import load_schedules
s = load_schedules(refresh=True)
raise SystemExit(0 if s[(s.season == CURRENT_SEASON) & s.home_score.isna()].shape[0] else 3)
"
rc=$?
if [ $rc -eq 3 ]; then
  echo "No upcoming games in the current season; nothing to do."
  exit 0
elif [ $rc -ne 0 ]; then
  echo "Season check failed (see the error above); no predictions this run."
  exit 1
fi

step predict
"$PY" -m src.predict --refresh --top 3 || { echo "predict failed"; exit 1; }
step track
"$PY" -m src.track | tail -8

# Prime-time betting splits: early DraftKings snapshot (pre-kickoff ones come from splits_run.sh)
# and Action Network splits for prime-time games that just finished.
step splits
"$PY" -m src.splits || echo "Splits capture failed."

# Update the hosted dashboard's data (GitHub Release "dashboard-data").
step publish
"$PY" -m src.publish 2>&1 | tail -1 || echo "Publishing dashboard data failed."

# Commit and push the pregame record (predictions + live-test tracking).
step push
git add reports/predictions reports/live_tracking.csv reports/prime_time_splits.csv reports/prime_time_games.csv reports/prime_time_public.csv
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -q -m "Pregame predictions $(date '+%Y-%m-%d %H:%M %Z')"
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -q origin main \
    && echo "Pushed $(git log --oneline -1)" || echo "Push failed; commit kept locally."
fi
