"""Live 2026 test: does blitz vulnerability improve the model?

Rule (set before any 2026 results were seen): after the 2026 regular season,
if the test model ("logistic_blitz") has lower log loss than the main model on
2026 games, blitz vulnerability joins the model.

Every prediction run records both models' probabilities for games not yet played
in reports/live_tracking.csv. Once a game has kicked off its row is frozen.
Weeks already played when tracking started are backfilled "as if live": each
week is predicted by models trained only on games before it.

Usage:
    python -m src.track            # scoreboard
    python -m src.track --backfill # (re)build as-if-live rows for played 2026 weeks
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, ROOT
from src.evaluate import TRAIN_START, metrics, moneyline_prob

TRACK_PATH = ROOT / "reports" / "live_tracking.csv"
MODELS = ("logistic", "logistic_blitz")
COLS = ["game_id", "season", "week", "gameday", "recorded", "source", "p_logistic", "p_logistic_blitz", "p_vegas"]


def _load() -> pd.DataFrame:
    if TRACK_PATH.exists():
        return pd.read_csv(TRACK_PATH, parse_dates=["gameday"])
    return pd.DataFrame(columns=COLS)


def record(games: pd.DataFrame, bundles: dict, source: str = "live"):
    """Record both models' probabilities for games that haven't kicked off yet."""
    from src.data_loader import load_schedules
    sched = load_schedules()[["game_id", "gameday", "gametime"]]
    kickoff = pd.to_datetime(sched.gameday + " " + sched.gametime.fillna("13:00"))  # US Eastern
    kickoff = games.game_id.map(dict(zip(sched.game_id, kickoff)))
    now_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    upcoming = games[games.home_score.isna() & (kickoff > now_et)]  # only games not yet kicked off
    if upcoming.empty:
        return
    rows = upcoming[["game_id", "season", "week", "gameday"]].copy()
    rows["recorded"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    rows["source"] = source
    for name, b in bundles.items():
        rows[f"p_{name}"] = b["model"].predict_proba(upcoming[b["features"]])[:, 1]
    rows["p_vegas"] = moneyline_prob(upcoming.home_moneyline, upcoming.away_moneyline)
    track = _load()
    # Replace earlier pregame rows for these games; rows for games already played stay frozen.
    track = track[~track.game_id.isin(rows.game_id)]
    pd.concat([track, rows[COLS]], ignore_index=True).sort_values(["gameday", "game_id"]).to_csv(TRACK_PATH, index=False)


def backfill(season: int = CURRENT_SEASON):
    """As-if-live predictions for weeks of `season` already played."""
    from src.train import MODELS as MAIN, TEST_MODELS
    factories = {"logistic": MAIN["logistic"], **TEST_MODELS}
    g = pd.read_parquet(ROOT / "data" / "processed" / "games_features.parquet")
    played = g[(g.season == season) & g.home_score.notna()]
    rows = []
    for week, wk in played.groupby("week"):
        train = g[g.home_win.notna() & (g.season >= TRAIN_START) & (g.gameday < wk.gameday.min())]
        r = wk[["game_id", "season", "week", "gameday"]].copy()
        r["recorded"] = f"backfill (trained through {train.gameday.max().date()})"
        r["source"] = "backfill"
        for name, (factory, cols) in factories.items():
            r[f"p_{name}"] = factory().fit(train[cols], train.home_win).predict_proba(wk[cols])[:, 1]
        r["p_vegas"] = moneyline_prob(wk.home_moneyline, wk.away_moneyline)
        rows.append(r)
    track = _load()
    new = pd.concat(rows)
    track = track[~track.game_id.isin(new.game_id)]
    pd.concat([track, new[COLS]], ignore_index=True).sort_values(["gameday", "game_id"]).to_csv(TRACK_PATH, index=False)
    print(f"Backfilled {len(new)} games from {season} weeks {sorted(new.week.unique())}")


def scoreboard(season: int = CURRENT_SEASON) -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "processed" / "games_features.parquet")
    t = _load()
    t = t[t.season == season].merge(g[["game_id", "home_win"]], on="game_id")
    t = t[t.home_win.notna()]
    rows = {}
    for name in MODELS + ("vegas",):
        col = f"p_{name}"
        d = t.dropna(subset=[col])
        rows[name] = {**metrics(d.home_win.to_numpy(), d[col].to_numpy()), "games": len(d)}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    if args.backfill:
        backfill()
    sb = scoreboard()
    print(f"\nLive {CURRENT_SEASON} test — blitz vulnerability (games played so far)")
    print(sb.round(4).to_string())
    if sb.loc["logistic", "games"]:
        lead = "logistic_blitz" if sb.loc["logistic_blitz", "log_loss"] < sb.loc["logistic", "log_loss"] else "logistic"
        gap = sb.loc["logistic", "log_loss"] - sb.loc["logistic_blitz", "log_loss"]
        print(f"\nLeading: {lead} (blitz model better by {gap:+.4f} log loss). "
              f"Decision after the {CURRENT_SEASON} regular season.")
