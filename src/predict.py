"""Predict upcoming games with win probabilities and the factors behind them.

Usage:
    python -m src.predict                          # next unplayed week
    python -m src.predict --week 5                 # a specific week
    python -m src.predict --game NYG@LA            # one game
    python -m src.predict --refresh                # pull the latest data first
    python -m src.predict --qb PHI="Tanner McKee"  # what if a different QB starts?

Predictions are also saved to reports/predictions/.
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, MODELS_DIR, ROOT
from src.evaluate import moneyline_prob

PRED_DIR = ROOT / "reports" / "predictions"


def refresh_data():
    """Re-download the current season's files and rebuild team-game stats."""
    from src import data_loader
    from src.team_stats import build_team_games
    from src.config import PROCESSED_DIR

    data_loader.load_schedules(refresh=True)
    for kind in ("pbp", "player_stats", "rosters", "injuries"):
        data_loader._season_file(kind, CURRENT_SEASON, refresh=True)
    build_team_games().to_parquet(PROCESSED_DIR / "team_games.parquet", index=False)


def resolve_qb(team: str, name: str):
    """Find a QB's player id from his name (full or last name) on the team's current roster."""
    from src.data_loader import load_rosters
    from src.players import ROSTER_TEAM_MAP

    r = load_rosters([CURRENT_SEASON])
    r = r.assign(team=r.team.replace(ROSTER_TEAM_MAP))
    r = r[(r.position == "QB") & (r.team == team)].drop_duplicates("gsis_id", keep="last")
    q = name.lower()
    hit = r[r.full_name.str.lower() == q]
    if hit.empty:
        hit = r[r.full_name.str.lower().str.contains(q, regex=False)]
    if len(hit) != 1:
        options = ", ".join(r.full_name.unique())
        raise SystemExit(f"Couldn't uniquely match QB '{name}' on {team}. Roster QBs: {options}")
    return hit.gsis_id.iloc[0], hit.full_name.iloc[0]


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--week", type=int)
    ap.add_argument("--game", help="e.g. NYG@LA (away@home)")
    ap.add_argument("--refresh", action="store_true", help="download the latest data before predicting")
    ap.add_argument("--qb", action="append", default=[], metavar='TEAM="QB Name"',
                    help="override a team's starting QB (repeatable)")
    ap.add_argument("--top", type=int, default=5, help="factors to show per game")
    ap.add_argument("--assume-backups", action="store_true",
                    help="also start the backup when the listed QB left his last game early")
    args = ap.parse_args()

    from src.explain import explain, format_explanation
    from src.features import rebuild

    if args.refresh:
        print("Refreshing data...")
        refresh_data()

    overrides = {}
    for spec in args.qb:
        team, name = spec.split("=", 1)
        overrides[team.upper()] = resolve_qb(team.upper(), name.strip().strip('"'))
        print(f"QB override: {team.upper()} -> {overrides[team.upper()][1]}")

    games, missing = rebuild(qb_overrides=overrides or None, save=not (overrides or args.assume_backups),
                             assume_backups=args.assume_backups)
    qs = games.attrs.get("qb_status")
    if qs is not None and len(qs) and (qs.status != "ok").any():
        print("\nQB status for each team's next game:")
        for r in qs[qs.status != "ok"].itertuples():
            if r.applied:
                print(f"  {r.team}: {r.qb_name} -> {r.backup_name} starting ({r.reason})")
            else:
                print(f"  {r.team}: WARNING {r.qb_name} {r.reason}. Model still uses {r.qb_name}; "
                      f'to start the backup: --qb {r.team}="{r.backup_name}"  (or --assume-backups)')
    season = games[games.season == args.season]
    week = args.week
    if week is None:
        upcoming = season[season.home_win.isna() & season.home_score.isna()]
        if upcoming.empty:
            raise SystemExit(f"No upcoming games in {args.season}.")
        week = int(upcoming.week.min())
    g = season[season.week == week]
    if args.game:
        away, home = args.game.upper().split("@")
        g = g[(g.away_team == away) & (g.home_team == home)]
        if g.empty:
            raise SystemExit(f"{args.game} is not in {args.season} week {week}.")

    bundle = joblib.load(MODELS_DIR / "logistic.joblib")
    if not overrides and not args.assume_backups:  # record both models for the live blitz test (src/track.py)
        from src.track import record
        test_path = MODELS_DIR / "logistic_blitz.joblib"
        if test_path.exists():
            record(season, {"logistic": bundle, "logistic_blitz": joblib.load(test_path)})
    exps = explain(bundle, g.reset_index(drop=True), missing, top=args.top)
    g = g.reset_index(drop=True)
    vegas = moneyline_prob(g.home_moneyline, g.away_moneyline)

    print(f"\n{args.season} week {week} — model trained through {bundle['trained_through']}\n")
    rows = []
    for e, v, (_, r) in zip(exps, vegas, g.iterrows()):
        print(format_explanation(e, v))
        if r.home_score == r.home_score:  # already played
            print(f"  Final: {r.away_team} {r.away_score:.0f} - {r.home_team} {r.home_score:.0f}")
        print()
        rows.append({"game_id": e["game_id"], "away": e["away_team"], "home": e["home_team"],
                     "p_home": round(e["p_home"], 4), "p_vegas": None if np.isnan(v) else round(float(v), 4),
                     **{f"factor_{i + 1}": f"{f['favors']} +{abs(f['pct_points']) * 100:.1f}% {f['factor']}"
                        for i, f in enumerate(e["factors"])}})
    if not overrides and not args.assume_backups:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out = PRED_DIR / f"{args.season}_week{week:02d}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
