"""Is the listed starting QB actually available for his team's next game?

For each team's next game:
- Out / Doubtful on the official injury report, or reserve / inactive on the
  weekly roster  ->  "out": the backup is swapped in automatically.
- Left his most recent game early (he played, but another QB took the team's
  final 5+ dropbacks)  ->  "left_early": a warning only (could be an injury or
  a blowout benching), with the suggested backup.

Standing starters (config/starters.json, set with `python -m src.qb_status --set`)
are your own information about who starts. They replace the listed starter for
every upcoming game and skip the roster and "left early" checks; only an official
Out/Doubtful injury report for that week overrides them (with a warning).

The backup must itself be available (not Out/Doubtful, reserve or inactive).
Preference: the available QB who took most of the team's dropbacks in its last
game, then the latest depth chart order. Depth charts don't always reflect
injuries (in 2026 Atlanta's listed QB1 and QB2 were both inactive).
"""
import json

import pandas as pd

from src.config import CURRENT_SEASON, ROOT
from src.players import OUT_REPORT, OUT_STATUSES, ROSTER_TEAM_MAP

MIN_BACKUP_DROPBACKS = 5
STARTERS_PATH = ROOT / "config" / "starters.json"


def resolve_qb(team: str, name: str):
    """Find a QB's player id from his name (full or last name) on the team's current roster."""
    from src.data_loader import load_rosters

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


def load_standing_starters() -> dict:
    """{team: (qb_id, qb_name)} from config/starters.json."""
    if not STARTERS_PATH.exists():
        return {}
    data = json.loads(STARTERS_PATH.read_text())
    return {t: (v["qb_id"], v["qb_name"]) for t, v in data.items()}


def load_standing_since() -> dict:
    """{team: date the standing starter was set}."""
    if not STARTERS_PATH.exists():
        return {}
    return {t: pd.Timestamp(v.get("since", "1900-01-01")) for t, v in json.loads(STARTERS_PATH.read_text()).items()}


def _save_standing(data: dict):
    STARTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STARTERS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _second_half_qbs(season: int) -> pd.DataFrame:
    """Per team-game: dropbacks by QB in the second half (and overtime)."""
    from src.data_loader import load_pbp
    p = load_pbp([season], columns=["game_id", "play_id", "week", "posteam", "qtr", "qb_dropback", "passer_player_id",
                                    "rusher_player_id", "passer_player_name", "rusher_player_name"])
    p = p[(p.qb_dropback == 1) & p.posteam.notna()]
    p = p.assign(qb_id=p.passer_player_id.fillna(p.rusher_player_id),
                 qb_name=p.passer_player_name.fillna(p.rusher_player_name), half2=p.qtr >= 3)
    return p


def qb_availability(sched: pd.DataFrame, season: int = CURRENT_SEASON, standing: dict = None) -> pd.DataFrame:
    """sched: normalized schedule with home/away_qb_id and _name. Returns one row per
    (game_id, team) for each team's next unplayed game with status and backup."""
    from src.data_loader import load_depth_charts, load_injuries, load_rosters

    up = sched[(sched.season == season) & sched.home_score.isna()].sort_values("gameday")
    if up.empty:
        return pd.DataFrame()
    rows = pd.concat([
        up[["game_id", "week", "gameday", "home_team", "home_qb_id", "home_qb_name"]].set_axis(
            ["game_id", "week", "gameday", "team", "qb_id", "qb_name"], axis=1),
        up[["game_id", "week", "gameday", "away_team", "away_qb_id", "away_qb_name"]].set_axis(
            ["game_id", "week", "gameday", "team", "qb_id", "qb_name"], axis=1),
    ]).sort_values("gameday").groupby("team").head(1)  # each team's next game

    inj = load_injuries([season]).assign(team=lambda d: d.team.replace(ROSTER_TEAM_MAP))
    ros = load_rosters([season]).assign(team=lambda d: d.team.replace(ROSTER_TEAM_MAP))
    ros_latest = ros[ros.week == ros.groupby("team").week.transform("max")]
    try:
        dc = load_depth_charts(season)
        dc = dc[dc.pos_abb == "QB"].assign(team=lambda d: d.team.replace(ROSTER_TEAM_MAP))
        dc = dc[dc.dt == dc.groupby("team").dt.transform("max")].sort_values(["team", "pos_rank"])
    except Exception:
        dc = pd.DataFrame(columns=["team", "gsis_id", "player_name", "pos_rank"])
    drops = _second_half_qbs(season)

    out = []
    for r in rows.itertuples():
        status, reason, backup_id, backup_name = "ok", "", None, None
        is_standing = r.team in (standing or {})
        report = inj[(inj.team == r.team) & (inj.week == r.week) & (inj.gsis_id == r.qb_id)]
        roster = ros_latest[(ros_latest.team == r.team) & (ros_latest.gsis_id == r.qb_id)]
        if len(report) and report.report_status.iloc[-1] in OUT_REPORT:
            status, reason = "out", f"{report.report_status.iloc[-1]} on the week {r.week} injury report"
            if is_standing:
                reason += " (overrides your standing starter)"
        elif is_standing:
            status, reason = "set", "your standing starter (config/starters.json)"
        elif len(roster) and roster.status.iloc[-1] in OUT_STATUSES:
            status, reason = "out", f"roster status {roster.status.iloc[-1]} (week {roster.week.iloc[-1]})"

        # Did he play the team's most recent game but leave before the end?
        team_drops = drops[drops.posteam == r.team]
        finisher = None
        if len(team_drops):
            last = team_drops[team_drops.week == team_drops.week.max()].sort_values("play_id")
            mine = last[last.qb_id == r.qb_id]
            if len(mine) and status != "set":
                after = last[last.play_id > mine.play_id.max()]
                if len(after) >= MIN_BACKUP_DROPBACKS and (after.qb_id != r.qb_id).all():
                    top = after.groupby(["qb_id", "qb_name"]).size().sort_values(ascending=False)
                    finisher = top.index[0]
                    if status == "ok":
                        status = "left_early"
                        reason = (f"left week {last.week.max()} early: {top.index[0][1]} took the team's "
                                  f"last {len(after)} dropbacks (injury or benching?)")
        if status in ("out", "left_early"):
            def available(pid):
                rep_ = inj[(inj.team == r.team) & (inj.week == r.week) & (inj.gsis_id == pid)]
                if len(rep_) and rep_.report_status.iloc[-1] in OUT_REPORT:
                    return False
                st = ros_latest[(ros_latest.team == r.team) & (ros_latest.gsis_id == pid)]
                return not (len(st) and st.status.iloc[-1] in OUT_STATUSES)

            candidates = []
            if len(team_drops):  # who actually ran the offense in the last game
                last = team_drops[team_drops.week == team_drops.week.max()]
                candidates += list(last.groupby("qb_id").size().sort_values(ascending=False).index)
            candidates += list(dc[dc.team == r.team].gsis_id)
            for pid in candidates:
                if pid != r.qb_id and isinstance(pid, str) and available(pid):
                    backup_id = pid
                    break
            if backup_id is not None:
                names = ros[ros.gsis_id == backup_id].full_name
                dnames = dc[dc.gsis_id == backup_id].player_name
                backup_name = names.iloc[-1] if len(names) else (dnames.iloc[0] if len(dnames) else backup_id)
        out.append({"game_id": r.game_id, "week": r.week, "team": r.team, "qb_id": r.qb_id, "qb_name": r.qb_name,
                    "status": status, "reason": reason, "backup_id": backup_id, "backup_name": backup_name})
    return pd.DataFrame(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Manage standing starting QBs used by every prediction run.")
    ap.add_argument("--set", nargs=2, metavar=("TEAM", "NAME"), help='e.g. --set ATL "Michael Penix Jr."')
    ap.add_argument("--clear", metavar="TEAM", help="remove a team's standing starter")
    args = ap.parse_args()
    data = json.loads(STARTERS_PATH.read_text()) if STARTERS_PATH.exists() else {}
    if args.set:
        team = args.set[0].upper()
        qb_id, qb_name = resolve_qb(team, args.set[1])
        data[team] = {"qb_id": qb_id, "qb_name": qb_name, "since": pd.Timestamp.now().strftime("%Y-%m-%d")}
        _save_standing(data)
        print(f"{team}: {qb_name} is the standing starter")
    elif args.clear:
        data.pop(args.clear.upper(), None)
        _save_standing(data)
        print(f"{args.clear.upper()}: standing starter cleared")
    print("Standing starters:", {t: v["qb_name"] for t, v in data.items()} or "none")
