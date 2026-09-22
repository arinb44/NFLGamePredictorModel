"""Is the listed starting QB actually available for his team's next game?

For each team's next game:
- Out / Doubtful on the official injury report, or reserve / inactive on the
  weekly roster  ->  "out": the backup is swapped in automatically.
- Left his most recent game early (he played, but another QB took the team's
  final 5+ dropbacks)  ->  "left_early": a warning only (could be an injury or
  a blowout benching), with the suggested backup.

The backup must itself be available (not Out/Doubtful, reserve or inactive).
Preference: the available QB who took most of the team's dropbacks in its last
game, then the latest depth chart order. Depth charts don't always reflect
injuries (in 2026 Atlanta's listed QB1 and QB2 were both inactive).
"""
import pandas as pd

from src.config import CURRENT_SEASON
from src.players import OUT_REPORT, OUT_STATUSES, ROSTER_TEAM_MAP

MIN_BACKUP_DROPBACKS = 5


def _second_half_qbs(season: int) -> pd.DataFrame:
    """Per team-game: dropbacks by QB in the second half (and overtime)."""
    from src.data_loader import load_pbp
    p = load_pbp([season], columns=["game_id", "play_id", "week", "posteam", "qtr", "qb_dropback", "passer_player_id",
                                    "rusher_player_id", "passer_player_name", "rusher_player_name"])
    p = p[(p.qb_dropback == 1) & p.posteam.notna()]
    p = p.assign(qb_id=p.passer_player_id.fillna(p.rusher_player_id),
                 qb_name=p.passer_player_name.fillna(p.rusher_player_name), half2=p.qtr >= 3)
    return p


def qb_availability(sched: pd.DataFrame, season: int = CURRENT_SEASON) -> pd.DataFrame:
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
        report = inj[(inj.team == r.team) & (inj.week == r.week) & (inj.gsis_id == r.qb_id)]
        roster = ros_latest[(ros_latest.team == r.team) & (ros_latest.gsis_id == r.qb_id)]
        if len(report) and report.report_status.iloc[-1] in OUT_REPORT:
            status, reason = "out", f"{report.report_status.iloc[-1]} on the week {r.week} injury report"
        elif len(roster) and roster.status.iloc[-1] in OUT_STATUSES:
            status, reason = "out", f"roster status {roster.status.iloc[-1]} (week {roster.week.iloc[-1]})"

        # Did he play the team's most recent game but leave before the end?
        team_drops = drops[drops.posteam == r.team]
        finisher = None
        if len(team_drops):
            last = team_drops[team_drops.week == team_drops.week.max()].sort_values("play_id")
            mine = last[last.qb_id == r.qb_id]
            if len(mine):
                after = last[last.play_id > mine.play_id.max()]
                if len(after) >= MIN_BACKUP_DROPBACKS and (after.qb_id != r.qb_id).all():
                    top = after.groupby(["qb_id", "qb_name"]).size().sort_values(ascending=False)
                    finisher = top.index[0]
                    if status == "ok":
                        status = "left_early"
                        reason = (f"left week {last.week.max()} early: {top.index[0][1]} took the team's "
                                  f"last {len(after)} dropbacks (injury or benching?)")
        if status != "ok":
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
