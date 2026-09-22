"""Skill-player (RB / WR / TE) profiles and availability.

Team rolling stats already contain the production of players who have been
playing. What they miss is *who is unavailable today*. So for every team-game:

1. Each skill player has a pregame value: his *role size*, the share of his
   team's RB/WR/TE targets + carries he gets, averaged over his earlier games
   (exponentially weighted, shrunk toward 0). Role size proved a steadier
   signal than per-game EPA, which is noisy and mixes in the QB's quality.
2. Each player has a participation score for his team: an exponentially
   weighted share of the team's recent games he played in.
3. A player is *missing* if he has participation with this team, is still on
   the team's weekly roster, and does not play in this game.
   - Played games: "plays" = he appears in that game's player stats. Inactive
     lists are public ~90 minutes before kickoff, so this is pregame knowledge.
   - Upcoming games: he is missing if the injury report lists him Out/Doubtful
     or his roster status is reserve/inactive.
4. missing_value = sum of value * participation over missing players, i.e. the
   share of the team's usual skill-player touches that is unavailable.
"""
import bisect
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON
from src.data_loader import load_injuries, load_player_stats, load_rosters

SKILL_GROUPS = ("RB", "WR", "TE")

# Roster files use their own historical codes.
ROSTER_TEAM_MAP = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA",
                   "STL": "LA", "OAK": "LV", "SD": "LAC"}
GONE_STATUSES = {"CUT", "TRD", "TRC", "TRT", "RET", "EXE", "RSN"}  # no longer with the team
OUT_STATUSES = {"RES", "INA", "PUP", "NON", "SUS"}                 # on the team but not playing
OUT_REPORT = {"Out", "Doubtful"}


def load_skill_log(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    """One row per skill player per game played."""
    ps = load_player_stats(seasons)
    ps = ps[ps.position_group.isin(SKILL_GROUPS)].copy()
    ps["team"] = ps.team.replace(ROSTER_TEAM_MAP)
    touches = ps.targets.fillna(0) + ps.carries.fillna(0)
    ps["value"] = touches / touches.groupby([ps.game_id, ps.team]).transform("sum")
    ps["epa"] = ps.receiving_epa.fillna(0) + ps.rushing_epa.fillna(0)  # kept for explanations
    return ps[["player_id", "player_display_name", "position_group", "team", "game_id",
               "season", "week", "value", "epa"]].rename(columns={"player_display_name": "name"})


def load_roster_status(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    r = load_rosters(seasons)
    r = r[r.position.isin(SKILL_GROUPS) | r.depth_chart_position.isin(SKILL_GROUPS)]
    r = r.assign(team=r.team.replace(ROSTER_TEAM_MAP))
    return r[["season", "week", "team", "gsis_id", "status"]].rename(columns={"gsis_id": "player_id"})


# Defense: same availability logic, valued by share of the team's defensive playmaking.
DEF_GROUPS = ("DL", "LB", "DB")
DEF_POSITIONS = {"DL", "LB", "DB", "DE", "DT", "NT", "OLB", "ILB", "MLB", "CB", "S", "SS", "FS", "SAF", "EDGE"}
DEF_WEIGHTS = {"def_sacks": 1.0, "def_interceptions": 1.0, "def_fumbles_forced": 1.0,
               "def_qb_hits": 0.5, "def_tackles_for_loss": 0.5, "def_pass_defended": 0.5}


def load_def_log(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    """One row per defender per game played. value = his share of the team's
    defensive playmaking that game (sacks, INTs, forced fumbles = 1; QB hits,
    tackles for loss, passes defended = 0.5)."""
    ps = load_player_stats(seasons)
    ps = ps[ps.position_group.isin(DEF_GROUPS)].copy()
    ps["team"] = ps.team.replace(ROSTER_TEAM_MAP)
    plays = sum(ps[c].fillna(0) * w for c, w in DEF_WEIGHTS.items() if c in ps)
    total = plays.groupby([ps.game_id, ps.team]).transform("sum")
    ps["value"] = (plays / total.where(total > 0)).fillna(0.0)
    ps["epa"] = plays  # raw playmaking count, kept for display
    return ps[["player_id", "player_display_name", "position_group", "team", "game_id",
               "season", "week", "value", "epa"]].rename(columns={"player_display_name": "name"})


def load_def_roster_status(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    r = load_rosters(seasons)
    r = r[r.position.isin(DEF_POSITIONS) | r.depth_chart_position.isin(DEF_POSITIONS)]
    r = r.assign(team=r.team.replace(ROSTER_TEAM_MAP))
    return r[["season", "week", "team", "gsis_id", "status"]].rename(columns={"gsis_id": "player_id"})


def load_report_outs(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    inj = load_injuries(seasons)
    inj = inj[inj.report_status.isin(OUT_REPORT)]
    inj = inj.assign(team=inj.team.replace(ROSTER_TEAM_MAP))
    return inj[["season", "week", "team", "gsis_id"]].rename(columns={"gsis_id": "player_id"})


class _PlayerValues:
    """Pregame value of any player as of any date (uses only earlier games)."""

    def __init__(self, log: pd.DataFrame, decay: float, season_decay: float, prior_games: float):
        self.decay, self.season_decay, self.prior_games = decay, season_decay, prior_games
        self.hist: Dict[str, Tuple[List, np.ndarray, np.ndarray]] = {}
        for pid, g in log.sort_values("gameday").groupby("player_id"):
            self.hist[pid] = (list(g.gameday), g.season.to_numpy(), g.value.to_numpy(dtype=float))

    def value(self, pid: str, day: pd.Timestamp, season: int) -> float:
        h = self.hist.get(pid)
        if h is None:
            return 0.0
        days, seasons, vals = h
        n = bisect.bisect_left(days, day)  # games strictly before `day`
        if n == 0:
            return 0.0
        v = vals[:n]
        w = self.decay ** np.arange(n - 1, -1, -1) * self.season_decay ** (season - seasons[:n])
        return float((w * v).sum() / (self.prior_games + w.sum()))


def skill_availability(
    tg: pd.DataFrame,
    log: pd.DataFrame,
    rosters: pd.DataFrame,
    report_outs: pd.DataFrame,
    value_decay: float = 0.95,
    value_season_decay: float = 0.7,
    value_prior_games: float = 4.0,
    part_decay: float = 0.8,
    part_season_decay: float = 0.5,
    return_details: bool = False,
):
    """Missing skill-player value for each row of tg (team-game rows with
    game_id, team, season, week, gameday). Rows whose game_id has no player
    stats are treated as upcoming games and use the injury report instead.

    Returns a frame aligned with tg.index (and optionally the list of missing
    players for explanations)."""
    log = log.merge(tg[["game_id", "gameday"]].drop_duplicates(), on="game_id")
    values = _PlayerValues(log, value_decay, value_season_decay, value_prior_games)
    played = log.groupby(["game_id", "team"]).player_id.apply(set).to_dict()
    names = log.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "position_group"]]

    roster = rosters.groupby(["season", "week", "team"])
    roster_status = {k: dict(zip(g.player_id, g.status)) for k, g in roster}
    latest_week = rosters.groupby(["season", "team"]).week.max().to_dict()
    outs = report_outs.groupby(["season", "week", "team"]).player_id.apply(set).to_dict()

    missing_total = np.zeros(len(tg))
    missing_top = np.zeros(len(tg))
    details = []
    pos_of = {ix: i for i, ix in enumerate(tg.index)}

    for team, rows in tg.sort_values("gameday").groupby("team", sort=False):
        part: Dict[str, float] = {}
        cur_season = None
        for r in rows.itertuples():
            if r.season != cur_season:
                part = {p: s * part_season_decay for p, s in part.items()}
                cur_season = r.season
            week = int(r.week)
            key = (r.season, week, team)
            last_week = latest_week.get((r.season, team), week)
            if key not in roster_status:  # e.g. upcoming week not published yet
                key = (r.season, last_week, team)
            status = roster_status.get(key)
            here = played.get((r.game_id, team))
            upcoming = here is None
            report = outs.get((r.season, week, team), set())

            total, top = 0.0, 0.0
            for pid, p in part.items():
                if p < 0.05:
                    continue
                st = status.get(pid) if status is not None else "ACT"
                if st is None or st in GONE_STATUSES:
                    continue  # left the team (or not on this week's roster)
                if upcoming:
                    # Roster status and injury reports only tell us about the next
                    # game; further out, assume no known absences.
                    is_missing = week <= last_week + 1 and (pid in report or st in OUT_STATUSES)
                else:
                    is_missing = pid not in here
                if not is_missing:
                    continue
                v = max(values.value(pid, r.gameday, r.season), 0.0)
                contrib = v * p
                total += contrib
                top = max(top, contrib)
                if return_details and contrib > 0:
                    info = names.loc[pid] if pid in names.index else None
                    details.append({"game_id": r.game_id, "team": team, "player_id": pid,
                                    "name": None if info is None else info["name"],
                                    "position": None if info is None else info["position_group"],
                                    "value": v, "participation": p, "missing_value": contrib})
            i = pos_of[r.Index]
            missing_total[i], missing_top[i] = total, top

            # Update participation after the game (only for games that were played).
            if not upcoming:
                part = {pid: s * part_decay for pid, s in part.items()}
                for pid in here:
                    part[pid] = part.get(pid, 0.0) + (1 - part_decay)

    out = pd.DataFrame({"skill_missing": missing_total, "skill_missing_top": missing_top},
                       index=tg.index)
    if return_details:
        return out, pd.DataFrame(details)
    return out
