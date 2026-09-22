"""Matchup features: how a specific QB or unit interacts with conditions or opponents.

QB weather sensitivity
    For each QB, his EPA/dropback in bad weather (wind >= 15 mph, 1+ mm/hr of
    rain/snow, or <= 32F outdoors) minus his normal games, *beyond* the league-wide
    drop, shrunk toward 0: excess * n_bad / (n_bad + K). Applied only when the
    game itself is bad weather. Only earlier games are used.

Pass protection vs. pass rush
    Pressure = QB hit or sacked (in play-by-play for every season). Expected
    pressure for an offense = its pressure rate allowed x the opponent's pressure
    rate generated / league average, both as pregame rolling averages.

QB clutch and prime-time performance
    Same shrunk "excess over the league" approach, for (a) late-and-close plays
    (4th quarter or overtime, within one score) and (b) night games (7 PM ET or
    later). Clutch applies to every game; prime time only to night games.

Blitz vulnerability (FTN charting, 2022+)
    An offense's EPA/dropback when blitzed minus when not, beyond the league gap,
    shrunk toward 0, times the opponent's blitz rate. Zero before data exists.
"""
import bisect
from typing import Dict

import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON, FTN_FIRST_SEASON

BAD_WIND, BAD_PRECIP, BAD_TEMP = 15.0, 1.0, 32.0


def bad_weather(weather: pd.DataFrame) -> pd.Series:
    return ((weather.wind_mph >= BAD_WIND) | (weather.precip_mm >= BAD_PRECIP)
            | (weather.temp_f <= BAD_TEMP))


def load_dropbacks(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    """One row per QB dropback: game, offense, QB, EPA, and (2022+) blitzers."""
    from src.data_loader import load_ftn, load_pbp
    cols = ["game_id", "play_id", "posteam", "qb_dropback", "epa", "passer_player_id", "rusher_player_id",
            "qtr", "score_differential"]
    frames = []
    for s in seasons:
        p = load_pbp([s], columns=cols)
        p = p[(p.qb_dropback == 1) & p.epa.notna() & p.posteam.notna()]
        p = p.assign(qb_id=p.passer_player_id.fillna(p.rusher_player_id))
        p = p.assign(clutch=(p.qtr >= 4) & (p.score_differential.abs() <= 8))
        frames.append(p[["game_id", "play_id", "posteam", "qb_id", "epa", "clutch"]])
    db = pd.concat(frames, ignore_index=True)
    ftn = load_ftn(range(FTN_FIRST_SEASON, CURRENT_SEASON + 1))[["nflverse_game_id", "nflverse_play_id", "n_blitzers"]]
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    return db.merge(ftn, on=["game_id", "play_id"], how="left")


class _History:
    """Per-key cumulative sums over dated games; query totals strictly before a date."""

    def __init__(self, df: pd.DataFrame, key: str, cols):
        self.data: Dict[str, tuple] = {}
        for k, g in df.sort_values("gameday").groupby(key):
            self.data[k] = (list(g.gameday), g[cols].to_numpy(dtype=float).cumsum(axis=0))
        self.width = len(cols)

    def before(self, k, day) -> np.ndarray:
        h = self.data.get(k)
        if h is None:
            return np.zeros(self.width)
        n = bisect.bisect_left(h[0], day)
        return h[1][n - 1] if n else np.zeros(self.width)


def _league_gap_by_season(games: pd.DataFrame, n_a, s_a, n_b, s_b) -> pd.Series:
    """League EPA gap (group a minus group b) using all *earlier* seasons."""
    t = games.groupby("season")[[n_a, s_a, n_b, s_b]].sum().cumsum().shift(1)
    return (t[s_a] / t[n_a] - t[s_b] / t[n_b]).fillna(0.0)


def qb_weather_sensitivity(tg: pd.DataFrame, dropbacks: pd.DataFrame, weather: pd.DataFrame,
                           indoor: pd.DataFrame, k: float = 400.0) -> pd.DataFrame:
    """Per team-game row of tg (game_id, team, season, gameday, qb_id)."""
    w = weather.merge(indoor, on="game_id")
    w["bad"] = bad_weather(w) & (w.indoor == 0)
    g = (dropbacks.groupby(["game_id", "qb_id"]).epa.agg(["size", "sum"]).reset_index()
         .merge(w[["game_id", "bad"]], on="game_id")
         .merge(tg[["game_id", "season", "gameday"]].drop_duplicates("game_id"), on="game_id"))
    g["n_bad"] = np.where(g.bad, g["size"], 0)
    g["s_bad"] = np.where(g.bad, g["sum"], 0.0)
    g["n_good"] = np.where(g.bad, 0, g["size"])
    g["s_good"] = np.where(g.bad, 0.0, g["sum"])
    hist = _History(g, "qb_id", ["n_bad", "s_bad", "n_good", "s_good"])
    league = _league_gap_by_season(g, "n_bad", "s_bad", "n_good", "s_good")

    bad_now = tg.game_id.map(w.set_index("game_id").bad).fillna(False).to_numpy()
    sens = np.zeros(len(tg))
    for i, (qb, day, season) in enumerate(zip(tg.qb_id, tg.gameday, tg.season)):
        nb, sb, ng, sg = hist.before(qb, day)
        if nb > 0 and ng > 0:
            excess = (sb / nb - sg / ng) - league.get(season, 0.0)
            sens[i] = excess * nb / (nb + k)
    return pd.DataFrame({"qb_weather_sens": sens, "qb_weather_adj": sens * bad_now}, index=tg.index)


def qb_split_sensitivity(tg: pd.DataFrame, dropbacks: pd.DataFrame, flag: pd.Series, k: float) -> np.ndarray:
    """Shrunk excess EPA/dropback of each row's QB on flagged dropbacks vs his others,
    beyond the league gap, using only games before each row's date."""
    d = dropbacks.assign(f=flag.to_numpy())
    d = d.assign(n_a=d.f.astype(int), s_a=np.where(d.f, d.epa, 0.0), n_b=(~d.f).astype(int), s_b=np.where(d.f, 0.0, d.epa))
    g = d.groupby(["game_id", "qb_id"])[["n_a", "s_a", "n_b", "s_b"]].sum().reset_index()
    g = g.merge(tg[["game_id", "season", "gameday"]].drop_duplicates("game_id"), on="game_id")
    hist = _History(g, "qb_id", ["n_a", "s_a", "n_b", "s_b"])
    league = _league_gap_by_season(g, "n_a", "s_a", "n_b", "s_b")
    out = np.zeros(len(tg))
    for i, (qb, day, season) in enumerate(zip(tg.qb_id, tg.gameday, tg.season)):
        na, sa, nb, sb = hist.before(qb, day)
        if na > 0 and nb > 0:
            out[i] = ((sa / na - sb / nb) - league.get(season, 0.0)) * na / (na + k)
    return out


def blitz_vulnerability(tg: pd.DataFrame, dropbacks: pd.DataFrame, k: float = 300.0) -> pd.DataFrame:
    """Offense's shrunk blitz EPA gap and the defense's blitz rate, pregame."""
    d = dropbacks.dropna(subset=["n_blitzers"])
    d = d.assign(blitz=d.n_blitzers > 0)
    d = d.assign(n_b=d.blitz.astype(int), s_b=np.where(d.blitz, d.epa, 0.0),
                 n_nb=(~d.blitz).astype(int), s_nb=np.where(d.blitz, 0.0, d.epa))
    off = d.groupby(["game_id", "posteam"])[["n_b", "s_b", "n_nb", "s_nb"]].sum().reset_index()
    dates = tg[["game_id", "season", "gameday"]].drop_duplicates("game_id")
    off = off.merge(dates, on="game_id")
    # The defense on each dropback is the other team in the game.
    teams = tg[["game_id", "team", "opponent"]].dropna() if "opponent" in tg else None
    off_hist = _History(off, "posteam", ["n_b", "s_b", "n_nb", "s_nb"])
    league = _league_gap_by_season(off, "n_b", "s_b", "n_nb", "s_nb")
    opp = off.merge(teams.rename(columns={"team": "posteam", "opponent": "defteam"}), on=["game_id", "posteam"]) \
        if teams is not None else off.assign(defteam=np.nan)
    opp["n_all"] = opp.n_b + opp.n_nb
    def_hist = _History(opp, "defteam", ["n_b", "n_all"])

    gap = np.zeros(len(tg))
    rate = np.zeros(len(tg))
    for i, (team, day, season) in enumerate(zip(tg.team, tg.gameday, tg.season)):
        nb, sb, nnb, snb = off_hist.before(team, day)
        if nb > 0 and nnb > 0:
            gap[i] = ((sb / nb - snb / nnb) - league.get(season, 0.0)) * nb / (nb + k)
        db, dall = def_hist.before(team, day)
        rate[i] = db / dall if dall > 0 else np.nan
    return pd.DataFrame({"blitz_gap": gap, "def_blitz_rate": rate}, index=tg.index)


def blitz_team_games(seasons=None) -> pd.DataFrame:
    """Per team-game blitzed / not-blitzed dropbacks and EPA (FTN charting, 2022+), for the dashboard."""
    seasons = seasons or range(FTN_FIRST_SEASON, CURRENT_SEASON + 1)
    d = load_dropbacks(seasons).dropna(subset=["n_blitzers"])
    blitz = d.n_blitzers > 0
    d = d.assign(season=d.game_id.str[:4].astype(int), team=d.posteam,
                 n_b=blitz.astype(int), s_b=np.where(blitz, d.epa, 0.0),
                 n_nb=(~blitz).astype(int), s_nb=np.where(blitz, 0.0, d.epa))
    return d.groupby(["season", "team", "game_id"])[["n_b", "s_b", "n_nb", "s_nb"]].sum().reset_index()
