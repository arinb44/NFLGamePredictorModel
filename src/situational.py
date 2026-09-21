"""Situational features per team-game: travel and body clock, fatigue, weather.

Travel / circadian
    travel_km       distance from the team's home stadium to the venue
    tz_shift        hours of time-zone change (+ = traveled east)
    early_clock     hours before noon on the team's *body clock* at kickoff
                    (a 1 PM ET game is 10 AM for a West Coast team -> 2.0)
    late_clock      hours after 8 PM on the team's body clock at kickoff
    altitude_gain   km the venue sits above the team's home stadium (Denver, Mexico City)

Fatigue (only from games already played)
    prev_snaps      offensive + defensive snaps in the previous game
    prev_def_snaps  defensive snaps in the previous game
    snap_load       average snaps over the previous 3 games
    def_snap_load   average defensive snaps over the previous 3 games
    prev_def_ot_snaps  defensive snaps in overtime last game
    prev_ot         previous game went to overtime
    road_streak     consecutive road games before this one

Weather (game level, see weather.py)
    cold_shock      how much colder than the team's usual home climate (deg F)
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.venues import VENUES, game_venues, haversine_km, team_home_venues, utc_offset_hours

WEATHER_PATH = PROCESSED_DIR / "game_weather.parquet"


def load_game_weather() -> pd.DataFrame:
    return pd.read_parquet(WEATHER_PATH)


def _team_rows(sched: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "gameday", "location", "overtime"]
    home = sched[cols].assign(team=sched.home_team, is_home=1)
    away = sched[cols].assign(team=sched.away_team, is_home=0)
    rows = pd.concat([home, away], ignore_index=True)
    rows.loc[rows.location == "Neutral", "is_home"] = 0
    return rows


def _travel(rows: pd.DataFrame, venues: pd.DataFrame, bases: pd.DataFrame) -> pd.DataFrame:
    r = rows.merge(venues[["game_id", "venue", "lat", "lon", "tz", "elev", "kickoff_utc"]], on="game_id")
    r = r.merge(bases, on=["season", "team"], how="left")
    base = pd.DataFrame(VENUES, index=["base_lat", "base_lon", "base_tz", "base_elev"]).T
    r = r.join(base, on="home_venue")

    r["travel_km"] = haversine_km(r.base_lat.astype(float), r.base_lon.astype(float),
                                  r.lat.astype(float), r.lon.astype(float))
    offsets = {}

    def off(tz, t):
        key = (tz, t)
        if key not in offsets:
            offsets[key] = utc_offset_hours(tz, pd.Timestamp(t))
        return offsets[key]

    venue_off = np.array([off(tz, t) for tz, t in zip(r.tz, r.kickoff_utc)])
    base_off = np.array([off(tz, t) for tz, t in zip(r.base_tz, r.kickoff_utc)])
    r["tz_shift"] = (venue_off - base_off + 12) % 24 - 12  # +17h ahead == 7h behind
    k = pd.to_datetime(r.kickoff_utc)
    utc_hour = k.dt.hour + k.dt.minute / 60
    body = (utc_hour + base_off) % 24
    r["early_clock"] = np.clip(12 - body, 0, None)
    r["late_clock"] = np.clip(body - 20, 0, None)
    r["altitude_gain"] = np.clip((r.elev.astype(float) - r.base_elev.astype(float)) / 1000, 0, None)
    return r[["game_id", "team", "travel_km", "tz_shift", "early_clock", "late_clock",
              "altitude_gain", "home_venue"]]


def _fatigue(rows: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    snaps = team_games[["game_id", "team", "off_plays", "def_plays", "def_ot_plays"]]
    r = rows.merge(snaps, on=["game_id", "team"], how="left").sort_values(["team", "gameday"])
    r["snaps"] = r.off_plays + r.def_plays
    g = r.groupby(["team", "season"])
    r["prev_snaps"] = g.snaps.shift(1)
    r["prev_def_snaps"] = g.def_plays.shift(1)
    r["snap_load"] = g.snaps.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    r["def_snap_load"] = g.def_plays.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    r["prev_def_ot_snaps"] = g.def_ot_plays.shift(1).fillna(0)
    r["prev_ot"] = g.overtime.shift(1).fillna(0)
    # Consecutive road games before this one (neutral counts as road).
    away = (r.is_home == 0).astype(int)
    streak = []
    for _, s in away.groupby([r.team, r.season]):
        run, out = 0, []
        for a in s:
            out.append(run)
            run = run + 1 if a else 0
        streak.extend(out)
    r["road_streak"] = streak
    # Week 1 (no previous game this season): use the league average.
    for c in ["prev_snaps", "prev_def_snaps", "snap_load", "def_snap_load"]:
        r[c] = r[c].fillna(r[c].mean())
    return r[["game_id", "team", "prev_snaps", "prev_def_snaps", "snap_load", "def_snap_load",
              "prev_def_ot_snaps", "prev_ot", "road_streak"]]


def _cold_shock(rows: pd.DataFrame, venues: pd.DataFrame, bases: pd.DataFrame,
                weather: pd.DataFrame) -> pd.DataFrame:
    """Team's usual home climate = average home-game temperature over the previous
    two seasons (indoor stadiums count as 70F). cold_shock = max(0, climate - game temp)."""
    w = venues[["game_id", "venue"]].merge(weather[["game_id", "temp_f"]], on="game_id")
    home_games = rows[rows.is_home == 1].merge(w, on="game_id").merge(bases, on=["season", "team"])
    home_games = home_games[home_games.venue == home_games.home_venue]
    by_season = home_games.groupby(["team", "season"]).temp_f.mean().rename("season_temp").reset_index()
    by_season = by_season.sort_values(["team", "season"])
    by_season["climate"] = (by_season.groupby("team").season_temp
                            .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()))
    by_season["climate"] = by_season.climate.fillna(by_season.season_temp)
    r = rows.merge(w[["game_id", "temp_f"]], on="game_id").merge(
        by_season[["team", "season", "climate"]], on=["team", "season"], how="left")
    r["cold_shock"] = np.clip(r.climate - r.temp_f, 0, None).fillna(0)
    return r[["game_id", "team", "cold_shock"]]


def situational_features(sched: pd.DataFrame, team_games: pd.DataFrame, weather: pd.DataFrame):
    """sched: normalized schedule (all games incl. future). Returns
    (team-level frame keyed by game_id+team, game-level weather frame)."""
    sched = sched.reset_index(drop=True)
    venues = game_venues(sched)
    bases = team_home_venues(sched, venues)
    rows = _team_rows(sched)
    team = (rows[["game_id", "team"]]
            .merge(_travel(rows, venues, bases), on=["game_id", "team"])
            .merge(_fatigue(rows, team_games), on=["game_id", "team"])
            .merge(_cold_shock(rows, venues, bases, weather), on=["game_id", "team"], how="left"))
    game = venues[["game_id", "indoor"]].merge(
        weather[["game_id", "temp_f", "wind_mph", "precip_mm"]], on="game_id", how="left")
    return team, game


def update_game_weather(refresh: bool = False) -> pd.DataFrame:
    """Rebuild data/processed/game_weather.parquet. Cached venue history is reused;
    forecasts for upcoming games are refreshed when older than STALE_AFTER_HOURS."""
    from src.config import CURRENT_SEASON, FIRST_SEASON
    from src.data_loader import load_schedules
    from src.team_stats import normalize_teams
    from src.weather import game_weather

    s = load_schedules()
    s = normalize_teams(s[s.season.between(FIRST_SEASON, CURRENT_SEASON)].copy(), ["home_team", "away_team"])
    w = game_weather(game_venues(s.reset_index(drop=True)), refresh)
    w.to_parquet(WEATHER_PATH, index=False)
    return w
