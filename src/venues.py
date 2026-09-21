"""Where and when each game is played: stadium location, time zone, altitude,
roof, and kickoff time in UTC. Also each team's home base for each season.

The nflverse schedule has a few venue errors (2025 international games are
listed at U.S. stadiums), which are corrected here.
"""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# code: (lat, lon, time zone, elevation in meters)
VENUES = {
    "ATL": (33.755, -84.401, "America/New_York", 300),
    "BAL": (39.278, -76.623, "America/New_York", 10),
    "BOS": (42.091, -71.264, "America/New_York", 80),
    "BUF": (42.774, -78.787, "America/New_York", 190),
    "TOR": (43.641, -79.389, "America/Toronto", 80),
    "CAR": (35.226, -80.853, "America/New_York", 230),
    "CHI": (41.862, -87.617, "America/Chicago", 180),
    "CIN": (39.095, -84.516, "America/New_York", 150),
    "CLE": (41.506, -81.700, "America/New_York", 180),
    "DAL": (32.748, -97.093, "America/Chicago", 180),
    "DAL_TX": (32.840, -96.911, "America/Chicago", 150),
    "DEN": (39.744, -105.020, "America/Denver", 1609),
    "DET": (42.340, -83.046, "America/Detroit", 180),
    "GB": (44.501, -88.062, "America/Chicago", 200),
    "HOU": (29.685, -95.411, "America/Chicago", 15),
    "IND": (39.760, -86.164, "America/Indiana/Indianapolis", 220),
    "JAX": (30.324, -81.637, "America/New_York", 5),
    "KC": (39.049, -94.484, "America/Chicago", 270),
    "LA_SOFI": (33.953, -118.339, "America/Los_Angeles", 30),
    "LA_CARSON": (33.864, -118.261, "America/Los_Angeles", 10),
    "LA_COL": (34.014, -118.288, "America/Los_Angeles", 60),
    "MIA": (25.958, -80.239, "America/New_York", 3),
    "MIN": (44.974, -93.258, "America/Chicago", 250),
    "MIN_TCF": (44.976, -93.225, "America/Chicago", 250),
    "NAS": (36.166, -86.771, "America/Chicago", 130),
    "NO": (29.951, -90.081, "America/Chicago", 3),
    "NYC": (40.813, -74.074, "America/New_York", 5),
    "OAK": (37.752, -122.201, "America/Los_Angeles", 5),
    "PHI": (39.901, -75.168, "America/New_York", 5),
    "PHO": (33.528, -112.263, "America/Phoenix", 330),
    "PIT": (40.447, -80.016, "America/New_York", 220),
    "SD": (32.783, -117.120, "America/Los_Angeles", 30),
    "SEA": (47.595, -122.332, "America/Los_Angeles", 5),
    "SF_CANDLE": (37.714, -122.386, "America/Los_Angeles", 5),
    "SF": (37.403, -121.970, "America/Los_Angeles", 5),
    "STL": (38.633, -90.188, "America/Chicago", 140),
    "TB": (27.976, -82.503, "America/New_York", 10),
    "LV": (36.091, -115.184, "America/Los_Angeles", 620),
    "WAS": (38.908, -76.864, "America/New_York", 50),
    # International
    "LON_WEMBLEY": (51.556, -0.280, "Europe/London", 40),
    "LON_TWICK": (51.456, -0.341, "Europe/London", 10),
    "LON_TOTT": (51.604, -0.066, "Europe/London", 30),
    "MUNICH": (48.219, 11.625, "Europe/Berlin", 500),
    "FRANKFURT": (50.069, 8.645, "Europe/Berlin", 110),
    "BERLIN": (52.515, 13.239, "Europe/Berlin", 50),
    "DUBLIN": (53.361, -6.251, "Europe/Dublin", 20),
    "MADRID": (40.453, -3.688, "Europe/Madrid", 650),
    "PARIS": (48.924, 2.360, "Europe/Paris", 40),
    "MEXICO": (19.303, -99.150, "America/Mexico_City", 2240),
    "SAO_PAULO": (-23.545, -46.474, "America/Sao_Paulo", 750),
    "RIO": (-22.912, -43.230, "America/Sao_Paulo", 10),
    "MELBOURNE": (-37.820, 144.983, "Australia/Melbourne", 30),
}

STADIUM_ID_MAP = {
    "ATL00": "ATL", "ATL97": "ATL", "BAL00": "BAL", "BOS00": "BOS", "BUF00": "BUF",
    "BUF01": "TOR", "CAR00": "CAR", "CHI98": "CHI", "CIN00": "CIN", "CLE00": "CLE",
    "DAL00": "DAL", "DAL99": "DAL_TX", "DEN00": "DEN", "DET00": "DET", "FRA00": "FRANKFURT",
    "GER00": "MUNICH", "MUN01": "MUNICH", "GNB00": "GB", "HOU00": "HOU", "IND00": "IND",
    "IND99": "IND", "JAX00": "JAX", "KAN00": "KC", "LAX01": "LA_SOFI", "LAX97": "LA_CARSON",
    "LAX99": "LA_COL", "LON00": "LON_WEMBLEY", "LON01": "LON_TWICK", "LON02": "LON_TOTT",
    "MAD01": "MADRID", "MEL00": "MELBOURNE", "MEX00": "MEXICO", "MIA00": "MIA", "MIN00": "MIN",
    "MIN01": "MIN", "MIN98": "MIN_TCF", "NAS00": "NAS", "NOR00": "NO", "NYC00": "NYC",
    "NYC01": "NYC", "OAK00": "OAK", "PAR00": "PARIS", "PHI00": "PHI", "PHO00": "PHO",
    "PIT00": "PIT", "RIO00": "RIO", "SAO00": "SAO_PAULO", "SDG00": "SD", "SEA00": "SEA",
    "SFO00": "SF_CANDLE", "SFO01": "SF", "STL00": "STL", "TAM00": "TB", "VEG00": "LV",
    "WAS00": "WAS",
}

# The stadium name wins over a wrong stadium_id (e.g. a JAX "home" game in London).
STADIUM_NAME_MAP = {
    "Tottenham Hotspur Stadium": "LON_TOTT", "Tottenham Stadium": "LON_TOTT",
    "Wembley Stadium": "LON_WEMBLEY", "Twickenham Stadium": "LON_TWICK",
}

# 2025 international games that the schedule lists at U.S. stadiums.
GAME_VENUE_FIXES = {
    "2025_01_KC_LAC": "SAO_PAULO",
    "2025_04_MIN_PIT": "DUBLIN",
    "2025_05_MIN_CLE": "LON_TOTT",
    "2025_06_DEN_NYJ": "LON_TOTT",
    "2025_07_LA_JAX": "LON_WEMBLEY",
    "2025_10_ATL_IND": "BERLIN",
    "2025_11_WAS_MIA": "MADRID",
}

# Open-air international venues the schedule marks as domes (or leaves blank).
OUTDOOR_VENUES = {"MELBOURNE", "PARIS", "MUNICH", "RIO", "SAO_PAULO", "DUBLIN", "BERLIN",
                  "MADRID", "FRANKFURT", "LON_WEMBLEY", "LON_TOTT", "LON_TWICK", "MEXICO"}
INDOOR_ROOFS = {"dome", "closed"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def game_venues(sched: pd.DataFrame) -> pd.DataFrame:
    """Per game: venue code, coordinates, tz, elevation, indoor flag, kickoff (UTC)."""
    code = sched.stadium.map(STADIUM_NAME_MAP).fillna(sched.stadium_id.map(STADIUM_ID_MAP))
    code = sched.game_id.map(GAME_VENUE_FIXES).fillna(code)
    if code.isna().any():
        raise ValueError(f"Unknown stadiums: {sched.loc[code.isna(), 'stadium'].unique()}")

    info = pd.DataFrame(VENUES, index=["lat", "lon", "tz", "elev"]).T
    v = info.loc[code].reset_index(drop=True)
    v.insert(0, "venue", code.to_numpy())
    v.insert(0, "game_id", sched.game_id.to_numpy())

    roof = sched.roof.fillna("").to_numpy()
    outdoor_fix = v.venue.isin(OUTDOOR_VENUES).to_numpy()
    v["indoor"] = np.where(outdoor_fix, 0, np.isin(roof, list(INDOOR_ROOFS)).astype(int))

    # Kickoff times in the schedule are US Eastern.
    et = ZoneInfo("America/New_York")
    local = pd.to_datetime(sched.gameday.astype(str) + " " + sched.gametime.fillna("13:00").to_numpy())
    v["kickoff_utc"] = [t.replace(tzinfo=et).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                        for t in local]
    return v


def team_home_venues(sched: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Each team's home venue per season (its most common non-neutral home venue)."""
    home = sched[["game_id", "season", "home_team", "location"]].merge(venues[["game_id", "venue"]])
    home = home[home.location != "Neutral"]
    home = home[~home.venue.isin(OUTDOOR_VENUES)]  # international "home" games aren't a home base
    base = (home.groupby(["season", "home_team"]).venue
            .agg(lambda x: x.value_counts().index[0]).rename("home_venue").reset_index()
            .rename(columns={"home_team": "team"}))
    # A team with no home game yet this season (e.g. early in the year, or only
    # when played games are included) keeps last season's home base.
    teams = pd.unique(pd.concat([sched.home_team, sched.away_team]))
    grid = pd.MultiIndex.from_product([sorted(sched.season.unique()), teams], names=["season", "team"])
    base = base.set_index(["season", "team"]).reindex(grid).reset_index().sort_values(["team", "season"])
    base["home_venue"] = base.groupby("team").home_venue.transform(lambda v: v.ffill().bfill())
    return base.reset_index(drop=True)


def utc_offset_hours(tz: str, when_utc: pd.Timestamp) -> float:
    return ZoneInfo(tz).utcoffset(when_utc.to_pydatetime()).total_seconds() / 3600
