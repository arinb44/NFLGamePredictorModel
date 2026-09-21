"""Game-time weather from Open-Meteo (https://open-meteo.com, CC BY 4.0).

- Past games: hourly ERA5 reanalysis from the archive API.
- Games in the next ~16 days: the forecast API.
- Games further out: the venue's historical average for that time of year.

Each game gets the average of the 4 hours from kickoff (temperature in F,
wind in mph, precipitation in mm/hour). Indoor games get fixed values.
Hourly data is cached per venue under data/raw/weather/.
"""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DIR, STALE_AFTER_HOURS
from src.venues import VENUES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,wind_speed_10m,precipitation"
WEATHER_DIR = RAW_DIR / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

INDOOR = {"temp_f": 70.0, "wind_mph": 0.0, "precip_mm": 0.0}
ARCHIVE_LAG_DAYS = 6


def _get(url: str, params: dict) -> pd.DataFrame:
    for attempt in range(6):
        try:
            resp = requests.get(url, params=params, timeout=180)
        except requests.exceptions.ConnectionError:
            time.sleep(10 * (attempt + 1))  # flaky network: wait and retry
            continue
        if resp.status_code == 429:  # rate limited: back off and retry
            time.sleep(30 * (attempt + 1))
            continue
        resp.raise_for_status()
        h = resp.json()["hourly"]
        return pd.DataFrame({
            "time": pd.to_datetime(h["time"]),
            "temp_f": h["temperature_2m"],
            "wind_mph": h["wind_speed_10m"],
            "precip_mm": h["precipitation"],
        })
    raise RuntimeError(f"Open-Meteo request failed repeatedly: {url}")


def _params(venue: str, **extra) -> dict:
    lat, lon, _, _ = VENUES[venue]
    return {"latitude": lat, "longitude": lon, "hourly": HOURLY, "timezone": "GMT",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", **extra}


def _covered(cache: pd.DataFrame, start: date, end: date) -> bool:
    if cache is None or cache.empty:
        return False
    times = set(cache.time)
    return (pd.Timestamp(start) in times) and (pd.Timestamp(end) + pd.Timedelta(hours=23) in times)


def venue_history(venue: str, game_days: pd.Series, refresh: bool = False) -> pd.DataFrame:
    """Hourly weather at a venue covering every game day in `game_days`.

    Downloads one window per season (first to last game there), which keeps each
    request small enough for Open-Meteo's free-tier limits. The cache grows
    incrementally; the current weeks are refreshed from the forecast API."""
    path = WEATHER_DIR / f"{venue}.parquet"
    cache = pd.read_parquet(path) if path.exists() and not refresh else None
    archive_end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    days = pd.to_datetime(game_days).dt.date
    windows = [(d.min() - timedelta(days=1), d.max() + timedelta(days=1))
               for _, d in days.groupby(pd.to_datetime(game_days).dt.year - (pd.to_datetime(game_days).dt.month < 7))]
    # Games beyond the forecast range fall back to a climate average, so make sure
    # the same calendar window from the previous 3 years is available.
    for start, end in list(windows):
        if start > archive_end:
            for y in (1, 2, 3):
                windows.append((start.replace(year=start.year - y), end.replace(year=end.year - y)))
    parts = [] if cache is None else [cache]
    for start, end in windows:
        end = min(end, archive_end)
        if start > archive_end or _covered(cache, start, end):
            continue
        print(f"Fetching weather for {venue} {start} to {end}")
        parts.append(_get(ARCHIVE_URL, _params(venue, start_date=str(start), end_date=str(end))))
        time.sleep(1)  # stay well under the per-minute limit

    stale = not path.exists() or (time.time() - path.stat().st_mtime) / 3600 > STALE_AFTER_HOURS
    if days.max() > archive_end and (stale or refresh):
        parts.append(_get(FORECAST_URL, _params(venue, past_days=ARCHIVE_LAG_DAYS + 2, forecast_days=16)))

    df = pd.concat(parts, ignore_index=True).dropna(subset=["temp_f"])
    # Newer downloads (forecast) replace older values for the same hour.
    df = df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    if len(parts) > (0 if cache is None else 1):
        df.to_parquet(path, index=False)
    return df


def _climate(hist: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    """Average weather at this hour of day within +/- 10 days of this date, all years."""
    doy = hist.time.dt.dayofyear
    near = (np.abs(doy - kickoff.dayofyear) <= 10) & (hist.time.dt.hour == kickoff.hour)
    return hist.loc[near, ["temp_f", "wind_mph", "precip_mm"]].mean().to_dict()


def game_weather(venues: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """venues: output of venues.game_venues. Returns game_id + weather columns."""
    rows = []
    outdoor = venues[venues.indoor == 0]
    for venue, games in outdoor.groupby("venue"):
        hist = venue_history(venue, games.kickoff_utc, refresh)
        hist = hist.set_index("time")
        last_hour = hist.index.max()
        for g in games.itertuples():
            k = pd.Timestamp(g.kickoff_utc).floor("h")
            window = hist.loc[k:k + pd.Timedelta(hours=3)]
            if k + pd.Timedelta(hours=3) <= last_hour and len(window) and window.temp_f.notna().all():
                w = window[["temp_f", "wind_mph", "precip_mm"]].mean().to_dict()
                source = "observed_or_forecast"
            else:
                w = _climate(hist.reset_index(), k)
                source = "climate"
            rows.append({"game_id": g.game_id, **w, "weather_source": source})
    out = pd.DataFrame(rows)
    indoor = venues.loc[venues.indoor == 1, ["game_id"]].assign(**INDOOR, weather_source="indoor")
    return pd.concat([out, indoor], ignore_index=True)
