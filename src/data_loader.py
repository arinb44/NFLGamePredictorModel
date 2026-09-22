"""Download and cache nflverse data files.

Files are stored under data/raw/. Historical seasons are downloaded once;
files for the current season (and the schedule) are refreshed when stale.

Usage:
    python -m src.data_loader                 # download everything
    python -m src.data_loader --start 2020    # only 2020+
"""
import argparse
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

from src.config import (
    CURRENT_SEASON,
    FIRST_SEASON,
    FTN_FIRST_SEASON,
    INJURIES_FIRST_SEASON,
    RAW_DIR,
    STALE_AFTER_HOURS,
    URLS,
)


def _is_stale(path: Path) -> bool:
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > STALE_AFTER_HOURS


def _download(url: str, dest: Path, refresh: bool = False) -> Path:
    """Download url to dest unless a usable cached copy exists."""
    if dest.exists() and not refresh:
        return dest
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def _season_file(kind: str, season: int, refresh: bool) -> Path:
    dest = RAW_DIR / kind / f"{kind}_{season}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    live = season >= CURRENT_SEASON and dest.exists() and _is_stale(dest)
    return _download(URLS[kind].format(season=season), dest, refresh or live)


def load_schedules(refresh: bool = False) -> pd.DataFrame:
    """All games since 1999: scores, spread/moneylines, rest, QBs, venue."""
    dest = RAW_DIR / "schedules.parquet"
    stale = dest.exists() and _is_stale(dest)
    _download(URLS["schedules"], dest, refresh or stale)
    return pd.read_parquet(dest)


def load_teams(refresh: bool = False) -> pd.DataFrame:
    """Team names, divisions, colors and logo URLs."""
    dest = RAW_DIR / "teams.csv"
    _download(URLS["teams"], dest, refresh)
    return pd.read_csv(dest)


def load_pbp(
    seasons: Iterable[int],
    columns: Optional[List[str]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Play-by-play for the given seasons. Pass `columns` to save memory
    (the full file has ~370 columns)."""
    frames = [
        pd.read_parquet(_season_file("pbp", s, refresh), columns=columns)
        for s in seasons
    ]
    return pd.concat(frames, ignore_index=True)


def load_player_stats(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly player stats (passing EPA, CPOE, etc.) for the given seasons."""
    frames = [pd.read_parquet(_season_file("player_stats", s, refresh)) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_rosters(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly rosters: every player on each team each week, with status
    (ACT active, RES reserve/IR, INA gameday inactive, CUT, TRD, ...)."""
    frames = [pd.read_parquet(_season_file("rosters", s, refresh)) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_injuries(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly injury reports (Out / Doubtful / Questionable), 2009+."""
    seasons = [s for s in seasons if s >= INJURIES_FIRST_SEASON]
    frames = [pd.read_parquet(_season_file("injuries", s, refresh)) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_depth_charts(season: int, refresh: bool = False) -> pd.DataFrame:
    """Daily depth-chart snapshots for a season (2025+ format: team, player, pos_abb, pos_rank, dt)."""
    return pd.read_parquet(_season_file("depth_charts", season, refresh))


def load_participation(season: int, refresh: bool = False) -> pd.DataFrame:
    """Per-play participation charting (2018+ has man/zone and coverage type).
    Published with a lag; the current season may not exist yet."""
    return pd.read_parquet(_season_file("participation", season, refresh))


def load_ftn(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """FTN play charting, 2022+: number of blitzers and pass rushers on each play."""
    seasons = [s for s in seasons if s >= FTN_FIRST_SEASON]
    frames = [pd.read_parquet(_season_file("ftn", s, refresh)) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_ngs(kind: str = "passing", refresh: bool = False) -> pd.DataFrame:
    """Next Gen Stats (2016+). kind is 'passing', 'rushing' or 'receiving'."""
    dest = RAW_DIR / f"ngs_{kind}.parquet"
    stale = dest.exists() and _is_stale(dest)
    _download(URLS["ngs"].format(kind=kind), dest, refresh or stale)
    return pd.read_parquet(dest)


def download_all(start: int = FIRST_SEASON, end: int = CURRENT_SEASON) -> None:
    seasons = range(start, end + 1)
    load_schedules()
    load_ngs("passing")
    for s in seasons:
        _season_file("pbp", s, refresh=False)
        _season_file("player_stats", s, refresh=False)
        _season_file("rosters", s, refresh=False)
        if s >= INJURIES_FIRST_SEASON:
            _season_file("injuries", s, refresh=False)
        if s >= FTN_FIRST_SEASON:
            _season_file("ftn", s, refresh=False)
    print(f"Done. Cached files are in {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download nflverse data")
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument("--end", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()
    download_all(args.start, args.end)
