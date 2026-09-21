"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons used for modeling. EPA in pbp goes back to 1999, but 2006+ keeps the
# data closer to the modern game while still giving ~5,000 games.
FIRST_SEASON = 2006
CURRENT_SEASON = 2026

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "schedules": f"{NFLVERSE_RELEASES}/schedules/games.parquet",
    "teams": f"{NFLVERSE_RELEASES}/teams/teams_colors_logos.csv",
    "pbp": f"{NFLVERSE_RELEASES}/pbp/play_by_play_{{season}}.parquet",
    "player_stats": f"{NFLVERSE_RELEASES}/stats_player/stats_player_week_{{season}}.parquet",
    "ngs": f"{NFLVERSE_RELEASES}/nextgen_stats/ngs_{{kind}}.parquet",
    "rosters": f"{NFLVERSE_RELEASES}/weekly_rosters/roster_weekly_{{season}}.parquet",
    "injuries": f"{NFLVERSE_RELEASES}/injuries/injuries_{{season}}.parquet",
    "ftn": f"{NFLVERSE_RELEASES}/ftn_charting/ftn_charting_{{season}}.parquet",
    "depth_charts": f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_{{season}}.parquet",
}
INJURIES_FIRST_SEASON = 2009
FTN_FIRST_SEASON = 2022  # FTN charting (blitzers, pass rushers per play)

# Files for an in-progress season change weekly; re-download them if older than this.
STALE_AFTER_HOURS = 12
