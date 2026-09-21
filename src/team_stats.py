"""Build a team-game table: one row per team per game with offensive and
defensive efficiency stats computed from play-by-play.

These are *in-game* stats (what happened in that game). They are turned into
pregame features in features.py by only looking at earlier games.

Usage:
    python -m src.team_stats      # writes data/processed/team_games.parquet
"""
import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON, PROCESSED_DIR
from src.data_loader import load_pbp, load_schedules

# Schedules use historical abbreviations; pbp uses the current franchise codes.
FRANCHISE_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

PBP_COLUMNS = [
    "game_id", "posteam", "defteam", "play_type", "epa", "wp", "success",
    "pass", "rush", "qb_dropback", "sack", "interception", "fumble_lost",
    "yards_gained", "qb_kneel", "qb_spike", "yardline_100", "fixed_drive",
    "fixed_drive_result", "special_teams_play", "two_point_attempt", "qtr",
]

ST_PLAY_TYPES = {"kickoff", "punt", "field_goal", "extra_point"}


def normalize_teams(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].replace(FRANCHISE_MAP)
    return df


def _offense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Offensive stats per (game_id, posteam) from scrimmage plays."""
    plays = pbp[
        pbp.play_type.isin(["pass", "run"])
        & (pbp.two_point_attempt == 0)
        & pbp.epa.notna()
        & pbp.posteam.notna()
    ].copy()

    plays["dropback"] = plays.qb_dropback == 1
    plays["designed_run"] = (plays.rush == 1) & ~plays.dropback
    # Neutral situations: drop garbage time, where EPA says little about team quality.
    plays["neutral"] = plays.wp.between(0.10, 0.90)
    plays["turnover"] = plays.interception.fillna(0) + plays.fumble_lost.fillna(0)
    plays["explosive"] = (
        (plays.dropback & (plays.yards_gained >= 20))
        | (plays.designed_run & (plays.yards_gained >= 10))
    )

    def _masked_mean(values, mask):
        return values.where(mask).groupby([plays.game_id, plays.posteam]).mean()

    g = plays.groupby(["game_id", "posteam"])
    out = pd.DataFrame({
        "plays": g.size(),
        "ot_plays": (plays.qtr == 5).groupby([plays.game_id, plays.posteam]).sum(),
        "epa_per_play": g.epa.mean(),
        "success_rate": g.success.mean(),
        "epa_neutral": _masked_mean(plays.epa, plays.neutral),
        "pass_epa": _masked_mean(plays.epa, plays.dropback),
        "rush_epa": _masked_mean(plays.epa, plays.designed_run),
        "pass_rate_neutral": _masked_mean(plays.dropback.astype(float), plays.neutral),
        "sack_rate": _masked_mean(plays.sack.astype(float), plays.dropback),
        "explosive_rate": g.explosive.mean(),
        "turnovers": g.turnover.sum(),
    })
    return out


def _red_zone_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Share of drives reaching the opponent's 20 that end in a touchdown."""
    d = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna()]
    drives = d.groupby(["game_id", "posteam", "fixed_drive"]).agg(
        min_yl=("yardline_100", "min"), result=("fixed_drive_result", "first")
    )
    rz = drives[drives.min_yl <= 20]
    out = rz.groupby(["game_id", "posteam"]).agg(
        rz_trips=("result", "size"),
        rz_td_rate=("result", lambda r: (r == "Touchdown").mean()),
    )
    return out


def _special_teams_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Net special-teams EPA per team per game (EPA is from posteam's view)."""
    st = pbp[pbp.play_type.isin(ST_PLAY_TYPES) & pbp.epa.notna() & pbp.posteam.notna()]
    as_pos = st.groupby(["game_id", "posteam"]).epa.sum()
    as_def = st.groupby(["game_id", "defteam"]).epa.sum()
    as_def.index = as_def.index.set_names(["game_id", "posteam"])
    return as_pos.sub(as_def, fill_value=0).rename("st_epa").to_frame()


def build_team_games(seasons=range(FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    frames = []
    for season in seasons:
        pbp = load_pbp([season], columns=PBP_COLUMNS)
        off = _offense_stats(pbp).join(_red_zone_stats(pbp)).join(_special_teams_epa(pbp))
        frames.append(off.reset_index().rename(columns={"posteam": "team"}))
    off = pd.concat(frames, ignore_index=True)
    off["rz_trips"] = off.rz_trips.fillna(0)

    # Schedule gives one row per game; make it one row per team per game.
    sched = load_schedules()
    sched = sched[sched.season.isin(list(seasons)) & sched.home_score.notna()]
    sched = normalize_teams(sched.copy(), ["home_team", "away_team"])
    base_cols = ["game_id", "season", "week", "game_type", "gameday", "location", "div_game", "roof"]
    home = sched[base_cols].assign(
        team=sched.home_team, opponent=sched.away_team, is_home=1,
        points_for=sched.home_score, points_against=sched.away_score,
        rest=sched.home_rest, qb_id=sched.home_qb_id, qb_name=sched.home_qb_name,
        coach=sched.home_coach,
    )
    away = sched[base_cols].assign(
        team=sched.away_team, opponent=sched.home_team, is_home=0,
        points_for=sched.away_score, points_against=sched.home_score,
        rest=sched.away_rest, qb_id=sched.away_qb_id, qb_name=sched.away_qb_name,
        coach=sched.away_coach,
    )
    tg = pd.concat([home, away], ignore_index=True)
    tg.loc[tg.location == "Neutral", "is_home"] = 0
    tg["win"] = np.select(
        [tg.points_for > tg.points_against, tg.points_for < tg.points_against], [1.0, 0.0], 0.5
    )
    tg["point_diff"] = tg.points_for - tg.points_against

    # Offense = this team's plays; defense = the opponent's offense in the same game.
    stat_cols = [c for c in off.columns if c not in ("game_id", "team")]
    tg = tg.merge(off.rename(columns={c: f"off_{c}" for c in stat_cols}),
                  on=["game_id", "team"], how="left")
    defense = off.rename(columns={"team": "opponent", **{c: f"def_{c}" for c in stat_cols}})
    tg = tg.merge(defense, on=["game_id", "opponent"], how="left")
    tg = tg.drop(columns=["def_st_epa"])  # net ST EPA is already symmetric

    tg["gameday"] = pd.to_datetime(tg.gameday)
    return tg.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)


if __name__ == "__main__":
    tg = build_team_games()
    path = PROCESSED_DIR / "team_games.parquet"
    tg.to_parquet(path, index=False)
    print(f"Wrote {len(tg):,} team-games to {path}")
