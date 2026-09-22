"""Coverage analytics (informational, not model features): how often each defense
plays man vs. zone and which coverages, and how each offense does against man and
zone. Source: nflverse pbp participation data, 2018+ (published after the season).

Usage:
    python -m src.coverage      # writes data/processed/coverage_*.parquet
"""
import pandas as pd

from src.config import COVERAGE_FIRST_SEASON, CURRENT_SEASON, PROCESSED_DIR
from src.data_loader import load_participation, load_pbp
from src.team_stats import FRANCHISE_MAP

COVERAGE_NAMES = {"COVER_0": "Cover 0", "COVER_1": "Cover 1", "COVER_2": "Cover 2", "2_MAN": "2-Man",
                  "COVER_3": "Cover 3", "COVER_4": "Cover 4", "COVER_6": "Cover 6", "COVER_9": "Cover 9",
                  "PREVENT": "Prevent", "BLOWN": "Blown"}


def charted_dropbacks(seasons=range(COVERAGE_FIRST_SEASON, CURRENT_SEASON + 1)) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            part = load_participation(s)
        except Exception:
            continue  # not published yet
        part = part.rename(columns={"nflverse_game_id": "game_id"})[
            ["game_id", "play_id", "defense_man_zone_type", "defense_coverage_type"]]
        p = load_pbp([s], columns=["game_id", "play_id", "posteam", "defteam", "qb_dropback", "epa", "season_type"])
        p = p[(p.qb_dropback == 1) & p.epa.notna() & p.posteam.notna()]
        d = p.merge(part, on=["game_id", "play_id"])
        d = d[d.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
        d["season"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam"):
        d[c] = d[c].replace(FRANCHISE_MAP)
    d["man"] = d.defense_man_zone_type == "MAN_COVERAGE"
    d["coverage"] = d.defense_coverage_type.map(COVERAGE_NAMES)
    return d


def build():
    d = charted_dropbacks()
    defense = (d.groupby(["season", "defteam"])
               .agg(dropbacks=("man", "size"), man_rate=("man", "mean"), epa_allowed=("epa", "mean"))
               .reset_index().rename(columns={"defteam": "team"}))
    types = (d.dropna(subset=["coverage"]).groupby(["season", "defteam", "coverage"]).size()
             .rename("plays").reset_index().rename(columns={"defteam": "team"}))
    types["share"] = types.plays / types.groupby(["season", "team"]).plays.transform("sum")
    offense = (d.groupby(["season", "posteam", "man"]).epa.agg(["mean", "size"]).unstack("man"))
    offense.columns = ["epa_vs_zone", "epa_vs_man", "n_vs_zone", "n_vs_man"]
    offense = offense.reset_index().rename(columns={"posteam": "team"})
    defense.to_parquet(PROCESSED_DIR / "coverage_defense.parquet", index=False)
    types.to_parquet(PROCESSED_DIR / "coverage_types.parquet", index=False)
    offense.to_parquet(PROCESSED_DIR / "coverage_offense.parquet", index=False)
    return defense, types, offense


if __name__ == "__main__":
    defense, types, offense = build()
    print(f"Coverage data for seasons {sorted(defense.season.unique())}")
    last = defense[defense.season == defense.season.max()].sort_values("man_rate")
    print(f"{defense.season.max()} most zone-heavy:", last.head(3)[["team", "man_rate"]].round(3).values.tolist())
    print(f"{defense.season.max()} most man-heavy:", last.tail(3)[["team", "man_rate"]].round(3).values.tolist())
