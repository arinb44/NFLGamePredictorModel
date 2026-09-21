"""Simple NFL Elo ratings (FiveThirtyEight-style).

Each team's rating is recorded *before* each game, so it can be used as a
pregame feature and as a baseline model.
"""
import numpy as np
import pandas as pd

K = 20.0
HOME_ADV = 48.0          # Elo points for home field in the baseline probability
MEAN_ELO = 1505.0
SEASON_REVERT = 1 / 3    # pull ratings toward the mean between seasons


def elo_prob(elo_diff: np.ndarray) -> np.ndarray:
    """P(team A wins) given A's Elo minus B's Elo (home field already included)."""
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def _mov_multiplier(point_diff: float, winner_elo_diff: float) -> float:
    return np.log(abs(point_diff) + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """games: one row per game with game_id, season, gameday, home_team, away_team,
    home_score, away_score, neutral. Returns game_id, home_elo_pre, away_elo_pre."""
    games = games.sort_values(["gameday", "game_id"])
    ratings = {}
    last_season = {}
    rows = []
    for g in games.itertuples(index=False):
        for team in (g.home_team, g.away_team):
            if team not in ratings:
                ratings[team] = MEAN_ELO
            elif last_season[team] != g.season:
                ratings[team] = ratings[team] + SEASON_REVERT * (MEAN_ELO - ratings[team])
            last_season[team] = g.season
        h, a = ratings[g.home_team], ratings[g.away_team]
        rows.append((g.game_id, h, a))

        if pd.isna(g.home_score):  # future game: no update
            continue
        hfa = 0.0 if g.neutral else HOME_ADV
        p_home = elo_prob(h - a + hfa)
        pd_ = g.home_score - g.away_score
        result = 1.0 if pd_ > 0 else 0.0 if pd_ < 0 else 0.5
        winner_diff = (h + hfa - a) if pd_ > 0 else (a - h - hfa)
        mult = _mov_multiplier(pd_, winner_diff) if pd_ != 0 else 1.0
        shift = K * mult * (result - p_home)
        ratings[g.home_team] = h + shift
        ratings[g.away_team] = a - shift
    return pd.DataFrame(rows, columns=["game_id", "home_elo_pre", "away_elo_pre"])
