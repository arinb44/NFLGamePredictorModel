"""Pregame features must not depend on anything that happened on or after kickoff.

For several cutoff dates we rebuild the features with all results from the cutoff
onward hidden, and check the features for games on the cutoff date are unchanged.
"""
import numpy as np
import pandas as pd
import pytest

import src.features as features
from src.config import PROCESSED_DIR
from src.features import FEATURE_COLUMNS, MATCHUP_FEATURES, build_game_features

CUTOFFS = ["2012-11-04", "2019-09-15", "2023-12-10", "2025-10-05"]

SCORE_COLS = ["home_score", "away_score", "result", "total", "overtime"]


@pytest.fixture(scope="module")
def team_games():
    return pd.read_parquet(PROCESSED_DIR / "team_games.parquet")


@pytest.fixture(scope="module")
def full(team_games):
    return build_game_features(team_games)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_ignore_future(team_games, full, cutoff, monkeypatch):
    cutoff = pd.Timestamp(cutoff)
    real_load = features.load_schedules

    def hidden_schedule():
        s = real_load()
        future = pd.to_datetime(s.gameday) >= cutoff
        s.loc[future, SCORE_COLS] = np.nan
        return s

    real_players = features.load_player_inputs

    def hidden_players():
        # Player stats from the cutoff on are unknown; only *who played* on the
        # cutoff day is known (inactive lists are public before kickoff).
        log, rosters, outs = real_players()
        day = log.game_id.map(team_games.drop_duplicates("game_id").set_index("game_id").gameday)
        log = log[day <= cutoff].copy()
        log.loc[day[day <= cutoff] == cutoff, "value"] = np.nan
        return log, rosters, outs

    real_matchups = features.load_matchup_inputs

    def hidden_matchups():
        d = real_matchups()
        day = d.game_id.map(team_games.drop_duplicates("game_id").set_index("game_id").gameday)
        return d[day < cutoff]

    monkeypatch.setattr(features, "load_schedules", hidden_schedule)
    monkeypatch.setattr(features, "load_player_inputs", hidden_players)
    monkeypatch.setattr(features, "load_matchup_inputs", hidden_matchups)
    truncated = build_game_features(team_games[team_games.gameday < cutoff])

    cols = FEATURE_COLUMNS + MATCHUP_FEATURES
    day = full[full.gameday == cutoff].set_index("game_id")[cols]
    assert len(day) > 0, "pick a cutoff that is a game day"
    again = truncated.set_index("game_id").loc[day.index, cols]
    pd.testing.assert_frame_equal(day, again, check_exact=False, rtol=1e-9)
