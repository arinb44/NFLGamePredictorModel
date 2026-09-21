"""Tune the feature-building settings (FeatureParams) by coordinate descent.

Each candidate is scored by the walk-forward log loss of a logistic regression
on the TUNING seasons only (2012-2018). The holdout seasons (2019-2025) are
never looked at here, so they stay an honest test of the final choice.

Usage:
    python -m src.tune_features
"""
import json
import warnings
from dataclasses import asdict, replace

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.evaluate import TRAIN_START, metrics, walk_forward
from src.features import FEATURE_COLUMNS, FeatureParams, build_game_features
from src.models import logistic

TUNE_SEASONS = range(2012, 2019)

GRID = {
    "decay": [0.8, 0.85, 0.9, 0.95, 0.98],
    "prior_games": [2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    "carryover": [0.3, 0.45, 0.6, 0.75, 0.9],
    "form_decay": [0.3, 0.5, 0.7],
    "qb_decay": [0.95, 0.97, 0.99, 0.995, 1.0],
    "qb_season_decay": [0.5, 0.7, 0.8, 0.9, 1.0],
    "qb_prior_plays": [50.0, 100.0, 150.0, 300.0, 500.0],
    "qb_prior_epa": [-0.15, -0.1, -0.05, 0.0],
}

PARAMS_PATH = MODELS_DIR / "feature_params.json"


def score(team_games: pd.DataFrame, params: FeatureParams) -> float:
    g = build_game_features(team_games, params, include_future=False)
    g = g[g.home_win.notna() & (g.season >= TRAIN_START)]
    preds = walk_forward(g, lambda: logistic(0.01), FEATURE_COLUMNS, TUNE_SEASONS)
    y = g.set_index("game_id").loc[preds.game_id, "home_win"].to_numpy()
    return metrics(y, preds.p_home.to_numpy())["log_loss"]


def tune(team_games: pd.DataFrame, passes: int = 2) -> FeatureParams:
    best = FeatureParams()
    best_score = score(team_games, best)
    print(f"start: {best_score:.5f}")
    for n in range(passes):
        for name, values in GRID.items():
            for v in values:
                if v == getattr(best, name):
                    continue
                cand = replace(best, **{name: v})
                s = score(team_games, cand)
                if s < best_score - 1e-5:
                    best, best_score = cand, s
            print(f"pass {n + 1} {name:16s} -> {getattr(best, name):<6} log loss {best_score:.5f}")
    return best


def load_params() -> FeatureParams:
    if PARAMS_PATH.exists():
        return FeatureParams(**json.loads(PARAMS_PATH.read_text()))
    return FeatureParams()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    best = tune(tg)
    PARAMS_PATH.write_text(json.dumps(asdict(best), indent=2))
    print("best:", asdict(best))
    print(f"saved to {PARAMS_PATH}")
