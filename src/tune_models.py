"""Grid-search tree-model settings on the TUNING seasons (2012-2018) only.

Usage:
    python -m src.tune_models          # writes models/model_params.json
"""
import itertools
import json
import time
import warnings

import pandas as pd

from src.config import MODELS_DIR
from src.evaluate import load_played_games, metrics, walk_forward
from src.features import ALL_FEATURES, FEATURE_COLUMNS
from src.models import gradient_boosting, random_forest

TUNE_SEASONS = range(2012, 2019)
PARAMS_PATH = MODELS_DIR / "model_params.json"
FEATURE_SETS = {"lr_features": FEATURE_COLUMNS, "all_features": ALL_FEATURES}

GBM_GRID = {
    "learning_rate": [0.01, 0.03],
    "max_iter": [150, 400],
    "max_leaf_nodes": [3, 6],
    "min_samples_leaf": [80, 200],
}
RF_GRID = {
    "min_samples_leaf": [50, 100, 200],
    "max_features": [0.2, 0.5],
}


def _score(games, make, cols):
    preds = walk_forward(games, make, cols, TUNE_SEASONS)
    y = games.set_index("game_id").loc[preds.game_id, "home_win"].to_numpy()
    return metrics(y, preds.p_home.to_numpy())["log_loss"]


def search(games, name, factory, grid):
    rows = []
    for fs_name, cols in FEATURE_SETS.items():
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            t = time.time()
            ll = _score(games, lambda: factory(**params), cols)
            rows.append({"model": name, "features": fs_name, **params, "log_loss": ll})
            print(f"{name} {fs_name} {params} -> {ll:.5f} ({time.time() - t:.0f}s)")
    return pd.DataFrame(rows).sort_values("log_loss")


def load_model_params() -> dict:
    return json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    games = load_played_games()
    results = pd.concat([search(games, "gbm", gradient_boosting, GBM_GRID),
                         search(games, "rf", random_forest, RF_GRID)])
    best = {}
    for model, r in results.groupby("model"):
        top = r.iloc[0].dropna().to_dict()
        best[model] = {"features": top.pop("features"), "log_loss": top.pop("log_loss"),
                       "params": {k: (int(v) if float(v).is_integer() and k != "learning_rate"
                                      and k != "max_features" else v)
                                  for k, v in top.items() if k != "model"}}
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print("\nBest per model (tuning seasons):")
    print(json.dumps(best, indent=2))
