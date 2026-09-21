"""Evaluate models walk-forward, then fit the final model on every played game
and save it to models/.

Usage:
    python -m src.train                 # evaluate + save logistic regression
"""
import argparse
import warnings

import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, ROOT
from src.evaluate import baseline_predictions, compare, load_played_games, walk_forward
from src.features import ALL_FEATURES, FEATURE_COLUMNS
from src.models import gradient_boosting, logistic, random_forest
from src.tune_features import load_params
from src.tune_models import load_model_params

TUNING = range(2012, 2019)
HOLDOUT = range(2019, 2026)

_MP = load_model_params()


def _tree(name, factory):
    cfg = _MP.get(name, {"features": "all_features", "params": {}})
    cols = ALL_FEATURES if cfg["features"] == "all_features" else FEATURE_COLUMNS
    return (lambda: factory(**cfg["params"])), cols


# name -> (factory, feature list). Settings were all chosen on the tuning seasons.
MODELS = {
    "logistic": (lambda: logistic(C=0.003), FEATURE_COLUMNS),
    "random_forest": _tree("rf", random_forest),
    "gradient_boosting": _tree("gbm", gradient_boosting),
}
FINAL_MODEL = "logistic"  # best log loss on the tuning seasons

# Live 2026 test (see src/track.py): the final model plus blitz vulnerability, whose
# data starts in 2022 and so can't be judged on the tuning seasons.
TEST_MODELS = {
    "logistic_blitz": (lambda: logistic(C=0.003), FEATURE_COLUMNS + ["diff_blitz_matchup"]),
}


def evaluate(games: pd.DataFrame, names) -> pd.DataFrame:
    tables = []
    for label, seasons in (("tuning 2012-18", TUNING), ("holdout 2019-25", HOLDOUT)):
        preds = {n: walk_forward(games, MODELS[n][0], MODELS[n][1], seasons) for n in names}
        t = compare(games, preds)
        t.insert(0, "period", label)
        tables.append(t)
    return pd.concat(tables)


def fit_final(games: pd.DataFrame, name: str):
    factory, cols = {**MODELS, **TEST_MODELS}[name]
    model = factory()
    model.fit(games[cols], games.home_win)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({
        "model": model,
        "features": cols,
        "feature_params": load_params(),
        "trained_through": str(games.gameday.max().date()),
    }, path)
    return model, path


def coefficients(model) -> pd.Series:
    """Logistic regression weights on standardized features: the change in
    log-odds of a home win for a one-standard-deviation change in the feature."""
    return pd.Series(model.named_steps["clf"].coef_[0], index=FEATURE_COLUMNS).sort_values(
        key=abs, ascending=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--final", default=FINAL_MODEL, help="model to fit on all data and save")
    args = parser.parse_args()

    games = load_played_games()
    results = evaluate(games, args.models)
    print(results.round(4).to_string())
    (ROOT / "reports" / "results.json").write_text(results.reset_index().to_json(orient="records", indent=2))

    # Out-of-sample predictions (each season predicted by a model trained only on
    # earlier seasons) - used by the dashboard.
    oos = walk_forward(games, *MODELS[args.final], range(TUNING.start, HOLDOUT.stop))
    oos = oos.merge(baseline_predictions(games), on="game_id")
    oos.to_parquet(PROCESSED_DIR / "oos_predictions.parquet", index=False)

    model, path = fit_final(games, args.final)
    print(f"\nSaved {args.final} trained on {len(games):,} games to {path}")
    for name in TEST_MODELS:
        print(f"Saved test model {name} to {fit_final(games, name)[1]}")
    if args.final == "logistic":
        print("\nTop coefficients (log-odds per 1 std dev):")
        print(coefficients(model).head(15).round(3).to_string())
