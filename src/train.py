"""Evaluate models walk-forward, then fit the final model on every played game
and save it to models/.

Usage:
    python -m src.train                 # evaluate + save logistic regression
"""
import argparse
import warnings

import joblib
import pandas as pd

from src.config import MODELS_DIR, ROOT
from src.evaluate import compare, load_played_games, walk_forward
from src.features import FEATURE_COLUMNS
from src.models import logistic
from src.tune_features import load_params

TUNING = range(2012, 2019)
HOLDOUT = range(2019, 2026)

MODELS = {
    "logistic": lambda: logistic(C=0.003),  # C chosen on the tuning seasons
}


def evaluate(games: pd.DataFrame, names) -> pd.DataFrame:
    tables = []
    for label, seasons in (("tuning 2012-18", TUNING), ("holdout 2019-25", HOLDOUT)):
        preds = {n: walk_forward(games, MODELS[n], FEATURE_COLUMNS, seasons) for n in names}
        t = compare(games, preds)
        t.insert(0, "period", label)
        tables.append(t)
    return pd.concat(tables)


def fit_final(games: pd.DataFrame, name: str):
    model = MODELS[name]()
    model.fit(games[FEATURE_COLUMNS], games.home_win)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump({
        "model": model,
        "features": FEATURE_COLUMNS,
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
    parser.add_argument("--final", default="logistic", help="model to fit on all data and save")
    args = parser.parse_args()

    games = load_played_games()
    results = evaluate(games, args.models)
    print(results.round(4).to_string())
    (ROOT / "reports" / "results.json").write_text(results.reset_index().to_json(orient="records", indent=2))

    model, path = fit_final(games, args.final)
    print(f"\nSaved {args.final} trained on {len(games):,} games to {path}")
    if args.final == "logistic":
        print("\nTop coefficients (log-odds per 1 std dev):")
        print(coefficients(model).head(15).round(3).to_string())
