"""Walk-forward evaluation shared by every model.

For each test season T: fit on all seasons before T, predict T. Nothing from
season T (or later) is ever seen during fitting.
"""
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.elo import HOME_ADV, elo_prob

TRAIN_START = 2007          # 2006 only warms up the rolling features
TEST_SEASONS = range(2012, 2026)


def load_played_games() -> pd.DataFrame:
    g = pd.read_parquet(PROCESSED_DIR / "games_features.parquet")
    return g[g.home_win.notna() & (g.season >= TRAIN_START)].reset_index(drop=True)


def moneyline_prob(home_ml: pd.Series, away_ml: pd.Series) -> np.ndarray:
    """Vegas home win probability from moneylines, with the bookmaker's cut removed."""
    def implied(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    ph, pa = implied(home_ml), implied(away_ml)
    return ph / (ph + pa)


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def walk_forward(
    games: pd.DataFrame,
    make_model: Callable,
    features: List[str],
    test_seasons: Iterable[int] = TEST_SEASONS,
) -> pd.DataFrame:
    """Out-of-sample predictions for every game in test_seasons.
    make_model() must return a fresh, unfitted scikit-learn estimator."""
    preds = []
    for season in test_seasons:
        train = games[games.season < season]
        test = games[games.season == season]
        model = make_model()
        model.fit(train[features], train.home_win)
        preds.append(pd.DataFrame({
            "game_id": test.game_id,
            "season": season,
            "p_home": model.predict_proba(test[features])[:, 1],
        }))
    return pd.concat(preds, ignore_index=True)


def baseline_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Elo and Vegas probabilities for every game (for side-by-side comparison)."""
    hfa = np.where(games.neutral, 0.0, HOME_ADV)
    return pd.DataFrame({
        "game_id": games.game_id,
        "p_elo": elo_prob(games.diff_elo + hfa),
        "p_vegas": moneyline_prob(games.home_moneyline, games.away_moneyline),
    })


def compare(games: pd.DataFrame, model_preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Metrics table for each model plus Elo/Vegas, all on the same games
    (test seasons only, games with a Vegas line)."""
    df = games[["game_id", "season", "home_win"]].merge(baseline_predictions(games), on="game_id")
    for name, p in model_preds.items():
        df = df.merge(p[["game_id", "p_home"]].rename(columns={"p_home": f"p_{name}"}), on="game_id")
    df = df.dropna(subset=["p_vegas"])

    rows = {}
    cols = [c for c in df.columns if c.startswith("p_")]
    for c in cols:
        rows[c[2:]] = metrics(df.home_win.to_numpy(), df[c].to_numpy())
    table = pd.DataFrame(rows).T
    table["games"] = len(df)
    return table.sort_values("log_loss")
