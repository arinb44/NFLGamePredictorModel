"""Point-spread predictions: the model's expected home margin, compared with the
Vegas spread (which stays a benchmark only, never an input).

Two approaches, compared on the tuning seasons:
- "from_prob": convert the win probability to a margin, assuming the final margin
  is normal around the expected margin with standard deviation SIGMA:
  margin = SIGMA * inverse_normal(p_home).
- "ridge": a separate symmetric linear regression on the same features, trained
  to predict the final margin directly.

nflverse convention: spread_line > 0 means the home team is favored by that many
points; result = home score - away score; the home team covers if result > spread_line.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SIGMA = 13.5  # typical standard deviation of NFL margins around the expected margin


def margin_model(alpha: float = 300.0) -> Pipeline:
    """Symmetric (no intercept) ridge regression on home-minus-away features."""
    return Pipeline([("scale", StandardScaler(with_mean=False)),
                     ("reg", Ridge(alpha=alpha, fit_intercept=False))])


def margin_from_prob(p_home, sigma: float = SIGMA):
    return sigma * norm.ppf(np.clip(p_home, 1e-6, 1 - 1e-6))


def cover_prob(pred_margin, spread_line, sigma: float = SIGMA):
    """P(home team covers) = P(margin > spread_line) given the predicted margin."""
    return 1 - norm.cdf((spread_line - pred_margin) / sigma)


def walk_forward_margin(games: pd.DataFrame, make_model, features, seasons) -> pd.DataFrame:
    out = []
    for season in seasons:
        train = games[(games.season < season) & games.result.notna()]
        test = games[games.season == season]
        m = make_model().fit(train[features], train.result)
        resid_sd = float(np.std(train.result - m.predict(train[features])))
        out.append(pd.DataFrame({"game_id": test.game_id, "season": season,
                                 "pred_margin": m.predict(test[features]), "sigma": resid_sd}))
    return pd.concat(out, ignore_index=True)


SPREAD_ALPHA = 1000.0  # ridge strength, chosen on the tuning seasons by margin RMSE


def fmt_spread(home: str, away: str, margin: float) -> str:
    """Home margin (+ = home favored) -> 'KC -6.5' style, rounded to the half point."""
    if margin != margin:
        return "–"
    m = round(margin * 2) / 2
    if abs(m) < 0.5:
        return "Pick'em"
    return f"{home} -{abs(m):g}" if m > 0 else f"{away} -{abs(m):g}"


def cover_prob_calibrated(pred_margin, spread_line, k: float):
    """P(home covers) from the model's disagreement with the line, calibrated on the
    tuning seasons (k = log-odds of covering per point of disagreement)."""
    return 1 / (1 + np.exp(-k * (np.asarray(pred_margin) - np.asarray(spread_line))))


def fit_spread_model(games: pd.DataFrame, features, tune_seasons=range(2012, 2019)) -> dict:
    """Fit the final margin model on every played game, plus the cover-probability
    calibration learned from walk-forward predictions on the tuning seasons."""
    from sklearn.linear_model import LogisticRegression
    played = games[games.result.notna()]
    wf = walk_forward_margin(played, lambda: margin_model(SPREAD_ALPHA), features, tune_seasons)
    d = played[["game_id", "result", "spread_line"]].merge(wf, on="game_id").dropna(subset=["spread_line"])
    d = d[d.result != d.spread_line]
    cal = LogisticRegression(fit_intercept=False).fit((d.pred_margin - d.spread_line).to_frame("edge"),
                                                     (d.result > d.spread_line).astype(int))
    model = margin_model(SPREAD_ALPHA).fit(played[features], played.result)
    sigma = float(np.std(played.result - model.predict(played[features])))
    return {"model": model, "features": list(features), "sigma": sigma, "cover_k": float(cal.coef_[0][0])}


def spread_report(df: pd.DataFrame, pred_col: str, thresholds=(0, 1.5, 3.0)) -> dict:
    """Error vs Vegas, and record against the spread when our number differs from Vegas.
    df needs result, spread_line and pred_col."""
    d = df.dropna(subset=["result", "spread_line", pred_col])
    rep = {
        "games": len(d),
        "rmse_model": float(np.sqrt(np.mean((d.result - d[pred_col]) ** 2))),
        "rmse_vegas": float(np.sqrt(np.mean((d.result - d.spread_line) ** 2))),
        "mae_model": float(np.mean(np.abs(d.result - d[pred_col]))),
        "mae_vegas": float(np.mean(np.abs(d.result - d.spread_line))),
    }
    for t in thresholds:
        edge = d[pred_col] - d.spread_line
        pick = d[edge.abs() > t]
        home_pick = (pick[pred_col] > pick.spread_line)
        ats = pick.result - pick.spread_line
        push = ats == 0
        won = np.where(home_pick, ats > 0, ats < 0)[~push.to_numpy()]
        rep[f"ats_edge>{t}"] = (int(won.sum()), int(len(won) - won.sum()), float(won.mean()) if len(won) else np.nan)
    return rep
