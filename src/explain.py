"""Explain logistic-regression predictions as a sum of factor contributions.

The model is  log-odds = intercept + sum_i w_i * (x_i - mean_i) / std_i.
Rewriting it relative to x_i = 0 ("the teams are equal on this factor"):

    log-odds = base + sum_i c_i,   c_i = w_i * x_i / std_i
    base     = intercept - sum_i w_i * mean_i / std_i

The final model has no intercept (see models.logistic), so `base` is 0 and
explanations start at 50%. For a model with an intercept, `base` is folded into
"Home field" ("Baseline" at neutral sites). The decomposition is exact.

Related features (offensive EPA, success rate, points scored...) overlap, so
the model's split of credit between them is arbitrary. Contributions are
therefore summed into factor groups, which are stable and readable.

Each group's effect in percentage points is its step in a waterfall: start at
50%, add groups one at a time (largest first), and record how far each moves the
probability. The steps add up exactly to the final probability.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

GROUPS = {
    "Quarterback": ["diff_qb_epa", "diff_qb_cpoe", "diff_qb_experience", "diff_qb_changed"],
    "QB in bad weather": ["diff_qb_weather_adj"],
    "Pass rush matchup": ["diff_exp_pressure"],
    "Blitz matchup": ["diff_blitz_matchup"],
    "Offense": ["diff_off_epa_per_play", "diff_off_success_rate", "diff_off_epa_neutral",
                "diff_off_pass_epa", "diff_off_rush_epa", "diff_off_sack_rate",
                "diff_off_explosive_rate", "diff_off_rz_td_rate", "diff_off_pass_rate_neutral"],
    "Defense": ["diff_def_epa_per_play", "diff_def_success_rate", "diff_def_epa_neutral",
                "diff_def_pass_epa", "diff_def_rush_epa", "diff_def_sack_rate",
                "diff_def_explosive_rate", "diff_def_rz_td_rate"],
    "Turnovers": ["diff_off_turnovers", "diff_def_turnovers"],
    "Special teams": ["diff_off_st_epa"],
    "Scoring & record": ["diff_points_for", "diff_points_against", "diff_point_diff", "diff_win"],
    "Recent form": ["diff_form_point_diff", "diff_form_off_epa_per_play", "diff_form_def_epa_per_play"],
    "Elo rating": ["diff_elo"],
    "Missing skill players": ["diff_skill_missing", "diff_skill_missing_top"],
    "Missing defenders": ["diff_def_missing", "diff_def_missing_top"],
    "Rest": ["diff_rest"],
    "Fatigue": ["diff_prev_def_snaps", "diff_def_snap_load", "diff_prev_def_ot_snaps", "diff_prev_ot",
                "diff_road_streak", "diff_prev_snaps", "diff_snap_load"],
    "Home field": ["home_field"],
    "Divisional game": ["div_game"],
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def decompose(bundle: dict, games: pd.DataFrame):
    """Returns (base log-odds per game, contributions frame [games x features])."""
    model, feats = bundle["model"], bundle["features"]
    scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
    w = clf.coef_[0] / scaler.scale_
    base = float(np.atleast_1d(clf.intercept_)[0])
    if scaler.with_mean:
        base -= np.sum(w * scaler.mean_)
    contrib = pd.DataFrame(games[feats].to_numpy() * w, columns=feats, index=games.index)
    return base, contrib


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    out = {}
    used = set()
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in contrib.columns]
        used.update(cols)
        if cols:
            out[g] = contrib[cols].sum(axis=1)
    other = [c for c in contrib.columns if c not in used]
    if other:
        out["Other"] = contrib[other].sum(axis=1)
    return pd.DataFrame(out)


def _details(row: pd.Series, missing: pd.DataFrame) -> Dict[str, str]:
    h, a = row.home_team, row.away_team
    d = {
        "Quarterback": f"{row.home_qb_name} {row.home_qb_epa:+.2f} vs {row.away_qb_name} "
                       f"{row.away_qb_epa:+.2f} EPA/play",
        "Offense": f"{h} {row.home_off_epa_per_play:+.3f} vs {a} {row.away_off_epa_per_play:+.3f} EPA/play",
        "Defense": f"{h} {row.home_def_epa_per_play:+.3f} vs {a} {row.away_def_epa_per_play:+.3f} "
                   f"EPA/play allowed (lower is better)",
        "Turnovers": f"{h} {row.home_off_turnovers:.1f} vs {a} {row.away_off_turnovers:.1f} giveaways/game",
        "Special teams": f"{h} {row.home_off_st_epa:+.1f} vs {a} {row.away_off_st_epa:+.1f} EPA/game",
        "Scoring & record": f"{h} {row.home_point_diff:+.1f} vs {a} {row.away_point_diff:+.1f} point diff/game",
        "Recent form": f"{h} {row.home_form_point_diff:+.1f} vs {a} {row.away_form_point_diff:+.1f} "
                       f"recent point diff",
        "Elo rating": f"{h} {row.home_elo_pre:.0f} vs {a} {row.away_elo_pre:.0f}",
        "Rest": f"{h} {row.home_rest:.0f} days vs {a} {row.away_rest:.0f} days",
        "QB in bad weather": (f"bad-weather game ({row.wind_mph:.0f} mph wind, {row.temp_f:.0f}F): "
                              f"{row.home_qb_name} {row.home_qb_weather_sens:+.3f} vs {row.away_qb_name} "
                              f"{row.away_qb_weather_sens:+.3f} EPA/dropback vs typical QB"),
        "Home field": f"{h} at home",
        "Baseline": "neutral site: small edge the model gives the listed home team",
        "Divisional game": "division rivals" if row.div_game else "",
    }
    fat = []
    for t, ot, ot_snaps in [(h, row.home_prev_ot, row.home_prev_def_ot_snaps),
                            (a, row.away_prev_ot, row.away_prev_def_ot_snaps)]:
        if ot:
            fat.append(f"{t} played OT last game ({ot_snaps:.0f} defensive OT snaps)")
    fat.append(f"defensive snaps/game, last 3: {h} {row.home_def_snap_load:.0f} vs {a} {row.away_def_snap_load:.0f}")
    d["Fatigue"] = "; ".join(fat)
    if missing is not None:
        m_all = missing[missing.game_id == row.game_id].sort_values("missing_value", ascending=False)
        for unit, key, what in (("offense", "Missing skill players", "touches"),
                                ("defense", "Missing defenders", "defensive playmaking")):
            m = m_all[m_all.unit == unit] if "unit" in m_all else (m_all if unit == "offense" else m_all.iloc[0:0])
            parts = []
            for team in (h, a):
                names = [f"{r['name']} ({r['position']})" for _, r in m[m.team == team].head(3).iterrows()]
                share = m[m.team == team].missing_value.sum()
                if names:
                    parts.append(f"{team} without {', '.join(names)} ({share:.0%} of {what})")
            d[key] = "; ".join(parts) if parts else "no notable absences"
    return d


def explain(bundle: dict, games: pd.DataFrame, missing: pd.DataFrame = None, top: int = 5) -> List[dict]:
    """One explanation dict per game row."""
    base, contrib = decompose(bundle, games)
    groups = group_contributions(contrib)
    logit = base + contrib.sum(axis=1)
    groups["Home field"] = groups.get("Home field", 0) + base  # start every explanation at 50%
    p = _sigmoid(logit)
    out = []
    for idx, row in games.iterrows():
        g = groups.loc[idx]
        order = g.abs().sort_values(ascending=False).index
        # Waterfall steps: add groups largest first; each step's probability change.
        cum = np.concatenate([[0.0], np.cumsum(g[order].to_numpy())])
        pp = pd.Series(np.diff(_sigmoid(cum)), index=order)
        details = _details(row, missing)
        if row.neutral:
            details["Home field"] = details["Baseline"]
        factors = [{"factor": "Baseline" if (f == "Home field" and row.neutral) else f,
                    "logit": float(g[f]), "pct_points": float(pp[f]),
                    "favors": row.home_team if pp[f] > 0 else row.away_team, "detail": details.get(f, "")}
                   for f in order]
        out.append({
            "game_id": row.game_id, "home_team": row.home_team, "away_team": row.away_team,
            "gameday": row.gameday, "p_home": float(p[idx]), "base_p": 0.5,
            "factors": factors[:top], "all_factors": factors,
        })
    return out


def waterfall(e: dict, min_pts: float = 0.5) -> pd.DataFrame:
    """Steps from 50% to the model's home win probability, largest factor first.

    Each factor's log-odds contribution is added in turn and converted to a
    probability, so the steps add up exactly to the final probability. Factors
    smaller than `min_pts` percentage points are combined into "Other factors".
    """
    factors = sorted(e["all_factors"], key=lambda f: -abs(f["logit"]))
    big = [f for f in factors if abs(f["pct_points"]) * 100 >= min_pts]
    small = [f for f in factors if abs(f["pct_points"]) * 100 < min_pts]
    if small:
        big.append({"factor": "Other factors", "logit": sum(f["logit"] for f in small),
                    "detail": ", ".join(f["factor"] for f in small)})
    rows, z = [], 0.0
    for f in big:
        start, end = float(_sigmoid(z)), float(_sigmoid(z + f["logit"]))
        rows.append({"factor": f["factor"], "start": start, "end": end, "delta": end - start,
                     "detail": f.get("detail", "")})
        z += f["logit"]
    rows.append({"factor": "Model probability", "start": 0.5, "end": float(_sigmoid(z)),
                 "delta": float(_sigmoid(z)) - 0.5, "detail": ""})
    return pd.DataFrame(rows)


def format_explanation(e: dict, vegas_p: float = None) -> str:
    h, a = e["home_team"], e["away_team"]
    ph = e["p_home"]
    fav, pf = (h, ph) if ph >= 0.5 else (a, 1 - ph)
    lines = [f"{a} @ {h}  ({pd.Timestamp(e['gameday']).strftime('%a %b %d')})",
             f"  {h} {ph:.1%}  |  {a} {1 - ph:.1%}   ->  {fav} favored at {pf:.1%}"]
    if vegas_p is not None and not np.isnan(vegas_p):
        lines[-1] += f"   (Vegas: {h} {vegas_p:.1%})"
    lines.append("  Top factors:")
    for f in e["factors"]:
        pts = abs(f["pct_points"]) * 100
        lines.append(f"    {f['favors']:>3} +{pts:4.1f}%  {f['factor']:<22} {f['detail']}")
    return "\n".join(lines)
