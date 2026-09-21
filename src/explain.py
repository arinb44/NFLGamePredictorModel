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
therefore summed into factor groups, which are stable and readable. Each
group's effect in percentage points is how much the probability would move if
that group were neutral.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

GROUPS = {
    "Quarterback": ["diff_qb_epa", "diff_qb_cpoe", "diff_qb_experience", "diff_qb_changed"],
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
    "Rest": ["diff_rest"],
    "Fatigue": ["diff_prev_snaps", "diff_prev_def_snaps", "diff_snap_load", "diff_prev_ot",
                "diff_road_streak"],
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
        "Home field": f"{h} at home",
        "Baseline": "neutral site: small edge the model gives the listed home team",
        "Divisional game": "division rivals" if row.div_game else "",
    }
    fat = []
    if row.home_prev_ot or row.away_prev_ot:
        fat.append("OT last game: " + ", ".join(t for t, ot in [(h, row.home_prev_ot), (a, row.away_prev_ot)] if ot))
    fat.append(f"defensive snaps last game {h} {row.home_prev_def_snaps:.0f} vs {a} {row.away_prev_def_snaps:.0f}")
    d["Fatigue"] = "; ".join(fat)
    if missing is not None:
        m = missing[missing.game_id == row.game_id].sort_values("missing_value", ascending=False)
        parts = []
        for team in (h, a):
            names = [f"{r['name']} ({r['position']})" for _, r in m[m.team == team].head(3).iterrows()]
            share = m[m.team == team].missing_value.sum()
            if names:
                parts.append(f"{team} without {', '.join(names)} ({share:.0%} of touches)")
        d["Missing skill players"] = "; ".join(parts) if parts else "no notable absences"
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
        pp = _sigmoid(logit[idx]) - _sigmoid(logit[idx] - g)  # effect of each group, in probability
        order = pp.abs().sort_values(ascending=False).index
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
