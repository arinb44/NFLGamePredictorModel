"""Interactive dashboard for the NFL win-probability project.

Run from the project root:
    streamlit run app/dashboard.py

It reads the files produced by the pipeline (data/processed/, reports/), so run
`python -m src.features` and `python -m src.train` first.
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR  # noqa: E402

# ---------------------------------------------------------------- style
# Validated categorical palette (first three slots are safe for all pairs and
# for color-vision deficiency). Blue <-> red is the diverging pair.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY = "#e34948", "#8a8984"
TEXT_2, GRID = "#52514e", "#e8e7e3"
FEATURE_LABELS = {
    "diff_qb_epa": "QB EPA per play", "diff_qb_cpoe": "QB completion % over expected",
    "diff_qb_experience": "QB experience", "diff_qb_changed": "QB change from last game",
    "diff_elo": "Elo rating", "diff_off_success_rate": "Offense success rate",
    "diff_off_epa_per_play": "Offense EPA/play", "diff_off_epa_neutral": "Offense EPA (close games)",
    "diff_off_pass_epa": "Pass offense EPA", "diff_off_rush_epa": "Rush offense EPA",
    "diff_off_sack_rate": "Sacks taken rate", "diff_off_explosive_rate": "Explosive plays (offense)",
    "diff_off_turnovers": "Turnovers committed", "diff_off_rz_td_rate": "Red-zone TD rate",
    "diff_off_pass_rate_neutral": "Pass rate (close games)", "diff_off_st_epa": "Special teams EPA",
    "diff_def_epa_per_play": "Defense EPA allowed", "diff_def_success_rate": "Defense success rate allowed",
    "diff_def_epa_neutral": "Defense EPA allowed (close)", "diff_def_pass_epa": "Pass defense EPA allowed",
    "diff_def_rush_epa": "Rush defense EPA allowed", "diff_def_sack_rate": "Sack rate (defense)",
    "diff_def_explosive_rate": "Explosive plays allowed", "diff_def_turnovers": "Takeaways",
    "diff_def_rz_td_rate": "Red-zone TD rate allowed", "diff_points_for": "Points scored",
    "diff_points_against": "Points allowed", "diff_point_diff": "Point differential", "diff_win": "Win %",
    "diff_form_point_diff": "Recent form: point diff", "diff_form_off_epa_per_play": "Recent form: offense EPA",
    "diff_form_def_epa_per_play": "Recent form: defense EPA", "diff_skill_missing": "Skill players missing",
    "diff_skill_missing_top": "Biggest single absence", "diff_rest": "Rest days", "home_field": "Home field",
    "div_game": "Divisional game", "diff_prev_snaps": "Snaps last game", "diff_prev_def_snaps": "Defensive snaps last game",
    "diff_snap_load": "Snaps, last 3 games", "diff_def_snap_load": "Defensive snaps, last 3 games",
    "diff_prev_def_ot_snaps": "Defensive OT snaps last game", "diff_prev_ot": "Overtime last game", "diff_road_streak": "Consecutive road games",
}
MODEL_COLORS = alt.Scale(domain=["Model", "Vegas", "Elo"], range=[BLUE, ORANGE, AQUA])

st.set_page_config(page_title="NFL Win Probability", page_icon="🏈", layout="wide")


def style(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (chart.properties(height=height)
            .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelColor=TEXT_2, titleColor=TEXT_2, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=TEXT_2, titleColor=TEXT_2, orient="top")
            .configure_view(strokeWidth=0))


def line_chart(df, x, y, color=None, color_scale=None, tooltip=None, y_title=None, x_title=None,
               zero=False, height=320, detail=None, x_format=None):
    """Lines (2px) with a vertical crosshair and tooltip on the nearest x.
    `detail` breaks the line into segments (e.g. one per season)."""
    base = alt.Chart(df).encode(
        x=alt.X(x, title=x_title, axis=alt.Axis(format=x_format, tickCount="year") if x_format else alt.Axis()),
        y=alt.Y(y, title=y_title, scale=alt.Scale(zero=zero)),
    )
    enc = {}
    if color:
        enc["color"] = alt.Color(color, scale=color_scale, legend=alt.Legend(title=None))
    line_enc = dict(enc, detail=detail) if detail else enc
    lines = base.mark_line(strokeWidth=2).encode(**line_enc)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x.split(":")[0]], empty=False)
    points = base.mark_point(size=70, filled=True, opacity=0).encode(
        tooltip=tooltip or [x, y], **enc).add_params(nearest)
    rule = base.mark_rule(color=GRAY, strokeWidth=1).encode(
        opacity=alt.condition(nearest, alt.value(0.6), alt.value(0))).transform_filter(nearest)
    dots = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)), **enc)
    return style(alt.layer(lines, rule, dots, points), height)


def bar_chart(df, x, y, tooltip, color=BLUE, horizontal=False, y_title=None, x_title=None,
              height=320, color_enc=None):
    mark = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=color)
    if horizontal:
        enc = dict(y=alt.Y(y, sort="-x", title=y_title, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                   x=alt.X(x, title=x_title))
    else:
        enc = dict(x=x if isinstance(x, alt.X) else alt.X(x, title=x_title), y=alt.Y(y, title=y_title))
    if color_enc is not None:
        enc["color"] = color_enc
    return style(mark.encode(tooltip=tooltip, **enc), height)


# ---------------------------------------------------------------- data
DATA_FILES = [PROCESSED_DIR / "games_features.parquet", PROCESSED_DIR / "oos_predictions.parquet",
              PROCESSED_DIR / "missing_players.parquet", ROOT / "reports" / "results.json"]


@st.cache_data
def load(versions):
    """Cached; `versions` (file modification times) makes the cache refresh after a pipeline rerun."""
    games = pd.read_parquet(PROCESSED_DIR / "games_features.parquet")
    oos_path = PROCESSED_DIR / "oos_predictions.parquet"
    oos = pd.read_parquet(oos_path) if oos_path.exists() else None
    missing_path = PROCESSED_DIR / "missing_players.parquet"
    missing = pd.read_parquet(missing_path) if missing_path.exists() else None
    results_path = ROOT / "reports" / "results.json"
    results = pd.DataFrame(json.loads(results_path.read_text())) if results_path.exists() else None
    return games, oos, missing, results


@st.cache_data
def team_long(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with that team's pregame features."""
    feats = [c[len("home_"):] for c in games.columns
             if c.startswith("home_") and f"away_{c[len('home_'):]}" in games.columns
             and c not in ("home_team", "home_score", "home_moneyline", "home_qb_name")]
    base = ["game_id", "season", "week", "gameday", "game_type", "home_win", "neutral"]
    home = games[base + ["home_team", "away_team", "home_score", "away_score", "home_qb_name"]
                 + [f"home_{f}" for f in feats]].copy()
    home.columns = base + ["team", "opponent", "points", "opp_points", "qb"] + feats
    home["venue"] = np.where(games.neutral, "Neutral", "Home")
    away = games[base + ["away_team", "home_team", "away_score", "home_score", "away_qb_name"]
                 + [f"away_{f}" for f in feats]].copy()
    away.columns = base + ["team", "opponent", "points", "opp_points", "qb"] + feats
    away["venue"] = np.where(games.neutral, "Neutral", "Away")
    t = pd.concat([home, away], ignore_index=True).sort_values("gameday")
    t["result"] = np.select([t.points > t.opp_points, t.points < t.opp_points], ["W", "L"], "T")
    t.loc[t.points.isna(), "result"] = ""
    return t


@st.cache_resource
def load_model(version):
    import joblib
    from src.config import MODELS_DIR
    return joblib.load(MODELS_DIR / "logistic.joblib")


games, oos, missing, results = load(tuple(f.stat().st_mtime if f.exists() else 0 for f in DATA_FILES))
teams = team_long(games)
played = games[games.home_win.notna()]

if oos is not None:
    pred = games.merge(oos[["game_id", "p_home", "p_elo", "p_vegas"]], on="game_id", how="left")
else:
    pred = games.assign(p_home=np.nan, p_elo=np.nan, p_vegas=np.nan)

# ---------------------------------------------------------------- layout
st.title("NFL Win Probability Model")
st.caption("Pregame win probabilities from team efficiency, quarterbacks, player availability, "
           "travel, fatigue and weather. Data: nflverse, Open-Meteo.")

page = st.sidebar.radio("Section", ["Overview", "Teams", "Games", "Players", "Situational", "Model"])

# ================================================================ Overview
if page == "Overview":
    c = st.columns(4)
    c[0].metric("Games", f"{len(played):,}")
    c[1].metric("Seasons", f"{played.season.min()}–{str(played.season.max())[2:]}")
    if results is not None:
        hold = results[results.period.str.startswith("holdout")].set_index("index")
        c[2].metric("Model accuracy", f"{hold.loc['logistic', 'accuracy']:.1%}",
                    f"{hold.loc['logistic', 'accuracy'] - hold.loc['elo', 'accuracy']:+.1%} vs Elo",
                    help="Holdout seasons 2019–2025, never used for any modeling choice.")
        c[3].metric("Vegas accuracy", f"{hold.loc['vegas', 'accuracy']:.1%}",
                    help="Vegas closing moneyline, same holdout games.")

    st.subheader("Home-field advantage is shrinking")
    st.caption("Share of regular-season games won by the home team (neutral sites excluded).")
    reg = played[(played.game_type == "REG") & ~played.neutral]
    complete = reg.groupby("season").size()
    reg = reg[reg.season.isin(complete[complete >= 200].index)]  # skip the season in progress
    hfa = reg.groupby("season").home_win.mean().rename("home_win_rate").reset_index()
    st.altair_chart(line_chart(hfa, "season:O", "home_win_rate:Q", y_title="Home win rate",
                               x_title="Season",
                               tooltip=[alt.Tooltip("season:O"),
                                        alt.Tooltip("home_win_rate:Q", format=".1%", title="Home win rate")]),
                    use_container_width=True)

    if results is not None:
        st.subheader("Model vs. benchmarks")
        st.caption("Walk-forward: every season is predicted by a model trained only on earlier seasons. "
                   "Holdout seasons were never used to make any modeling choice.")
        r = results.rename(columns={"index": "model"})
        names = {"logistic": "Model", "vegas": "Vegas", "elo": "Elo",
                 "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
        r["model"] = r.model.map(names)
        order = ["Model", "Random forest", "Gradient boosting", "Vegas", "Elo"]
        st.caption("**Model** is the final logistic regression. Random forest and gradient boosting "
                   "(gray) were tuned the same way but did worse on the tuning seasons.")
        r["period"] = r.period.str.replace("holdout", "Holdout").str.replace("tuning", "Tuning")
        for metric, title, fmt in [("accuracy", "Accuracy (higher is better)", ".1%"),
                                   ("log_loss", "Log loss (lower is better)", ".3f")]:
            base = alt.Chart(r).encode(
                x=alt.X(f"{metric}:Q", title=title, scale=alt.Scale(zero=False, nice=True, padding=40)),
                y=alt.Y("period:N", title=None, sort="descending"),
                yOffset=alt.YOffset("model:N", sort=order),
                color=alt.Color("model:N", legend=alt.Legend(title=None, labelLimit=200),
                                scale=alt.Scale(domain=order, range=[BLUE, GRAY, GRAY, ORANGE, AQUA])),
                shape=alt.Shape("model:N", legend=alt.Legend(title=None, labelLimit=200),
                                scale=alt.Scale(domain=order, range=["circle", "square", "triangle-up",
                                                                     "circle", "circle"])),
                tooltip=["period", "model", alt.Tooltip(f"{metric}:Q", format=fmt), "games"])
            dots = base.mark_point(size=140, filled=True, opacity=1)
            labels = base.mark_text(dx=12, align="left", fontSize=11).encode(
                text=alt.Text(f"{metric}:Q", format=fmt), color=alt.value(TEXT_2))
            st.altair_chart(style(dots + labels, 300), use_container_width=True)

# ================================================================ Teams
elif page == "Teams":
    all_teams = sorted(teams.team.unique())
    f1, f2 = st.columns([1, 2])
    team = f1.selectbox("Team", all_teams, index=all_teams.index("KC"))
    lo, hi = int(teams.season.min()), int(teams.season.max())
    seasons = f2.slider("Seasons", lo, hi, (max(lo, hi - 4), hi))
    t = teams[(teams.team == team) & teams.season.between(*seasons) & (teams.result != "")]

    st.subheader(f"{team} pregame strength over time")
    st.caption("Values the model saw *before* each game.")
    elo = t[["gameday", "elo_pre", "opponent", "result", "season", "week"]].dropna(subset=["elo_pre"])
    st.altair_chart(line_chart(elo, "gameday:T", "elo_pre:Q", y_title="Elo rating", x_title=None,
                               detail="season:N", x_format="%Y",
                               tooltip=["gameday:T", "season", "week", "opponent", "result",
                                        alt.Tooltip("elo_pre:Q", format=".0f", title="Elo")]),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    epa = t.melt(id_vars=["gameday", "opponent", "season", "week"],
                 value_vars=["off_epa_per_play", "def_epa_per_play"], var_name="side", value_name="epa")
    epa["side"] = epa.side.map({"off_epa_per_play": "Offense", "def_epa_per_play": "Defense allowed"})
    epa["series"] = epa.side + " " + epa.season.astype(str)
    c1.markdown("**EPA per play**")
    c1.caption("Offense: higher is better. Defense allowed: lower is better.")
    c1.altair_chart(line_chart(
        epa, "gameday:T", "epa:Q", color="side:N", detail="series:N", x_format="%Y",
        color_scale=alt.Scale(domain=["Offense", "Defense allowed"], range=[BLUE, ORANGE]),
        tooltip=["gameday:T", "opponent", "side", alt.Tooltip("epa:Q", format=".3f")],
        y_title="EPA per play"), use_container_width=True)
    c2.markdown("**Starting QB: EPA per play**")
    c2.caption("Career history, shrunk toward a replacement-level prior. Sharp dips are backup starts.")
    c2.altair_chart(line_chart(t, "gameday:T", "qb_epa:Q", detail="season:N", x_format="%Y",
                               tooltip=["gameday:T", "qb", "opponent", alt.Tooltip("qb_epa:Q", format=".3f")],
                               y_title="QB EPA per play"), use_container_width=True)

    st.subheader("Game log")
    log = t.merge(pred[["game_id", "p_home", "p_vegas"]], on="game_id", how="left")
    home_side = log.venue != "Away"
    log["model_win_prob"] = np.where(home_side, log.p_home, 1 - log.p_home)
    log["vegas_win_prob"] = np.where(home_side, log.p_vegas, 1 - log.p_vegas)
    show = log[["season", "week", "gameday", "venue", "opponent", "qb", "result", "points", "opp_points",
                "model_win_prob", "vegas_win_prob"]].sort_values("gameday", ascending=False)
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={"gameday": st.column_config.DateColumn("date"),
                                "model_win_prob": st.column_config.ProgressColumn(
                                    "model win %", min_value=0, max_value=1, format="percent"),
                                "vegas_win_prob": st.column_config.NumberColumn("Vegas win %", format="percent")})

# ================================================================ Games
elif page == "Games":
    f1, f2 = st.columns(2)
    season_opts = sorted(pred.season.unique(), reverse=True)
    latest = int(pred.loc[pred.p_home.notna(), "season"].max()) if pred.p_home.notna().any() else season_opts[0]
    season = f1.selectbox("Season", season_opts, index=season_opts.index(latest))
    weeks = sorted(pred[pred.season == season].week.unique())
    played_weeks = sorted(pred[(pred.season == season) & pred.home_win.notna()].week.unique())
    default_week = played_weeks[-1] if played_weeks else weeks[0]
    week = f2.selectbox("Week", weeks, index=weeks.index(default_week))
    g = pred[(pred.season == season) & (pred.week == week)].copy()
    from src.config import MODELS_DIR
    from src.explain import explain
    model_path = MODELS_DIR / "logistic.joblib"
    bundle = load_model(model_path.stat().st_mtime) if model_path.exists() else None
    if bundle is not None:
        # Games without an out-of-sample prediction (e.g. upcoming) get the final model's prediction.
        live = bundle["model"].predict_proba(g[bundle["features"]])[:, 1]
        g["p_home"] = np.where(g.p_home.isna(), live, g.p_home)
        vg = g.home_moneyline.notna() & g.p_vegas.isna()
        if vg.any():
            from src.evaluate import moneyline_prob
            g.loc[vg, "p_vegas"] = moneyline_prob(g.loc[vg, "home_moneyline"], g.loc[vg, "away_moneyline"])
    g["matchup"] = g.away_team + " @ " + g.home_team
    g["winner"] = np.select([g.home_score > g.away_score, g.home_score < g.away_score],
                            [g.home_team, g.away_team], "")
    g["model_pick"] = np.where(g.p_home >= 0.5, g.home_team, g.away_team)
    g.loc[g.p_home.isna(), "model_pick"] = ""
    g["correct"] = np.where(g.winner == "", None, g.model_pick == g.winner)
    st.caption("Past games show out-of-sample probabilities (each season predicted by a model trained only on "
               "earlier seasons). Upcoming games use the final model.")
    st.dataframe(g[["gameday", "matchup", "home_qb_name", "away_qb_name", "p_home", "p_vegas", "p_elo",
                    "model_pick", "winner", "correct", "home_score", "away_score"]],
                 hide_index=True, use_container_width=True,
                 column_config={"gameday": st.column_config.DateColumn("date"),
                                "home_qb_name": "home QB", "away_qb_name": "away QB",
                                "model_pick": "model pick", "home_score": "home pts", "away_score": "away pts",
                                "p_home": st.column_config.ProgressColumn("model: home win %", min_value=0,
                                                                          max_value=1, format="percent"),
                                "p_vegas": st.column_config.NumberColumn("Vegas: home win %", format="percent"),
                                "p_elo": st.column_config.NumberColumn("Elo: home win %", format="percent")})

    if bundle is not None and len(g):
        st.subheader("Why? Factors behind a prediction")
        pick = st.selectbox("Game", list(g.matchup))
        row = g[g.matchup == pick]
        e = explain(bundle, row.reset_index(drop=True), missing, top=20)[0]
        r0 = row.iloc[0]
        ph = e["p_home"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{r0.home_team} (home)", f"{ph:.1%}")
        c2.metric(f"{r0.away_team} (away)", f"{1 - ph:.1%}")
        if r0.p_vegas == r0.p_vegas:
            c3.metric(f"Vegas: {r0.home_team}", f"{r0.p_vegas:.1%}")
        st.caption("Starting point: two evenly matched teams = 50%. Each bar is how much "
                   "the probability would move if that factor were even. Blue pushes toward the home team, red toward "
                   "the away team. Hover for details.")
        f = pd.DataFrame(e["all_factors"])
        f = f[f.pct_points.abs() >= 0.001]
        f["pts"] = f.pct_points * 100
        f["direction"] = np.where(f.pts >= 0, f"Favors {r0.home_team}", f"Favors {r0.away_team}")
        f["label"] = f.pts.map(lambda v: f"{v:+.1f}%")
        bars = alt.Chart(f).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("factor:N", sort=alt.EncodingSortField("pts", op="max", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("pts:Q", title="Percentage points",
                    scale=alt.Scale(domain=[min(f.pts.min(), 0) - 2.5, max(f.pts.max(), 0) + 2.5])),
            color=alt.Color("direction:N", legend=alt.Legend(title=None),
                            scale=alt.Scale(domain=[f"Favors {r0.home_team}", f"Favors {r0.away_team}"],
                                            range=[BLUE, RED])),
            tooltip=["factor", alt.Tooltip("pts:Q", format="+.1f", title="points"), "detail"])
        def labels(align, dx, cond):
            return alt.Chart(f).transform_filter(cond).mark_text(
                align=align, dx=dx, fontSize=11, color=TEXT_2).encode(
                y=alt.Y("factor:N", sort=alt.EncodingSortField("pts", op="max", order="descending")),
                x="pts:Q", text="label:N")
        chart = bars + labels("left", 4, "datum.pts >= 0") + labels("right", -4, "datum.pts < 0")
        st.altair_chart(style(chart, 28 * len(f) + 40), use_container_width=True)
        st.dataframe(f[["factor", "label", "direction", "detail"]].rename(columns={"label": "effect"}),
                     hide_index=True, use_container_width=True)

    st.subheader(f"{season}: model vs. Vegas")
    s = pred[(pred.season == season) & pred.p_home.notna() & pred.p_vegas.notna()].copy()
    if len(s):
        s["matchup"] = s.away_team + " @ " + s.home_team
        s["outcome"] = np.where(s.home_win == 1, "Home won", "Away won")
        pts = alt.Chart(s).mark_circle(size=70, opacity=0.75, stroke="white", strokeWidth=1).encode(
            x=alt.X("p_vegas:Q", title="Vegas home win probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("p_home:Q", title="Model home win probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["Home won", "Away won"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(title=None)),
            tooltip=["week", "matchup", alt.Tooltip("p_home:Q", format=".1%", title="model"),
                     alt.Tooltip("p_vegas:Q", format=".1%", title="Vegas"), "outcome"])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=GRAY, strokeDash=[4, 4], strokeWidth=1).encode(x="x", y="y")
        st.altair_chart(style(diag + pts, 420), use_container_width=True)
    else:
        st.info("No out-of-sample predictions for this season yet.")

# ================================================================ Players
elif page == "Players":
    if missing is None or missing.empty:
        st.info("Run `python -m src.features` to build the missing-player table.")
    else:
        m = missing.merge(games[["game_id", "season", "week", "home_team", "away_team"]], on="game_id")
        m["opponent"] = np.where(m.team == m.home_team, m.away_team, m.home_team)
        f1, f2 = st.columns(2)
        season = f1.selectbox("Season", sorted(m.season.unique(), reverse=True))
        team = f2.selectbox("Team", ["All teams"] + sorted(m.team.unique()))
        sm = m[m.season == season]

        st.caption("**Role** = share of the team's RB/WR/TE targets + carries he usually gets. "
                   "**Missing share** = role × how regularly he had been playing.")
        if team == "All teams":
            st.subheader(f"Biggest absences of {season}")
            top = sm.sort_values("missing_value", ascending=False).head(20)
            top["label"] = top.name + " (" + top.team + ", wk " + top.week.astype(str) + ")"
            st.altair_chart(bar_chart(top, "missing_value:Q", "label:N", horizontal=True,
                                      x_title="Share of team touches missing", y_title=None, height=520,
                                      tooltip=["name", "position", "team", "week", "opponent",
                                               alt.Tooltip("value:Q", format=".1%", title="role"),
                                               alt.Tooltip("participation:Q", format=".2f"),
                                               alt.Tooltip("missing_value:Q", format=".1%", title="missing share")]),
                            use_container_width=True)
        else:
            tm = sm[sm.team == team]
            weekly = tm.groupby("week").missing_value.sum().reindex(
                sorted(games[(games.season == season) & ((games.home_team == team) | (games.away_team == team))]
                       .week.unique()), fill_value=0).rename("missing").reset_index()
            st.subheader(f"{team} {season}: share of skill-player touches missing each week")
            st.altair_chart(bar_chart(weekly, "week:O", "missing:Q", x_title="Week", y_title="Missing share",
                                      tooltip=["week", alt.Tooltip("missing:Q", format=".1%")]),
                            use_container_width=True)
            st.dataframe(tm.sort_values(["week", "missing_value"], ascending=[True, False])[
                ["week", "opponent", "name", "position", "value", "participation", "missing_value"]],
                hide_index=True, use_container_width=True,
                column_config={"value": st.column_config.NumberColumn("role", format="percent"),
                               "missing_value": st.column_config.NumberColumn("missing share", format="percent")})

# ================================================================ Situational
elif page == "Situational":
    need = {"away_early_clock", "away_tz_shift", "temp_f", "wind_mph", "away_cold_shock", "diff_prev_def_snaps"}
    if not need.issubset(games.columns):
        st.info("Situational features are still being built.")
    else:
        p = played[played.season >= 2007].copy()
        p["away_win"] = 1 - p.home_win

        overall = p.away_win.mean()

        def rate_bars(df, group, title, x_title):
            col = df[group]
            order = [str(c) for c in col.cat.categories] if hasattr(col, "cat") else sorted(col.unique())
            r = df.groupby(group, observed=True).agg(rate=("away_win", "mean"), games=("away_win", "size")).reset_index()
            r = r[r.games >= 30]
            r[group] = r[group].astype(str)
            bars = alt.Chart(r).mark_bar(cornerRadiusEnd=4, color=BLUE).encode(
                x=alt.X(f"{group}:O", sort=[o for o in order if o in set(r[group])], title=x_title,
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("rate:Q", title="Away team win rate", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip(f"{group}:N", title=x_title),
                         alt.Tooltip("rate:Q", format=".1%", title="away win rate"), "games"])
            ref = alt.Chart(pd.DataFrame({"y": [overall]})).mark_rule(
                color=GRAY, strokeDash=[4, 4], strokeWidth=1).encode(y="y:Q")
            st.markdown(f"**{title}**")
            st.altair_chart(style(bars + ref, 280), use_container_width=True)

        st.caption(f"Dashed line = all road teams ({overall:.1%}). Bars with fewer than 30 games are hidden. "
                   "These are raw rates: they don't control for team strength, which the model does.")
        st.subheader("Travel and body clock")
        st.caption("A 1 PM Eastern kickoff is 10 AM on a West Coast team's body clock.")
        c1, c2 = st.columns(2)
        p["body_clock"] = pd.cut(p.away_early_clock, [-0.1, 0.01, 1.5, 2.5, 10],
                                 labels=["noon or later", "~1 hr early", "~2 hrs early", "3+ hrs early"])
        with c1:
            rate_bars(p, "body_clock", "Away team, by how early kickoff feels", "Body-clock kickoff")
        p["tz"] = p.away_tz_shift.round().clip(-3, 3).astype(int)
        with c2:
            rate_bars(p[~p.neutral], "tz", "Away team, by time zones traveled (+ = east)", "Time-zone shift")

        st.subheader("Weather")
        c1, c2 = st.columns(2)
        out = p[p.indoor == 0].copy()
        out["cold"] = pd.cut(out.away_cold_shock, [-0.1, 5, 15, 25, 100],
                             labels=["<5°F", "5–15°F", "15–25°F", ">25°F"])
        with c1:
            rate_bars(out, "cold", "Away team, by how much colder than home", "Colder than usual")
        out["total_points"] = out.home_score + out.away_score
        out["wind"] = pd.cut(out.wind_mph, [-0.1, 5, 10, 15, 20, 60], labels=["0–5", "5–10", "10–15", "15–20", "20+"])
        wp = out.groupby("wind", observed=True).agg(points=("total_points", "mean"), games=("total_points", "size")).reset_index()
        with c2:
            st.markdown("**Scoring drops in the wind (outdoor games)**")
            wp["wind"] = wp.wind.astype(str)
            st.altair_chart(bar_chart(wp, alt.X("wind:O", title="Wind (mph)", sort=["0–5", "5–10", "10–15", "15–20", "20+"],
                                                axis=alt.Axis(labelAngle=0)), "points:Q", y_title="Avg total points",
                                      height=280, tooltip=["wind", alt.Tooltip("points:Q", format=".1f"), "games"]),
                            use_container_width=True)

        st.subheader("Fatigue")
        c1, c2 = st.columns(2)
        p["def_load"] = pd.cut(p.away_prev_def_snaps, [0, 55, 62, 70, 200],
                               labels=["<55", "55–62", "62–70", "70+"])
        with c1:
            rate_bars(p, "def_load", "Away team, by defensive snaps last game", "Defensive snaps last game")
        p["ot"] = np.where(p.away_prev_ot == 1, "OT last game", "No OT")
        with c2:
            rate_bars(p, "ot", "Away team, after an overtime game", "Previous game")

# ================================================================ Model
elif page == "Model":
    if oos is None:
        st.info("Run `python -m src.train` to produce out-of-sample predictions.")
    else:
        d = played[["game_id", "season", "home_win"]].merge(
            oos.drop(columns="season"), on="game_id").dropna(subset=["p_vegas"])
        long = d.melt(id_vars=["game_id", "season", "home_win"], value_vars=["p_home", "p_vegas", "p_elo"],
                      var_name="model", value_name="p")
        long["model"] = long.model.map({"p_home": "Model", "p_vegas": "Vegas", "p_elo": "Elo"})
        period = st.radio("Seasons", ["Holdout 2019–2025", "Tuning 2012–2018", "All"], horizontal=True)
        if period.startswith("Holdout"):
            long = long[long.season >= 2019]
        elif period.startswith("Tuning"):
            long = long[long.season < 2019]

        c1, c2 = st.columns(2)
        long["bin"] = pd.cut(long.p, np.linspace(0, 1, 11))
        cal = long.groupby(["model", "bin"], observed=True).agg(
            predicted=("p", "mean"), actual=("home_win", "mean"), games=("p", "size")).reset_index()
        cal = cal[cal.games >= 20].drop(columns="bin")
        with c1:
            st.markdown("**Calibration:** when we say 70%, does it happen 70% of the time?")
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                color=GRAY, strokeDash=[4, 4], strokeWidth=1).encode(x="x", y="y")
            lines = alt.Chart(cal).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
                x=alt.X("predicted:Q", title="Predicted home win probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("actual:Q", title="Actual home win rate", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N", scale=MODEL_COLORS, legend=alt.Legend(title=None)),
                tooltip=["model", alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("actual:Q", format=".1%"), "games"])
            st.altair_chart(style(diag + lines, 380), use_container_width=True)

        def ll(g):
            p = g.p.clip(1e-6, 1 - 1e-6)
            return -np.mean(g.home_win * np.log(p) + (1 - g.home_win) * np.log(1 - p))
        by_season = long.groupby(["model", "season"]).apply(ll).rename("log_loss").reset_index()
        with c2:
            st.markdown("**Log loss by season** (lower is better)")
            st.altair_chart(line_chart(by_season, "season:O", "log_loss:Q", color="model:N",
                                       color_scale=MODEL_COLORS, y_title="Log loss", x_title="Season",
                                       tooltip=["season", "model", alt.Tooltip("log_loss:Q", format=".4f")],
                                       height=380), use_container_width=True)

        st.subheader("What the model weighs most")
        st.caption("Logistic regression weights on standardized features: change in log-odds of a home win "
                   "per one standard deviation. Blue favors the home team, red the away team.")
        try:
            import joblib
            from src.config import MODELS_DIR
            bundle = joblib.load(MODELS_DIR / "logistic.joblib")
            coef = pd.Series(bundle["model"].named_steps["clf"].coef_[0], index=bundle["features"])
            cdf = coef.rename("weight").reset_index().rename(columns={"index": "feature"})
            cdf["feature"] = cdf.feature.map(FEATURE_LABELS).fillna(cdf.feature)
            cdf["direction"] = np.where(cdf.weight >= 0, "Favors home team", "Favors away team")
            cdf = cdf.reindex(cdf.weight.abs().sort_values(ascending=False).index).head(25)
            st.altair_chart(bar_chart(
                cdf, "weight:Q", "feature:N", horizontal=True, x_title="Weight (log-odds per std dev)",
                height=25 * 24, tooltip=["feature", alt.Tooltip("weight:Q", format=".3f")],
                color_enc=alt.Color("direction:N", scale=alt.Scale(domain=["Favors home team", "Favors away team"],
                                                                   range=[BLUE, RED]),
                                    legend=alt.Legend(title=None))), use_container_width=True)
        except FileNotFoundError:
            st.info("Run `python -m src.train` to save the model.")

st.sidebar.markdown("---")
st.sidebar.caption("Every chart has a hover tooltip. Tables can be sorted and downloaded.")
