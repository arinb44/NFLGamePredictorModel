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
    "diff_skill_missing_top": "Biggest single absence", "diff_def_missing": "Defensive playmakers missing", "diff_rest": "Rest days", "home_field": "Home field",
    "div_game": "Divisional game", "diff_prev_snaps": "Snaps last game", "diff_prev_def_snaps": "Defensive snaps last game",
    "diff_qb_weather_adj": "QB in bad weather", "diff_exp_pressure": "Pass rush matchup",
    "diff_blitz_matchup": "Blitz matchup",
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
        x=alt.X(x, title=x_title, axis=alt.Axis(format=x_format, tickCount="year") if x_format else alt.Axis(labelAngle=0)),
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
              height=320, color_enc=None, x_format=None):
    mark = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=color)
    if horizontal:
        enc = dict(y=alt.Y(y, sort="-x", title=y_title, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                   x=alt.X(x, title=x_title, axis=alt.Axis(format=x_format) if x_format else alt.Axis()))
    else:
        enc = dict(x=x if isinstance(x, alt.X) else alt.X(x, title=x_title), y=alt.Y(y, title=y_title))
    if color_enc is not None:
        enc["color"] = color_enc
    return style(mark.encode(tooltip=tooltip, **enc), height)


# ---------------------------------------------------------------- data
DATA_FILES = [PROCESSED_DIR / "games_features.parquet", PROCESSED_DIR / "oos_predictions.parquet",
              PROCESSED_DIR / "missing_players.parquet", ROOT / "reports" / "results.json",
              PROCESSED_DIR / "team_games.parquet"]


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
    team_games = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    return games, oos, missing, results, team_games


@st.cache_data
def load_coverage(version):
    paths = {k: PROCESSED_DIR / f"coverage_{k}.parquet" for k in ("defense", "types", "offense")}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_parquet(p) for k, p in paths.items()}


def team_scatter(df, x, y, x_title, y_title, focus, x_fmt="%", y_fmt="%", reverse_y=False, diagonal=False,
                 tooltip=None, height=440):
    """32-team scatter with league-average reference lines, the focus team highlighted and labeled."""
    df = df.copy()
    df["group"] = np.where(df.team == focus, "Selected team", "Other teams")
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=x_fmt), scale=alt.Scale(zero=False, padding=24)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_fmt),
                scale=alt.Scale(zero=False, padding=24, reverse=reverse_y)),
        tooltip=tooltip or ["team"])
    dots = base.mark_circle(size=90, stroke="white", strokeWidth=1).encode(
        color=alt.Color("group:N", legend=None, scale=alt.Scale(domain=["Selected team", "Other teams"],
                                                                range=[ORANGE, BLUE])),
        size=alt.condition(alt.datum.group == "Selected team", alt.value(220), alt.value(90)))
    # Label the selected team plus the 3 most extreme teams on each axis, to keep labels readable.
    ext = set(df.nlargest(3, x).team) | set(df.nsmallest(3, x).team) | set(df.nlargest(3, y).team) | \
        set(df.nsmallest(3, y).team) | {focus}
    names = base.transform_filter(alt.FieldOneOfPredicate(field="team", oneOf=sorted(ext))).mark_text(
        dy=-11, fontSize=11, color=TEXT_2).encode(text="team:N")
    layers = [alt.Chart(pd.DataFrame({"v": [float(df[x].mean())]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="v:Q"),
              alt.Chart(pd.DataFrame({"v": [float(df[y].mean())]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(y="v:Q")]
    if diagonal:
        lo, hi = float(min(df[x].min(), df[y].min())), float(max(df[x].max(), df[y].max()))
        layers.append(alt.Chart(pd.DataFrame({"a": [lo, hi], "b": [lo, hi]})).mark_line(
            color=GRAY, strokeWidth=1).encode(x="a:Q", y="b:Q"))
    return style(alt.layer(*layers, dots, names), height)


@st.cache_data
def prime_time_records():
    """Each QB's prime-time record vs the wins the model expected (out-of-sample, 2012+)."""
    from src.data_loader import load_schedules
    s = load_schedules()
    s = s[(s.season >= 2012) & s.home_score.notna() & (s.gametime.fillna("13:00") >= "19:00")]
    oos_path = PROCESSED_DIR / "oos_predictions.parquet"
    if not oos_path.exists():
        return pd.DataFrame()
    s = s.merge(pd.read_parquet(oos_path)[["game_id", "p_home"]], on="game_id")
    t = pd.concat([s.assign(qb=s.home_qb_name, win=(s.home_score > s.away_score) * 1.0, p=s.p_home),
                   s.assign(qb=s.away_qb_name, win=(s.away_score > s.home_score) * 1.0, p=1 - s.p_home)])
    r = t.groupby("qb").agg(W=("win", "sum"), G=("win", "size"), expected=("p", "sum")).reset_index()
    r = r[r.G >= 5]
    r["record"] = r.W.astype(int).astype(str) + "-" + (r.G - r.W).astype(int).astype(str)
    r["vs_expected"] = r.W - r.expected
    return r[["qb", "record", "expected", "vs_expected"]]


@st.cache_data
def load_blitz():
    """Per team-game blitzed / not-blitzed dropbacks and EPA (FTN charting, 2022+)."""
    from src.config import CURRENT_SEASON, FTN_FIRST_SEASON
    from src.matchups import load_dropbacks
    d = load_dropbacks(range(FTN_FIRST_SEASON, CURRENT_SEASON + 1)).dropna(subset=["n_blitzers"])
    blitz = d.n_blitzers > 0
    d = d.assign(season=d.game_id.str[:4].astype(int), team=d.posteam,
                 n_b=blitz.astype(int), s_b=np.where(blitz, d.epa, 0.0),
                 n_nb=(~blitz).astype(int), s_nb=np.where(blitz, 0.0, d.epa))
    return d.groupby(["season", "team", "game_id"])[["n_b", "s_b", "n_nb", "s_nb"]].sum().reset_index()


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


@st.cache_resource
def load_spread(version):
    import joblib
    from src.config import MODELS_DIR
    p = MODELS_DIR / "spread.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def team_meta():
    from src.data_loader import load_teams
    t = load_teams().set_index("team_abbr")
    return {k: {"name": r.team_name, "nick": r.team_nick, "logo": r.team_logo_espn} for k, r in t.iterrows()}


@st.cache_data
def kickoffs():
    """game_id -> kickoff (US Eastern) from the schedule."""
    from src.data_loader import load_schedules
    s = load_schedules()[["game_id", "gameday", "gametime"]]
    return dict(zip(s.game_id, pd.to_datetime(s.gameday + " " + s.gametime.fillna("13:00"))))


def waterfall_chart(e, home, away, height=None):
    """Horizontal waterfall from 50% to the model's home win probability."""
    from src.explain import waterfall
    w = waterfall(e).reset_index(drop=True)
    w["order"] = range(len(w))
    final = w.factor == "Model probability"
    w["kind"] = np.where(final, "Model probability", np.where(w.delta >= 0, f"Pushes toward {home}", f"Pushes toward {away}"))
    w["label"] = np.where(final, w.end.map(lambda v: f"{v:.1%}"), (w.delta * 100).map(lambda v: f"{v:+.1f}"))
    w["lo"], w["hi"] = w[["start", "end"]].min(axis=1), w[["start", "end"]].max(axis=1)
    lo = max(0.0, w.lo.min() - 0.06)
    hi = min(1.0, w.hi.max() + 0.06)
    y = alt.Y("factor:N", sort=alt.EncodingSortField("order"), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False))
    x_scale = alt.Scale(domain=[lo, hi])
    bars = alt.Chart(w).mark_bar(cornerRadius=3, height=18).encode(
        y=y, x=alt.X("start:Q", title=f"{home} win probability", scale=x_scale, axis=alt.Axis(format="%")),
        x2="end:Q",
        color=alt.Color("kind:N", legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=[f"Pushes toward {home}", f"Pushes toward {away}", "Model probability"],
                                        range=[BLUE, RED, "#52514e"])),
        tooltip=["factor", alt.Tooltip("start:Q", format=".1%", title="from"),
                 alt.Tooltip("end:Q", format=".1%", title="to"), "detail"])
    labels = alt.Chart(w).mark_text(align="left", dx=5, fontSize=11, color=TEXT_2).encode(
        y=y, x=alt.X("hi:Q", scale=x_scale), text="label:N")
    even = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(
        x=alt.X("x:Q", scale=x_scale))
    return style(even + bars + labels, height or 30 * len(w) + 50)


def prob_bar(home, away, p_home, p_vegas=None):
    """Two-segment win-probability bar (away left, home right) with a Vegas tick."""
    a, h = (1 - p_home) * 100, p_home * 100
    tick = ""
    if p_vegas is not None and p_vegas == p_vegas:
        tick = (f'<div title="Vegas" style="position:absolute;top:-4px;bottom:-4px;left:calc({(1 - p_vegas) * 100:.1f}% - 1.5px);'
                f'width:3px;background:#0b0b0b;border-radius:2px"></div>')
    seg = 'display:flex;align-items:center;white-space:nowrap;overflow:hidden'
    left = f"{away} {a:.0f}%" if a >= 14 else ""
    right = f"{home} {h:.0f}%" if h >= 14 else ""
    return (f'<div style="position:relative;margin:6px 0 2px 0">'
            f'<div style="display:flex;height:28px;border-radius:6px;overflow:hidden;font-size:13px;font-weight:600;color:#fff">'
            f'<div style="width:{a:.1f}%;background:{ORANGE};{seg};padding-left:8px">{left}</div>'
            f'<div style="width:{h:.1f}%;background:{BLUE};{seg};justify-content:flex-end;padding-right:8px">{right}</div>'
            f'</div>{tick}</div>')


games, oos, missing, results, team_games = load(tuple(f.stat().st_mtime if f.exists() else 0 for f in DATA_FILES))
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

page = st.sidebar.radio("Section", ["This Week", "Overview", "Teams", "Team Analytics", "Games", "Players",
                                    "Situational", "Matchups", "Model"])

# ================================================================ This Week
if page == "This Week":
    from src.config import MODELS_DIR
    from src.explain import explain
    from src.evaluate import moneyline_prob
    from src.matchups import bad_weather
    meta = team_meta()
    ko = kickoffs()
    model_path = MODELS_DIR / "logistic.joblib"
    bundle = load_model(model_path.stat().st_mtime)

    cur = int(games.season.max())
    wk_all = sorted(games[games.season == cur].week.unique())
    upcoming_weeks = sorted(games[(games.season == cur) & games.home_score.isna()].week.unique())
    f1, f2 = st.columns([1, 3])
    week = f1.selectbox("Week", wk_all, index=wk_all.index(upcoming_weeks[0]) if upcoming_weeks else len(wk_all) - 1,
                        format_func=lambda w: f"Week {w}" if w <= 18 else {19: "Wild Card", 20: "Divisional",
                                                                           21: "Conference", 22: "Super Bowl"}.get(w, f"Week {w}"))
    wg = games[(games.season == cur) & (games.week == week)].copy()
    wg["kickoff"] = wg.game_id.map(ko)
    wg = wg.sort_values(["kickoff", "game_id"]).reset_index(drop=True)
    wg["p_vegas"] = moneyline_prob(wg.home_moneyline, wg.away_moneyline)

    # Played games show the probability recorded before kickoff; upcoming games the current model.
    live_p = bundle["model"].predict_proba(wg[bundle["features"]])[:, 1]
    from src.track import _load as load_tracking
    try:
        tracked = load_tracking().set_index("game_id").p_logistic
    except Exception:
        tracked = pd.Series(dtype=float)
    played_now = wg.home_score.notna()
    wg["p_show"] = np.where(played_now, wg.game_id.map(tracked).fillna(pd.Series(live_p)), live_p)
    exps = {e["game_id"]: e for e in explain(bundle, wg, missing, top=20)}
    from src.spread import cover_prob_calibrated, fmt_spread
    sp_path = MODELS_DIR / "spread.joblib"
    sp = load_spread(sp_path.stat().st_mtime) if sp_path.exists() else None
    if sp is not None:
        live_margin = sp["model"].predict(wg[sp["features"]])
        tracked_margin = load_tracking().set_index("game_id").pred_margin if "pred_margin" in load_tracking() else pd.Series(dtype=float)
        wg["margin"] = np.where(played_now, wg.game_id.map(tracked_margin).fillna(pd.Series(live_margin)), live_margin)
        wg["p_cover"] = np.where(wg.spread_line.notna(),
                                 cover_prob_calibrated(wg.margin, wg.spread_line.fillna(0), sp["cover_k"]), np.nan)
    else:
        wg["margin"], wg["p_cover"] = np.nan, np.nan

    def spread_line_txt(r):
        if r.margin != r.margin:
            return ""
        txt = f"Spread: model **{fmt_spread(r.home_team, r.away_team, r.margin)}**"
        if r.spread_line == r.spread_line:
            txt += f" · Vegas **{fmt_spread(r.home_team, r.away_team, r.spread_line)}**"
            side, ps = (r.home_team, r.p_cover) if r.p_cover >= 0.5 else (r.away_team, 1 - r.p_cover)
            txt += f" · {side} covers {ps:.0%}"
        return txt

    detail = st.session_state.get("detail_game")
    if detail not in set(wg.game_id):
        detail = None

    def title_html(r, size=34):
        a, h = meta.get(r.away_team, {}), meta.get(r.home_team, {})
        img = lambda m: f'<img src="{m.get("logo", "")}" style="height:{size}px;vertical-align:middle">' if m else ""
        return (f'<div style="display:flex;align-items:center;gap:10px;font-size:{17 if size < 40 else 24}px;font-weight:600">'
                f'{img(a)}<span>{a.get("nick", r.away_team)}</span><span style="color:{GRAY};font-weight:400">at</span>'
                f'{img(h)}<span>{h.get("nick", r.home_team)}</span></div>')

    qb_path = PROCESSED_DIR / "qb_status.parquet"
    qb_status = pd.read_parquet(qb_path) if qb_path.exists() else pd.DataFrame(columns=["game_id", "status"])

    def badges(r):
        out = []
        for q in qb_status[(qb_status.game_id == r.game_id) & (qb_status.status != "ok")].itertuples():
            last = lambda n: str(n).replace(" Jr.", "").replace(" II", "").split()[-1]
            if q.applied:
                out.append((f"{q.team} QB: {last(q.backup_name)} starts, {last(q.qb_name)} out",
                            ":material/sports_football:", "violet"))
            else:
                out.append((f"{q.team} QB: {last(q.qb_name)} left last game early",
                            ":material/warning:", "orange"))
        if r.p_vegas == r.p_vegas and abs(r.p_show - r.p_vegas) >= 0.10:
            out.append(("Model vs Vegas: {:.0f} pts".format(abs(r.p_show - r.p_vegas) * 100), ":material/compare_arrows:", "orange"))
        wx = pd.DataFrame([{"wind_mph": r.wind_mph, "precip_mm": r.precip_mm, "temp_f": r.temp_f}])
        if not r.indoor and bool(bad_weather(wx).iloc[0]):
            txt = f"{r.temp_f:.0f}°F, {r.wind_mph:.0f} mph wind" + (", rain/snow" if r.precip_mm >= 1 else "")
            out.append((txt, ":material/thunderstorm:", "blue"))
        m = missing[missing.game_id == r.game_id] if missing is not None else pd.DataFrame()
        for team in (r.away_team, r.home_team):
            for unit in ("offense", "defense"):
                mt = m[(m.team == team) & (m.get("unit", "offense") == unit)] if len(m) else m
                mt = mt.sort_values("missing_value", ascending=False) if len(mt) else mt
                if len(mt) and mt.missing_value.sum() >= 0.03:
                    out.append((f"{team} without {mt.iloc[0]['name']}", ":material/personal_injury:", "red"))
        if r.neutral:
            out.append(("Neutral site", ":material/public:", "gray"))
        return out

    if detail is None:
        st.header(f"{cur} — " + (f"Week {week}" if week <= 18 else "Playoffs"))
        n_up = int((~played_now).sum())
        st.caption(f"{len(wg)} games · {n_up} still to play. Bars show the model's win probability "
                   f"(orange = away, blue = home); the black tick is Vegas. Played games show the probability recorded "
                   f"before kickoff.")
        cols = st.columns(2)
        for i, r in wg.iterrows():
            e = exps[r.game_id]
            with cols[i % 2].container(border=True):
                st.markdown(title_html(r), unsafe_allow_html=True)
                when = r.kickoff.strftime("%a %b %-d · %-I:%M %p ET") if pd.notna(r.kickoff) else ""
                st.caption(f"{when}  ·  {r.away_qb_name} vs {r.home_qb_name}")
                st.markdown(prob_bar(r.home_team, r.away_team, r.p_show, r.p_vegas), unsafe_allow_html=True)
                vegas_txt = ""
                if r.p_vegas == r.p_vegas:
                    vfav, vp = (r.home_team, r.p_vegas) if r.p_vegas >= 0.5 else (r.away_team, 1 - r.p_vegas)
                    vegas_txt = f"Vegas: {vfav} {vp:.0%}"
                if played_now[i]:
                    winner = r.home_team if r.home_score > r.away_score else r.away_team
                    pick = r.home_team if r.p_show >= 0.5 else r.away_team
                    mark = "✓" if winner == pick else "✗"
                    ats = ""
                    if r.margin == r.margin and r.spread_line == r.spread_line:
                        diff = (r.home_score - r.away_score) - r.spread_line
                        side = r.home_team if r.margin > r.spread_line else r.away_team
                        ok = (diff > 0) if r.margin > r.spread_line else (diff < 0)
                        ats = " · spread: push" if diff == 0 else f" · spread pick {side} {'✓' if ok else '✗'}"
                    st.caption(f"**Final: {r.away_team} {r.away_score:.0f} – {r.home_team} {r.home_score:.0f}** "
                               f"(model picked {pick} {mark}{ats}) · {vegas_txt}")
                else:
                    st.caption(vegas_txt)
                if spread_line_txt(r):
                    st.caption(spread_line_txt(r))
                b = badges(r)
                if b:
                    st.markdown(" ".join(f":{c}-badge[{ic} {t}]" for t, ic, c in b))
                reasons = "".join(
                    f"<div style='font-size:13px;line-height:1.35;margin:3px 0'><b>{f['favors']} "
                    f"+{abs(f['pct_points']) * 100:.1f}%</b> {f['factor']} "
                    f"<span style='color:{TEXT_2}'>— {f['detail']}</span></div>" for f in e["factors"][:3])
                st.markdown(f"<div style='margin:4px 0 8px 0'>{reasons}</div>", unsafe_allow_html=True)
                if st.button("Full breakdown", key=f"open_{r.game_id}", icon=":material/insights:"):
                    st.session_state["detail_game"] = r.game_id
                    st.rerun()
    else:
        r = wg[wg.game_id == detail].iloc[0]
        e = exps[detail]
        if st.button("All games", icon=":material/arrow_back:"):
            st.session_state.pop("detail_game", None)
            st.rerun()
        st.markdown(title_html(r, size=56), unsafe_allow_html=True)
        when = r.kickoff.strftime("%A %B %-d · %-I:%M %p ET") if pd.notna(r.kickoff) else ""
        st.caption(f"{when} · {r.away_qb_name} vs {r.home_qb_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{r.away_team} win", f"{1 - r.p_show:.1%}")
        c2.metric(f"{r.home_team} win", f"{r.p_show:.1%}")
        if r.p_vegas == r.p_vegas:
            c3.metric(f"Vegas: {r.home_team}", f"{r.p_vegas:.1%}", f"{(r.p_show - r.p_vegas) * 100:+.1f} pts model vs Vegas",
                      delta_color="off")
        st.markdown(prob_bar(r.home_team, r.away_team, r.p_show, r.p_vegas), unsafe_allow_html=True)
        if r.margin == r.margin:
            s1, s2, s3 = st.columns(3)
            s1.metric("Model spread", fmt_spread(r.home_team, r.away_team, r.margin))
            s2.metric("Vegas spread", fmt_spread(r.home_team, r.away_team, r.spread_line))
            if r.p_cover == r.p_cover:
                side, ps = (r.home_team, r.p_cover) if r.p_cover >= 0.5 else (r.away_team, 1 - r.p_cover)
                s3.metric(f"{side} covers", f"{ps:.0%}")
            st.caption("Cover chances are calibrated on 2012–2018: historically, disagreeing with Vegas has been worth "
                       "little (a 3-point gap ≈ 53%), and on 2019–2025 the model's spread did not beat the line.")
        b = badges(r)
        if b:
            st.markdown(" ".join(f":{c}-badge[{ic} {t}]" for t, ic, c in b))
        if r.p_vegas == r.p_vegas and abs(r.p_show - r.p_vegas) >= 0.10:
            fav = r.home_team if r.p_vegas >= 0.5 else r.away_team
            model_lower_on_fav = (r.p_vegas >= 0.5) == (r.p_show < r.p_vegas)
            if model_lower_on_fav:
                st.info(f"**The model is much lower on {fav} than Vegas.** In past games like this (2012–2025, Vegas "
                        f"15+ points higher on the favorite), the favorite won 60% of the time: Vegas said 66%, the model "
                        f"46%. The truth usually lands between the two, closer to Vegas. The model deliberately regresses "
                        f"teams toward average and can't see things like daily injury news.", icon=":material/info:")
            else:
                st.info(f"**The model is much higher on {fav} than Vegas.** When the model likes the favorite more than "
                        f"Vegas does, the favorite has won about as often as Vegas implied, so lean toward the Vegas number.",
                        icon=":material/info:")
        st.subheader("How the prediction is built")
        st.caption(f"Starts at 50% (evenly matched teams at a neutral site). Each bar adds one factor, biggest first, "
                   f"and the last bar is the model's probability for {r.home_team}. Hover a bar for the numbers behind it.")
        if played_now.loc[r.name] and r.game_id in tracked.index:
            st.caption("This game has been played: the breakdown uses the current model, so it may differ slightly "
                       "from the probability recorded before kickoff.")
        st.altair_chart(waterfall_chart(e, r.home_team, r.away_team), use_container_width=True)
        f = pd.DataFrame(e["all_factors"])
        f = f[f.pct_points.abs() >= 0.001]
        f["effect"] = f.apply(lambda x: f"{x.favors} +{abs(x.pct_points) * 100:.1f}%", axis=1)
        st.dataframe(f[["factor", "effect", "detail"]], hide_index=True, use_container_width=True)
        m = missing[missing.game_id == detail] if missing is not None else pd.DataFrame()
        if len(m):
            st.markdown("**Missing players**")
            cols_m = ["team", "unit", "name", "position", "value", "missing_value"] if "unit" in m else \
                ["team", "name", "position", "value", "missing_value"]
            st.dataframe(m.sort_values("missing_value", ascending=False)[cols_m], hide_index=True, use_container_width=True,
                         column_config={"value": st.column_config.NumberColumn("usual share", format="percent",
                                                                               help="offense: share of RB/WR/TE touches; "
                                                                                    "defense: share of playmaking"),
                                        "missing_value": st.column_config.NumberColumn("missing share", format="percent")})

# ================================================================ Overview
elif page == "Overview":
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

# ================================================================ Team Analytics
elif page == "Team Analytics":
    st.caption("Team tendencies and situational efficiency. These are **not model features**: 3rd/4th-down rates "
               "made predictions worse on the tuning seasons (they mostly repeat EPA plus luck), and coverage data "
               "only exists for 2018–2025. They're here to explain *how* teams win.")
    tg = team_games[team_games.game_type == "REG"]
    c1, c2 = st.columns(2)
    per_team = tg.groupby("season").size() / 32
    seasons_a = sorted(tg.season.unique(), reverse=True)
    default_a = max(int(x) for x in per_team[per_team >= 8].index)
    season_a = c1.selectbox("Season", seasons_a, index=seasons_a.index(default_a), key="ta_season")
    team_opts = sorted(tg.team.unique())
    focus = c2.selectbox("Team", team_opts, index=team_opts.index("LAC"), key="ta_team")
    ts = tg[tg.season == season_a]

    # ---- 3rd downs
    st.subheader("3rd downs")
    d3 = ts.groupby("team")[["off_third_conv", "off_third_att", "def_third_conv", "def_third_att"]].sum().reset_index()
    d3["off_rate"] = d3.off_third_conv / d3.off_third_att
    d3["def_rate"] = d3.def_third_conv / d3.def_third_att
    r = d3.set_index("team").loc[focus]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{focus} offense converts", f"{r.off_rate:.1%}", f"{(r.off_rate - d3.off_rate.mean()) * 100:+.1f} pts vs league")
    m2.metric(f"{focus} defense allows", f"{r.def_rate:.1%}", f"{(r.def_rate - d3.def_rate.mean()) * 100:+.1f} pts vs league",
              delta_color="inverse")
    m3.metric("Offense rank", f"{int(d3.off_rate.rank(ascending=False)[d3.team == focus].iloc[0])} of 32")
    st.caption("Right = offense converts more. Up = defense allows fewer (axis flipped so up is better). "
               "Dashed lines are league averages.")
    st.altair_chart(team_scatter(
        d3, "off_rate", "def_rate", "Offense 3rd-down conversion rate", "Defense 3rd-down rate allowed", focus,
        reverse_y=True, tooltip=["team", alt.Tooltip("off_rate:Q", format=".1%", title="offense converts"),
                                 alt.Tooltip("off_third_att:Q", title="offense attempts"),
                                 alt.Tooltip("def_rate:Q", format=".1%", title="defense allows")]),
        use_container_width=True)

    # ---- 4th downs
    st.subheader("4th downs")
    d4 = ts.groupby("team").agg(att=("off_fourth_att", "sum"), conv=("off_fourth_conv", "sum"),
                                games=("game_id", "size")).reset_index()
    d4["att_per_game"] = d4.att / d4.games
    d4["rate"] = d4.conv / d4.att.where(d4.att > 0)
    r4 = d4.set_index("team").loc[focus]
    m1, m2 = st.columns(2)
    m1.metric(f"{focus} 4th-down attempts per game", f"{r4.att_per_game:.2f}")
    m1.caption(f"{r4.att:.0f} attempts; league average {d4.att_per_game.mean():.2f} per game")
    m2.metric(f"{focus} 4th-down conversion", f"{r4.rate:.0%}" if r4.rate == r4.rate else "–")
    m2.caption(f"league {d4.conv.sum() / d4.att.sum():.0%}")
    st.caption("Right = more aggressive (more 4th-down attempts). Up = converts more often. "
               "Small samples: a team might try only 15–30 in a season.")
    st.altair_chart(team_scatter(
        d4.dropna(subset=["rate"]), "att_per_game", "rate", "4th-down attempts per game", "4th-down conversion rate",
        focus, x_fmt=".1f", tooltip=["team", alt.Tooltip("att:Q", title="attempts"), alt.Tooltip("conv:Q", title="converted"),
                                     alt.Tooltip("rate:Q", format=".0%", title="conversion rate")], height=380),
        use_container_width=True)

    # ---- Coverage
    st.subheader("Coverage: man vs. zone")
    cov_path = PROCESSED_DIR / "coverage_defense.parquet"
    cov = load_coverage(cov_path.stat().st_mtime if cov_path.exists() else 0)
    if cov is None:
        st.info("Run `python -m src.coverage` to build coverage analytics.")
    else:
        cseasons = sorted(cov["defense"].season.unique())
        season_c = season_a if season_a in cseasons else cseasons[-1]
        if season_c != season_a:
            st.caption(f"Coverage charting for {season_a} isn't published yet (nflverse releases it after the season); "
                       f"showing {season_c}.")
        dfn = cov["defense"][cov["defense"].season == season_c].copy()
        league_man = float((dfn.man_rate * dfn.dropbacks).sum() / dfn.dropbacks.sum())
        rc = dfn.set_index("team").loc[focus] if focus in set(dfn.team) else None
        if rc is not None:
            m1, m2 = st.columns(2)
            m1.metric(f"{focus} man coverage", f"{rc.man_rate:.0%}")
            m1.caption(f"of charted dropbacks; league {league_man:.0%}")
            m2.metric(f"{focus} zone coverage", f"{1 - rc.man_rate:.0%}")
            m2.caption(f"league {1 - league_man:.0%}")
        dfn["group"] = np.where(dfn.team == focus, "Selected team", "Other teams")
        bars = alt.Chart(dfn).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("team:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("man_rate:Q", title="Share of dropbacks in man coverage", axis=alt.Axis(format="%")),
            color=alt.Color("group:N", legend=None, scale=alt.Scale(domain=["Selected team", "Other teams"],
                                                                    range=[ORANGE, BLUE])),
            tooltip=["team", alt.Tooltip("man_rate:Q", format=".1%", title="man"),
                     alt.Tooltip("dropbacks:Q", title="charted dropbacks")])
        ref = alt.Chart(pd.DataFrame({"v": [league_man]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="v:Q")
        st.caption(f"{season_c}: how often each defense plays man (the rest is zone). Dashed line = league average.")
        st.altair_chart(style(bars + ref, 20 * len(dfn) + 40), use_container_width=True)

        types = cov["types"][cov["types"].season == season_c]
        lg = types.groupby("coverage").plays.sum()
        mix = pd.DataFrame({"league": lg / lg.sum()})
        mix[focus] = types[types.team == focus].set_index("coverage").share
        mix = mix.fillna(0).reset_index().melt(id_vars="coverage", var_name="who", value_name="share")
        order = list((lg / lg.sum()).sort_values(ascending=False).index)
        st.markdown(f"**{focus} coverage mix vs. league ({season_c})**")
        st.altair_chart(style(alt.Chart(mix).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("coverage:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("who:N", sort=[focus, "league"]),
            y=alt.Y("share:Q", title="Share of charted dropbacks", axis=alt.Axis(format="%")),
            color=alt.Color("who:N", legend=alt.Legend(title=None),
                            scale=alt.Scale(domain=[focus, "league"], range=[ORANGE, GRAY])),
            tooltip=["who", "coverage", alt.Tooltip("share:Q", format=".1%")]), 300), use_container_width=True)

        st.subheader("Offense vs. man and zone")
        off = cov["offense"][cov["offense"].season == season_c].dropna()
        st.caption(f"{season_c}: EPA per dropback against each coverage. Above the diagonal = better against zone; "
                   "below = better against man. Dashed lines are league averages.")
        ro = off.set_index("team").loc[focus] if focus in set(off.team) else None
        if ro is not None:
            m1, m2 = st.columns(2)
            lm = float((off.epa_vs_man * off.n_vs_man).sum() / off.n_vs_man.sum())
            lz = float((off.epa_vs_zone * off.n_vs_zone).sum() / off.n_vs_zone.sum())
            m1.metric(f"{focus} EPA/dropback vs man", f"{ro.epa_vs_man:+.3f}")
            m1.caption(f"{ro.n_vs_man:.0f} dropbacks; league {lm:+.3f}")
            m2.metric(f"{focus} EPA/dropback vs zone", f"{ro.epa_vs_zone:+.3f}")
            m2.caption(f"{ro.n_vs_zone:.0f} dropbacks; league {lz:+.3f}")
        st.altair_chart(team_scatter(
            off, "epa_vs_man", "epa_vs_zone", "EPA/dropback vs man coverage", "EPA/dropback vs zone coverage", focus,
            x_fmt="+.2f", y_fmt="+.2f", diagonal=True,
            tooltip=["team", alt.Tooltip("epa_vs_man:Q", format="+.3f", title="vs man"),
                     alt.Tooltip("n_vs_man:Q", title="dropbacks vs man"),
                     alt.Tooltip("epa_vs_zone:Q", format="+.3f", title="vs zone"),
                     alt.Tooltip("n_vs_zone:Q", title="dropbacks vs zone")]), use_container_width=True)

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
        st.caption("Starts at 50% (evenly matched teams at a neutral site). Each bar adds one factor, biggest first; "
                   "the last bar is the model's probability for the home team. Hover for details.")
        st.altair_chart(waterfall_chart(e, r0.home_team, r0.away_team), use_container_width=True)
        f = pd.DataFrame(e["all_factors"])
        f = f[f.pct_points.abs() >= 0.001]
        f["label"] = (f.pct_points * 100).map(lambda v: f"{v:+.1f}%")
        f["direction"] = np.where(f.pct_points >= 0, f"Favors {r0.home_team}", f"Favors {r0.away_team}")
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
        unit = st.radio("Unit", ["Offense (RB/WR/TE)", "Defense"], horizontal=True)
        if "unit" in m:
            m = m[m.unit == ("defense" if unit == "Defense" else "offense")]
        f1, f2 = st.columns(2)
        season = f1.selectbox("Season", sorted(m.season.unique(), reverse=True))
        team = f2.selectbox("Team", ["All teams"] + sorted(m.team.unique()))
        sm = m[m.season == season]

        if unit == "Defense":
            st.caption("**Role** = share of the team's defensive playmaking he usually produces (sacks, interceptions, "
                       "forced fumbles = 1; QB hits, tackles for loss, passes defended = ½). "
                       "**Missing share** = role × how regularly he had been playing.")
        else:
            st.caption("**Role** = share of the team's RB/WR/TE targets + carries he usually gets. "
                       "**Missing share** = role × how regularly he had been playing.")
        if team == "All teams":
            st.subheader(f"Biggest absences of {season}")
            top = sm.sort_values("missing_value", ascending=False).head(20)
            top["label"] = top.name + " (" + top.team + ", wk " + top.week.astype(str) + ")"
            st.altair_chart(bar_chart(top, "missing_value:Q", "label:N", horizontal=True,
                                      x_title="Share of team role missing", y_title=None, height=520, x_format=".1%",
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
            st.subheader(f"{team} {season}: share of {'defensive playmaking' if unit == 'Defense' else 'skill-player touches'} missing each week")
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

# ================================================================ Matchups
elif page == "Matchups":
    st.caption("How specific QBs and units match up with conditions and opponents. "
               "Only *QB in bad weather* is in the model; the others are shown for context (see README).")

    # ---- QB bad-weather sensitivity
    st.subheader("QBs in bad weather")
    st.caption("EPA per dropback in bad weather (15+ mph wind, rain/snow, or 32°F or colder) compared with the same QB's "
               "normal games, beyond the league-wide drop. Shrunk toward 0 until a QB has ~2,500 bad-weather dropbacks, "
               "so short careers sit near 0. Values are what the model used for each QB's latest start.")
    qb_cols = ["season", "gameday", "qb", "team", "qb_weather_sens"] + \
        [c for c in ("qb_clutch", "qb_prime_sens") if c in teams.columns]
    qbs = teams[qb_cols].dropna(subset=["qb"])
    seasons_qb = sorted(qbs.season.unique(), reverse=True)
    c1, c2 = st.columns([1, 2])
    season_q = c1.selectbox("Entering season", seasons_qb, key="qb_season")
    n_show = c2.slider("QBs to show at each end", 5, 20, 10)
    latest = qbs[qbs.season == season_q].sort_values("gameday").groupby("qb").tail(1)
    latest = latest[latest.qb_weather_sens != 0]
    pick = pd.concat([latest.nsmallest(n_show, "qb_weather_sens"), latest.nlargest(n_show, "qb_weather_sens")]).drop_duplicates("qb")
    pick["label"] = pick.qb + " (" + pick.team + ")"
    pick["direction"] = np.where(pick.qb_weather_sens >= 0, "Better in bad weather", "Worse in bad weather")
    st.altair_chart(bar_chart(
        pick, "qb_weather_sens:Q", "label:N", horizontal=True, x_title="EPA per dropback vs. typical QB (bad weather)",
        height=24 * len(pick) + 40,
        tooltip=["qb", "team", alt.Tooltip("qb_weather_sens:Q", format="+.3f", title="EPA/dropback vs typical")],
        color_enc=alt.Color("direction:N", legend=alt.Legend(title=None),
                            scale=alt.Scale(domain=["Better in bad weather", "Worse in bad weather"],
                                            range=[BLUE, RED]))), use_container_width=True)

    # ---- Prime time & clutch (informational)
    st.subheader("QBs in prime time and in the clutch")
    st.caption("**Not used by the model.** Prime-time edges don't carry over from one set of seasons to the next "
               "(correlation −0.09), and clutch edges carry over only a little (+0.28), too little to improve "
               "predictions. Records are shown against the wins the model expected, because night games are often "
               "against strong opponents. Prime time = kickoff 7 PM ET or later; clutch = 4th quarter or OT within one score.")
    pt = prime_time_records()
    if len(pt):
        qb_pick = st.multiselect("QBs", sorted(pt.qb.unique()),
                                 default=[q for q in ["Patrick Mahomes", "Daniel Jones", "Josh Allen", "Jalen Hurts",
                                                      "Lamar Jackson", "Joe Burrow"] if q in set(pt.qb)])
        latest_r = qbs.sort_values("gameday").groupby("qb").tail(1).set_index("qb")
        show = pt[pt.qb.isin(qb_pick)].set_index("qb")
        show = show.join(latest_r[["qb_prime_sens", "qb_clutch"]] if "qb_clutch" in latest_r else pd.DataFrame())
        st.dataframe(show.reset_index(), hide_index=True, use_container_width=True,
                     column_config={"qb": "QB", "record": "prime-time record",
                                    "expected": st.column_config.NumberColumn("expected wins", format="%.1f"),
                                    "vs_expected": st.column_config.NumberColumn("wins vs expected", format="%+.1f"),
                                    "qb_prime_sens": st.column_config.NumberColumn("prime-time EPA edge", format="%+.3f"),
                                    "qb_clutch": st.column_config.NumberColumn("clutch EPA edge", format="%+.3f")})
        st.caption("EPA edges are per dropback vs. a typical QB, shrunk toward 0 for small samples. "
                   "Expected wins use the model's pregame probabilities (2012 onward).")

    # ---- Pressure map
    st.subheader("Pressure: protection vs. pass rush")
    st.caption("Share of dropbacks where the QB was hit or sacked. Right = offense gets hit more (worse protection). "
               "Up = defense hits the QB more (better pass rush). Dashed lines are league averages.")
    tg = team_games
    c1, c2 = st.columns(2)
    per_team = tg.groupby("season").size() / 32
    complete = [int(x) for x in per_team[per_team >= 8].index]  # skip the season until teams have ~8 games
    opts_p = sorted(tg.season.unique(), reverse=True)
    season_p = c1.selectbox("Season", opts_p, index=opts_p.index(max(complete)), key="pr_season")
    focus = c2.selectbox("Highlight team", ["None"] + sorted(tg.team.unique()), key="pr_team")
    pr = (tg[tg.season == season_p].groupby("team")
          .agg(allowed=("off_pressure_rate", "mean"), generated=("def_pressure_rate", "mean"), games=("game_id", "size"))
          .reset_index())
    pr["group"] = np.where(pr.team == focus, focus, "Other teams")
    base = alt.Chart(pr).encode(
        x=alt.X("allowed:Q", title="Offense: hit or sacked rate (lower is better)", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y("generated:Q", title="Defense: hit or sack rate (higher is better)", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False, padding=20)),
        tooltip=["team", alt.Tooltip("allowed:Q", format=".1%", title="offense hit/sacked"),
                 alt.Tooltip("generated:Q", format=".1%", title="defense hit/sack"), "games"])
    dots = base.mark_circle(size=90, stroke="white", strokeWidth=1).encode(
        color=alt.Color("group:N", legend=None,
                        scale=alt.Scale(domain=[focus, "Other teams"], range=[ORANGE, BLUE])))
    names = base.mark_text(dy=-10, fontSize=10, color=TEXT_2).encode(text="team:N")
    avg_x = alt.Chart(pd.DataFrame({"v": [pr.allowed.mean()]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="v:Q")
    avg_y = alt.Chart(pd.DataFrame({"v": [pr.generated.mean()]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(y="v:Q")
    st.altair_chart(style(avg_x + avg_y + dots + names, 460), use_container_width=True)
    if focus != "None":
        trend = (tg[tg.team == focus].groupby("season")
                 .agg(allowed=("off_pressure_rate", "mean"), generated=("def_pressure_rate", "mean")).reset_index()
                 .melt(id_vars="season", var_name="side", value_name="rate"))
        trend["side"] = trend.side.map({"allowed": "Offense hit/sacked", "generated": "Defense hit/sack"})
        st.markdown(f"**{focus} by season**")
        st.altair_chart(line_chart(trend, "season:O", "rate:Q", color="side:N", x_title="Season", y_title="Rate",
                                   color_scale=alt.Scale(domain=["Offense hit/sacked", "Defense hit/sack"],
                                                         range=[ORANGE, BLUE]),
                                   tooltip=["season", "side", alt.Tooltip("rate:Q", format=".1%")], height=260),
                        use_container_width=True)

    # ---- Blitz vulnerability
    st.subheader("Offense vs. the blitz (FTN charting, 2022+)")
    st.caption("EPA per dropback when blitzed minus when not blitzed, for the season. The dashed line is the league "
               "average; left of it = hurt by the blitz more than most offenses, right = handles it better. "
               "One season is a small sample, so expect big swings year to year.")
    bl = load_blitz()
    opts_b = sorted(bl.season.unique(), reverse=True)
    games_per = bl.groupby("season").game_id.nunique()
    default_b = max(int(x) for x in games_per[games_per >= 100].index)
    season_b = st.selectbox("Season", opts_b, index=opts_b.index(default_b), key="bl_season")
    b = bl[bl.season == season_b]
    league_gap = float((b.s_b.sum() / b.n_b.sum()) - (b.s_nb.sum() / b.n_nb.sum()))  # plain float: numpy repr breaks Vega expressions
    bt = b.groupby("team")[["n_b", "s_b", "n_nb", "s_nb"]].sum().reset_index()
    bt["gap"] = bt.s_b / bt.n_b - bt.s_nb / bt.n_nb
    bt["blitz_rate_faced"] = bt.n_b / (bt.n_b + bt.n_nb)
    bars = alt.Chart(bt).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("team:N", sort="x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("gap:Q", title="EPA/dropback: blitzed minus not blitzed"),
        color=alt.condition(alt.datum.gap < league_gap, alt.value(RED), alt.value(BLUE)),
        tooltip=["team", alt.Tooltip("gap:Q", format="+.3f", title="blitz gap"),
                 alt.Tooltip("blitz_rate_faced:Q", format=".0%", title="blitzed on"),
                 alt.Tooltip("n_b:Q", title="blitzed dropbacks")])
    ref = alt.Chart(pd.DataFrame({"v": [league_gap]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="v:Q")
    st.altair_chart(style(bars + ref, 20 * len(bt) + 40), use_container_width=True)
    st.caption(f"League-average drop when blitzed: {league_gap:+.3f} EPA/dropback. "
               "Red = worse than the league-average drop; blue = better.")

    # ---- Live test over time
    st.subheader("Live 2026 test: week by week")
    from src.track import _load as load_tracking
    tr = load_tracking()
    tr = tr[tr.season == tr.season.max()].merge(played[["game_id", "home_win"]], on="game_id") if len(tr) else tr
    if len(tr):
        rows = []
        for name, label in [("p_logistic", "Main model"), ("p_logistic_blitz", "Main + blitz"), ("p_vegas", "Vegas")]:
            d = tr.dropna(subset=[name]).sort_values("week")
            p = d[name].clip(1e-6, 1 - 1e-6)
            d = d.assign(ll=-(d.home_win * np.log(p) + (1 - d.home_win) * np.log(1 - p)))
            cum = d.groupby("week").agg(ll_sum=("ll", "sum"), n=("ll", "size")).cumsum()
            for wk, r in cum.iterrows():
                rows.append({"week": int(wk), "model": label, "log_loss": r.ll_sum / r.n, "games": int(r.n)})
        live = pd.DataFrame(rows)
        st.caption("Season-to-date log loss after each week (lower is better). The decision is made after the regular season.")
        st.altair_chart(line_chart(live, "week:O", "log_loss:Q", color="model:N", x_title="Week", y_title="Log loss so far",
                                   color_scale=alt.Scale(domain=["Main model", "Main + blitz", "Vegas"],
                                                         range=[BLUE, AQUA, ORANGE]),
                                   tooltip=["week", "model", alt.Tooltip("log_loss:Q", format=".4f"), "games"],
                                   height=300), use_container_width=True)
    else:
        st.info("No live tracking yet. Run `python -m src.predict`.")

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

        st.subheader("Spreads: model vs. Vegas")
        sp_rep_path = ROOT / "reports" / "spread_results.json"
        if sp_rep_path.exists():
            sr = json.loads(sp_rep_path.read_text())
            rows_s = []
            for period, v in sr.items():
                row = {"period": period, "games": v["games"], "model error (RMSE)": round(v["rmse_model"], 2),
                       "Vegas error (RMSE)": round(v["rmse_vegas"], 2)}
                for key, label in [("ats_edge>0", "ATS: all games"), ("ats_edge>3.0", "ATS: 3+ pt disagreement"),
                                   ("ats_edge>5.0", "ATS: 5+ pt disagreement")]:
                    w, l, pct = v[key]
                    row[label] = f"{w}-{l} ({pct:.1%})"
                rows_s.append(row)
            st.dataframe(pd.DataFrame(rows_s), hide_index=True, use_container_width=True)
            st.caption("Error = how far the predicted margin misses the final margin, in points. ATS = record picking "
                       "the side our spread favors against the Vegas line (break-even at -110 odds is 52.4%). "
                       "The tuning seasons chose the spread model; the holdout is the honest test. "
                       "Vegas's spread is more accurate, and the holdout ATS record is about 50%.")
        from src.track import ats_record
        a = ats_record()
        if a.get("games"):
            st.caption(f"Live {int(games.season.max())} record against the spread (recorded before kickoff): "
                       f"**{a['wins']}-{a['losses']}** ({a['wins'] / a['games']:.1%}).")

        st.subheader("Live 2026 test: blitz vulnerability")
        st.caption("Rule set before any 2026 results: after the regular season, if the model *with* blitz "
                   "vulnerability has lower log loss on 2026 games, it joins the model. Predictions are recorded "
                   "before kickoff each week (weeks already played were backfilled as if live).")
        try:
            from src.track import TRACK_PATH, scoreboard
            sb = scoreboard().rename(index={"logistic": "Main model", "logistic_blitz": "Main + blitz", "vegas": "Vegas"})
            sb["games"] = sb.games.astype(int)
            st.dataframe(sb.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}),
                         use_container_width=True)
            if sb.loc["Main model", "games"] < 100:
                st.caption(f"Only {sb.loc['Main model', 'games']} games so far — far too few to judge. "
                           "Differences this early are mostly luck.")
        except Exception as exc:  # tracking file not created yet
            st.info(f"No live tracking yet ({exc}). Run `python -m src.predict`.")

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
