"""Turn in-game team stats into *pregame* features, then build the modeling table
(one row per game, home-minus-away differences).

Leakage rule: a feature for a game played on date t only uses games before t.

Rolling team stats use an exponentially weighted mean over the current season,
blended with a prior from last season (regressed toward the league average):

    pregame = (prior_games * prior + sum(w_i * x_i)) / (prior_games + sum(w_i))

so early-season values lean on last year and later values on this year.

Usage:
    python -m src.features      # writes data/processed/games_features.parquet
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, FIRST_SEASON, PROCESSED_DIR
from src.data_loader import load_player_stats, load_schedules
from src.elo import compute_elo
from src.players import load_report_outs, load_roster_status, load_skill_log, skill_availability
from src.situational import load_game_weather, situational_features
from src.team_stats import normalize_teams


@dataclass(frozen=True)
class FeatureParams:
    """Settings for how pregame features are built (tuned in src/tune_features.py)."""
    decay: float = 0.90           # weight multiplier per game back (0.9 = half-life ~6.6 games)
    prior_games: float = 3.0      # how many games' worth of weight last season's value gets
    carryover: float = 0.6        # share of last season's deviation from average that carries over
    form_decay: float = 0.5       # short-memory "recent form" features
    form_prior_games: float = 1.0
    qb_decay: float = 0.97        # per QB game
    qb_season_decay: float = 0.8  # extra discount per season back
    qb_prior_plays: float = 150.0 # plays of weight given to the replacement-level prior
    qb_prior_epa: float = -0.05   # unknown QBs are assumed slightly below average
    qb_prior_cpoe: float = -2.0
    player_decay: float = 0.95        # skill-player value: weight per game back
    player_season_decay: float = 0.7  # ... and per season back
    player_prior_games: float = 4.0   # shrinkage toward 0 for players with few games
    part_decay: float = 0.8           # participation memory (per team game)
    part_season_decay: float = 0.5    # participation carried into a new season


TEAM_STATS = [
    "off_epa_per_play", "off_success_rate", "off_epa_neutral", "off_pass_epa",
    "off_rush_epa", "off_sack_rate", "off_explosive_rate", "off_turnovers",
    "off_rz_td_rate", "off_pass_rate_neutral", "off_st_epa",
    "def_epa_per_play", "def_success_rate", "def_epa_neutral", "def_pass_epa",
    "def_rush_epa", "def_sack_rate", "def_explosive_rate", "def_turnovers",
    "def_rz_td_rate",
    "points_for", "points_against", "point_diff", "win",
]
FORM_STATS = ["point_diff", "off_epa_per_play", "def_epa_per_play"]


def _rolling_pregame(tg: pd.DataFrame, stats, decay, prior_games, carryover) -> pd.DataFrame:
    """Pregame weighted averages of `stats` for each team-game row in tg.
    tg must be sorted by gameday. Returns a frame aligned with tg.index."""
    league_mean = tg.groupby("season")[stats].mean()
    out = np.full((len(tg), len(stats)), np.nan)

    for team, idx in tg.groupby("team").groups.items():
        rows = tg.loc[idx].sort_values("gameday")
        vals = rows[stats].to_numpy(dtype=float)
        seasons = rows.season.to_numpy()
        pos = tg.index.get_indexer(rows.index)

        S = np.zeros(len(stats))
        W = np.zeros(len(stats))
        prior = None
        cur_season = None
        for i in range(len(rows)):
            s = seasons[i]
            if s != cur_season:
                prev = s - 1
                if prior is None or prev not in league_mean.index:
                    prior = league_mean.loc[s].to_numpy()   # first season: burn-in only
                else:
                    final = (prior_games * prior + S) / (prior_games + W)
                    lm = league_mean.loc[prev].to_numpy()
                    prior = lm + carryover * (final - lm)
                S[:] = 0.0
                W[:] = 0.0
                cur_season = s
            out[pos[i]] = (prior_games * prior + S) / (prior_games + W)
            x = vals[i]
            ok = ~np.isnan(x)
            S *= decay
            W *= decay
            S[ok] += x[ok]
            W[ok] += 1.0
    return pd.DataFrame(out, index=tg.index, columns=stats)


def _qb_game_log() -> pd.DataFrame:
    ps = load_player_stats(range(FIRST_SEASON, CURRENT_SEASON + 1))
    ps = ps[(ps.attempts.fillna(0) + ps.sacks_suffered.fillna(0)) > 0].copy()
    ps["plays"] = ps.attempts + ps.sacks_suffered.fillna(0) + ps.carries.fillna(0)
    ps["epa"] = ps.passing_epa.fillna(0) + ps.rushing_epa.fillna(0)
    ps["cpoe_sum"] = ps.passing_cpoe.fillna(0) * ps.attempts
    ps["cpoe_att"] = np.where(ps.passing_cpoe.notna(), ps.attempts, 0)
    return ps[["player_id", "game_id", "season", "week", "plays", "epa", "cpoe_sum", "cpoe_att"]]


def _qb_pregame(tg: pd.DataFrame, p: FeatureParams) -> pd.DataFrame:
    """Pregame QB EPA/play and CPOE for each team-game's listed starting QB,
    using only that QB's earlier games (for any team)."""
    log = _qb_game_log().merge(tg[["game_id", "gameday"]].drop_duplicates(), on="game_id")
    log = log.sort_values("gameday")

    history = {}  # player_id -> list of (gameday, season, plays, epa, cpoe_sum, cpoe_att)
    for r in log.itertuples(index=False):
        history.setdefault(r.player_id, []).append(
            (r.gameday, r.season, r.plays, r.epa, r.cpoe_sum, r.cpoe_att))

    epa_out = np.full(len(tg), np.nan)
    cpoe_out = np.full(len(tg), np.nan)
    exp_out = np.zeros(len(tg))
    for i, (qb, day, season) in enumerate(zip(tg.qb_id, tg.gameday, tg.season)):
        games = [g for g in history.get(qb, []) if g[0] < day]
        if games:
            n = len(games)
            w = p.qb_decay ** np.arange(n - 1, -1, -1)
            seasons_back = season - np.array([g[1] for g in games])
            w = w * p.qb_season_decay ** seasons_back
            arr = np.array([g[2:] for g in games], dtype=float)
            plays, epa, csum, catt = (w[:, None] * arr).sum(axis=0)
            exp_out[i] = arr[:, 0].sum()
        else:
            plays = epa = csum = catt = 0.0
        k = p.qb_prior_plays
        epa_out[i] = (k * p.qb_prior_epa + epa) / (k + plays)
        cpoe_out[i] = (k * p.qb_prior_cpoe + csum) / (k + catt)

    return pd.DataFrame({
        "qb_epa": epa_out,
        "qb_cpoe": cpoe_out,
        "qb_experience": np.log1p(exp_out),  # career plays before this game
    }, index=tg.index)


def _qb_changed(tg: pd.DataFrame) -> pd.Series:
    """1 if the listed starter differs from the team's previous game starter."""
    prev = tg.sort_values("gameday").groupby("team").qb_id.shift(1).reindex(tg.index)
    return (prev.notna() & (prev != tg.qb_id)).astype(int)


@lru_cache(maxsize=1)
def _player_inputs():
    """(skill log, roster status, injury-report outs). Cached: loaded once per process."""
    return load_skill_log(), load_roster_status(), load_report_outs()


def load_player_inputs():
    return _player_inputs()


def build_team_features(tg: pd.DataFrame, p: FeatureParams = FeatureParams()) -> pd.DataFrame:
    tg = tg.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    roll = _rolling_pregame(tg, TEAM_STATS, p.decay, p.prior_games, p.carryover)
    form = _rolling_pregame(tg, FORM_STATS, p.form_decay, p.form_prior_games, 0.0)
    form.columns = [f"form_{c}" for c in FORM_STATS]
    qb = _qb_pregame(tg, p)
    log, rosters, outs = load_player_inputs()
    skill = skill_availability(tg, log, rosters, outs, p.player_decay, p.player_season_decay,
                               p.player_prior_games, p.part_decay, p.part_season_decay)
    feats = pd.concat([tg[["game_id", "team", "season", "gameday", "rest"]], roll, form, qb, skill],
                      axis=1)
    feats["qb_changed"] = _qb_changed(tg)
    return feats


def _league_home_margin(games: pd.DataFrame) -> pd.Series:
    """Trailing 3-season average home point margin (non-neutral games), per season.
    Captures the decline in home-field advantage without looking ahead."""
    hm = games[~games.neutral & games.home_score.notna()]
    by_season = (hm.home_score - hm.away_score).groupby(hm.season).mean()
    trailing = by_season.rolling(3, min_periods=1).mean().shift(1)
    return trailing.fillna(by_season.iloc[0])


def _fill_future_qbs(sched: pd.DataFrame, team_games: pd.DataFrame, qb_overrides=None) -> pd.DataFrame:
    """Upcoming games often have no listed starter yet. Assume each team's most
    recent starter, unless `qb_overrides` ({team: (qb_id, qb_name)}) says otherwise."""
    last = (team_games.sort_values("gameday").dropna(subset=["qb_id"])
            .groupby("team").tail(1).set_index("team")[["qb_id", "qb_name"]])
    future = sched.home_score.isna()
    for side in ("home", "away"):
        team = sched[f"{side}_team"]
        missing = future & sched[f"{side}_qb_id"].isna()
        sched.loc[missing, f"{side}_qb_id"] = team[missing].map(last.qb_id)
        sched.loc[missing, f"{side}_qb_name"] = team[missing].map(last.qb_name)
        for t, (qb_id, qb_name) in (qb_overrides or {}).items():
            hit = future & (team == t)
            sched.loc[hit, f"{side}_qb_id"] = qb_id
            sched.loc[hit, f"{side}_qb_name"] = qb_name
    return sched


def with_placeholders(team_games: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows plus placeholder rows for games not played yet, so upcoming
    games get pregame features too."""
    future = sched[~sched.game_id.isin(team_games.game_id)]
    placeholders = pd.concat([
        pd.DataFrame({"game_id": future.game_id, "season": future.season, "week": future.week,
                      "gameday": future.gameday, "team": future.home_team,
                      "rest": future.home_rest, "qb_id": future.home_qb_id}),
        pd.DataFrame({"game_id": future.game_id, "season": future.season, "week": future.week,
                      "gameday": future.gameday, "team": future.away_team,
                      "rest": future.away_rest, "qb_id": future.away_qb_id}),
    ])
    return pd.concat([team_games, placeholders], ignore_index=True)


def build_game_features(
    team_games: pd.DataFrame,
    params: FeatureParams = FeatureParams(),
    include_future: bool = True,
    qb_overrides=None,
) -> pd.DataFrame:
    """One row per game (home perspective) with home-minus-away feature diffs,
    the label, and Vegas lines kept only for benchmarking."""
    sched = load_schedules()
    sched = sched[sched.season.between(FIRST_SEASON, CURRENT_SEASON)].copy()
    sched = normalize_teams(sched, ["home_team", "away_team"])
    sched["gameday"] = pd.to_datetime(sched.gameday)
    sched["neutral"] = sched.location == "Neutral"
    if not include_future:
        sched = sched[sched.home_score.notna()]
    sched = _fill_future_qbs(sched, team_games, qb_overrides)

    tg_all = with_placeholders(team_games, sched)
    feats = build_team_features(tg_all, params)

    feat_cols = [c for c in feats.columns if c not in ("game_id", "team", "season", "gameday")]
    home = feats[["game_id", "team"] + feat_cols].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in feat_cols}})
    away = feats[["game_id", "team"] + feat_cols].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in feat_cols}})

    team_sit, game_sit = situational_features(sched, team_games, load_game_weather())
    sit_cols = [c for c in team_sit.columns if c not in ("game_id", "team", "home_venue")]
    home = home.merge(team_sit[["game_id", "team"] + sit_cols].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in sit_cols}}), on=["game_id", "home_team"])
    away = away.merge(team_sit[["game_id", "team"] + sit_cols].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in sit_cols}}), on=["game_id", "away_team"])
    feat_cols = feat_cols + sit_cols

    keep = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
            "home_score", "away_score", "neutral", "div_game", "roof",
            "spread_line", "home_moneyline", "away_moneyline", "home_qb_name", "away_qb_name"]
    g = sched[keep].merge(home, on=["game_id", "home_team"]).merge(away, on=["game_id", "away_team"])

    elo = compute_elo(sched[["game_id", "season", "gameday", "home_team", "away_team",
                             "home_score", "away_score", "neutral"]])
    g = g.merge(elo, on="game_id").merge(game_sit, on="game_id", how="left")

    for c in feat_cols:
        g[f"diff_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    g["diff_elo"] = g.home_elo_pre - g.away_elo_pre
    g["home_field"] = np.where(g.neutral, 0.0, g.season.map(_league_home_margin(sched)))
    g["div_game"] = g.div_game.astype(int)
    g["diff_travel_km"] = g.diff_travel_km / 1000  # thousands of km
    g["diff_tz_east"] = np.clip(g.home_tz_shift, 0, None) - np.clip(g.away_tz_shift, 0, None)
    g["diff_tz_west"] = np.clip(-g.home_tz_shift, 0, None) - np.clip(-g.away_tz_shift, 0, None)
    # Strong wind or rain blunts a passing-offense edge.
    g["wind_pass_edge"] = np.clip(g.wind_mph - 10, 0, None) * g.diff_off_pass_epa
    g["precip_pass_edge"] = np.clip(g.precip_mm, 0, 3) * g.diff_off_pass_epa

    g["home_win"] = np.where(g.home_score > g.away_score, 1.0,
                             np.where(g.home_score < g.away_score, 0.0, np.nan))
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


TRAVEL_FEATURES = ["diff_travel_km", "diff_tz_east", "diff_tz_west", "diff_early_clock",
                   "diff_late_clock", "diff_altitude_gain"]
# Defense-focused: defensive snaps are the opponent's plays, and the back seven
# rarely rotates. Beat total snaps on the tuning seasons (0.6154 vs 0.6156).
FATIGUE_FEATURES = ["diff_prev_def_snaps", "diff_def_snap_load", "diff_prev_def_ot_snaps",
                    "diff_prev_ot", "diff_road_streak"]
WEATHER_FEATURES = ["indoor", "temp_f", "wind_mph", "precip_mm", "diff_cold_shock",
                    "wind_pass_edge", "precip_pass_edge"]

BASE_FEATURES = (
    [f"diff_{c}" for c in TEAM_STATS]
    + [f"diff_form_{c}" for c in FORM_STATS]
    + ["diff_qb_epa", "diff_qb_cpoe", "diff_qb_experience", "diff_qb_changed",
       "diff_skill_missing", "diff_skill_missing_top",
       "diff_rest", "diff_elo", "home_field", "div_game"]
)
# The logistic regression uses base + fatigue: travel and weather did not improve
# log loss on the tuning seasons. All groups stay available to the tree models,
# which can pick up interactions (e.g. a warm-weather team in the cold).
FEATURE_COLUMNS = BASE_FEATURES + FATIGUE_FEATURES
ALL_FEATURES = BASE_FEATURES + FATIGUE_FEATURES + TRAVEL_FEATURES + WEATHER_FEATURES


def rebuild(qb_overrides=None, save: bool = True):
    """Refresh weather, rebuild the game feature table and the missing-player list.
    Returns (games, missing_players)."""
    from src.situational import update_game_weather
    from src.tune_features import load_params  # tuned settings, if tuning has been run

    update_game_weather()
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    params = load_params()
    games = build_game_features(tg, params, qb_overrides=qb_overrides)

    # Who was missing in each game, including upcoming ones (for explanations and the dashboard).
    log, rosters, outs = load_player_inputs()
    sched = load_schedules()
    sched = normalize_teams(sched[sched.season.between(FIRST_SEASON, CURRENT_SEASON)].copy(),
                            ["home_team", "away_team"])
    sched["gameday"] = pd.to_datetime(sched.gameday)
    sched = _fill_future_qbs(sched, tg, qb_overrides)
    tg_all = with_placeholders(tg, sched).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    _, details = skill_availability(tg_all, log, rosters, outs, params.player_decay,
                                    params.player_season_decay, params.player_prior_games,
                                    params.part_decay, params.part_season_decay, return_details=True)
    if save:
        games.to_parquet(PROCESSED_DIR / "games_features.parquet", index=False)
        details.to_parquet(PROCESSED_DIR / "missing_players.parquet", index=False)
    return games, details


if __name__ == "__main__":
    games, _ = rebuild()
    played = games.home_win.notna().sum()
    print(f"Wrote {len(games):,} games ({played:,} played) with "
          f"{len(FEATURE_COLUMNS)} features to {PROCESSED_DIR / 'games_features.parquet'}")
