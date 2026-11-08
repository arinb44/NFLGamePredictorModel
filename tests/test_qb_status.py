"""Who starts at QB in each team's next game (no network: the nflverse loaders are faked)."""
import pandas as pd
import pytest

import src.data_loader as dl
from src.qb_status import qb_availability

S = 2026


def _game(gid, week, day, away, home, away_qb, home_qb, played=False):
    return {"game_id": gid, "season": S, "week": week, "gameday": day, "away_team": away, "home_team": home,
            "away_qb_id": away_qb, "away_qb_name": f"QB {away_qb}", "home_qb_id": home_qb,
            "home_qb_name": f"QB {home_qb}", "home_score": 20.0 if played else None}


SCHED = pd.DataFrame([
    _game("2026_09_KC_BUF", 9, "2026-11-01", "KC", "BUF", "K1", "B1", played=True),
    _game("2026_10_KC_BUF", 10, "2026-11-08", "KC", "BUF", "K1", "B1"),
    _game("2026_10_DAL_NYG", 10, "2026-11-08", "DAL", "NYG", "D1", "N1"),
    _game("2026_10_ATL_PHI", 10, "2026-11-09", "ATL", "PHI", "A2", "P1"),
    _game("2026_11_KC_DAL", 11, "2026-11-15", "KC", "DAL", "K1", "D1"),
])

INJURIES = pd.DataFrame([
    ("KC", 10, "K1", "Out"),
    ("KC", 10, "K2", "Doubtful"),
    ("NYG", 9, "N1", "Out"),  # last week's report: doesn't count for week 10
    ("PHI", 10, "P1", "Out"),
], columns=["team", "week", "gsis_id", "report_status"])

ROSTERS = pd.DataFrame(
    [("BUF", 9, "B1", "ACT")]  # older week: only the latest roster week counts
    + [(t, 10, q, "ACT") for t, q in [("KC", "K1"), ("KC", "K2"), ("KC", "K3"), ("BUF", "B2"), ("BUF", "B3"),
                                     ("DAL", "D1"), ("DAL", "D2"), ("NYG", "N1"), ("NYG", "N2"),
                                     ("ATL", "A1"), ("PHI", "P1"), ("PHI", "P2")]]
    + [("BUF", 10, "B1", "RES"), ("ATL", 10, "A2", "INA")],
    columns=["team", "week", "gsis_id", "status"],
).assign(full_name=lambda d: "QB " + d.gsis_id, position="QB")

DEPTH = pd.DataFrame(
    [(t, q, i + 1) for t, qbs in {"KC": ["K1", "K2", "K3"], "BUF": ["B1", "B3", "B2"], "PHI": ["P1", "P2"]}.items()
     for i, q in enumerate(qbs)],
    columns=["team", "gsis_id", "pos_rank"],
).assign(pos_abb="QB", dt=pd.Timestamp("2026-11-05"), player_name=lambda d: "QB " + d.gsis_id)


def _dropbacks():
    """Week 9 dropbacks in play order. PHI has none (depth chart only)."""
    order = {"KC": [("K1", 20)], "BUF": [("B2", 30)], "DAL": [("D1", 15), ("D2", 6)],
             "NYG": [("N1", 20), ("N2", 4)], "ATL": [("A2", 10), ("A1", 8)]}
    rows = [(team, qb) for team, seq in order.items() for qb, n in seq for _ in range(n)]
    d = pd.DataFrame(rows, columns=["posteam", "passer_player_id"])
    return d.assign(game_id="g", play_id=range(len(d)), week=9, qtr=4, qb_dropback=1, rusher_player_id=None,
                    passer_player_name="QB " + d.passer_player_id, rusher_player_name=None)


@pytest.fixture
def status(monkeypatch):
    monkeypatch.setattr(dl, "load_injuries", lambda seasons, refresh=False: INJURIES.copy())
    monkeypatch.setattr(dl, "load_rosters", lambda seasons, refresh=False: ROSTERS.copy())
    monkeypatch.setattr(dl, "load_depth_charts", lambda season, refresh=False: DEPTH.copy())
    monkeypatch.setattr(dl, "load_pbp", lambda seasons, columns=None, **kw: _dropbacks())
    standing = {"ATL": ("A2", "QB A2"), "PHI": ("P1", "QB P1")}
    return qb_availability(SCHED, season=S, standing=standing).set_index("team")


def test_one_row_per_team_for_its_next_unplayed_game(status):
    assert sorted(status.index) == ["ATL", "BUF", "DAL", "KC", "NYG", "PHI"]
    assert (status.week == 10).all()


def test_injured_starter_skips_a_backup_who_is_also_out(status):
    kc = status.loc["KC"]
    assert kc.status == "out" and "Out on the week 10 injury report" in kc.reason
    assert (kc.backup_id, kc.backup_name) == ("K3", "QB K3")  # K2 is Doubtful


def test_reserve_starter_backup_is_who_ran_the_offense_not_depth_order(status):
    buf = status.loc["BUF"]
    assert buf.status == "out" and buf.reason == "roster status RES (week 10)"
    assert buf.backup_id == "B2"  # depth chart has B3 ahead of B2


def test_starter_who_left_early_is_flagged_with_the_finisher(status):
    dal = status.loc["DAL"]
    assert dal.status == "left_early" and dal.backup_id == "D2"
    assert "QB D2 took the team's last 6 dropbacks" in dal.reason


def test_short_cameo_and_last_weeks_report_leave_the_starter_in(status):
    nyg = status.loc["NYG"]
    assert nyg.status == "ok" and nyg.backup_id is None  # 4 dropbacks < MIN_BACKUP_DROPBACKS


def test_standing_starter_ignores_roster_and_left_early_but_not_an_out_report(status):
    atl = status.loc["ATL"]
    assert atl.status == "set" and atl.backup_id is None  # A2 is INA and left early, still the starter
    phi = status.loc["PHI"]
    assert phi.status == "out" and "overrides your standing starter" in phi.reason
    assert phi.backup_id == "P2"
