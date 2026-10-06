"""DraftKings splits parsing and grading of prime-time bets (no network)."""
import pandas as pd

from src.splits import disagreements, graded, majority_record, parse_dk


def _side(name, odds, handle, bets):
    return (f'<div class="tb-sodd flex"><div class="tb-slipline flex-1 font-medium">{name}</div>'
            f'<div class="flex-1"><a class="tb-odd-s bg-neutral-900" href="#">{odds}</a></div>'
            f'<div class="flex-1">{handle}%<div class="tb-progress"><div style="width: {handle}%"></div></div></div>'
            f'<div class="flex-1">{bets}%<div class="tb-progress"><div style="width: {bets}%"></div></div></div></div>')


def _market(name, *sides):
    return (f'<div><div class="tb-se-head flex"><div class="flex-1">{name}</div><div class="flex-1">Odds</div>'
            f'<div class="flex-1">% Handle</div><div class="flex-1">% Bets</div></div>'
            f'<div class="tb-sm">{"".join(sides)}</div></div>')


HTML = ('<div class="tb-se border-b"><div class="tb-se-title flex">'
        '<h5 class="tb-se-title-new mt-0"><a href="#"><img src="ATL.png"/> ATL Falcons @ <img src="NO.png"/>NO Saints</a></h5>'
        '<span>10/5, 08:15PM</span></div><div class="tb-market-wrap">'
        + _market("Moneyline", _side("NO Saints", "−122", 55, 59), _side("ATL Falcons", "+102", 45, 41))
        + _market("Spread", _side("NO Saints -1.5", "−108", 70, 62), _side("ATL Falcons +1.5", "−112", 30, 38))
        + _market("Total", _side("Over 47.5", "−115", 90, 71), _side("Under 47.5", "−105", 10, 29))
        + "</div></div>")


def test_parse_dk_reads_every_market_and_side():
    d = parse_dk(HTML)
    assert len(d) == 6
    ml = d[d.market == "moneyline"].set_index("side_name")
    assert ml.loc["NO Saints", "odds"] == -122 and ml.loc["ATL Falcons", "odds"] == 102
    assert ml.loc["NO Saints", "handle_pct"] == 55 and ml.loc["NO Saints", "bets_pct"] == 59
    sp = d[d.market == "spread"].set_index("side_name")
    assert sp.loc["NO Saints", "line"] == -1.5 and sp.loc["ATL Falcons", "line"] == 1.5
    tot = d[d.market == "total"].set_index("side_name")
    assert tot.loc["Over", "line"] == 47.5 and tot.loc["Under", "handle_pct"] == 10


def _graded(away_score, home_score):
    sched = pd.DataFrame({"game_id": ["G"], "gameday": ["2026-10-05"], "gametime": ["20:15"],
                          "away_team": ["ATL"], "home_team": ["NO"],
                          "away_score": [away_score], "home_score": [home_score]})
    pt = pd.DataFrame({"game_id": ["G"], "week": [4], "slot": ["MNF"]})
    rows = [("moneyline", "NO", None, 55, 59), ("moneyline", "ATL", None, 45, 41),
            ("spread", "NO", -1.5, 40, 62), ("spread", "ATL", 1.5, 60, 38),
            ("total", "over", 47.5, 90, 71), ("total", "under", 47.5, 10, 29)]
    early = pd.DataFrame([("2026-10-05T16:00:00Z", "G", m, s, l, -110, 50, 50, "dk") for m, s, l, _, _ in rows],
                         columns=["captured_at", "game_id", "market", "side", "line", "odds", "handle_pct",
                                  "bets_pct", "source"])
    late = early.assign(captured_at="2026-10-05T23:55:00Z", handle_pct=[r[3] for r in rows],
                        bets_pct=[r[4] for r in rows])
    after_kickoff = late.assign(captured_at="2026-10-06T01:00:00Z", handle_pct=99, bets_pct=99)
    return graded(pd.concat([early, late, after_kickoff]), pt, sched).set_index(["market", "side"])


def test_grades_last_snapshot_before_kickoff():
    g = _graded(20, 21)  # NO wins by 1: doesn't cover -1.5, total 41 stays under
    assert (g.handle_pct != 99).all() and (g.hours_before > 0).all()
    assert g.loc[("moneyline", "NO"), "result"] == "won"
    assert g.loc[("spread", "NO"), "result"] == "lost" and g.loc[("spread", "ATL"), "result"] == "won"
    assert g.loc[("total", "under"), "result"] == "won"


def test_push_and_pending():
    assert _graded(24, 24).loc[("moneyline", "NO"), "result"] == "push"
    assert (_graded(None, None).result == "pending").all()


def test_public_side_vs_money_side():
    g = _graded(20, 21).reset_index()
    money = majority_record(g, "handle_pct").set_index("market").side
    bets = majority_record(g, "bets_pct").set_index("market").side
    assert money["spread"] == "ATL" and bets["spread"] == "NO"
    split = disagreements(g)
    assert list(split.market) == ["spread"] and split.iloc[0].result == "won"


def _an_outcome(market, side, team_id, value, odds, money, tickets):
    return {"type": market, "side": side, "team_id": team_id, "value": value, "odds": odds,
            "bet_info": {"money": {"value": 0, "percent": money}, "tickets": {"value": 0, "percent": tickets}}}


def test_parse_action_maps_teams_and_drops_draws_and_empty_markets():
    from src.splits import parse_action
    lookup = {"Atlanta Falcons": "ATL", "New Orleans Saints": "NO"}
    game = {"status": "complete", "season": 2026, "week": 4, "away_team_id": 1, "home_team_id": 2,
            "teams": [{"id": 1, "full_name": "Atlanta Falcons"}, {"id": 2, "full_name": "New Orleans Saints"}],
            "markets": {"15": {"event": {
                "moneyline": [_an_outcome("moneyline", "away", 1, 0, -105, 0, 0),
                              _an_outcome("moneyline", "home", 2, 0, -115, 0, 0),
                              _an_outcome("moneyline", "draw", None, 0, 6000, 0, 0)],
                "spread": [_an_outcome("spread", "away", 1, 1.5, -117, 75, 61),
                           _an_outcome("spread", "home", 2, -1.5, -103, 25, 39)],
                "total": [_an_outcome("total", "over", None, 47.5, -110, 75, 67),
                          _an_outcome("total", "under", None, 47.5, -110, 25, 33)]}}}}
    upcoming = {**game, "status": "scheduled"}
    d = parse_action({"games": [game, upcoming]}, lookup)
    assert set(d.market) == {"spread", "total"}  # the moneyline had no splits (all 0%)
    sp = d[d.market == "spread"].set_index("side")
    assert sp.loc["ATL", "line"] == 1.5 and sp.loc["ATL", "bets_pct"] == 61 and sp.loc["NO", "handle_pct"] == 25
    assert set(d[d.market == "total"].side) == {"over", "under"}


def test_units_pays_at_the_listed_odds():
    from src.splits import units
    df = pd.DataFrame({"odds": [150, -200, -110, 120], "result": ["won", "won", "lost", "push"]})
    assert abs(units(df) - (1.5 + 0.5 - 1.0)) < 1e-9
