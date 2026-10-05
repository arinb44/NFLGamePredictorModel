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
