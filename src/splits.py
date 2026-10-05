"""Prime-time betting splits: how much of the money and how many of the bets went on
each side of TNF, SNF and MNF games, and which side won.

Sources:
  - Which games are prime time: Champs or Chumps' TNF/SNF/MNF schedule pages
    (champsorchumps.us/nfl). Falls back to the nflverse schedule (Thu/Sun/Mon night
    kickoffs) if the site can't be reached.
  - Splits: the DraftKings Sportsbook Betting Splits table (dknetwork.draftkings.com),
    which shows, for the moneyline, spread and total of each upcoming game,
    % Handle (share of the money wagered) and % Bets (share of the number of wagers),
    all jurisdictions combined.

DraftKings only shows upcoming games, so splits must be captured before kickoff.
Every capture appends a snapshot to reports/prime_time_splits.csv; the last snapshot
taken before kickoff is the one graded. Results are graded from nflverse final scores
against the line at that snapshot (the spread or total the bettors actually got).

Usage:
    python -m src.splits             # capture current splits for upcoming prime-time games
    python -m src.splits --report    # record of the public side vs the money side
"""
import argparse
import re
from html import unescape

import numpy as np
import pandas as pd
import requests

from src.config import CURRENT_SEASON, ROOT

DK_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
DK_NFL = "88808"  # DraftKings event group for the NFL
COC_URL = "https://champsorchumps.us/nfl/{}"
SLOT_PAGES = {"TNF": "thursday-night-football-games", "SNF": "sunday-night-football-games",
              "MNF": "monday-night-football-games"}
SLOT_DAYS = {"Thursday": "TNF", "Sunday": "SNF", "Monday": "MNF"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}

SPLITS_PATH = ROOT / "reports" / "prime_time_splits.csv"
GAMES_PATH = ROOT / "reports" / "prime_time_games.csv"
SPLIT_COLS = ["captured_at", "game_id", "market", "side", "line", "odds", "handle_pct", "bets_pct", "source"]
MARKETS = ("moneyline", "spread", "total")


def _get(url: str, **params) -> str:
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def _text(html: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


def _team_lookup() -> dict:
    """Nickname / full name -> current nflverse abbreviation ("Saints" -> NO, "Los Angeles Rams" -> LA)."""
    from src.data_loader import load_schedules, load_teams
    s = load_schedules()
    current = set(s.loc[s.season == s.season.max(), "home_team"])
    t = load_teams()
    t = t[t.team_abbr.isin(current)]
    lookup = {**dict(zip(t.team_nick, t.team_abbr)), **dict(zip(t.team_name, t.team_abbr))}
    lookup["Bucs"] = "TB"
    return lookup


def kickoff_times(sched: pd.DataFrame) -> pd.Series:
    """game_id -> kickoff as a UTC timestamp (nflverse gametime is US Eastern)."""
    local = pd.to_datetime(sched.gameday + " " + sched.gametime.fillna("13:00"))
    return local.dt.tz_localize("America/New_York").dt.tz_convert("UTC").set_axis(sched.game_id.values)


# ---------------------------------------------------------------- which games are prime time
def _coc_slot_games(slot: str, lookup: dict) -> list:
    """(date, away, home) for every game listed on a Champs or Chumps prime-time page."""
    html = _get(COC_URL.format(SLOT_PAGES[slot]))
    games = []
    for table in html.split('<table class="table table-sm')[1:]:
        table = table.split("</table>")[0]
        date = re.search(r"[A-Z][a-z]{2}, ([A-Z][a-z]{2}\s+\d{1,2}, \d{4})", _text(table))
        teams = re.findall(r'title="\d{4} ([^"]+)" href="/team/nfl/', table)
        if date and len(teams) == 2 and all(t in lookup for t in teams):
            games.append((pd.to_datetime(date.group(1).replace("  ", " ")), lookup[teams[0]], lookup[teams[1]]))
    return games


def prime_time_games(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """game_id, season, week, slot for the season's TNF, SNF and MNF games."""
    from src.data_loader import load_schedules
    s = load_schedules()
    s = s[(s.season == season) & (s.game_type == "REG")].copy()
    s["date"] = pd.to_datetime(s.gameday)
    try:
        lookup = _team_lookup()
        rows = []
        for slot in SLOT_PAGES:
            for date, away, home in _coc_slot_games(slot, lookup):
                hit = s[(s.date == date) & (s.away_team == away) & (s.home_team == home)]
                rows += [(gid, slot) for gid in hit.game_id]
        if not rows:
            raise ValueError("no games parsed")
        slots = pd.DataFrame(rows, columns=["game_id", "slot"]).drop_duplicates("game_id")
        source = "champsorchumps"
    except (requests.RequestException, ValueError) as e:
        print(f"Champs or Chumps unavailable ({e}); using the nflverse schedule.")
        night = s[s.weekday.isin(SLOT_DAYS) & (s.gametime >= "19:00")]
        slots = pd.DataFrame({"game_id": night.game_id, "slot": night.weekday.map(SLOT_DAYS)})
        source = "nflverse"
    out = s[["game_id", "season", "week"]].merge(slots, on="game_id")
    return out.assign(slot_source=source).sort_values("game_id").reset_index(drop=True)


def update_games(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Refresh the season's prime-time game list in reports/prime_time_games.csv."""
    new = prime_time_games(season)
    if GAMES_PATH.exists():
        old = pd.read_csv(GAMES_PATH)
        new = pd.concat([old[old.season != season], new], ignore_index=True)
    new.to_csv(GAMES_PATH, index=False)
    return new


# ---------------------------------------------------------------- DraftKings splits
def _odds(s: str):
    s = s.replace("−", "-").replace("+", "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else np.nan


def _pct(cell: str):
    m = re.search(r"(\d+(?:\.\d+)?)%", _text(cell))
    return float(m.group(1)) if m else np.nan


def parse_dk(html: str) -> pd.DataFrame:
    """One row per game, market and side from a DraftKings splits page."""
    rows = []
    for block in re.split(r'<div class="tb-se ', html)[1:]:
        title = re.search(r'<h5[^>]*class="tb-se-title-new[^"]*"[^>]*>(.*?)</h5>', block, re.S)
        if not title or "@" not in _text(title.group(1)):
            continue
        away, home = (x.strip() for x in _text(title.group(1)).split("@", 1))
        for market in block.split('class="tb-se-head')[1:]:
            name = _text(re.search(r'<div class="flex-1">(.*?)</div>', market, re.S).group(1)).lower()
            if name not in MARKETS:
                continue
            for odd in market.split('class="tb-sodd')[1:]:
                side = _text(re.search(r'tb-slipline[^>]*>(.*?)</div>', odd, re.S).group(1))
                price = re.search(r'class="tb-odd-s[^"]*"[^>]*>(.*?)</a>', odd, re.S)
                cells = re.split(r'<div class="flex-1">', odd)[1:]  # odds, % handle, % bets
                line = re.search(r"([+-]?\d+(?:\.\d+)?)$", side) if name != "moneyline" else None
                rows.append({"away_name": away, "home_name": home, "market": name,
                             "side_name": side[: line.start()].strip() if line else side,
                             "line": float(line.group(1)) if line else np.nan,
                             "odds": _odds(_text(price.group(1))) if price else np.nan,
                             "handle_pct": _pct(cells[1]) if len(cells) > 2 else np.nan,
                             "bets_pct": _pct(cells[2]) if len(cells) > 2 else np.nan})
    return pd.DataFrame(rows)


def fetch_dk(max_pages: int = 10) -> pd.DataFrame:
    """Current NFL splits for the next 7 days (all pages)."""
    frames, seen = [], set()
    for page in range(1, max_pages + 1):
        df = parse_dk(_get(DK_URL, tb_eg=DK_NFL, tb_edate="n7days", tb_emt="0", tb_page=page))
        keys = set(zip(df.away_name, df.home_name)) if len(df) else set()
        if not keys or keys <= seen:
            break
        seen |= keys
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else parse_dk("")


def _to_games(dk: pd.DataFrame, games: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Attach game_id and normalize sides to team abbreviations / over / under."""
    def abbr(name):
        return lookup.get(name.split()[-1]) if isinstance(name, str) and name else None

    dk = dk.assign(away=dk.away_name.map(abbr), home=dk.home_name.map(abbr))
    dk = dk.merge(games[["game_id", "away_team", "home_team"]],
                  left_on=["away", "home"], right_on=["away_team", "home_team"])
    side = dk.side_name.map(abbr)
    total = dk.market == "total"
    dk["side"] = np.where(total, dk.side_name.str.split().str[0].str.lower(), side)
    return dk[dk.side.notna()][["game_id", "market", "side", "line", "odds", "handle_pct", "bets_pct"]]


def capture(source: str = "dk") -> pd.DataFrame:
    """Snapshot DraftKings splits for prime-time games that haven't kicked off."""
    from src.data_loader import load_schedules
    now = pd.Timestamp.now(tz="UTC")
    pt = update_games()
    sched = load_schedules()
    kick = kickoff_times(sched)
    games = sched[sched.game_id.isin(pt.game_id)]
    games = games[games.game_id.map(kick) > now]
    snap = _to_games(fetch_dk(), games, _team_lookup())
    snap = snap.assign(captured_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"), source=source)[SPLIT_COLS]
    if len(snap):
        old = load_splits()
        (pd.concat([old, snap], ignore_index=True) if len(old) else snap).to_csv(SPLITS_PATH, index=False)
    return snap


def load_splits() -> pd.DataFrame:
    if SPLITS_PATH.exists():
        return pd.read_csv(SPLITS_PATH)
    return pd.DataFrame(columns=SPLIT_COLS)


# ---------------------------------------------------------------- grading
def graded(splits: pd.DataFrame, pt_games: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """The last pre-kickoff snapshot of every prime-time game, each side graded.

    result: "won" / "lost" / "push", or "pending" until the game is final.
    """
    if splits.empty:
        return splits.assign(result=pd.Series(dtype=str))
    kick = kickoff_times(sched)
    s = splits.copy()
    s["captured_at"] = pd.to_datetime(s.captured_at, utc=True)
    s["kickoff"] = s.game_id.map(kick)
    s = s[s.captured_at < s.kickoff]
    last = s.groupby("game_id").captured_at.transform("max")
    s = s[s.captured_at == last]
    sc = sched.set_index("game_id")
    s = s.join(sc[["away_team", "home_team", "away_score", "home_score", "gameday"]], on="game_id")
    s = s.merge(pt_games[["game_id", "week", "slot"]], on="game_id", how="left")

    home_side = s.side == s.home_team
    margin = np.where(home_side, s.home_score - s.away_score, s.away_score - s.home_score)
    total = s.home_score + s.away_score
    edge = np.select([s.market == "moneyline", s.market == "spread", s.side == "over"],
                     [margin, margin + s.line, total - s.line], default=s.line - total)
    s["result"] = np.select([pd.isna(edge), edge > 0, edge < 0], ["pending", "won", "lost"], default="push")
    s["hours_before"] = (s.kickoff - s.captured_at).dt.total_seconds() / 3600
    return s.sort_values(["kickoff", "game_id", "market", "side"]).reset_index(drop=True)


def majority_record(g: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per game and market, the side holding the majority of `by` (handle_pct or bets_pct)
    and how it did. Even splits (50/50) are left out."""
    g = g[g.result != "pending"]
    lead = g[g[by] == g.groupby(["game_id", "market"])[by].transform("max")]
    return lead[lead.groupby(["game_id", "market"])[by].transform("size") == 1]


def disagreements(g: pd.DataFrame) -> pd.DataFrame:
    """Markets where the money and the tickets are on different sides: the side with the
    larger share of handle than of bets (fewer, bigger wagers) is returned."""
    g = g[g.result != "pending"]
    money, bets = majority_record(g, "handle_pct"), majority_record(g, "bets_pct")
    both = money.merge(bets[["game_id", "market", "side"]], on=["game_id", "market"], suffixes=("", "_bets"))
    return both[both.side != both.side_bets].drop(columns="side_bets")


def record(df: pd.DataFrame) -> str:
    w, l, p = (df.result == "won").sum(), (df.result == "lost").sum(), (df.result == "push").sum()
    rate = f" ({w / (w + l):.0%})" if w + l else ""
    return f"{w}-{l}" + (f"-{p}" if p else "") + rate


def report():
    from src.data_loader import load_schedules
    g = graded(load_splits(), pd.read_csv(GAMES_PATH), load_schedules())
    done = g[g.result != "pending"]
    print(f"{done.game_id.nunique()} graded prime-time games, {g[g.result == 'pending'].game_id.nunique()} pending")
    for m in MARKETS:
        gm = done[done.market == m]
        print(f"  {m:<9}  majority of bets: {record(majority_record(gm, 'bets_pct')):<14}"
              f"majority of money: {record(majority_record(gm, 'handle_pct')):<14}"
              f"money vs bets split, money side: {record(disagreements(gm))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print the record instead of capturing")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        snap = capture()
        print(f"Captured {snap.game_id.nunique()} prime-time game(s): {', '.join(sorted(snap.game_id.unique())) or '-'}")
