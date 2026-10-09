"""Prime-time betting splits: how much of the money and how many of the bets went on
each side of TNF, SNF and MNF games, and which side won.

Sources:
  - Which games are prime time: Champs or Chumps' TNF/SNF/MNF schedule pages
    (champsorchumps.us/nfl). Falls back to the nflverse schedule (Thu/Sun/Mon night
    kickoffs) if the site can't be reached.
  - Historical splits: Action Network's public betting data (actionnetwork.com/public-betting),
    consensus % of money and % of tickets for the moneyline, spread and total of finished
    games, 2023 onward. Saved to reports/prime_time_public.csv.
  - Live splits: the DraftKings Sportsbook Betting Splits table (dknetwork.draftkings.com),
    which shows, for the moneyline, spread and total of each upcoming game,
    % Handle (share of the money wagered) and % Bets (share of the number of wagers),
    all jurisdictions combined.

DraftKings only shows upcoming games, so its splits must be captured before kickoff.
Every capture appends a snapshot to reports/prime_time_splits.csv; the last snapshot
taken before kickoff is the one graded. Action Network keeps finished games, so those
are fetched after the game. Results are graded from nflverse final scores against the
source's line (the spread or total the bettors got).

"The public" is the side with the majority of bets (tickets). When it loses, the
sportsbooks win: "Vegas won".

Usage:
    python -m src.splits             # capture DraftKings splits + add finished games from Action Network
    python -m src.splits --public 2023   # backfill Action Network splits from 2023 on
    python -m src.splits --report    # record of the public side vs the money side
"""
import argparse
import re
import time
from html import unescape

import numpy as np
import pandas as pd
import requests

from src.config import CURRENT_SEASON, ROOT

DK_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
DK_NFL = "88808"  # DraftKings event group for the NFL
AN_URL = "https://api.actionnetwork.com/web/v2/scoreboard/publicbetting/nfl"
AN_BOOK = "15"  # Action Network's consensus book
AN_FIRST_SEASON = 2023  # earlier seasons have no splits
COC_URL = "https://champsorchumps.us/nfl/{}"
SLOT_PAGES = {"TNF": "thursday-night-football-games", "SNF": "sunday-night-football-games",
              "MNF": "monday-night-football-games"}
SLOT_DAYS = {"Thursday": "TNF", "Sunday": "SNF", "Monday": "MNF"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"}

SPLITS_PATH = ROOT / "reports" / "prime_time_splits.csv"
PUBLIC_PATH = ROOT / "reports" / "prime_time_public.csv"
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


# ---------------------------------------------------------------- Action Network (historical)
def _get_json(url: str, tries: int = 4, **params) -> dict:
    """GET with backoff; the API answers bursts with 504s or empty bodies."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, "Accept": "application/json"}, timeout=60)
            if r.ok and r.text.strip():
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    raise requests.RequestException(f"no response from {url} {params}")


def parse_action(data: dict, lookup: dict) -> pd.DataFrame:
    """One row per finished game, market and side: consensus % money and % tickets."""
    rows = []
    for g in data.get("games", []):
        if g.get("status") != "complete":
            continue
        team = {t["id"]: lookup.get(t["full_name"]) for t in g.get("teams", [])}
        event = g.get("markets", {}).get(AN_BOOK, {}).get("event", {})
        for market in MARKETS:
            for o in event.get(market, []):
                info = o.get("bet_info") or {}
                rows.append({"season": g["season"], "week": g["week"], "away": team.get(g["away_team_id"]),
                             "home": team.get(g["home_team_id"]), "market": market,
                             "side": o["side"] if market == "total" else team.get(o.get("team_id")),
                             "line": o.get("value") if market != "moneyline" else np.nan, "odds": o.get("odds"),
                             "handle_pct": (info.get("money") or {}).get("percent"),
                             "bets_pct": (info.get("tickets") or {}).get("percent")})
    df = pd.DataFrame(rows, columns=["season", "week", "away", "home", "market", "side", "line", "odds",
                                     "handle_pct", "bets_pct"])
    return clean_action(df, ["season", "week", "away", "home", "market"])


def clean_action(df: pd.DataFrame, market_key: list) -> pd.DataFrame:
    """Drop sides that aren't a team or over/under (some moneylines list a draw) and markets
    with no splits recorded (every side at 0% of money and bets)."""
    df = df[df.side.notna()]
    empty = df.assign(z=(df.handle_pct.fillna(0) == 0) & (df.bets_pct.fillna(0) == 0)) \
        .groupby(market_key).z.transform("all")
    return df[~empty]


def load_public() -> pd.DataFrame:
    if PUBLIC_PATH.exists():
        return pd.read_csv(PUBLIC_PATH)
    return pd.DataFrame(columns=SPLIT_COLS)


def update_public(seasons, pause: float = 3.0) -> pd.DataFrame:
    """Add Action Network splits for finished prime-time games not saved yet.
    Only weeks that have such games are requested."""
    from src.data_loader import load_schedules
    sched = load_schedules(refresh=True)  # needs the latest final scores
    lookup = _team_lookup()
    stored = load_public()
    new = []
    for season in seasons:
        pt = update_games(season)
        pt = pt[pt.season == season].merge(sched[["game_id", "away_team", "home_team", "home_score"]], on="game_id")
        todo = pt[pt.home_score.notna() & ~pt.game_id.isin(stored.game_id)]
        for week in sorted(todo.week.unique()):
            an = parse_action(_get_json(AN_URL, bookIds=AN_BOOK, season=season, week=int(week),
                                        seasonType="reg"), lookup)
            # match on the pair of teams: neutral-site games can be listed home/away the other way round
            an["pair"] = ["-".join(sorted(map(str, t))) for t in zip(an.away, an.home)]
            wk = todo[todo.week == week]
            wk = wk.assign(pair=["-".join(sorted(t)) for t in zip(wk.away_team, wk.home_team)])
            hit = an.merge(wk[["game_id", "pair"]], on="pair")
            new.append(hit[hit.bets_pct.notna()])
            time.sleep(pause)
    if not new or not sum(len(n) for n in new):
        return stored.iloc[0:0]
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    add = pd.concat(new, ignore_index=True).assign(captured_at=now, source="action_network")[SPLIT_COLS]
    (pd.concat([stored, add], ignore_index=True) if len(stored) else add).to_csv(PUBLIC_PATH, index=False)
    return add


# ---------------------------------------------------------------- grading
def graded(splits: pd.DataFrame, pt_games: pd.DataFrame, sched: pd.DataFrame,
           before_kickoff: bool = True) -> pd.DataFrame:
    """The last snapshot of every prime-time game, each side graded.

    before_kickoff: only snapshots taken before kickoff count (DraftKings captures).
    Action Network's numbers for finished games are already as of kickoff, so pass False.
    result: "won" / "lost" / "push", or "pending" until the game is final.
    """
    if splits.empty:
        return splits.assign(result=pd.Series(dtype=str))
    kick = kickoff_times(sched)
    s = splits.copy()
    s["captured_at"] = pd.to_datetime(s.captured_at, utc=True)
    s["kickoff"] = s.game_id.map(kick)
    if before_kickoff:
        s = s[s.captured_at < s.kickoff]
    last = s.groupby("game_id").captured_at.transform("max")
    s = s[s.captured_at == last]
    sc = sched.set_index("game_id")
    s = s.join(sc[["away_team", "home_team", "away_score", "home_score", "gameday"]], on="game_id")
    s = s.merge(pt_games[["game_id", "season", "week", "slot"]], on="game_id", how="left")

    home_side = s.side == s.home_team
    margin = np.where(home_side, s.home_score - s.away_score, s.away_score - s.home_score)
    total = s.home_score + s.away_score
    edge = np.select([s.market == "moneyline", s.market == "spread", s.side == "over"],
                     [margin, margin + s.line, total - s.line], default=s.line - total)
    s["result"] = np.select([pd.isna(edge), edge > 0, edge < 0], ["pending", "won", "lost"], default="push")
    s["hours_before"] = (s.kickoff - s.captured_at).dt.total_seconds() / 3600 if before_kickoff else np.nan
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


def units(df: pd.DataFrame) -> float:
    """Profit from betting 1 unit on every row's side at its listed odds (pushes return the stake)."""
    odds = df.odds.astype(float)
    win = np.where(odds > 0, odds / 100, 100 / odds.abs())
    return float(np.select([df.result == "won", df.result == "lost"], [win, -1.0], 0.0).sum())


def win_rate(df: pd.DataFrame) -> float:
    w, l = (df.result == "won").sum(), (df.result == "lost").sum()
    return w / (w + l) if w + l else np.nan


def record(df: pd.DataFrame, rate: bool = True) -> str:
    w, l, p = (df.result == "won").sum(), (df.result == "lost").sum(), (df.result == "push").sum()
    pct = f" ({win_rate(df):.0%})" if rate and w + l else ""
    return f"{w}-{l}" + (f"-{p}" if p else "") + pct


def report():
    from src.data_loader import load_schedules
    sched, pt = load_schedules(), pd.read_csv(GAMES_PATH)
    for name, splits, before in [("Action Network", load_public(), False), ("DraftKings", load_splits(), True)]:
        g = graded(splits, pt, sched, before_kickoff=before)
        done = g[g.result != "pending"] if len(g) else g
        print(f"{name}: {done.game_id.nunique() if len(done) else 0} graded prime-time games")
        for m in MARKETS:
            gm = done[done.market == m] if len(done) else done
            if len(gm):
                print(f"  {m:<9}  public side (most bets): {record(majority_record(gm, 'bets_pct')):<16}"
                      f"money side: {record(majority_record(gm, 'handle_pct')):<16}"
                      f"money vs bets split, money side: {record(disagreements(gm))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print the record instead of capturing")
    ap.add_argument("--public", type=int, metavar="FIRST_SEASON",
                    help=f"backfill Action Network splits from this season on (earliest {AN_FIRST_SEASON})")
    args = ap.parse_args()
    if args.report:
        report()
    elif args.public:
        add = update_public(range(max(args.public, AN_FIRST_SEASON), CURRENT_SEASON + 1))
        print(f"Added Action Network splits for {add.game_id.nunique()} prime-time game(s)")
    else:
        snap = capture()
        print(f"Captured {snap.game_id.nunique()} prime-time game(s): {', '.join(sorted(snap.game_id.unique())) or '-'}")
        try:
            add = update_public([CURRENT_SEASON])
            print(f"Action Network: added {add.game_id.nunique()} finished prime-time game(s)")
        except requests.RequestException as e:
            print(f"Action Network unavailable ({e}); will retry next run.")
