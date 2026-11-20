"""This Week markup: the ruled game list and the selected game's panel (header, win-probability
split, the factor "play" diagram and its caption). Approved as round 5/6 mockups in output/fourth-and-why.

Pure HTML/CSS: st.html runs no scripts and strips inline <svg>, so icons are data-URI images and the
play diagram goes through st.markdown. Team colors are lightened until they hold 3:1 on the ground.
"""
from html import escape

import pandas as pd

from theme import GOOD, GROUND, BAD, INK, MUTED, SOFT, svg_uri


def _lin(h: str) -> float:
    c = [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    x, y = sorted((_lin(a), _lin(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)


def _mix(c: str, w: float) -> str:
    r, g, b = (int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(round(v + (255 - v) * w) for v in (r, g, b))


def team_color(c: str) -> str:
    """The team's primary, lightened until it holds 3:1 against the ground as a mark."""
    try:
        w = 0.0
        while contrast(_mix(c, w), GROUND) < 3:
            w += 0.04
        return _mix(c, w)
    except (ValueError, AttributeError):
        return MUTED


def pair_colors(meta: dict, away: str, home: str) -> tuple:
    """(away, home) colors. When the two primaries look alike (Seahawks and Patriots both end up slate),
    the away team takes its secondary color so home and away never share one."""
    ca, ch = meta[away]["color"], meta[home]["color"]
    rgb = lambda c: [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    if sum((x - y) ** 2 for x, y in zip(rgb(ca), rgb(ch))) ** 0.5 < 80:
        ca = meta[away]["color2"]
    return ca, ch


def ink_on(bg: str) -> str:
    return GROUND if contrast(bg, GROUND) > contrast(bg, "#ffffff") else "#ffffff"


def _icon(ok: bool) -> str:
    c = GOOD if ok else BAD
    mark = ('<path d="M4.2 7.2l1.9 1.9 3.8-4" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>' if ok else
            '<path d="M4.8 4.8l4.4 4.4M9.2 4.8l-4.4 4.4" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>')
    inner = f'<circle cx="7" cy="7" r="6.2" fill="none" stroke="{c}" stroke-width="1.6"/>' + mark.format(c=c)
    return f'<img src="{svg_uri(inner, "0 0 14 14")}" width="14" height="14" alt="">'


_INFO = svg_uri('<circle cx="9" cy="9" r="8" fill="none" stroke="#e9c47c" stroke-width="1.5"/>'
                '<path d="M9 8v5M9 5.2v.1" stroke="#e9c47c" stroke-width="1.8" stroke-linecap="round"/>', "0 0 18 18")


def _logo(meta: dict, team: str, size: int) -> str:
    return f'<img src="{escape(str(meta.get(team, {}).get("logo", "")))}" width="{size}" height="{size}" alt="" loading="lazy">'


def pct(p: float) -> str:
    return f"{p * 100:.0f}"


def result(r) -> tuple:
    """(pick right?, final-score text, spread text) for a played game."""
    home_won = r.home_score > r.away_score
    right = (r.p_show >= 0.5) == home_won
    ats = ""
    if r.margin == r.margin and r.spread_line == r.spread_line:
        diff = (r.home_score - r.away_score) - r.spread_line
        side = r.home_team if r.margin > r.spread_line else r.away_team
        ok = (diff > 0) if r.margin > r.spread_line else (diff < 0)
        ats = "spread: push" if diff == 0 else f"spread pick {side} {'covered' if ok else 'did not cover'}"
    return right, f"Final: {r.away_team} {r.away_score:.0f}, {r.home_team} {r.home_score:.0f}", ats


def row_html(r, meta: dict, top: dict, tags: list, played: bool) -> str:
    """One game in the list: time, matchup, a 0-100 track (dot = model, short line = Vegas, tall line = 50%),
    the favorite's chance, the quarterbacks, and the biggest factor."""
    a, h = r.away_team, r.home_team
    fav, pf = (h, r.p_show) if r.p_show >= 0.5 else (a, 1 - r.p_show)
    when = "Final" if played else (r.kickoff.strftime("%a %-I:%M %p") if pd.notna(r.kickoff) else "")
    if played:
        hw = r.home_score > r.away_score
        side = lambda t, s, won: f'{_logo(meta, t, 22)}{t} <b class="{"w" if won else "l"}">{s:.0f}</b>'
        m = f'{side(a, r.away_score, not hw)} <span class="at">at</span> {side(h, r.home_score, hw)}'
        right = result(r)[0]
        why = f'<span class="res">{_icon(right)}Pick {"right" if right else "wrong"}</span>'
    else:
        m = f'{_logo(meta, a, 22)}{a} <span class="at">at</span> {_logo(meta, h, 22)}{h}'
        why = ""
    if top:
        why += f'<span>{top["favors"]} +{abs(top["pct_points"]) * 100:.1f} {escape(top["factor"])}</span>'
    why += "".join(f'<span class="tag{" alert" if alert else ""}">{escape(t)}</span>' for t, alert in tags)
    vegas = (f'<span class="v" style="left:{r.p_vegas * 100:.1f}%"></span>' if r.p_vegas == r.p_vegas else "")
    label = f"{h} {pct(r.p_show)}%" + (f", Vegas {h} {pct(r.p_vegas)}%" if r.p_vegas == r.p_vegas else "")
    return (f'<div class="gm{" played" if played else ""}"><span class="when">{when}</span><span class="m">{m}</span>'
            f'<span class="trk" role="img" aria-label="{label}"><span class="mid"></span>{vegas}'
            f'<span class="dot" style="left:{r.p_show * 100:.1f}%;background:{meta[fav]["color"]}"></span></span>'
            f'<span class="p">{fav} {pct(pf)}</span><span class="qb">{escape(f"{r.away_qb_name} vs {r.home_qb_name}")}</span>'
            f'<span class="why">{why}</span></div>')


def caption(e: dict, meta: dict) -> str:
    """One plain sentence on what carries the favorite and what the other side has."""
    h, a, ph = e["home_team"], e["away_team"], e["p_home"]
    fav, dog = (h, a) if ph >= 0.5 else (a, h)
    fs = [f for f in e["all_factors"] if abs(f["pct_points"]) >= 0.005]
    pro = [f["factor"].lower() for f in fs if f["favors"] == fav][:2]
    con = [f for f in fs if f["favors"] == dog]
    nick = lambda t: meta[t]["nick"]
    if not pro:
        return f"Nothing moves the model far from even: the {nick(fav)} edge it by a hair."
    lead = (" and ".join(pro)).capitalize()
    out = f"{lead} {'carry' if len(pro) > 1 else 'carries'} the {nick(fav)}."
    if con:
        pts = sum(abs(f["pct_points"]) for f in con) * 100
        names = [f["factor"].lower() for f in con[:3]]
        named = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        out += (f" The {nick(dog)}' edge{'s' if len(con) > 1 else ''} in {named} "
                f"{'add' if len(con) > 1 else 'adds'} up to {pts:.1f} points, not enough to swing it.")
    else:
        out += f" No factor worth half a point favors the {nick(dog)}."
    return out


def panel_html(r, e: dict, meta: dict, venue: str, tags: list, spread: dict, note: str, played: bool) -> str:
    """The selected game: when and where, the matchup, the win-probability split with Vegas, spreads,
    the model-vs-Vegas note, the caption, and the heading for the play diagram below it."""
    a, h = r.away_team, r.home_team
    when = r.kickoff.strftime("%a %b %-d · %-I:%M %p ET") if pd.notna(r.kickoff) else ""
    meta_line = " · ".join(x for x in (("Final · " + r.kickoff.strftime("%a %b %-d")) if played and pd.notna(r.kickoff)
                                       else when, venue) if x)
    out = [f'<div class="fw-panel"><p class="fw-meta">{escape(meta_line)}</p>',
           f'<h2 class="fw-h2">{meta[a]["nick"]} <span class="and">at</span> {meta[h]["nick"]}</h2>',
           f'<p class="fw-qbs">{escape(f"{r.away_qb_name} vs {r.home_qb_name}")}</p>']
    if played:
        right, final, ats = result(r)
        out.append(f'<p class="fw-final"><span class="res">{_icon(right)}Pick {"right" if right else "wrong"}</span>'
                   f'{final}{" · " + ats if ats else ""}</p>')
    if tags:
        out.append('<p class="fw-tags">' + "".join(f'<span class="tag{" alert" if al else ""}">{escape(t)}</span>'
                                                   for t, al in tags) + "</p>")
    pa, ph = (1 - r.p_show) * 100, r.p_show * 100
    ca, ch = pair_colors(meta, a, h)
    seg = lambda t, p, c, logo_first: (f'<div style="width:{p:.1f}%;background:{c};color:{ink_on(c)}">'
                                       + ((_logo(meta, t, 26) + f"{t} {p:.1f}") if logo_first else (f"{t} {p:.1f}" + _logo(meta, t, 26)))
                                       + "</div>") if p >= 14 else f'<div style="width:{p:.1f}%;background:{c}"></div>'
    vg, aria = "", f"{a} {pa:.1f}%, {h} {ph:.1f}%"
    if r.p_vegas == r.p_vegas:
        vfav, vp = (h, r.p_vegas) if r.p_vegas >= 0.5 else (a, 1 - r.p_vegas)
        left = (1 - r.p_vegas) * 100
        vg = (f'<span class="vg" style="left:{left:.1f}%"></span>'
              f'<span class="vgl" style="left:{left:.1f}%;transform:{"none" if left < 20 else "translateX(-100%)" if left > 80 else "translateX(-50%)"}">'
              f'Vegas {vfav} {vp * 100:.1f}</span>')
        aria += f", Vegas {vfav} {vp * 100:.1f}%"
    out.append(f'<div class="fw-split-wrap"><div class="fw-split" role="img" aria-label="{aria}">'
               f'{seg(a, pa, ca, True)}{seg(h, ph, ch, False)}{vg}</div></div>')
    if spread:
        out.append("<dl>" + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in spread.items()) + "</dl>"
                   '<p class="fw-fine">Cover chances are calibrated on 2012–2018: historically, disagreeing with Vegas has '
                   "been worth little (a 3-point gap ≈ 53%), and on 2019–2025 the model's spread did not beat the line.</p>")
    if note:
        out.append(f'<div class="fw-note" role="note"><img src="{_INFO}" width="18" height="18" alt=""><p>{note}</p></div>')
    out.append(f'<p class="fw-cap">{escape(caption(e, meta))}</p><h3 class="fw-h3">How the prediction is built</h3>'
               f'<p class="fw-fine">Starts at 50%: evenly matched teams at a neutral site. Each route is one factor, biggest '
               f'first; O marks a factor for {h}, X one for {a}. The dashed line is Vegas. Hover a route for the details.'
               + (" This game has been played: the routes use the current model, so the end point may differ slightly "
                  "from the probability recorded before kickoff." if played else "") + "</p></div>")
    return "".join(out)


def route_svg(w: pd.DataFrame, home: str, away: str, p_vegas: float, meta: dict) -> str:
    """The factor breakdown as a play diagram: from 50%, one route per factor, biggest first, each starting
    where the last one ended; O for a home factor, X for an away one. `w` is src.explain.waterfall()."""
    steps = w[w.factor != "Model probability"].reset_index(drop=True)
    final = float(w.end.iloc[-1]) * 100
    vals = [50.0, final, *(steps.start * 100), *(steps.end * 100)] + ([p_vegas * 100] if p_vegas == p_vegas else [])
    step = 10 if max(vals) - min(vals) > 40 else 5
    lo, hi = step * int((min(vals) - 2) // step), step * -int(-(max(vals) + 2) // step)
    W, x0, rowh, top = 760, 196, 36, 32
    X = lambda v: x0 + (v - lo) * (W - x0 - 96) / (hi - lo)
    n = len(steps)
    bottom = top + n * rowh + 6
    H = bottom + 50
    ca, ch = pair_colors(meta, away, home)
    t = lambda x, y, s, fill=MUTED, anchor="start", size=13, weight=400: (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')
    out = [f'<div class="fw-play"><svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="fw-play-t">'
           f'<title id="fw-play-t">Factor routes from 50% to {home} {final:.1f}%</title>']
    for v in range(lo, hi + 1, step):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="12" y2="{bottom}" stroke="{"#7c7365" if v == 50 else "#2e2822"}" '
                   f'stroke-width="{1.5 if v == 50 else 1}"/>' + t(X(v), bottom + 20, v, anchor="middle"))
    out.append(t(X(lo), H - 6, f"← {away}") + t((X(lo) + X(hi)) / 2, H - 6, f"{meta[home]['nick']} win chance, %", anchor="middle")
               + t(X(hi), H - 6, f"{home} →", anchor="end"))
    if p_vegas == p_vegas:
        out.append(f'<line x1="{X(p_vegas * 100):.1f}" x2="{X(p_vegas * 100):.1f}" y1="12" y2="{bottom}" stroke="{SOFT}" '
                   'stroke-dasharray="4 4"/>' + t(X(p_vegas * 100) + 6, 22, f"Vegas {p_vegas * 100:.1f}", SOFT))
    for k, s in steps.iterrows():
        y = top + 12 + k * rowh
        cur, nxt = s.start * 100, s.end * 100
        is_home = nxt >= cur
        col, team = (ch, home) if is_home else (ca, away)
        x1, x2 = X(cur), X(nxt)
        tip = f"{s.factor}: {team} +{abs(nxt - cur):.1f} points" + (f". {s.detail}" if s.detail else "")
        g = [f'<g class="rt"><title>{escape(tip)}</title><rect x="0" y="{y - rowh / 2:.1f}" width="{W}" height="{rowh}" rx="3"/>',
             t(0, y + 5, escape(s.factor), INK, size=14)]
        if abs(x2 - x1) >= 9:  # long enough for an arrowhead
            d = 1 if x2 > x1 else -1
            g.append(f'<path d="M{x1:.1f} {y}L{x2 - d * 5:.1f} {y}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>'
                     f'<path d="M{x2:.1f} {y}l{-d * 8} -4.5v9z" fill="{col}"/>')
        else:
            g.append(f'<path d="M{x1:.1f} {y}L{x2:.1f} {y}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
        g.append(f'<circle cx="{x1:.1f}" cy="{y}" r="4.5" fill="{GROUND}" stroke="{col}" stroke-width="2"/>' if is_home else
                 f'<path d="M{x1 - 4:.1f} {y - 4}l8 8m0-8l-8 8" stroke="{col}" stroke-width="2"/>')
        lx, anc = (max(x1, x2) + 12, "start") if is_home else (min(x1, x2) - 12, "end")
        g.append(t(lx, y + 5, f"{team} +{abs(nxt - cur):.1f}", SOFT, anc) + "</g>")
        out.append("".join(g))
        if k < n - 1:
            out.append(f'<line x1="{x2:.1f}" x2="{x2:.1f}" y1="{y + 6}" y2="{y + rowh - 6}" stroke="#4a4239" stroke-dasharray="2 3"/>')
    yb = top + 12 + n * rowh - 4
    out.append(f'<ellipse cx="{X(final):.1f}" cy="{yb}" rx="9" ry="6" fill="#8a5a33" stroke="{INK}" stroke-width="1"/>'
               + t(X(final) + 16, yb + 5, f"{home} {final:.1f}%", INK, size=14, weight=700) + "</svg></div>")
    return "".join(out)


def css(selected: str, picked: bool) -> str:
    """This Week styles. `selected` is the game whose row is highlighted; once a game has been picked,
    its panel moves above the list on narrow screens so the answer shows first."""
    order = ('@media (max-width:1180px){.st-key-tw [data-testid="stColumn"]:last-child{order:-1}.fw-lh{display:block}}'
             if picked else "")
    return f"""<style>
.st-key-tw [data-testid="stHorizontalBlock"]{{gap:44px;align-items:flex-start}}
.st-key-tw [data-testid="stColumn"]:first-child{{flex:0 0 min(500px,40%)!important;min-width:420px}}
.st-key-tw [data-testid="stColumn"]:last-child{{flex:1 1 0!important;min-width:0}}
.st-key-games{{gap:0;border-top:1px solid var(--rule)}}
[class*="st-key-g_"]{{position:relative;gap:0;border-bottom:1px solid var(--rule);border-radius:2px}}
[class*="st-key-g_"]:hover{{background:#1b1714}}
[class*="st-key-g_"]:has(button:focus-visible){{outline:2px solid var(--gold);outline-offset:-2px}}
[class*="st-key-gb_"]{{position:absolute;inset:0;width:auto!important}}
[class*="st-key-gb_"] :is(.stButton,button){{width:100%;height:100%}}
.stApp [class*="st-key-gb_"] button{{background:transparent;border:0;color:transparent;cursor:pointer;outline:none!important}}
.stApp [class*="st-key-gb_"] button p{{color:transparent}}
.st-key-g_{selected}{{background:#221d18!important;box-shadow:inset 0 0 0 1px var(--gold)}}
.gm{{display:grid;grid-template-columns:78px 1fr 104px 66px;gap:4px 12px;align-items:center;padding:12px 10px}}
.gm .when{{color:var(--muted);font-size:13px;line-height:1.3}}
.gm .m{{display:flex;align-items:center;gap:7px;font-weight:600;white-space:nowrap}}.gm .m img{{width:22px;height:22px}}
.gm .at{{color:var(--muted);font-weight:400}}
.gm.played .m b{{font-weight:700}}.gm.played .m b.l{{color:var(--muted);font-weight:500}}
.gm .qb{{grid-column:2/5;color:var(--muted);font-size:13px;margin-top:-2px}}
.gm .p{{text-align:right;font-weight:700}}
.gm .why{{grid-column:2/5;display:flex;flex-wrap:wrap;gap:6px;align-items:center;color:var(--soft);font-size:13px}}
.tag{{border:1px solid #3d362f;border-radius:999px;padding:1px 8px;color:var(--muted);font-size:13px}}
.tag.alert{{border-color:#6b5223;color:#e9c47c}}
.res{{display:inline-flex;align-items:center;gap:6px;color:var(--text);font-weight:600;margin-right:4px}}
.trk{{position:relative;height:16px}}.trk:before{{content:"";position:absolute;left:0;right:0;top:7px;height:2px;background:#3d362f}}
.trk .mid{{position:absolute;left:50%;top:2px;width:1px;height:12px;background:#7c7365}}
.trk .v{{position:absolute;top:3px;width:2px;height:10px;margin-left:-1px;background:var(--muted)}}
.trk .dot{{position:absolute;top:3px;width:10px;height:10px;margin-left:-5px;border-radius:50%}}
.fw-lh{{display:none;font:600 15px/1.4 var(--sans);margin:22px 0 8px}}
.fw-meta{{color:var(--muted);margin:0}}
.fw-h2{{font:500 56px/1.02 var(--serif);letter-spacing:-.025em;margin:6px 0 4px}}
.fw-qbs{{color:var(--soft);margin:0 0 12px}}
.fw-final{{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;color:var(--soft);margin:0 0 10px}}
.fw-tags{{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 6px}}
.fw-split-wrap{{padding-top:26px}}
.fw-split{{position:relative;display:flex;height:44px;border-radius:4px;font-weight:700}}
.fw-split div{{display:flex;align-items:center;gap:8px;padding:0 12px;white-space:nowrap;overflow:hidden}}
.fw-split div+div{{justify-content:flex-end}}.fw-split img{{width:26px;height:26px}}
.fw-split div:first-child{{border-radius:4px 0 0 4px}}.fw-split div:last-of-type{{border-radius:0 4px 4px 0}}
.fw-split .vg{{position:absolute;top:-7px;bottom:-7px;width:2px;margin-left:-1px;background:var(--text)}}
.fw-split .vgl{{position:absolute;bottom:calc(100% + 9px);transform:translateX(-50%);white-space:nowrap;color:var(--muted);font-size:13px;font-weight:400}}
.fw-panel dl{{display:flex;flex-wrap:wrap;gap:6px 30px;margin:18px 0 4px;padding:0}}.fw-panel dl div{{display:flex;gap:8px}}
.fw-panel dt{{color:var(--muted)}}.fw-panel dd{{font-weight:700;margin:0}}
.fw-fine{{color:var(--muted);font-size:13px;max-width:70ch;line-height:1.5;margin:6px 0 0}}
.fw-note{{display:grid;grid-template-columns:20px 1fr;gap:10px;margin:20px 0 0;padding:14px 16px;border:1px solid #4a3c22;
  border-radius:6px;background:#1c1812;color:var(--soft)}}.fw-note p{{margin:0;line-height:1.5}}.fw-note b{{color:var(--text)}}
.fw-cap{{font:italic 400 21px/1.45 var(--serif);color:var(--soft);margin:24px 0 0;max-width:58ch}}
.fw-h3{{font:500 27px/1.15 var(--serif);margin:34px 0 4px}}
@media (max-width:1180px){{.st-key-tw [data-testid="stHorizontalBlock"]{{flex-direction:column;gap:28px}}
  .st-key-tw [data-testid="stColumn"]{{width:100%!important;flex:1 1 auto!important;min-width:0!important}}}}
@media (max-width:860px){{.fw-h2{{font-size:40px}}}}
@media (max-width:480px){{.gm{{grid-template-columns:78px 1fr 58px}}.gm .when{{white-space:nowrap}}.gm .trk{{display:none}}
  .gm .why,.gm .qb{{grid-column:1/4}}.fw-h2{{font-size:34px}}}}
@media (prefers-reduced-motion:no-preference){{[class*="st-key-g_"]{{transition:background-color .15s ease-out}}}}
{order}
</style>"""
