"""Away @ Home matchup card: a dark split card where each team's logo fades in from
its own edge, lit by a glow in the team's color, with the win probabilities in the
middle. Adapted from the MLB project's docs/matchup-card.md.

Pure HTML/CSS (st.html doesn't run scripts). `card_css()` is injected once per page;
`card_html(...)` renders one card, full size (breakdown page) or compact (week grid).
"""
from html import escape


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def glow_color(c1: str, c2: str) -> str:
    """Team color that still shows on a near-black card. Dark primaries (Packers green,
    the navy teams) switch to the secondary color when it's brighter, otherwise lighten."""
    try:
        if _lum(c1) >= 0.16:
            return c1
        if c2 and _lum(c2) >= 0.16:
            return c2
        return f"color-mix(in srgb, {c1} 55%, white)"
    except (ValueError, AttributeError):
        return "#888888"


def card_css() -> str:
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800;900&display=swap');
.mc { --away-c:#888; --home-c:#888; position:relative; overflow:hidden; border-radius:16px; height:320px;
  user-select:none; font-family:"Barlow Condensed", sans-serif;
  background:linear-gradient(135deg,#0a0a12 0%,#111118 50%,#0a0a12 100%);
  box-shadow:0 0 0 1px rgba(255,255,255,.06), 0 16px 40px rgba(0,0,0,.35),
    0 0 60px color-mix(in srgb, var(--away-c) 13%, transparent),
    0 0 60px color-mix(in srgb, var(--home-c) 13%, transparent); }
.mc-grain { position:absolute; inset:0; opacity:.03; pointer-events:none; z-index:10; background-size:128px auto;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.mc-divider { position:absolute; top:0; bottom:0; left:50%; width:1px; transform:translateX(-50%); opacity:.2; z-index:5;
  background:linear-gradient(transparent, white 30%, white 70%, transparent); }
.mc-half { position:absolute; top:0; bottom:0; width:50%; display:flex; flex-direction:column;
  justify-content:flex-end; padding-bottom:28px; z-index:1; }
.mc-half.away { left:0; --c:var(--away-c); }
.mc-half.home { right:0; --c:var(--home-c); align-items:flex-end; }
.mc-glow { position:absolute; inset:0; }
.mc-half.away .mc-glow { background:radial-gradient(at 20% 50%, color-mix(in srgb, var(--c) 33%, transparent) 0%, transparent 70%); }
.mc-half.home .mc-glow { background:radial-gradient(at 80% 50%, color-mix(in srgb, var(--c) 33%, transparent) 0%, transparent 70%); }
.mc-logo-wrap { position:absolute; top:0; bottom:0; width:60%; display:flex; align-items:center; }
.mc-half.away .mc-logo-wrap { left:0; padding-left:32px;
  -webkit-mask-image:linear-gradient(to right,#000 0%,rgba(0,0,0,.85) 30%,rgba(0,0,0,.4) 60%,transparent 100%);
          mask-image:linear-gradient(to right,#000 0%,rgba(0,0,0,.85) 30%,rgba(0,0,0,.4) 60%,transparent 100%); }
.mc-half.home .mc-logo-wrap { right:0; padding-right:32px; justify-content:flex-end;
  -webkit-mask-image:linear-gradient(to left,#000 0%,rgba(0,0,0,.85) 30%,rgba(0,0,0,.4) 60%,transparent 100%);
          mask-image:linear-gradient(to left,#000 0%,rgba(0,0,0,.85) 30%,rgba(0,0,0,.4) 60%,transparent 100%); }
.mc-logo { width:208px; height:208px; object-fit:contain; opacity:.9;
  filter:drop-shadow(0 0 32px color-mix(in srgb, var(--c) 53%, transparent)); }
.mc-label { position:relative; z-index:10; padding:0 28px; display:flex; flex-direction:column; }
.mc-half.home .mc-label { align-items:flex-end; text-align:right; }
.mc-city { font-size:12px; line-height:16px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:rgba(255,255,255,.55); margin-bottom:2px; }
.mc-name { font-size:36px; line-height:1; font-weight:900; letter-spacing:-.025em; text-transform:uppercase; color:#fff; }
.mc-abbr { font-size:14px; line-height:20px; font-weight:700; letter-spacing:.1em; margin-top:4px; text-transform:uppercase;
  color:color-mix(in srgb, var(--c) 65%, white); filter:drop-shadow(0 0 8px var(--c)); }
.mc-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
  justify-content:center; gap:6px; z-index:8; pointer-events:none; }
.mc-pill { padding:2px 12px; border-radius:999px; font-size:12px; line-height:16px; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
  color:rgba(255,255,255,.5); margin-bottom:4px; }
.mc-odds { display:grid; grid-template-columns:1fr 1fr; gap:36px; align-items:baseline; }
.mc-pct { font-size:44px; font-weight:800; line-height:1; color:rgba(255,255,255,.75); font-variant-numeric:tabular-nums; }
.mc-pct.away { text-align:right; --c:var(--away-c); }
.mc-pct.home { --c:var(--home-c); }
.mc-pct.fav { color:#fff; text-shadow:0 0 18px color-mix(in srgb, var(--c) 70%, white); }
.mc-date { font-size:14px; line-height:20px; font-weight:700; letter-spacing:.1em; color:#fff; opacity:.9;
  text-transform:uppercase; margin-top:4px; }
.mc-sub { font-size:13px; line-height:16px; font-weight:600; letter-spacing:.05em; color:rgba(255,255,255,.5); }
.mc-venue { position:absolute; bottom:0; left:0; right:0; display:flex; justify-content:center; padding-bottom:12px;
  z-index:9; font-size:12px; line-height:16px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:rgba(255,255,255,.3); }

/* compact: the week grid */
.mc.compact { height:150px; border-radius:12px; margin-bottom:4px; }
.mc.compact .mc-half { padding-bottom:14px; }
.mc.compact .mc-logo { width:104px; height:104px; filter:drop-shadow(0 0 18px color-mix(in srgb, var(--c) 53%, transparent)); }
.mc.compact .mc-half.away .mc-logo-wrap { padding-left:10px; }
.mc.compact .mc-half.home .mc-logo-wrap { padding-right:10px; }
.mc.compact .mc-label { padding:0 14px; }
.mc.compact .mc-city { font-size:10px; line-height:12px; }
.mc.compact .mc-name { font-size:22px; }
.mc.compact .mc-abbr { font-size:11px; line-height:14px; margin-top:2px; }
.mc.compact .mc-center { justify-content:flex-start; padding-top:18px; gap:2px; }
.mc.compact .mc-pct { font-size:30px; }
.mc.compact .mc-odds { gap:22px; }
.mc.compact .mc-date { font-size:11px; line-height:14px; }

@media (max-width: 720px) {
  .mc:not(.compact) { height:290px; }
  .mc:not(.compact) .mc-center { justify-content:flex-start; padding-top:22px; }
  .mc:not(.compact) .mc-logo { width:140px; height:140px; }
  .mc:not(.compact) .mc-half.away .mc-logo-wrap { padding-left:12px; }
  .mc:not(.compact) .mc-half.home .mc-logo-wrap { padding-right:12px; }
  .mc:not(.compact) .mc-label { padding:0 16px; }
  .mc:not(.compact) .mc-pct { font-size:30px; }
  .mc:not(.compact) .mc-odds { gap:24px; }
  .mc:not(.compact) .mc-name { font-size:26px; }
  .mc-venue { display:none; }
}
</style>"""


def _half(side: str, t: dict, abbr: str) -> str:
    logo = f'<img class="mc-logo" src="{escape(t.get("logo_dark") or t.get("logo", ""))}" alt="">' if t.get("logo") else ""
    return (f'<div class="mc-half {side}"><div class="mc-glow"></div><div class="mc-logo-wrap">{logo}</div>'
            f'<div class="mc-label"><span class="mc-city">{escape(t.get("city", ""))}</span>'
            f'<span class="mc-name">{escape(t.get("nick", abbr))}</span>'
            f'<span class="mc-abbr">{escape(abbr)}</span></div></div>')


def card_html(away: str, home: str, meta: dict, p_home: float, date: str = "", sub: str = "",
              venue: str = "", pill: str = "Win probability", compact: bool = False) -> str:
    a, h = meta.get(away, {}), meta.get(home, {})
    fmt = (lambda x: f"{x:.0%}") if compact else (lambda x: f"{x:.1%}")
    fav_home = p_home >= 0.5
    center = [f'<span class="mc-pill">{escape(pill)}</span>' if pill and not compact else "",
              f'<div class="mc-odds"><span class="mc-pct away{"" if fav_home else " fav"}">{fmt(1 - p_home)}</span>'
              f'<span class="mc-pct home{" fav" if fav_home else ""}">{fmt(p_home)}</span></div>',
              f'<span class="mc-date">{escape(date)}</span>' if date else "",
              f'<span class="mc-sub">{escape(sub)}</span>' if sub else ""]
    return (f'<div class="mc{" compact" if compact else ""}" style="--away-c:{a.get("glow", "#888")};'
            f'--home-c:{h.get("glow", "#888")}">'
            f'{_half("away", a, away)}{_half("home", h, home)}<div class="mc-divider"></div>'
            f'<div class="mc-center">{"".join(center)}</div>'
            + (f'<div class="mc-venue">{escape(venue)}</div>' if venue and not compact else "")
            + '<div class="mc-grain"></div></div>')
