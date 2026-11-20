"""Fourth and Why: the design system in one place.

Colors, fonts and page CSS for Streamlit, the Altair chart style, and the formation navigation:
every section is a player in an offense at the line of scrimmage. The look was approved as static
mockups (output/fourth-and-why, rounds 2-6); this module reproduces it with native Streamlit
widgets, because st.html runs no scripts and links would reload the in-browser (stlite) app.
"""
import math
import re
from urllib.parse import quote

import altair as alt
import streamlit as st

# ---------------------------------------------------------------- tokens
GROUND, PANEL, RAIL, RULE, AXIS = "#14110f", "#1d1915", "#181411", "#2e2822", "#4a4239"
INK, SOFT, MUTED, TICK, GOLD = "#f1e9dc", "#d9cfbf", "#b3a897", "#8f8676", "#e3a83c"
REF = "#6f675b"  # dashed reference lines: league average, 50%, perfect calibration
# Series colors, checked with the data-viz palette validator on GROUND (dark mode, all pairs).
MODEL, VEGAS, ELO, OTHER = "#3987e5", "#d95926", "#199e70", TICK
HOME, AWAY = "#3987e5", "#e66767"
GOOD, BAD = "#0ca30c", "#e66767"
SANS = '"Hanken Grotesk", system-ui, sans-serif'
SERIF = '"Source Serif 4", Georgia, serif'
FONTS = ("https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400..700;"
         "1,8..60,400..700&family=Hanken+Grotesk:wght@400;600;700&display=swap")

# The mark: a receiver's route curls into a question mark, and the player's O is its dot.
MARK_PATHS = ('<path d="M16 20.2V17C16 13.6 22.2 13.4 22.2 8.9A6.2 6.2 0 0 0 9.8 8.9" fill="none" '
              f'stroke="{GOLD}" stroke-width="2.6" stroke-linecap="round"/><path d="M7 8.6H12.6L9.8 12.9Z" '
              f'fill="{GOLD}"/><circle cx="16" cy="25.6" r="3" fill="none" stroke="{INK}" stroke-width="2.2"/>')


def svg_uri(inner: str, viewbox: str) -> str:
    """st.html strips inline <svg>, so static drawings go in as data-URI images."""
    return "data:image/svg+xml," + quote(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">{inner}</svg>')


_MARK_URI = svg_uri(MARK_PATHS, "0 0 32 32")
MARK = f'<img class="lm" src="{_MARK_URI}" alt="">'
WORD = 'Fourth <span class="and">and</span> Why'

# ---------------------------------------------------------------- formation
# section, position, position name, home-field (x, y), strip (x, y), assignment drawn on the home field
FORMATION = [
    ("Home", "C", "Center", (720, 452), (550, 30), "block"),
    ("This Week", "QB", "Quarterback", (720, 566), (550, 64), "M720 566Q760 280 1008 262"),
    ("Overview", "LT", "Left tackle", (528, 452), (330, 30), "block"),
    ("Teams", "X", "Split end", (170, 452), (24, 30), "M170 438V215"),
    ("Team Analytics", "Z", "Flanker", (1250, 500), (970, 64), "M1250 486V340L1140 228"),
    ("Games", "LG", "Left guard", (624, 452), (440, 30), "block"),
    ("Players", "RB", "Running back", (816, 566), (660, 64), "M830 566Q930 586 1060 520"),
    ("Situational", "RG", "Right guard", (816, 452), (660, 30), "block"),
    ("Matchups", "TE", "Tight end", (1008, 452), (900, 30), "M1008 438V225"),
    ("Prime Time", "Slot", "Slot receiver", (340, 500), (150, 64), "M340 486V440L476 326"),
    ("Model", "RT", "Right tackle", (912, 452), (790, 30), "block"),
]
SECTIONS = [f[0] for f in FORMATION]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _nav_css() -> str:
    """Per-player positions on the strip and the home field, and the routes drawn on hover."""
    out = []
    for name, pos, pname, (hx, hy), (sx, sy), _ in FORMATION:
        s = slug(name)
        out.append(f".st-key-s_{s}{{left:{sx / 1110 * 100:.2f}%;top:{sy / 84 * 100:.2f}%}}"
                   f'.st-key-s_{s} button::after{{content:"{pos}"}}'
                   f".st-key-h_{s}{{left:{hx / 1440 * 100:.2f}%;top:{(hy - 130) / 510 * 100:.2f}%}}"
                   f'.st-key-h_{s} button::after{{content:"{pname}"}}'
                   f".st-key-field:has(.st-key-h_{s} button:is(:hover,:focus-visible)) .r-{s} .route{{stroke-dashoffset:0}}"
                   f".st-key-field:has(.st-key-h_{s} button:is(:hover,:focus-visible)) .r-{s} .tip{{opacity:1}}")
    return "".join(out)


CSS = f"""
@import url("{FONTS}");
:root{{--ground:{GROUND};--panel:{PANEL};--rail:{RAIL};--rule:{RULE};--axis:{AXIS};--text:{INK};--soft:{SOFT};
  --muted:{MUTED};--tick:{TICK};--gold:{GOLD};--serif:{SERIF};--sans:{SANS};--pad:44px}}
html,body,.stApp{{background:var(--ground);color:var(--text)}}
.stApp{{font-family:var(--sans);font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}}
.stApp :is(p,li,label,input,textarea,button,table,summary){{font-family:var(--sans)}}
::selection{{background:var(--gold);color:var(--ground)}}
*{{scrollbar-color:var(--axis) transparent}}
.stApp :focus-visible{{outline:2px solid var(--gold)!important;outline-offset:2px}}
[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],[data-testid="stHeaderActionElements"]{{display:none!important}}
.stMainBlockContainer,.block-container{{padding:0 var(--pad) 64px!important;max-width:none!important}}

/* type */
.stApp :is(h1,h2,h3,h4){{color:var(--text);padding:0}}
.stApp [data-testid="stHeading"] :is(h2,h3),.stApp [data-testid="stMarkdownContainer"] h3{{
  font:500 27px/1.15 var(--serif)!important;letter-spacing:0;margin:30px 0 0}}
.stApp [data-testid="stCaptionContainer"]{{color:var(--soft);font-size:15px;line-height:1.5;max-width:80ch}}
.stApp [data-testid="stCaptionContainer"] p{{font-size:15px;line-height:1.5}}
.stApp [data-testid="stMarkdownContainer"] :is(p,li){{line-height:1.5}}
.stApp a{{color:var(--text);text-underline-offset:3px;text-decoration-color:var(--axis)}}
.stApp a:hover{{text-decoration-color:var(--gold)}}
.fw-top{{padding:30px 0 4px}}
.fw-top h1{{font:500 46px/1 var(--serif);letter-spacing:-.02em;margin:0}}
.fw-top p{{color:var(--muted);margin:8px 0 0;max-width:80ch;line-height:1.5}}
.and{{font-weight:400;color:var(--muted)}}
.lm{{flex:none;width:1.15em;height:1.15em}}

/* stat tiles: a row of metrics between two hairlines */
[data-testid="stHorizontalBlock"]:has(>[data-testid="stColumn"] [data-testid="stMetric"]){{
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:16px 0 18px}}
[data-testid="stMetricLabel"] p{{color:var(--muted);font-size:13px!important}}
[data-testid="stMetricValue"]{{font:600 34px/1.15 var(--sans)!important;font-variant-numeric:normal;color:var(--text)}}
[data-testid="stMetricDelta"]{{font-size:13px;color:var(--soft)!important;background:none!important;padding:0;margin-top:2px}}
[data-testid="stMetricDelta"] svg{{display:none}}

.fw-legend{{display:flex;flex-wrap:wrap;gap:6px 20px;color:var(--soft);font-size:13px;margin-top:4px}}
.fw-legend span{{display:inline-flex;align-items:center;gap:7px}}
.fw-legend .dk{{width:10px;height:10px;border-radius:50%;background:var(--soft)}}.fw-legend .dk.ring{{background:none;border:2px solid var(--soft)}}

/* controls */
.stApp [data-testid="stWidgetLabel"] p{{color:var(--muted);font-size:13px}}
.stApp [data-baseweb="select"]>div{{background:var(--panel);border-color:var(--rule);border-radius:6px}}
.stApp [data-baseweb="select"]>div:hover{{border-color:var(--axis)}}
.stApp [data-baseweb="select"] [title],.stApp [data-baseweb="select"] input{{font-weight:600}}
[data-baseweb="popover"] li{{font-family:var(--sans)}}
.stApp .stButton>button{{background:var(--panel);border:1px solid var(--rule);color:var(--text);border-radius:6px}}
.stApp .stButton>button:hover{{border-color:var(--axis);color:var(--text)}}
/* horizontal radios become a segmented control, the selected choice underlined in gold */
.stApp [data-testid="stRadio"] [role="radiogroup"]:has(>label:nth-child(2)){{flex-direction:row;display:inline-flex;gap:2px;
  border:1px solid var(--rule);border-radius:6px;padding:2px}}
.stApp [data-testid="stRadio"] [role="radiogroup"]>label{{margin:0;padding:6px 12px;border-radius:4px;color:var(--muted);cursor:pointer}}
.stApp [data-testid="stRadio"] [role="radiogroup"]>label>div:first-child{{display:none}}
.stApp [data-testid="stRadio"] [role="radiogroup"]>label:hover{{color:var(--text);background:#211c18}}
.stApp [data-testid="stRadio"] [role="radiogroup"]>label:has(input:checked){{color:var(--text);background:#241f1a;box-shadow:inset 0 -2px 0 var(--gold)}}
.stApp [data-testid="stRadio"] [role="radiogroup"]>label:has(input:focus-visible){{outline:2px solid var(--gold);outline-offset:1px}}
.stApp [data-testid="stExpander"] details{{border-color:var(--rule);border-radius:6px}}
.stApp [data-testid="stExpander"] summary:hover{{color:var(--text)}}

/* notes: a quiet warm box, never a colored side stripe */
.stApp [data-testid="stAlertContainer"]{{background:#1c1812;border:1px solid #4a3c22;color:var(--soft);border-radius:6px}}
.stApp [data-testid="stAlertContainer"] :is(p,[data-testid="stMarkdownContainer"]){{color:var(--soft)}}
.stApp [data-testid="stAlertContainer"] strong{{color:var(--text)}}

/* the factor "play" diagram (This Week, Games): scrolls sideways on phones so its labels stay readable */
.fw-play{{overflow-x:auto;margin-top:4px}}.fw-play svg{{display:block;width:100%;min-width:560px;height:auto;font-family:var(--sans)}}
.fw-play .rt rect{{fill:transparent}}.fw-play .rt:hover rect{{fill:#1b1714}}

/* chart tooltips */
#vg-tooltip-element{{background:var(--panel)!important;border:1px solid var(--axis)!important;color:var(--text)!important;
  font:13px/1.4 var(--sans)!important;border-radius:6px!important;box-shadow:0 8px 24px rgba(0,0,0,.45)!important;padding:8px 10px!important}}
#vg-tooltip-element td.key{{color:var(--muted)!important}}

/* masthead: the wordmark and the formation strip */
.st-key-mast{{flex-direction:row;align-items:center;gap:36px;margin:0 calc(-1*var(--pad)) 0;padding:12px var(--pad) 8px;
  width:calc(100% + 2*var(--pad))!important;max-width:none;
  background:var(--rail);border-bottom:1px solid var(--rule)}}
.st-key-mast>*{{width:auto!important}}.st-key-mast>:not(.st-key-mark){{flex:1;min-width:0}}
.st-key-mark button,[class*="st-key-s_"] button,[class*="st-key-h_"] button{{background:none!important;border:0!important;
  box-shadow:none!important;min-height:0;border-radius:4px}}
.st-key-mark button{{padding:2px 0;gap:0}}
.st-key-mark button p{{font:500 24px/1.05 var(--serif)!important;letter-spacing:-.01em;white-space:nowrap;display:flex;align-items:center;gap:.3em}}
.st-key-mark button p::before{{content:"";width:1.15em;height:1.15em;background:url("{_MARK_URI}") center/contain no-repeat}}
.st-key-mark button em{{font-style:normal;font-weight:400;color:var(--muted)}}
.st-key-strip{{position:relative;width:100%;max-width:1110px;aspect-ratio:1110/84;gap:0}}
.st-key-strip::before{{content:"";position:absolute;left:0;right:0;top:14%;border-top:1.5px solid #5a5148}}
[class*="st-key-s_"]{{position:absolute;width:auto!important;transform:translate(-11px,-50%)}}
[class*="st-key-s_"] button{{padding:3px 6px 3px 22px;color:var(--muted)!important;position:relative}}
[class*="st-key-s_"] button p{{font:400 14px/1.2 var(--sans)!important;white-space:nowrap}}
[class*="st-key-s_"] button::before{{content:"";position:absolute;left:4px;top:50%;width:13px;height:13px;margin-top:-6.5px;
  border-radius:50%;border:2px solid var(--tick);background:var(--ground);box-sizing:border-box}}
[class*="st-key-s_"] button::after{{display:none;color:var(--tick);font-size:12px;margin-left:6px}}
[class*="st-key-s_"] button:hover{{color:var(--text)!important}}
[class*="st-key-s_"] button:hover::before{{border-color:var(--gold)}}
[class*="st-key-s_"] button:disabled{{color:var(--text)!important;opacity:1;cursor:default}}
[class*="st-key-s_"] button:disabled p{{font-weight:700!important}}
[class*="st-key-s_"] button:disabled::before{{background:var(--gold);border-color:var(--gold)}}
.st-key-s_home button p::before{{content:"";position:absolute;left:5.5px;top:-6px;width:12px;height:8px;border-radius:50%;
  background:#8a5a33;border:.8px solid var(--text);box-sizing:border-box}}

/* home: the whole offense at the line; each player opens a section */
.fw-hero{{display:grid;grid-template-columns:1fr auto;gap:12px 48px;align-items:end;padding:40px 20px 0}}
.fw-hero h1{{font:500 84px/.95 var(--serif);letter-spacing:-.03em;display:flex;align-items:center;gap:.22em;margin:0}}
.fw-hero p{{color:var(--soft);font-size:18px;line-height:1.5;max-width:52ch;margin:16px 0 0}}
.fw-week{{border:1px solid var(--rule);border-radius:6px;padding:14px 18px;color:var(--soft);min-width:300px;max-width:360px;line-height:1.5}}
.fw-week b{{display:block;color:var(--text);font:500 26px/1.1 var(--serif);margin-bottom:4px}}
.fw-hint{{grid-column:1/3;color:var(--muted);margin:6px 0 0}}
.st-key-field{{position:relative;aspect-ratio:1440/510;margin:-8px calc(-1*var(--pad)) 0;gap:0}}
.st-key-field>div{{position:absolute;width:auto!important}}
.st-key-field>div:first-child{{inset:0}}
.st-key-field svg{{display:block;width:100%;height:100%}}.st-key-field>div:first-child *{{margin:0}}
.fm .ghost{{fill:none;stroke:#3d362f;stroke-width:2}}
.fm .route{{fill:none;stroke:var(--gold);stroke-width:3;stroke-dasharray:1;stroke-dashoffset:1}}
.fm .tip{{fill:var(--gold);opacity:0}}.fm .gtip{{fill:#3d362f}}
[class*="st-key-h_"]{{transform:translate(-50%,-17px)}}
[class*="st-key-h_"] button{{padding:0;color:var(--text)!important;display:flex;flex-direction:column;align-items:center;gap:0}}
[class*="st-key-h_"] button::before{{content:"";width:31px;height:31px;border-radius:50%;border:2.5px solid var(--tick);
  background:var(--ground);box-sizing:border-box;margin-bottom:6px}}
[class*="st-key-h_"] button p{{font:600 16px/1.3 var(--sans)!important;white-space:nowrap}}
[class*="st-key-h_"] button::after{{color:var(--tick);font-size:13px;line-height:1.4}}
[class*="st-key-h_"] button:hover::before{{border-color:var(--gold)}}
[class*="st-key-h_"] button:hover::after{{color:var(--muted)}}
.st-key-h_home button::before{{background:var(--gold);border-color:var(--gold)}}
.fw-foot{{color:var(--muted);font-size:13px;margin-top:48px;padding-top:16px;border-top:1px solid var(--rule)}}
@media (prefers-reduced-motion:no-preference){{
  .fm .route{{transition:stroke-dashoffset .35s cubic-bezier(.16,1,.3,1)}}.fm .tip{{transition:opacity .1s .25s}}
  [class*="st-key-s_"] button,[class*="st-key-s_"] button::before,[class*="st-key-h_"] button::before{{transition:color .15s ease-out,border-color .15s ease-out}}}}
/* below 1100px the field would shrink its labels under 12px, so the players become a roster */
@media (max-width:1100px){{
  .fw-hero h1{{font-size:68px}}.fw-hov{{display:none}}
  .st-key-field{{aspect-ratio:auto;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:24px 0 0}}
  .st-key-field>div{{position:static}}.st-key-field>div:first-child{{display:none}}
  [class*="st-key-h_"]{{transform:none}}
  [class*="st-key-h_"] button{{width:100%;display:grid;grid-template-columns:14px 1fr;gap:2px 10px;justify-items:start;
    padding:10px 12px;border:1px solid var(--rule)!important;border-radius:6px;text-align:left}}
  [class*="st-key-h_"] button::before{{width:14px;height:14px;border-width:2px;margin:0;grid-row:1}}
  [class*="st-key-h_"] button::after{{grid-column:2;font-size:12px}}
  [class*="st-key-h_"] button:hover{{border-color:var(--gold)!important}}
  .st-key-mast{{flex-direction:column;align-items:flex-start;gap:6px}}.st-key-strip{{width:100%}}}}
@media (max-width:860px){{
  :root{{--pad:18px}}
  .fw-hero{{grid-template-columns:1fr;padding:32px 0 0}}.fw-hero h1{{font-size:clamp(32px,10.5vw,52px);white-space:nowrap}}.fw-hint{{grid-column:1}}
  .fw-week{{min-width:0}}.st-key-field{{grid-template-columns:1fr 1fr}}
  .fw-top h1{{font-size:36px}}
  .st-key-mark button p{{font-size:20px!important}}
  .st-key-strip{{aspect-ratio:auto;flex-direction:row;overflow-x:auto;gap:6px;padding:2px 0 4px}}.st-key-strip::before{{display:none}}
  [class*="st-key-s_"]{{position:static;transform:none;flex:none}}
  [class*="st-key-s_"] button{{border:1px solid var(--rule)!important;border-radius:999px;padding:4px 10px 4px 24px}}
  [class*="st-key-s_"] button::before{{left:7px;width:12px;height:12px;margin-top:-6px}}
  [class*="st-key-s_"] button::after{{display:inline}}
  [class*="st-key-s_"] button:disabled{{border-color:var(--gold)!important}}
  .st-key-s_home button p::before{{display:none}}}}
{_nav_css()}
"""


def inject_css():
    st.html(f"<style>{CSS}</style>")


def go(section: str):
    st.session_state.page = section


def masthead(current: str):
    """The wordmark and the slim formation strip, with the current section's player in gold."""
    with st.container(key="mast"):
        st.button("Fourth *and* Why", key="mark", on_click=go, args=("Home",))
        with st.container(key="strip"):
            for name, *_ in sorted(FORMATION, key=lambda f: f[4]):  # tab order left to right
                st.button(name, key=f"s_{slug(name)}", on_click=go, args=(name,), disabled=name == current)


def _last_segment(path: str):
    """Start and end of a route's last segment (M, L, V, H and Q commands), to aim its arrowhead."""
    t, i, prev, cur = re.findall(r"[MLVHQ]|[-\d.]+", path), 0, (0.0, 0.0), (0.0, 0.0)
    while i < len(t):
        c = t[i]
        a = [float(v) for v in t[i + 1:i + 1 + {"M": 2, "L": 2, "V": 1, "H": 1, "Q": 4}[c]]]
        i += 1 + len(a)
        if c == "V":
            prev, cur = cur, (cur[0], a[0])
        elif c == "H":
            prev, cur = cur, (a[0], cur[1])
        elif c == "Q":
            prev, cur = (a[0], a[1]), (a[2], a[3])
        else:
            prev, cur = cur, (a[0], a[1])
    return prev, cur


def _field_svg() -> str:
    out = ['<svg class="fm" viewBox="0 130 1440 510" aria-hidden="true">']
    for y in range(120, 650, 80):
        out.append(f'<line x1="0" x2="1440" y1="{y}" y2="{y}" stroke="#221d19"/>')
    for y in range(142, 640, 16):  # hash marks
        out.append(f'<line x1="560" x2="572" y1="{y}" y2="{y}" stroke="#221d19"/>'
                   f'<line x1="868" x2="880" y1="{y}" y2="{y}" stroke="#221d19"/>')
    out.append('<line x1="0" x2="1440" y1="420" y2="420" stroke="#5a5148" stroke-width="1.5"/>'
               '<rect x="28" y="376" width="96" height="30" rx="3" fill="none" stroke="#5a5148"/>'
               f'<text x="76" y="396" font-size="14" font-weight="700" fill="{SOFT}" text-anchor="middle" '
               'font-family="Hanken Grotesk,sans-serif">4th &amp; Why</text>')
    for name, pos, _, (x, y), _, path in FORMATION:
        tip = ""
        if path == "block":
            path = f"M{x} {y - 22}V{y - 40}M{x - 11} {y - 40}H{x + 11}"
        else:  # an arrowhead along the last segment of the route
            (px, py), (ex, ey) = _last_segment(path)
            a = math.atan2(ey - py, ex - px)
            tri = " ".join(f"{ex + r * math.cos(a + d):.1f},{ey + r * math.sin(a + d):.1f}"
                           for r, d in ((4, 0), (9, 2.6), (9, -2.6)))
            tip = f'<polygon class="gtip" points="{tri}"/><polygon class="tip" points="{tri}"/>'
        dash = ' stroke-dasharray="6 6"' if pos == "QB" else ""
        ball = (f'<ellipse cx="{x}" cy="{y - 30}" rx="9" ry="6" fill="#8a5a33" stroke="{INK}" stroke-width="1"/>'
                if pos == "C" else "")
        out.append(f'<g class="r-{slug(name)}"><path class="ghost" d="{path}"{dash}/>'
                   f'<path class="route" d="{path}" pathLength="1"/>{tip}{ball}</g>')
    out.append("</svg>")
    return "".join(out)


def home_field():
    """The home screen's formation: hover a player to draw its assignment, click to open the section."""
    with st.container(key="field"):
        st.markdown(_field_svg(), unsafe_allow_html=True)  # st.html would strip the SVG
        for name, *_ in sorted(FORMATION, key=lambda f: f[3]):  # tab order left to right
            st.button(name, key=f"h_{slug(name)}", on_click=go, args=(name,), disabled=name == "Home")


def page_header(title: str, lede: str):
    st.html(f'<div class="fw-top"><h1>{title}</h1><p>{lede}</p></div>')


def footer():
    st.html('<p class="fw-foot">Updates Thursday, Sunday and Monday. Every chart has a hover tooltip; '
            'tables sort and download. Data: nflverse, Open-Meteo.</p>')


# ---------------------------------------------------------------- charts
def style(chart: alt.Chart, height: int = 320) -> alt.Chart:
    """The chart theme: Hanken Grotesk, solid hairline grid, recessive axes, horizontal axis titles."""
    font = "Hanken Grotesk"
    return (chart.properties(height=height)
            .configure(font=font, background="transparent")
            .configure_axis(gridColor=RULE, gridWidth=1, domainColor=AXIS, tickColor=AXIS, labelColor=TICK,
                            titleColor=MUTED, labelFontSize=12, titleFontSize=12, titleFontWeight=400,
                            labelFont=font, titleFont=font, labelPadding=6)
            .configure_axisY(domain=False, ticks=False, titleAngle=0, titleAlign="left", titleBaseline="bottom",
                             titleX=0, titleY=-10)
            .configure_legend(labelColor=SOFT, titleColor=MUTED, orient="top", labelFontSize=13, labelFont=font,
                              titleFont=font, symbolSize=90, symbolStrokeWidth=2)
            .configure_text(font=font, color=SOFT)
            .configure_title(font=font, color=INK)
            .configure_line(strokeWidth=2)
            .configure_view(strokeWidth=0))
