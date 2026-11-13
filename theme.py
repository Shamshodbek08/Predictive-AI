"""Visual system for the Library Wi-Fi dashboard (v2, English only).

Same colours, fonts and chart template as v1; labels are now English only and the page
is organised into tabs. Everything visual lives here so app.py keeps its data logic untouched.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------- tokens
GROUND = "#f2f4f7"
SURFACE = "#ffffff"
BORDER = "#e3e6eb"
RULE = "#eef0f3"
CONTROL = "#d4d4d4"
INK = "#0a0a0a"
MUTED = "#5f6368"
FAINT = "#8a8f98"
HUE = "#1d5fbf"          # the one chart hue: the main data element
HUE_TINT = "#e8effa"
POSITIVE = "#2e7d32"
NEGATIVE = "#c4242a"
NEUTRAL_BAR = "#c9ced6"  # non-selected / comparison bars

FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

# Categorical hues follow the ENTITY, never its rank (fixed order, same as v1).
GROUP_COLORS = {
    "UG": "#2a78d6",
    "TPG": "#eb6834",
    "RPG": "#1baf7a",
    "Staff": "#eda100",
    "Alumni": "#e87ba4",
    "Non-HKUST": "#008300",
    "Others(HKUST)": "#6250d6",
}
BUCKET_COLORS = {"UG": GROUP_COLORS["UG"], "PG (RPG+TPG)": GROUP_COLORS["TPG"], "Staff": GROUP_COLORS["Staff"]}

# Sequential blue ramp (light -> dark) for magnitude.
SEQ_BLUE = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ------------------------------------------------------- plain-English glossary
GROUP_NAMES = {
    "UG": "Undergraduates (UG)",
    "TPG": "Taught postgrads (TPG)",
    "RPG": "Research postgrads (RPG)",
    "Staff": "Staff",
    "Alumni": "Alumni",
    "Non-HKUST": "Visitors (Non-HKUST)",
    "Others(HKUST)": "Other HKUST",
    "PG (RPG+TPG)": "Postgrads (avg of RPG & TPG)",
}
GROUP_PHRASE = {
    "UG": "undergraduate",
    "TPG": "taught-postgraduate",
    "RPG": "research-postgraduate",
    "Staff": "staff",
    "Alumni": "alumni",
    "Non-HKUST": "visitor (Non-HKUST)",
    "Others(HKUST)": "other HKUST",
}
DOW_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DOW_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]


def group_label(g: str) -> str:
    return GROUP_NAMES.get(g, str(g))


def dow_label(i: int) -> str:
    return DOW_EN[i]


def month_label(m: int) -> str:
    return MONTH_FULL[m - 1]


# ---------------------------------------------------------------- plotly
_axis = dict(
    showgrid=False,
    zeroline=False,
    showline=True,
    linecolor=CONTROL,
    linewidth=1,
    ticks="",
    tickfont=dict(size=12, color=MUTED),
    title=dict(font=dict(size=12, color=MUTED), standoff=10),
    automargin=True,
)

pio.templates["ledger"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=list(GROUP_COLORS.values()),
        margin=dict(l=10, r=14, t=52, b=40),
        title=dict(x=0, xanchor="left", xref="container", y=1, yanchor="top", yref="container",
                   pad=dict(t=14, l=10), font=dict(size=15, color=INK)),
        xaxis=_axis,
        yaxis={**_axis, "showgrid": True, "gridcolor": RULE, "gridwidth": 1, "showline": False, "nticks": 6},
        legend=dict(
            orientation="h", x=0, xanchor="left", y=0, yanchor="bottom", yref="container",
            font=dict(size=12, color=INK), title=dict(text=""), bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=CONTROL, font=dict(family=FONT, size=12, color=INK)),
        hovermode="closest",
        bargap=0.28,
        bargroupgap=0.08,
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(size=11, color=MUTED))),
    )
)
TEMPLATE = "ledger"
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}


def finish(fig: go.Figure, title: str | None = None, subtitle: str | None = None, height: int = 360) -> go.Figure:
    """Apply the template, a title (+ grey sub-line) and a fixed height."""
    fig.update_layout(template=TEMPLATE, height=height)
    if title:
        text = f"<b style='font-weight:500'>{title}</b>"
        if subtitle:
            text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
        fig.update_layout(title=dict(text=text))
        fig.update_layout(margin=dict(t=76 if subtitle else 52))
    return fig


# ---------------------------------------------------------------- CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --ground:{GROUND}; --surface:{SURFACE}; --border:{BORDER}; --rule:{RULE};
  --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --hue:{HUE}; --hue-tint:{HUE_TINT};
  --positive:{POSITIVE}; --negative:{NEGATIVE};
  --radius-card:10px; --radius-control:6px;
}}

html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, .stCaption,
button, input, textarea, select {{
  font-family: {FONT} !important;
}}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--ground); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1240px; }}

/* type */
h1 {{ font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.02em; line-height: 1.2 !important; padding: 0 !important; color: var(--ink); }}
h2, h3 {{ font-size: 19px !important; font-weight: 600 !important; letter-spacing: -0.01em; color: var(--ink); padding-top: .25rem !important; }}
p, li {{ font-size: 15px; line-height: 1.6; }}
[data-testid="stCaptionContainer"], .stCaption p {{ color: var(--muted) !important; font-size: 13px !important; line-height: 1.5; }}
[data-testid="stHeadingWithActionElements"] a, h1 a, h2 a, h3 a {{ display: none !important; }}

/* page lead */
.lede {{ font-size: 16px; color: var(--muted); margin: .35rem 0 0 0; max-width: 80ch; }}
.team {{ font-size: 13px; color: var(--faint); margin: .35rem 0 0 0; }}
.section-gap {{ height: 1.25rem; }}
.eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--hue); margin: 0 0 .2rem 0; }}
.qhead {{ font-size: 21px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 .5rem 0; color: var(--ink); }}
.qbody {{ font-size: 15px; line-height: 1.6; color: #22252a; margin: 0; }}
.qbody b {{ font-weight: 600; }}
ul.plain {{ margin: .35rem 0 0 1.1rem; padding: 0; }}
ul.plain li {{ font-size: 15px; line-height: 1.6; margin-bottom: .2rem; color: #22252a; }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
[data-baseweb="tab"] {{ font-size: 15px !important; padding: 10px 16px !important; }}
[data-baseweb="tab"] p {{ font-size: 15px !important; font-weight: 500; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--hue) !important; }}

/* KPI strip: type, not tiles */
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px 24px; margin: .6rem 0 .4rem 0; }}
.kpi .l {{ font-size: 13px; color: var(--muted); }}
.kpi .v {{ font-size: 28px; font-weight: 500; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin-top: 2px; line-height: 1.15; }}
.kpi .s {{ font-size: 12.5px; color: var(--faint); font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

/* cards = keyed bordered containers */
[class*="st-key-card"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-card) !important; box-shadow: none !important;
}}
[class*="st-key-card"] [data-testid="stVerticalBlockBorderWrapper"] {{ border: 0 !important; }}

/* insight cards */
.ins-fig {{ font-size: 36px; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.1; color: var(--hue); }}
.ins-figcap {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
.ins-title {{ font-size: 18px; font-weight: 600; margin: .8rem 0 .35rem 0; line-height: 1.35; }}
.ins-body {{ font-size: 15px; line-height: 1.6; color: #22252a; margin: 0 0 .4rem 0; }}
.ins-body b {{ font-weight: 600; }}
.sowhat {{ font-size: 14px; line-height: 1.55; color: var(--ink); background: var(--hue-tint); border-radius: 6px; padding: .55rem .75rem; margin: .5rem 0 .25rem 0; }}
.sowhat b {{ font-weight: 600; }}

.note {{ font-size: 13.5px; color: var(--muted); line-height: 1.55; margin: .25rem 0 .75rem 0; }}
.note b {{ color: var(--ink); font-weight: 600; }}

/* predictive panel */
.pred-sentence {{ font-size: 19px; line-height: 1.5; margin: .2rem 0 .9rem 0; color: var(--ink); }}
.pred-sentence b {{ color: var(--hue); font-weight: 600; }}
[data-testid="stMetric"] {{ padding: 0; }}
[data-testid="stMetricLabel"] p {{ font-size: 13px !important; color: var(--muted) !important; white-space: normal !important; }}
[data-testid="stMetricLabel"] {{ white-space: normal !important; }}
[data-testid="stMetricValue"] {{ font-size: 30px !important; font-weight: 500 !important; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
.st-key-kpi_pred [data-testid="stMetricValue"] {{ color: var(--hue) !important; }}
[data-testid="stMetricDelta"] {{ font-size: 12.5px !important; font-variant-numeric: tabular-nums; }}
.honest {{ font-size: 14px; color: #33363b; line-height: 1.6; margin: .5rem 0 0 0; max-width: 120ch; }}
.honest b {{ font-weight: 600; }}

.foot {{ font-size: 12.5px; color: var(--muted); margin-top: 2.5rem; }}
.foot a {{ color: var(--ink); }}

/* focus stays visible */
:focus-visible {{ outline: 2px solid var(--hue) !important; outline-offset: 2px; }}
</style>
"""
