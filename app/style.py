"""Presentation layer: one CSS block and two small helpers.

Kept separate from common.py so styling can be changed without touching data
loading. Everything here is cosmetic; nothing affects results.

Sizing approach: `html { font-size }` sets the global scale, so everything
measured in rem grows together. Individual selectors below only adjust
elements whose Streamlit defaults are out of proportion once scaled. Raise
the one number in ROOT_FONT_PX if the whole app should be larger.
"""

import streamlit as st

ROOT_FONT_PX = 17          # browser default is 16

CSS = f"""
<style>
/* ---------- global scale ---------- */
/* everything sized in rem follows this one number */
html {{ font-size: {ROOT_FONT_PX}px; }}

/* ---------- layout ---------- */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- body prose ---------- */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {{
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(232,234,237,0.95);
}}

/* captions carry most of the argument in this app, so they sit just under
   body text rather than fading out */
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] li,
small, .stCaption {{
    font-size: 0.97rem !important;
    line-height: 1.65 !important;
    color: rgba(232,234,237,0.85) !important;
}}

div[data-testid="stMarkdownContainer"] table {{ font-size: 1rem; }}
div[data-testid="stMarkdownContainer"] th {{ font-size: 1rem; opacity: 0.9; }}

/* ---------- metrics ---------- */
div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 14px 16px;
}}
div[data-testid="stMetricLabel"] p {{
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgba(232,234,237,0.78) !important;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.85rem; font-weight: 600; }}
div[data-testid="stMetricDelta"] {{ font-size: 0.95rem; }}

/* ---------- sidebar ----------
   The sidebar is permanent context, so it needs to be as readable as the
   main column, not a footnote. */
section[data-testid="stSidebar"] {{ border-right: 1px solid rgba(255,255,255,0.09); }}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li {{
    font-size: 1.02rem !important;
    line-height: 1.65;
}}
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {{
    font-size: 1.0rem !important;
    line-height: 1.6 !important;
    color: rgba(232,234,237,0.86) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
    background: transparent; border: none; padding: 6px 0;
}}
section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] p {{
    font-size: 0.88rem !important;
}}
section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {{
    font-size: 1.6rem;
}}
section[data-testid="stSidebar"] h3 {{ font-size: 1.25rem; }}

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {{ font-size: 1.05rem; padding: 11px 20px; }}
button[data-baseweb="tab"] p {{ font-size: 1.05rem !important; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ font-weight: 600; }}

/* ---------- controls ---------- */
label p, div[data-testid="stWidgetLabel"] p {{ font-size: 1rem !important; }}
div[data-testid="stFileUploader"] label p {{ font-size: 1.02rem !important; }}
div[data-testid="stFileUploader"] small {{ font-size: 0.95rem !important; }}
div[data-baseweb="select"] {{ font-size: 1rem; }}
div[data-testid="stSlider"] div[data-testid="stTickBar"] {{ font-size: 0.9rem; }}

/* ---------- multiselect tags ----------
   White text on the yellow accent is unreadable. Broad selectors here on
   purpose: the tag element is not always a span across Streamlit versions. */
[data-baseweb="tag"] {{
    background-color: rgba(255,255,255,0.14) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
}}
[data-baseweb="tag"],
[data-baseweb="tag"] span,
[data-baseweb="tag"] div {{
    color: #E8EAED !important;
    font-size: 0.97rem !important;
}}
[data-baseweb="tag"] svg,
[data-baseweb="tag"] path {{ fill: #E8EAED !important; stroke: #E8EAED !important; }}

/* anything else filled with the accent needs dark text on top */
button[kind="primary"], button[data-testid="baseButton-primary"] {{
    color: #0F1116 !important; font-weight: 600;
}}

/* ---------- dataframes ---------- */
div[data-testid="stDataFrame"] {{
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    font-size: 0.98rem;
}}
div[data-testid="stDataFrame"] [aria-selected="true"] {{
    background: rgba(255,230,0,0.12) !important;
}}

/* ---------- callouts, expanders, code ---------- */
div[data-testid="stAlert"] p {{ font-size: 1.04rem; line-height: 1.65; }}
div[data-testid="stExpander"] summary p {{ font-size: 1.02rem !important; }}
code, pre, div[data-testid="stCode"] {{ font-size: 0.95rem; }}
div[data-testid="stJson"] {{ font-size: 0.95rem; }}

/* ---------- rules ---------- */
hr {{ margin: 1.6rem 0 1.2rem 0; opacity: 0.14; }}
</style>
"""


def inject_css():
    """Apply the CSS block above to the page. Call once from Home.py,
    immediately after set_page_config, so it lands before anything renders."""
    st.markdown(CSS, unsafe_allow_html=True)
