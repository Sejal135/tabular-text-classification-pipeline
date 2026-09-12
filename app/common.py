"""Paths, cached loaders, and file helpers shared by both tabs.

Also the single place the project root is added to sys.path, so both tabs can
import from src/ regardless of which directory Streamlit was launched from.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # must precede any src import

# re-exported for the tabs: unpickling pipeline.joblib also needs these classes
# importable under the exact module path they were saved under
from src.preprocessing import (FEATURE_COLS, RAW_PATH, clean_frame,  # noqa: E402,F401
                               load_and_clean, normalize_label)
from src.explain import Contributions  # noqa: E402
from src.explain import load_metrics as _load_metrics_raw  # noqa: E402
from src.explain import load_token_stats as _load_token_stats_raw  # noqa: E402

ART = ROOT / "artifacts"
FIGS = ART / "figures"
HOLDOUT_DIR = ROOT / "data" / "holdout"

# header spellings that mean "this column holds the true label"
LABEL_ALIASES = {"classificationlabel", "label", "col8", "columnh", "h",
                 "target", "actual", "truelabel", "class"}


# ---------- upload handling ----------

def canon(name):
    """Normalize a header so 'Col 1', 'COL_1' and 'col1' all match Col1."""
    return re.sub(r"[\s_]+", "", str(name)).lower()


def read_any(upload):
    """Read an uploaded file, trying the reader its extension suggests first
    and falling back to the others. Returns (dataframe, reader_used, errors)."""
    name = upload.name.lower()
    csv_plain = ("csv", lambda b: pd.read_csv(b))
    csv_sniff = ("csv (delimiter sniffed)",
                 lambda b: pd.read_csv(b, sep=None, engine="python"))
    excel = ("excel", lambda b: pd.read_excel(b))
    order = [csv_plain, csv_sniff, excel] if name.endswith(".csv") \
        else [excel, csv_plain, csv_sniff]

    errors = []
    for label, fn in order:
        try:
            upload.seek(0)                 # rewind, or the retry reads nothing
            return fn(upload), label, None
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    return None, None, errors


def resolve_columns(df):
    """Match the required columns to whatever the uploaded file calls them.
    Returns (rename_map, missing_columns, detected_label_column)."""
    lookup = {}
    for actual in df.columns:
        lookup.setdefault(canon(actual), actual)

    rename, missing = {}, []
    for req in FEATURE_COLS:
        actual = lookup.get(canon(req))
        if actual is None:
            missing.append(req)
        else:
            rename[actual] = req

    label_col = next((lookup[a] for a in LABEL_ALIASES if a in lookup), None)
    return rename, missing, label_col


# ---------- artifact rendering ----------

def show_csv(name, caption=None, index_col=0, **kwargs):
    """Render a saved artifact table, or say plainly that it is absent, so a
    missing file degrades one block instead of crashing the tab."""
    p = ART / name
    if not p.exists():
        st.caption(f"Not found: artifacts/{name}")
        return None
    df = pd.read_csv(p, index_col=index_col)
    st.dataframe(df, use_container_width=True, **kwargs)
    if caption:
        st.caption(caption)
    return df


def show_fig(name, caption=None):
    """Render a saved PNG from artifacts/figures, or note that it is absent.
    Kept as a fallback; the dashboard builds its charts live instead."""
    p = FIGS / f"{name}.png"
    if not p.exists():
        st.caption(f"Not found: artifacts/figures/{name}.png")
        return
    st.image(str(p), use_container_width=True)
    if caption:
        st.caption(caption)


# ---------- cached loaders ----------

@st.cache_resource
def load_model():
    """Load the fitted pipeline and label encoder once per session. Streamlit
    re-runs the script on every interaction, so without caching a 433-tree
    forest would reload on every click."""
    return (joblib.load(ART / "pipeline.joblib"),
            joblib.load(ART / "label_encoder.joblib"))


@st.cache_data
def load_metrics():
    """Load metrics.json: model config, cross-validation and holdout scores,
    baselines, and the approaches that were tested and rejected."""
    return json.loads((ART / "metrics.json").read_text())


@st.cache_data
def load_raw():
    """Load the source spreadsheet exactly as delivered, before any cleaning.
    Used for the before side of every before/after view."""
    return pd.read_excel(RAW_PATH)


@st.cache_data
def load_clean():
    """Load the dataset after label normalization, cleaning, and derived
    features. The after side of every before/after view."""
    return load_and_clean()


@st.cache_data
def load_holdout_labels():
    """Load the holdout labels if present, for the split-strategy table.
    Returns None rather than raising when the file is missing."""
    p = HOLDOUT_DIR / "holdout_labels.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_contributions(_pipe):
    """Build the per-row attribution helper once. The leading underscore tells
    Streamlit not to try hashing the pipeline, which it cannot hash."""
    return Contributions(_pipe)


@st.cache_data
def load_explain_artifacts():
    """Load the token statistics and metrics the explainer and its agent use
    as evidence. Cached because both are read on every Explain click."""
    return _load_token_stats_raw(), _load_metrics_raw()