"""Loading, cleaning, derived features, and the feature pipeline.

The important structural choice here: cleaning and feature derivation live
inside the pipeline as the RawPrep step, not in a script that has to be run
first. That is what makes artifacts/pipeline.joblib self-contained, so the
dashboard can score an arbitrary uploaded CSV and the notebook and the app
cannot drift apart.

Everything in add_derived is computed from a single row's own values, so no
derived feature can leak across the train/test boundary. The one exception is
add_group_key, which is a splitting artifact rather than a feature and is
deliberately kept out of the pipeline.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Challenge_Data.xls"

PHRASE_COLS = ["Col1", "Col4", "Col6"]
FLAT_COLS = ["Col7"]
ID_COL = "Col2"
FEATURE_COLS = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]

TOKEN = r"[^\s]+"
NUMERIC = ["col3_signed_log", "is_round_amount", "is_negative",
           "col2_len", "col2_is_numeric"]
ONEHOT = ["Col7", "col6_first_token", "col2_shape_coarse"]


# ---------- row-wise helpers ----------

def normalize_label(s):
    """Extract the digit from any label spelling. Handles casing, spacing, typos."""
    d = re.search(r"\d+", str(s))
    return f"Category_{d.group()}" if d else np.nan


def clean_text(s):
    """Collapse internal whitespace and trim. Deliberately does not lowercase:
    it merged no values here and can conflate genuinely distinct codes."""
    if pd.isna(s):
        return np.nan
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_id(x):
    """Cast mixed int/str/float codes to text without gluing on a false .0"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).strip()


def shape_mask(s):
    """Character-class skeleton: WE1243 -> AA9999, 082218-9 -> 999999-9"""
    return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", str(s)))


def shape_mask_coarse(s):
    """Same, runs collapsed: WE1243 -> A9, 082218-9 -> 9-9"""
    return re.sub(r"(.)\1+", r"\1", shape_mask(s))


# ---------- frame-level, all row-wise and leak-free ----------

def clean_frame(df):
    """Normalize the seven input columns to consistent types and text. Also
    survives a CSV round trip, where Col3 comes back as text and Col4's empty
    strings come back as NaN."""
    df = df.copy()
    df[ID_COL] = df[ID_COL].map(clean_id)
    for c in PHRASE_COLS + FLAT_COLS:
        df[c] = df[c].map(clean_text)
    df["Col4"] = df["Col4"].fillna("")          # vectorizers reject NaN
    df["Col3"] = pd.to_numeric(df["Col3"], errors="coerce")
    return df


def add_derived(df):
    """Row-wise features only. Nothing reads the label or any other row.

    Builds more columns than the model uses: col2_has_alpha, col2_has_hyphen,
    col2_is_480_family, and col2_shape are computed as evidence for the
    notebook and dashboard, while only the columns named in NUMERIC and ONEHOT
    reach the ColumnTransformer.
    """
    df = df.copy()
    c2 = df[ID_COL].astype(str)
    df["col2_is_numeric"] = c2.str.fullmatch(r"\d+")
    df["col2_len"] = c2.str.len()
    df["col2_has_alpha"] = c2.str.contains(r"[A-Za-z]", regex=True)
    df["col2_has_hyphen"] = c2.str.contains("-", regex=False)
    df["col2_is_480_family"] = c2.str.fullmatch(r"4\.80[A-Za-z]\+11")
    df["col2_shape"] = c2.map(shape_mask)
    df["col2_shape_coarse"] = c2.map(shape_mask_coarse)

    x = df["Col3"]
    df["col3_signed_log"] = np.sign(x) * np.log1p(x.abs())
    df["is_round_amount"] = (x % 10 == 0)
    df["is_negative"] = (x < 0)

    df["col6_first_token"] = df["Col6"].str.split().str[0]
    return df


def add_group_key(df):
    """Splitting artifact, NOT a feature. factorize() codes depend on the
    whole frame, so this must never run inside the pipeline."""
    df = df.copy()
    df["group_key"] = (df[FEATURE_COLS].astype(object).fillna("__NA__")
                       .astype(str).agg("|".join, axis=1).factorize()[0])
    return df


def load_and_clean(path=RAW_PATH):
    """Read the source spreadsheet and return it fully prepared: labels
    normalized, columns cleaned, derived features and group key added. The
    asserts fail loudly if the label parsing ever stops producing six clean
    classes."""
    df = pd.read_excel(path)
    df["label"] = df["ClassificationLabel"].map(normalize_label)
    assert df["label"].nunique() == 6, "expected 6 classes"
    assert df["label"].isna().sum() == 0, "unparsed label"
    return add_group_key(add_derived(clean_frame(df)))


# ---------- transformers ----------

class RawPrep(BaseEstimator, TransformerMixin):
    """Clean and derive inside the pipeline, so a saved model accepts
    raw Col1-Col7 rows. Stateless, therefore cannot leak."""

    def fit(self, X, y=None):
        """Nothing to learn. Present because scikit-learn requires it."""
        return self

    def transform(self, X):
        """Run the same cleaning and derivation the training data went
        through, so an uploaded row is prepared identically."""
        return add_derived(clean_frame(pd.DataFrame(X)))


class DateFeatures(BaseEstimator, TransformerMixin):
    """Calendar parts plus cyclical encodings. Stateless: only 17 distinct
    dates exist, so days-since-earliest carries nothing and is skipped."""

    def fit(self, X, y=None):
        """Nothing to learn, since the only feature needing a reference point
        was days-since-earliest and that was dropped."""
        return self

    def transform(self, X):
        """Expand a date column into calendar parts plus sine and cosine
        encodings, so month 12 and month 1 sit next to each other rather than
        at opposite ends of a scale."""
        d = pd.to_datetime(X.iloc[:, 0])
        return pd.DataFrame({
            "year": d.dt.year,
            "month": d.dt.month,
            "day": d.dt.day,
            "dayofweek": d.dt.dayofweek,
            "quarter": d.dt.quarter,
            "month_sin": np.sin(2 * np.pi * d.dt.month / 12),
            "month_cos": np.cos(2 * np.pi * d.dt.month / 12),
            "dow_sin": np.sin(2 * np.pi * d.dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * d.dt.dayofweek / 7),
        }, index=X.index).values

    def get_feature_names_out(self, input_features=None):
        """Name the nine output columns, so get_feature_names_out on the whole
        ColumnTransformer stays readable."""
        return np.array(["year", "month", "day", "dayofweek", "quarter",
                         "month_sin", "month_cos", "dow_sin", "dow_cos"])


# ---------- assembly ----------

def build_features(min_df=2, use_dates=True):
    """Assemble the ColumnTransformer: a TF-IDF branch per text column, a
    one-hot branch, a scaled numeric branch, and optionally the date branch.
    min_df and use_dates are arguments rather than constants because both were
    settled by cross-validation rather than assumed."""
    branches = [
        ("col1_tfidf", TfidfVectorizer(token_pattern=TOKEN, min_df=min_df), "Col1"),
        ("col4_tfidf", TfidfVectorizer(token_pattern=TOKEN, min_df=min_df), "Col4"),
        ("col6_tfidf", TfidfVectorizer(token_pattern=TOKEN, min_df=min_df), "Col6"),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ONEHOT),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ]
    if use_dates:
        branches.append(("date", DateFeatures(), ["Col5"]))
    return ColumnTransformer(branches, remainder="drop")


def build_pipeline(clf, min_df=2, use_dates=True, dense=False):
    """Raw Col1-Col7 in, prediction out. Set dense=True for estimators
    that reject sparse input, such as HistGradientBoosting."""
    steps = [
        ("prep", RawPrep()),
        ("features", build_features(min_df, use_dates)),
    ]
    if dense:
        steps.append(("dense", FunctionTransformer(to_dense, accept_sparse=True)))
    steps.append(("clf", clf))
    return Pipeline(steps)


# TF-IDF branches produce sparse but HistGradientBoosting cannot accept sparse input
def to_dense(X):
    """Module-level, not a lambda, so the fitted pipeline stays picklable."""
    return X.toarray() if hasattr(X, "toarray") else X