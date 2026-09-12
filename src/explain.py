"""Evidence layer for the prediction explainer.

The text columns are anonymized WordN tokens, so no language model can say
what a feature means. What it can do is turn measured evidence into readable
prose. This module produces that evidence and nothing else: per-feature
attribution for one row, the training-set behaviour of the tokens in that
row, and the model's known weaknesses for the predicted class.

Everything here is deterministic. The LLM layer sits on top and never
invents a number.

Attribution method
------------------
Default is leave-one-feature-out ablation: zero each active feature in turn
and measure how far the predicted probability moves. For a single row this
is exact, needs no extra dependency, and yields a claim anyone can check by
hand ("removing this token drops the probability from 0.73 to 0.41").

SHAP is supported but only used when it passes an additivity check. On this
model, shap 0.52's TreeExplainer returns values around 1e6 for the two
classes carrying probability mass while the other four are correct, so the
check matters.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

_SOURCE = {
    "col1_tfidf": "Col1",
    "col4_tfidf": "Col4",
    "col6_tfidf": "Col6",
    "onehot": "category",
    "num": "numeric",
}


# ---------------------------------------------------------------- naming

def parse_feature(name):
    """'col4_tfidf__word575' -> ('Col4', 'word575')."""
    branch, _, rest = name.partition("__")
    return _SOURCE.get(branch, branch), rest


def pretty_feature(name):
    """Turn a pipeline feature name into something a reader can act on, so
    'col4_tfidf__word575' reads as 'Col4 contains Word575'."""
    source, detail = parse_feature(name)
    if source in ("Col1", "Col4", "Col6"):
        # vectorizers lowercase, so word575 came from Word575
        return f"{source} contains {re.sub(r'^word', 'Word', detail)}"
    if source == "category":
        detail = re.sub(r"^(Col7|col6_first_token|col2_shape_coarse)_", r"\1 = ", detail)
        return detail.replace("_", " ")
    return detail


# ---------------------------------------------------------------- artifacts

def load_token_stats(path=None):
    """Load per-token class counts, built from the modeling set only so an
    explanation never quotes a statistic derived from holdout rows. Returns
    None when the file is absent rather than raising."""
    p = Path(path) if path else ART / "token_stats.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_metrics(path=None):
    """Load metrics.json for the model-level figures quoted in caveats.
    Returns an empty dict when the file is absent."""
    p = Path(path) if path else ART / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- attribution

class Contributions:
    """Per-feature attribution for a single row."""

    ADDITIVITY_TOL = 0.05

    def __init__(self, pipe, prefer="ablation"):
        """Cache the classifier and feature names off the fitted pipeline.
        Building the SHAP explainer is attempted only when prefer='shap', and
        a failure there leaves ablation as the method rather than raising."""
        self.pipe = pipe
        self.clf = pipe.named_steps["clf"]
        self.feature_names = pipe.named_steps["features"].get_feature_names_out()
        self.prefer = prefer
        self._explainer = None
        if prefer == "shap":
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.clf)
            except Exception:
                self._explainer = None

    # ---- internals

    def _transform(self, row_df):
        """Run the row through every pipeline step except the classifier, to
        get the dense feature vector the attribution operates on."""
        X = row_df
        for _, step in self.pipe.steps[:-1]:
            X = step.transform(X)
        dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return np.asarray(dense, dtype=float)

    def _ablate(self, dense, class_index):
        """Zero each active feature in turn; record the probability drop.

        Built as one batch so all variants are scored in a single
        predict_proba call rather than a loop.

        Zero is the neutral value for every branch here: absent token for
        TF-IDF, category off for one-hot, and the training mean for the
        standardized numeric columns.
        """
        base = float(self.clf.predict_proba(dense)[0, class_index])
        active = np.flatnonzero(dense[0] != 0)
        if active.size == 0:
            return base, np.zeros(dense.shape[1]), np.array([], dtype=int)

        batch = np.repeat(dense, active.size, axis=0)
        batch[np.arange(active.size), active] = 0.0
        probs = self.clf.predict_proba(batch)[:, class_index]

        contrib = np.zeros(dense.shape[1])
        contrib[active] = base - probs        # positive means it supported the class
        return base, contrib, active

    def _shap(self, dense, class_index, target_sum):
        """Try SHAP, and return None unless its contributions sum to the
        probability gap. Without that gate this model returns values around
        1e6 for the classes that actually carry probability."""
        if self._explainer is None:
            return None
        try:
            arr = np.asarray(self._explainer.shap_values(dense, check_additivity=False))
            contrib = arr[0, :, class_index] if arr.ndim == 3 else arr[class_index][0]
            if abs(float(contrib.sum()) - target_sum) > self.ADDITIVITY_TOL:
                return None                    # library returned something unusable
            return contrib
        except Exception:
            return None

    # ---- public

    def top(self, row_df, class_index, n=8):
        """Features supporting the class, features opposing it, and the method
        actually used."""
        dense = self._transform(row_df)
        base, contrib, _ = self._ablate(dense, class_index)
        method = "ablation"

        if self.prefer == "shap":
            ev = np.asarray(self._explainer.expected_value) if self._explainer is not None else None
            if ev is not None:
                alt = self._shap(dense, class_index, base - float(ev[class_index]))
                if alt is not None:
                    contrib, method = alt, "shap"

        d = pd.DataFrame({
            "feature": self.feature_names,
            "readable": [pretty_feature(f) for f in self.feature_names],
            "value": dense[0],
            "contribution": contrib,
        })
        d = d[d["contribution"] != 0]
        d = d.reindex(d["contribution"].abs().sort_values(ascending=False).index)

        if method == "ablation":
            d["probability without it"] = (base - d["contribution"]).round(4)

        return d[d["contribution"] > 0].head(n), d[d["contribution"] < 0].head(n), method


# ---------------------------------------------------------------- tokens

def token_evidence(row, stats, min_count=5, top_n=8):
    """How each token in this row behaved during training. This is what makes
    an explanation checkable rather than plausible.

    Tokens seen fewer than min_count times are skipped, since a token appearing
    twice looks perfectly predictive by chance. Ranked by lift over the class
    base rate rather than by raw share, so a token that is 86% Category_1 in a
    dataset that is 88% Category_1 does not look like signal.
    """
    if stats is None:
        return pd.DataFrame()

    total = stats["n_rows"]
    priors = {k: v / total for k, v in stats["class_counts"].items()}

    out = []
    for col in ["Col1", "Col4", "Col6"]:
        text = row.get(col)
        if not isinstance(text, str):
            continue
        table = stats["tokens"].get(col, {})
        for tok in dict.fromkeys(text.split()):
            counts = table.get(tok)
            if not counts:
                continue
            n = sum(counts.values())
            if n < min_count:
                continue
            cls, k = max(counts.items(), key=lambda kv: kv[1])
            share = k / n
            out.append({
                "column": col,
                "token": tok,
                "rows in training": n,
                "dominant class": cls,
                "share": round(share, 3),
                "lift vs base rate": round(share / priors.get(cls, 1e-9), 1),
            })

    if not out:
        return pd.DataFrame()
    return (pd.DataFrame(out)
            .sort_values(["lift vs base rate", "rows in training"],
                         ascending=[False, False])
            .head(top_n))


# ---------------------------------------------------------------- caveats

CLASS_ROWS = {"Category_1": 4693, "Category_2": 568, "Category_3": 21,
              "Category_4": 14, "Category_6": 11, "Category_5": 2}
CLASS_F1 = {"Category_1": 0.965, "Category_2": 0.768, "Category_3": 0.894,
            "Category_4": 0.500, "Category_6": 0.545}


def class_caveats(cls, metrics):
    """What is known to be shaky about predictions of this class, so the agent
    cannot sound confident where the model is not."""
    notes = []
    n = CLASS_ROWS.get(cls)
    if n is not None:
        notes.append(f"{cls} has {n} rows in the modeling set.")
    if cls in CLASS_F1:
        notes.append(f"Cross-validated F1 for {cls} is {CLASS_F1[cls]:.3f}.")
    if n is not None and n < 30:
        notes.append("With this few training examples, fold-to-fold variance is "
                     "large. Treat a single prediction as a suggestion for "
                     "review, not a decision.")
    if cls == "Category_2":
        notes.append("The model's largest error block is Category_1 rows "
                     "predicted as Category_2, about 5% of all Category_1 rows.")
    if cls == "Category_3":
        notes.append("Category_3 has perfect recall but precision of 0.808, "
                     "because it absorbs misclassified Category_4 and "
                     "Category_6 rows.")
    if cls == "Category_5":
        notes.append("Category_5 has 2 rows in the entire dataset and is never "
                     "validated. This prediction carries no measured "
                     "reliability at all.")
    cv = metrics.get("cv", {})
    if cv.get("macro_f1_mean"):
        notes.append(f"Overall cross-validated macro F1 is "
                     f"{cv['macro_f1_mean']:.4f} plus or minus "
                     f"{cv.get('macro_f1_std', 0):.4f} across seeds.")
    return notes


# ---------------------------------------------------------------- assembly

def build_evidence(row_df, pipe, le, contributions, stats, metrics, top_n=8):
    """Everything the explainer knows about one row, as plain data.

    Combines the ranked class probabilities, the ablation contributions for
    and against, the training behaviour of the row's tokens, and the caveats
    for the predicted class into one dict. This is the only thing the agent
    ever sees, which is why it can be told never to state a number that is not
    in it.
    """
    proba = pipe.predict_proba(row_df)[0]
    order = np.argsort(proba)[::-1]
    pred = le.classes_[order[0]]

    toward, against, method = contributions.top(row_df, order[0], n=top_n)
    keep = ["readable", "contribution"] + \
           (["probability without it"] if "probability without it" in toward else [])

    row = row_df.iloc[0].to_dict()
    tokens = token_evidence(row, stats)

    return {
        "prediction": pred,
        "agreement": round(float(proba[order[0]]), 4),
        "runner_up": le.classes_[order[1]] if len(order) > 1 else None,
        "runner_up_probability": round(float(proba[order[1]]), 4) if len(order) > 1 else None,
        "ranked_classes": [
            {"class": le.classes_[i], "probability": round(float(proba[i]), 4)}
            for i in order if proba[i] > 0
        ],
        "row": {k: (None if pd.isna(v) else v) for k, v in row.items()},
        "attribution_method": method,
        "pushing_toward": toward[keep].round(4).to_dict("records"),
        "pushing_against": against[keep].round(4).to_dict("records"),
        "token_evidence": tokens.to_dict("records") if len(tokens) else [],
        "caveats": class_caveats(pred, metrics),
    }