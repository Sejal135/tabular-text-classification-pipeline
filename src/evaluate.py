"""Cross-validation scoring on the custom group-aware splitter.

One scoring path for the whole project, so the ladder, the class-weight
comparison, the randomized search, and the final model are all measured the
same way and their numbers can be put in one table.

Metric scope: macro F1 is computed over the classes actually present in
validation, which is five rather than six. Category_5 sits in every training
fold and never in validation, so counting it as a zero would drag every score
down by a fifth for a class that was deliberately not validated.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.splits import make_cv_splits
from sklearn.metrics import make_scorer


def cv_evaluate(pipe, data, le, n_splits=3, seed=42, label_col="label"):
    """Fit and score `pipe` across the group-aware folds, collecting
    predictions from the fold where each row was held out.

    Returns a dict with macro F1, balanced accuracy, accuracy, the per-fold
    scores as a list rather than an average, a classification report, a
    confusion matrix, and the raw out-of-fold prediction vector.

    Out-of-fold scoring. Returns metrics plus the OOF prediction vector.

    Category_5 sits in every training fold and never in validation, so its
    rows carry no OOF prediction and are excluded from every metric.
    """
    y = le.transform(data[label_col])
    splits = make_cv_splits(data, n_splits=n_splits, seed=seed)

    oof = np.full(len(data), -1, dtype=int)
    fold_scores = []

    for tr, va in splits:
        m = clone(pipe)                      # fresh model per fold, no state carry-over
        m.fit(data.iloc[tr], y[tr])
        pred = m.predict(data.iloc[va])
        oof[va] = pred
        seen = np.unique(y[va])
        fold_scores.append(
            f1_score(y[va], pred, labels=seen, average="macro", zero_division=0))

    # -1 marks rows that were never in a validation fold, i.e. Category_5
    mask = oof >= 0
    y_true, y_pred = y[mask], oof[mask]
    seen = np.unique(y_true)

    return {
        "macro_f1": f1_score(y_true, y_pred, labels=seen, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "fold_macro_f1": fold_scores,
        "n_validated": int(mask.sum()),
        "report": classification_report(y_true, y_pred, labels=seen,
                                        target_names=le.classes_[seen],
                                        zero_division=0, digits=3),
        "confusion": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=seen),
            index=le.classes_[seen], columns=le.classes_[seen]),
        "oof": oof,
    }


def macro_f1_seen(y_true, y_pred):
    """Macro F1 over classes present in y_true only. Category_5 never appears
    in validation, so counting it would penalize every candidate identically."""
    seen = np.unique(y_true)
    return f1_score(y_true, y_pred, labels=seen, average="macro", zero_division=0)


# wrapped for RandomizedSearchCV, so the search optimizes exactly the metric
# cv_evaluate reports rather than a near-miss variant of it
macro_f1_scorer = make_scorer(macro_f1_seen)


# For out-of-fold(oof) probabilities
def cv_oof_proba(pipe, data, le, n_splits=3, seed=42, label_col="label"):
    """Same folds as cv_evaluate, but collecting probabilities instead of hard
    labels. Used by the threshold-tuning experiment, which needs the margin
    between classes rather than just the winner.

    Out-of-fold probability matrix. Rows never validated stay all-zero.
    """
    y = le.transform(data[label_col])
    proba = np.zeros((len(data), len(le.classes_)))
    for tr, va in make_cv_splits(data, n_splits=n_splits, seed=seed):
        m = clone(pipe)
        m.fit(data.iloc[tr], y[tr])
        p = m.predict_proba(data.iloc[va])
        # a fold may not contain every class, so map its columns back by class
        # code rather than by position
        for j, c in enumerate(m.classes_):
            proba[va, c] = p[:, j]
    return proba, y