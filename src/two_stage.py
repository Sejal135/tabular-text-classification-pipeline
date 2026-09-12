"""Two-stage classifier: Category_1 vs rest, then a rest-only multiclass model.

Built, measured, and rejected. Scored 0.6509 macro F1 plain and 0.6791 with
weighted stage A, against 0.7554 for the single-stage random forest. Kept in
the repository because the reason it lost is itself a finding: collapsing to a
binary first stage flattens Category_6's eleven rows into the same bucket as
Category_2's 568, and eight of those eleven never reach stage B.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """Stage A separates the majority class. Stage B classifies what A rejects,
    trained only on minority rows so their proportions rise sharply."""

    def __init__(self, stage_a, stage_b, majority_code, a_weights=None):
        """Store the two estimators unmodified, as scikit-learn's clone
        requires. a_weights optionally maps each original class code to a
        sample weight for stage A."""
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.majority_code = majority_code
        self.a_weights = a_weights          # {class_code: weight} or None

    def fit(self, X, y):
        """Fit stage A on the binary majority-vs-rest target, then fit stage B
        on the minority rows alone. Both estimators are cloned first so a
        single instance can be reused across cross-validation folds."""
        y = np.asarray(y)
        self.a_ = clone(self.stage_a)
        self.b_ = clone(self.stage_b)

        binary = (y == self.majority_code).astype(int)
        if self.a_weights is None:
            self.a_.fit(X, binary)
        else:
            # weight each row by its ORIGINAL class, so 11 Category_6 rows
            # are not flattened into the same bucket as 568 Category_2 rows
            sw = np.array([self.a_weights[c] for c in y])
            self.a_.fit(X, binary, clf__sample_weight=sw)

        minor = np.flatnonzero(y != self.majority_code)
        self.b_.fit(X.iloc[minor], y[minor])
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict the majority class by default, and ask stage B only for the
        rows stage A rejected. A row stage A keeps never reaches stage B, which
        is where this design loses its rare-class recall."""
        out = np.full(len(X), self.majority_code, dtype=int)
        idx = np.flatnonzero(self.a_.predict(X) == 0)
        if len(idx):
            out[idx] = self.b_.predict(X.iloc[idx])
        return out

    def predict_proba(self, X):
        """Combine both stages into one probability matrix: the majority class
        takes stage A's probability, and the remainder is split between the
        minority classes according to stage B. Columns are aligned to
        self.classes_, since stage B only knows the minority labels."""
        p_major = self.a_.predict_proba(X)[:, 1]
        pb = self.b_.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)))
        for j, c in enumerate(self.b_.classes_):
            out[:, np.searchsorted(self.classes_, c)] = (1 - p_major) * pb[:, j]
        out[:, np.searchsorted(self.classes_, self.majority_code)] = p_major
        return out