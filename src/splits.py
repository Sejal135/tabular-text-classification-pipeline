"""Group-aware CV splitter with Category_5 forced into every training fold.

Two constraints shaped this. Rows with identical Col1 through Col7 share a
group_key and must move together, or a duplicate lands in training and its twin
in validation and the model scores from memory. And Category_5 has two rows in
the entire dataset, so it cannot be split at all.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

FORCED_CLASS = "Category_5"


def make_cv_splits(data, n_splits=3, seed=42):
    """Build stratified, group-aware cross-validation folds for `data`.

    Returns a list of (train_idx, val_idx) pairs as positional indices, ready
    to pass to cv_evaluate or straight into RandomizedSearchCV.

    Rows of FORCED_CLASS are appended to every training fold and never
    appear in validation. With 2 examples they cannot be split.
    """
    pos = np.arange(len(data))
    labels = data["label"].values
    forced = pos[labels == FORCED_CLASS]
    rest = pos[labels != FORCED_CLASS]

    # fold the remaining rows, keeping each group_key whole and each class
    # proportional; n_splits is capped at 3 because Category_6 has 11 rows
    sub = data.iloc[rest]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for tr, va in sgkf.split(sub, sub["label"], groups=sub["group_key"]):
        splits.append((np.concatenate([rest[tr], forced]), rest[va]))
    return splits