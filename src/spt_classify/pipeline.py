"""Feature selection plus a bagged ensemble for 3D trajectory classification."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES, extract_many

SEARCH_SPACE = {
    "clf__n_estimators": [30, 50, 100, 200],
    "clf__max_samples": [0.5, 0.7, 1.0],
    "clf__max_features": [0.5, 0.7, 1.0],
    "clf__estimator__max_depth": [4, 8, 16, None],
    "clf__estimator__min_samples_leaf": [1, 2, 5],
}


def build_pipeline(random_state: int = 0) -> Pipeline:
    """Impute, scale, then bag decision trees."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(random_state=random_state),
                    n_estimators=100,
                    random_state=random_state,
                ),
            ),
        ]
    )


def rank_features(X: np.ndarray, y, random_state: int = 0):
    """Rank descriptors by mutual information with the class label.

    A fast stand-in for the mRMR / NCA / ReliefF consensus used in the paper.
    Returns a list of (name, score), highest first.
    """
    X = np.asarray(X, dtype=float)
    # Impute per descriptor, as the pipeline's SimpleImputer does. A single
    # median pooled over all six columns would fill a missing kurtosis (~3)
    # with a jump length (~0.5) and invent structure that isn't there.
    finite = np.isfinite(X)
    clean = np.where(finite, X, np.nan)
    medians = np.zeros(X.shape[1])  # a descriptor that is NaN everywhere -> 0
    usable = finite.any(axis=0)
    if usable.any():
        medians[usable] = np.nanmedian(clean[:, usable], axis=0)
    X = np.where(finite, X, medians)

    scores = mutual_info_classif(X, y, random_state=random_state)
    # mutual_info_classif jitters continuous features before its kNN estimate,
    # so a descriptor with no variance still scores ~0.09 off pure noise. A
    # constant column carries no information by definition; say so.
    scores[X.std(axis=0) == 0] = 0.0
    order = np.argsort(scores)[::-1]
    return [(FEATURE_NAMES[i], float(scores[i])) for i in order]


def tune(X: np.ndarray, y, n_iter: int = 20, cv: int = 5, random_state: int = 0):
    """Randomized hyperparameter search. Returns the fitted search object."""
    search = RandomizedSearchCV(
        build_pipeline(random_state),
        SEARCH_SPACE,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def classify_tracks(tracks, labels, tune_model: bool = False, cv: int = 5, random_state: int = 0):
    """End-to-end: trajectories in, cross-validated accuracy and a fitted model out.

    With ``tune_model=True`` the search and the reported score share the same
    folds, so ``accuracy_mean`` is optimistic and is a model-selection number,
    not a generalization estimate. Hold out a test set, or wrap :func:`tune` in
    an outer ``cross_val_score``, before quoting it. The default path does no
    tuning and the score is honest.
    """
    X = extract_many(tracks)
    y = np.asarray(labels)
    model = tune(X, y, cv=cv, random_state=random_state).best_estimator_ if tune_model else build_pipeline(random_state)
    scores = cross_val_score(model, X, y, cv=cv)
    model.fit(X, y)
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "feature_ranking": rank_features(X, y, random_state),
        "model": model,
    }
