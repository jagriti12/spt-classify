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
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    scores = mutual_info_classif(X, y, random_state=random_state)
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
    """End-to-end: trajectories in, cross-validated accuracy and a fitted model out."""
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
