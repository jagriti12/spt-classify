import warnings

import numpy as np
import pytest

from spt_classify import classify_tracks, make_dataset
from spt_classify.features import FEATURE_NAMES, extract_many
from spt_classify.pipeline import rank_features


def test_classifier_beats_chance_on_simulated_data():
    tracks, labels = make_dataset(n_per_class=60, n_steps=80, seed=1)
    result = classify_tracks(tracks, labels, cv=3)
    assert result["accuracy_mean"] > 0.6


def test_feature_ranking_covers_every_descriptor():
    tracks, labels = make_dataset(n_per_class=40, n_steps=80, seed=2)
    result = classify_tracks(tracks, labels, cv=3)
    assert {name for name, _ in result["feature_ranking"]} == set(FEATURE_NAMES)


def test_ranking_imputes_each_descriptor_separately():
    """A pooled median would fill a missing kurtosis (~3) with a jump length (~0.5)."""
    tracks, labels = make_dataset(n_per_class=40, n_steps=60, seed=4)
    X = extract_many(tracks)
    holes = X.copy()
    holes[::7, 0] = np.nan  # knock out some kurtosis values

    by_column = holes.copy()
    by_column[np.isnan(by_column)] = np.nanmedian(holes[:, 0])
    pooled = holes.copy()
    pooled[np.isnan(pooled)] = np.nanmedian(holes)  # what the old code did

    assert not np.allclose(by_column, pooled)  # the two really do differ
    assert rank_features(holes, labels) == rank_features(by_column, labels)


def test_ranking_survives_an_all_nan_descriptor():
    tracks, labels = make_dataset(n_per_class=20, n_steps=40, seed=3)
    X = extract_many(tracks)
    X[:, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an all-NaN median warns if unguarded
        ranking = rank_features(X, labels)
    assert {name for name, _ in ranking} == set(FEATURE_NAMES)
    assert dict(ranking)["kurtosis"] == pytest.approx(0.0, abs=1e-9)
