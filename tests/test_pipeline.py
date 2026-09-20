from spt_classify import classify_tracks, make_dataset
from spt_classify.features import FEATURE_NAMES


def test_classifier_beats_chance_on_simulated_data():
    tracks, labels = make_dataset(n_per_class=60, n_steps=80, seed=1)
    result = classify_tracks(tracks, labels, cv=3)
    assert result["accuracy_mean"] > 0.6


def test_feature_ranking_covers_every_descriptor():
    tracks, labels = make_dataset(n_per_class=40, n_steps=80, seed=2)
    result = classify_tracks(tracks, labels, cv=3)
    assert {name for name, _ in result["feature_ranking"]} == set(FEATURE_NAMES)
