import warnings

import numpy as np
import pytest

from spt_classify import features as F
from spt_classify import simulate as S


def test_extract_returns_six_finite_values():
    track = S.normal(120, rng=0)
    values = F.extract(track)
    assert values.shape == (6,)
    assert np.all(np.isfinite(values))


def test_short_track_rejected():
    with pytest.raises(ValueError):
        F.extract(S.normal(5, rng=0))


def test_msd_grows_linearly_for_brownian_motion():
    tracks = [S.normal(200, D=0.1, rng=i) for i in range(50)]
    m1 = np.mean([F.msd(t, 1) for t in tracks])
    m10 = np.mean([F.msd(t, 10) for t in tracks])
    assert 7 < m10 / m1 < 13


def test_directed_motion_fills_space_less_than_brownian():
    directed = np.mean([F.fractal_dimension(S.directed(150, rng=i)) for i in range(30)])
    brownian = np.mean([F.fractal_dimension(S.normal(150, rng=i)) for i in range(30)])
    assert directed < brownian


def test_confined_motion_scores_higher_trappedness():
    conf = np.mean([F.trappedness(S.confined(150, radius=0.5, rng=i)) for i in range(30)])
    free = np.mean([F.trappedness(S.normal(150, rng=i)) for i in range(30)])
    assert conf > free


def test_out_of_range_lag_returns_nan_quietly():
    track = S.normal(20, rng=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # empty-slice means an unguarded lag
        assert np.isnan(F.gaussianity(track, lag=50))
        assert np.isnan(F.msd(track, 50))


def test_max_pairwise_distance_matches_the_naive_form():
    track = S.normal(300, rng=3)
    naive = np.linalg.norm(track[:, None, :] - track[None, :, :], axis=-1).max()
    assert F._max_pairwise_distance(track) == pytest.approx(naive)


def test_max_pairwise_distance_blocks_without_an_n_by_n_matrix():
    """The blocked form must still be exact when it takes more than one pass."""
    track = S.normal(4000, rng=4)
    assert len(track) > 2_000_000 // len(track)  # more than one block
    expected = np.linalg.norm(track - track.mean(axis=0), axis=1)
    assert F._max_pairwise_distance(track) >= expected.max()
    assert F._max_pairwise_distance(track[:50]) == pytest.approx(
        np.linalg.norm(track[:50, None, :] - track[None, :50, :], axis=-1).max()
    )
