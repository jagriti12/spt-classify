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
