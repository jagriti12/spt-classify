import numpy as np
import pytest

from spt_classify import simulate as S


@pytest.mark.parametrize("radius", [0.2, 0.3, 0.5, 1.0])
def test_confined_stays_inside_the_sphere(radius):
    """A single fold only works while the overshoot is under one diameter.

    With radius=0.3 and the default D, the old one-shot reflection put a third
    of all points outside the sphere, some of them 9x the confinement radius
    away -- so 'confined' tracks were not confined at all.
    """
    for seed in range(40):
        track = S.confined(200, radius=radius, rng=seed)
        r = np.linalg.norm(track, axis=1)
        assert r.max() <= radius + 1e-9, f"escaped to {r.max():.3f} with radius={radius}"


def test_confined_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        S.confined(50, radius=0.0)


def test_tighter_confinement_gives_a_smaller_cloud():
    tight = np.mean([np.linalg.norm(S.confined(200, radius=0.3, rng=i), axis=1).max() for i in range(20)])
    loose = np.mean([np.linalg.norm(S.confined(200, radius=1.0, rng=i), axis=1).max() for i in range(20)])
    assert tight < loose


def test_simulators_return_the_requested_shape():
    for name, sim in S._SIMULATORS.items():
        track = sim(60, rng=0)
        assert track.shape == (60, 3), name


def test_make_dataset_is_balanced_and_reproducible():
    tracks_a, labels_a = S.make_dataset(n_per_class=5, n_steps=40, seed=7)
    tracks_b, labels_b = S.make_dataset(n_per_class=5, n_steps=40, seed=7)
    assert labels_a == labels_b
    assert all(np.array_equal(a, b) for a, b in zip(tracks_a, tracks_b))
    assert {labels_a.count(c) for c in S.CLASSES} == {5}
