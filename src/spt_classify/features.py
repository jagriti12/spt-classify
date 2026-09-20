"""The six trajectory descriptors selected in Chatterjee et al. (2025).

Selection used mRMR, NCA and ReliefF over a larger candidate set; these six
survived all three. Each function takes an (n_steps, d) array of positions.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "kurtosis",
    "fractal_dimension",
    "msd_ratio",
    "gaussianity",
    "jump_length",
    "trappedness",
)


def _steps(track: np.ndarray) -> np.ndarray:
    return np.diff(track, axis=0)


def msd(track: np.ndarray, lag: int) -> float:
    """Mean squared displacement at a given lag, in the track's length units."""
    if lag < 1 or lag >= len(track):
        return np.nan
    disp = track[lag:] - track[:-lag]
    return float(np.mean(np.sum(disp**2, axis=1)))


def kurtosis(track: np.ndarray) -> float:
    """Kurtosis of positions projected onto the dominant axis of the gyration tensor.

    Separates confined motion (platykurtic) from directed motion (leptokurtic).
    """
    centered = track - track.mean(axis=0)
    gyration = centered.T @ centered / len(track)
    _, vecs = np.linalg.eigh(gyration)
    projection = centered @ vecs[:, -1]
    sd = projection.std()
    if sd == 0:
        return np.nan
    return float(np.mean(((projection - projection.mean()) / sd) ** 4))


def fractal_dimension(track: np.ndarray) -> float:
    """Path-filling dimension: ~1 for directed, ~2 for random, ~3 for confined."""
    n = len(track)
    dists = np.linalg.norm(track[:, None, :] - track[None, :, :], axis=-1)
    d_max = dists.max()
    path_length = np.sum(np.linalg.norm(_steps(track), axis=1))
    if d_max <= 0 or path_length <= 0:
        return np.nan
    denom = np.log(n * d_max / path_length)
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.log(n) / denom)


def msd_ratio(track: np.ndarray, lag1: int = 1, lag2: int = 10) -> float:
    """Departure from linear MSD growth. Zero for normal diffusion."""
    m1, m2 = msd(track, lag1), msd(track, lag2)
    if not np.isfinite(m1) or not np.isfinite(m2) or m2 == 0:
        return np.nan
    return float(m1 / m2 - lag1 / lag2)


def gaussianity(track: np.ndarray, lag: int = 1) -> float:
    """Fourth-moment deviation from a Gaussian displacement distribution."""
    disp = track[lag:] - track[:-lag]
    r2 = np.sum(disp**2, axis=1)
    mean_r2 = np.mean(r2)
    if mean_r2 == 0:
        return np.nan
    d = track.shape[1]
    return float(np.mean(r2**2) / ((1 + 2 / d) * mean_r2**2) - 1)


def jump_length(track: np.ndarray) -> float:
    """Mean frame-to-frame displacement."""
    return float(np.mean(np.linalg.norm(_steps(track), axis=1)))


def trappedness(track: np.ndarray, dt: float = 1.0) -> float:
    """Probability that a particle of this diffusivity stayed within its observed radius.

    Follows Saxton's confinement estimator: near 1 means trapped.
    """
    n = len(track)
    m1 = msd(track, 1)
    if not np.isfinite(m1) or m1 <= 0:
        return np.nan
    d = track.shape[1]
    diffusivity = m1 / (2 * d * dt)
    dists = np.linalg.norm(track[:, None, :] - track[None, :, :], axis=-1)
    r0 = dists.max() / 2
    if r0 <= 0:
        return np.nan
    return float(1 - np.exp(0.2048 - 0.25117 * (diffusivity * n * dt / r0**2)))


_FEATURES = (kurtosis, fractal_dimension, msd_ratio, gaussianity, jump_length, trappedness)


def extract(track: np.ndarray) -> np.ndarray:
    """All six descriptors for one trajectory, in FEATURE_NAMES order."""
    track = np.asarray(track, dtype=float)
    if track.ndim != 2 or len(track) < 12:
        raise ValueError("track must be (n_steps, d) with at least 12 points")
    return np.array([f(track) for f in _FEATURES], dtype=float)


def extract_many(tracks) -> np.ndarray:
    """Feature matrix of shape (n_tracks, 6)."""
    return np.vstack([extract(t) for t in tracks])
