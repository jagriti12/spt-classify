"""Simulate 3D single-particle trajectories with known ground-truth motion type.

Four classes, matching Chatterjee et al., Chem. Biomed. Imaging (2025):
normal, directed, confined and anomalous (subdiffusive) motion.
"""

from __future__ import annotations

import numpy as np

CLASSES = ("normal", "directed", "confined", "anomalous")


def normal(n_steps: int, D: float = 0.1, dt: float = 1.0, rng=None) -> np.ndarray:
    """Brownian motion. Returns an (n_steps, 3) array of positions."""
    rng = np.random.default_rng(rng)
    sigma = np.sqrt(2 * D * dt)
    steps = rng.normal(0.0, sigma, size=(n_steps - 1, 3))
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


def directed(n_steps: int, D: float = 0.1, v: float = 0.15, dt: float = 1.0, rng=None) -> np.ndarray:
    """Brownian motion plus constant drift along a random direction."""
    rng = np.random.default_rng(rng)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    drift = v * dt * direction
    sigma = np.sqrt(2 * D * dt)
    steps = rng.normal(0.0, sigma, size=(n_steps - 1, 3)) + drift
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


def confined(n_steps: int, D: float = 0.1, radius: float = 1.0, dt: float = 1.0, rng=None) -> np.ndarray:
    """Diffusion inside a reflecting sphere of the given radius.

    A step that overshoots is folded back through the surface. One fold is only
    enough while the overshoot is less than the diameter; a step longer than
    that lands outside again, so keep folding until the particle is genuinely
    inside. This matters whenever sigma is comparable to the radius, which is
    exactly the tightly-confined regime worth simulating.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    rng = np.random.default_rng(rng)
    sigma = np.sqrt(2 * D * dt)
    pos = np.zeros(3)
    out = [pos.copy()]
    for _ in range(n_steps - 1):
        trial = pos + rng.normal(0.0, sigma, size=3)
        r = np.linalg.norm(trial)
        while r > radius:  # fold back through the sphere surface
            trial = trial * (2 * radius - r) / r
            r = np.linalg.norm(trial)
        pos = trial
        out.append(pos.copy())
    return np.asarray(out)


def _fgn(n: int, hurst: float, rng) -> np.ndarray:
    """One realization of fractional Gaussian noise via Cholesky factorization."""
    k = np.arange(n)
    cov = 0.5 * (
        np.abs(k[:, None] - k[None, :] + 1) ** (2 * hurst)
        - 2 * np.abs(k[:, None] - k[None, :]) ** (2 * hurst)
        + np.abs(k[:, None] - k[None, :] - 1) ** (2 * hurst)
    )
    cov[np.diag_indices(n)] = 1.0
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    return chol @ rng.normal(size=n)


def anomalous(n_steps: int, D: float = 0.1, hurst: float = 0.3, dt: float = 1.0, rng=None) -> np.ndarray:
    """Subdiffusive fractional Brownian motion (hurst < 0.5)."""
    rng = np.random.default_rng(rng)
    sigma = np.sqrt(2 * D * dt)
    steps = np.column_stack([sigma * _fgn(n_steps - 1, hurst, rng) for _ in range(3)])
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


_SIMULATORS = {
    "normal": normal,
    "directed": directed,
    "confined": confined,
    "anomalous": anomalous,
}


def add_localization_noise(track: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """Add isotropic Gaussian localization error, as a real microscope would."""
    rng = np.random.default_rng(rng)
    return track + rng.normal(0.0, sigma, size=track.shape)


def make_dataset(
    n_per_class: int = 250,
    n_steps: int = 100,
    loc_error: float = 0.02,
    seed: int | None = 0,
):
    """Build a labeled dataset of trajectories.

    Returns
    -------
    tracks : list of (n_steps, 3) arrays
    labels : list of str, one per track
    """
    rng = np.random.default_rng(seed)
    tracks, labels = [], []
    for name in CLASSES:
        sim = _SIMULATORS[name]
        for _ in range(n_per_class):
            track = sim(n_steps, rng=rng)
            if loc_error > 0:
                track = add_localization_noise(track, loc_error, rng=rng)
            tracks.append(track)
            labels.append(name)
    return tracks, labels
