"""Fitness function (paper Eqs. 1-4)."""
from __future__ import annotations

import numpy as np

from .motion import SparseMap

H_DEFAULT = 50.0     # desired upper AUV-shark distance h (m)
GAMMA = 0.05         # smoothing factor


def D(l, h=H_DEFAULT, gamma=GAMMA):
    """Eq. (3): ~1 for h/5 < l < h, ~0 when too close or too far."""
    l = np.asarray(l, float)
    a = np.clip((l - h) / (gamma * h), -60, 60)
    b = np.clip(((h / 5) - l) / (gamma * h), -60, 60)
    return 1.0 / (1.0 + np.exp(a)) + 1.0 / (1.0 + np.exp(b)) - 1.0


def C(theta_bar):
    """Eq. (4): reward for AUVs evenly spread around the target."""
    return (1.0 / (1.0 + np.asarray(theta_bar))) ** 3


def angle_error(bx, by, rx, ry):
    """theta_bar for every cell: mean |gap - 2pi/A| of the bearings from the
    cell to each AUV, taken between angularly adjacent AUVs (zero when the AUVs
    perfectly surround the cell)."""
    A = rx.shape[-1]
    beta = np.arctan2(ry[None, :] - by[:, None], rx[None, :] - bx[:, None])
    beta = np.sort(beta, axis=1)
    gaps = np.diff(np.concatenate([beta, beta[:, :1] + 2 * np.pi], axis=1), axis=1)
    return np.abs(gaps - 2 * np.pi / A).mean(axis=1)


def step_fitness(robots_xy: np.ndarray, pmap: SparseMap, h=H_DEFAULT, pairs=None):
    """Eq. (2) for one time step, normalised to [0, 1] by the number of pairs.

    robots_xy : (A, 2)
    """
    rx, ry = robots_xy[:, 0], robots_xy[:, 1]
    A = len(rx)
    q = np.hypot(rx[None, :] - pmap.x[:, None], ry[None, :] - pmap.y[:, None])   # (K, A)
    d = D(q, h)
    if pairs is None:
        pairs = np.triu_indices(A, 1)
    pair_term = np.minimum(d[:, pairs[0]], d[:, pairs[1]]).mean(axis=1)            # (K,)
    c = C(angle_error(pmap.x, pmap.y, rx, ry))
    return float(np.sum(pmap.p * c * pair_term))


def time_weights(times, tmax):
    """rho * t / tmax with rho = 2 (slides)."""
    return 2.0 * np.asarray(times, float) / tmax


def trajectory_fitness(robot_xy_seq, pmaps, times, tmax, h=H_DEFAULT):
    """Eq. (1), normalised by the weight sum so the maximum is 1."""
    w = time_weights(times, tmax)
    f = np.array([step_fitness(r, p, h) for r, p in zip(robot_xy_seq, pmaps)])
    return float((w * f).sum() / max(w.sum(), 1e-12)), f
