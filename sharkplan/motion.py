"""Shark motion models and the probability map P(X_s,t) (paper Section 4).

* MarkovModel      - empirical, heading-normalised transition kernel from the
                     shark data (optionally conditioned on speed or depth),
                     propagated by Markovian expansion (Eq. 5).
* ConstantVelocity - shark keeps its heading and speed.
* Stationary       - shark stays where it was last observed.

All models expose  `prob_maps(x0, y0, heading, times) -> list[SparseMap]`
(sparse cell lists used by the fitness function) and
`sample_next(state, dt, rng)` used to simulate the *actual* shark.

Sub-minute prediction: the data gives one-minute transitions. For t <= 60 s the
one-minute displacement samples are scaled by t/60 (straight-line motion within
a minute); beyond 60 s whole-minute kernels are composed by 2-D convolution,
which is exactly the Markovian sum of Eq. (5) for a translation-invariant
kernel. The kernel is rotated to the shark heading at the last measurement
(the paper: "translate and rotate the transition probabilities to be centred
on the state X_s,0").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .data import Transitions, build_transitions

SPEED_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 15.0)]


@dataclass
class SparseMap:
    """Probability mass on a set of cell centres (world frame)."""
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray

    @staticmethod
    def from_grid(P, xs, ys, mass=0.995, max_cells=400):
        flat = P.ravel()
        order = np.argsort(flat)[::-1]
        c = np.cumsum(flat[order])
        k = int(min(np.searchsorted(c, mass * c[-1]) + 1, max_cells, len(order)))
        idx = order[:k]
        iy, ix = np.unravel_index(idx, P.shape)
        p = flat[idx]
        return SparseMap(xs[ix], ys[iy], p / p.sum())

    @staticmethod
    def point(x, y):
        return SparseMap(np.array([x], float), np.array([y], float), np.array([1.0]))

    def argmax(self):
        i = int(np.argmax(self.p))
        return self.x[i], self.y[i]

    def mean(self):
        return float(self.p @ self.x), float(self.p @ self.y)


def histogram_kernel(dx, dy, cell=5.0, half=150.0, weights=None):
    """2-D histogram of displacements on a square grid centred at 0."""
    n = int(round(half / cell))
    edges = (np.arange(-n, n + 2) - 0.5) * cell
    H, _, _ = np.histogram2d(dy, dx, bins=[edges, edges], weights=weights)
    centers = np.arange(-n, n + 1) * cell
    s = H.sum()
    return (H / s if s > 0 else H), centers


class MotionModel:
    name = "base"

    def prob_maps(self, x0, y0, heading, times, **kw):
        raise NotImplementedError

    def sample_next(self, state, dt, rng):
        raise NotImplementedError


class Stationary(MotionModel):
    name = "Stationary"

    def prob_maps(self, x0, y0, heading, times, **kw):
        return [SparseMap.point(x0, y0) for _ in times]

    def sample_next(self, state, dt, rng):
        return np.array(state, float)


class ConstantVelocity(MotionModel):
    name = "Constant Velocity"

    def __init__(self, speed=0.75):
        self.speed = speed

    def prob_maps(self, x0, y0, heading, times, **kw):
        c, s = np.cos(heading), np.sin(heading)
        return [SparseMap.point(x0 + self.speed * t * c, y0 + self.speed * t * s) for t in times]

    def sample_next(self, state, dt, rng):
        x, y, th = state
        return np.array([x + self.speed * dt * np.cos(th), y + self.speed * dt * np.sin(th), th])


class MarkovModel(MotionModel):
    """Empirical Markov transition model from the shark data."""
    name = "Markov"

    def __init__(self, transitions: Transitions | None = None, speed: float | None = 0.75,
                 cell=2.0, rng_seed=0, scale: float = 1.0):
        tr = transitions if transitions is not None else build_transitions()
        if scale != 1.0:
            # scale < 1 reproduces the hypothesis that the original code used
            # 5 m kernel *cells* as metres (scale = 1/5); see README.
            tr = Transitions(tr.dx * scale, tr.dy * scale, tr.speed0, tr.speed1 * scale,
                             tr.turn, tr.depth, tr.shark, tr.dt)
        self.all = tr
        if speed is not None:
            lo, hi = next(b for b in SPEED_BINS if b[0] <= speed < b[1])
            tr = tr.by_speed(lo, hi)
        self.tr = tr
        self.speed = speed
        self.cell = max(0.5, cell * min(scale, 1.0))
        self._cache = {}
        self._rng = np.random.default_rng(rng_seed)

    # --- distributions -------------------------------------------------
    def heading_frame_grid(self, t, half=None):
        """P(dx,dy) after t seconds, shark at origin heading +x."""
        cell = self.cell
        if half is None:
            half = max(30.0, 1.6 * 60 * np.percentile(self.tr.speed1, 99) * max(t, 60) / 60)
        m, r = divmod(t, 60.0)
        m = int(m)
        if m == 0:
            return histogram_kernel(self.tr.dx * (r / 60), self.tr.dy * (r / 60), cell, half)
        K, cen = histogram_kernel(self.tr.dx, self.tr.dy, cell, half)
        P = K
        for _ in range(m - 1):
            P = fftconvolve(P, K, mode="same")
        if r > 1e-9:
            Kr, _ = histogram_kernel(self.tr.dx * (r / 60), self.tr.dy * (r / 60), cell, half)
            P = fftconvolve(P, Kr, mode="same")
        P = np.clip(P, 0, None)
        return P / P.sum(), cen

    def prob_maps(self, x0, y0, heading, times, mass=0.99, max_cells=300):
        out = []
        c, s = np.cos(heading), np.sin(heading)
        for t in times:
            if t <= 0:
                out.append(SparseMap.point(x0, y0))
                continue
            key = (round(float(t), 3), mass, max_cells)
            if key not in self._cache:
                P, cen = self.heading_frame_grid(t)
                self._cache[key] = SparseMap.from_grid(P, cen, cen, mass, max_cells)
            sm = self._cache[key]
            out.append(SparseMap(x0 + c * sm.x - s * sm.y, y0 + s * sm.x + c * sm.y, sm.p))
        return out

    def sample_next(self, state, dt, rng):
        """Draw the actual shark's next state from P(X_s,dt) (paper Sec. 6)."""
        x, y, th = state
        k = rng.integers(len(self.tr))
        f = dt / 60.0
        dx, dy = self.tr.dx[k] * f, self.tr.dy[k] * f
        c, s = np.cos(th), np.sin(th)
        nx, ny = x + c * dx - s * dy, y + s * dx + c * dy
        nth = th + np.arctan2(dy, dx) if np.hypot(dx, dy) > 1e-6 else th
        return np.array([nx, ny, nth])


def make_model(name: str, speed=0.75, transitions=None, scale=1.0) -> MotionModel:
    n = name.lower().replace(" ", "").replace("_", "")
    if n.startswith("stat"):
        return Stationary()
    if n.startswith("const") or n == "cv":
        return ConstantVelocity(speed)
    if n.startswith("markov"):
        return MarkovModel(transitions, speed=speed, scale=scale)
    raise ValueError(name)
