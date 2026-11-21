"""Closed-loop simulation (paper Section 6).

The planner replans every `replan` seconds with a `tmax`-second horizon.  At the
start of every cycle the simulated (actual) shark is advanced by its own motion
model (for the Markov model: a sample from P(X_s,10)), and the new shark state
is given to the planner as a zero-uncertainty measurement.  The executed
trajectory is scored with the fitness function evaluated against the *actual*
shark position (a point distribution).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .fitness import step_fitness
from .motion import MotionModel, SparseMap
from .planner import ExpansivePlanner, PlannerConfig, Plan


@dataclass
class SimResult:
    times: np.ndarray            # executed sample times (every cfg.dt)
    auv: np.ndarray              # (N, A, 3)
    auv_dense: np.ndarray        # (M, A, 2) dense executed polyline
    shark: np.ndarray            # (N, 3)
    f_actual: np.ndarray         # (N,)
    score: float
    plans: list = field(default_factory=list)
    plan_times: list = field(default_factory=list)
    shark_at_plan: list = field(default_factory=list)
    n_fail: int = 0
    plan_runtime: float = 0.0


def default_obstacles(rng, n=4, keep_out=45.0, extent=(-70, 140, -70, 70)):
    obs = []
    while len(obs) < n:
        x, y = rng.uniform(extent[0], extent[1]), rng.uniform(extent[2], extent[3])
        r = rng.uniform(4, 10)
        if np.hypot(x, y) > keep_out + r and abs(y) > r + 8:
            obs.append([x, y, r])
    return np.array(obs)


def run_closed_loop(cfg: PlannerConfig, predict: MotionModel, actual: MotionModel,
                    total_time=100.0, replan=10.0, obstacles=None, seed=0,
                    keep_plans=False, shark0=(0.0, 0.0, 0.0)) -> SimResult:
    rng = np.random.default_rng(seed)
    planner = ExpansivePlanner(cfg, obstacles)
    shark = np.array(shark0, float)
    auv = planner.initial_states(shark[:2], rng=rng)
    A = cfg.n_auvs
    steps_per = int(round(replan / cfg.dt))
    ts, auvs, sharks, dense = [0.0], [auv.copy()], [shark.copy()], [auv[:, :2].copy()]
    plans, ptimes, splan = [], [], []
    n_fail, rt = 0, 0.0
    t = 0.0
    cyc = 0
    while t < total_time - 1e-9:
        pcfg = replace(cfg, seed=seed * 1000 + cyc)
        planner.cfg = pcfg
        plan = planner.plan(auv, predict, shark, keep_tree=keep_plans)
        rt += plan.runtime
        n_fail += (not plan.success)
        if keep_plans:
            plans.append(plan); ptimes.append(t); splan.append(shark.copy())
        # actual shark moves during the cycle
        shark_next = actual.sample_next(shark, replan, rng)
        S = cfg.edge_samples
        for k in range(1, steps_per + 1):
            if k < len(plan.states):
                auv = plan.states[k].copy()
                dense.extend(plan.paths[(k - 1) * S + 1: k * S + 1])
            else:
                dense.extend([auv[:, :2].copy()] * S)
            a = k / steps_per
            sh = np.array([shark[0] + a * (shark_next[0] - shark[0]),
                           shark[1] + a * (shark_next[1] - shark[1]), shark_next[2]])
            ts.append(t + k * cfg.dt); auvs.append(auv.copy()); sharks.append(sh)
        shark = shark_next
        t += replan
        cyc += 1
    ts = np.array(ts); auvs = np.array(auvs); sharks = np.array(sharks)
    f = np.array([step_fitness(a[:, :2], SparseMap.point(s[0], s[1]), cfg.h) for a, s in zip(auvs, sharks)])
    w = 2 * ts / ts[-1]
    score = float((w * f).sum() / w.sum())
    return SimResult(ts, auvs, np.array(dense), sharks, f, score, plans, ptimes, splan, n_fail, rt)
