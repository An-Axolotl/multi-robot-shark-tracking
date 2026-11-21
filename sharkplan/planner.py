"""Expansive multi-AUV RRT planners (paper Section 5, Algorithm 1).

    variant="ISE"  Independent State Expansion - every AUV gets its own random
                   arc+straight motion from its parent state (Eq. 6).
    variant="JSE"  Joint State Expansion - the arc+straight motion is applied to
                   the cluster centre X_C and the AUVs are placed on a circle of
                   radius rho around it (Eq. 7), radius and per-AUV offsets
                   jittered by +/-0.5 m, velocities rescaled to respect v_max.
    variant="RRT"  Adapted baseline RRT - node = collection of robots, nearest
                   node by mean robot distance to a random configuration, each
                   robot steers toward its random target, no high-fitness test.

Algorithm 1:
    G <- InitializeTree(n0)
    while planTime < budget:
        n_rand <- NodeSelection(G)
        n_new  <- NodeExpansion(n_rand)
        if NoCollision(n_rand, n_new) and HighFitness(n_new):
            G <- n_new
            if InEndGame(n_new) and Fitness(n_new) > F*: n*, F* <- n_new, Fitness(n_new)
    return TrajectoryFromNode(n*)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .fitness import C, D, H_DEFAULT, step_fitness
from .motion import MotionModel, SparseMap


@dataclass
class PlannerConfig:
    variant: str = "JSE"
    n_auvs: int = 4
    dt: float = 2.0                 # expansion time (s)
    tmax: float = 30.0              # trajectory horizon (s)
    v_min: float = 0.25
    v_max: float = 2.5              # Iver3
    w_max: float = np.pi / 4        # rad/s (RRT baseline uses pi/2)
    time_budget: float = 1.0        # planTimeBudget (s)
    max_nodes: int = 20000
    high_fitness: float | None = 0.8
    # 'parent': f(n_new) >= thr * f(n_random)   (child keeps >= 80% of parent fitness)
    # 'best'  : f(n_new) >= thr * best f found so far at that time step
    high_fitness_mode: str = "best"
    # NodeSelection only expands from high-fitness nodes: open nodes with
    # f >= high_fitness * best f at their time step (when high_fitness is set)
    expand_high_fitness: bool = True
    h: float = H_DEFAULT
    rho0: float = 30.0              # JSE initial radius
    rho_min: float = 12.0
    rho_max: float = 48.0
    jitter: float = 0.5             # JSE radius / per-AUV jitter (m)
    sample_margin: float = 20.0     # node-selection sampling box margin (m)
    edge_samples: int = 5
    batch: int = 64                 # expansions evaluated per vectorised round
    obstacle_margin: float = 1.0
    seed: int = 0


@dataclass
class Plan:
    times: np.ndarray               # (S+1,)
    states: np.ndarray              # (S+1, A, 3) x, y, theta
    paths: np.ndarray               # (S*k+1, A, 2) dense polyline
    fitness: float
    success: bool
    n_nodes: int
    iterations: int
    runtime: float
    time_to_first: float | None
    tree: "Tree | None" = None


class Tree:
    def __init__(self, A, cap=4096, S=5):
        self.A, self.S, self.n = A, S, 0
        self.state = np.zeros((cap, A, 3))
        self.center = np.zeros((cap, 3))
        self.rho = np.zeros(cap)
        self.off = np.zeros((cap, A))
        self.tidx = np.zeros(cap, int)
        self.score = np.zeros(cap)          # accumulated sum w_t f_t
        self.f = np.zeros(cap)
        self.parent = np.full(cap, -1)
        self.edge = np.zeros((cap, A, S + 1, 2))

    def _grow(self):
        cap = len(self.tidx) * 2
        for k in ["state", "center", "rho", "off", "tidx", "score", "f", "parent", "edge"]:
            a = getattr(self, k)
            b = np.zeros((cap,) + a.shape[1:], a.dtype)
            if k == "parent":
                b[:] = -1
            b[: len(a)] = a
            setattr(self, k, b)

    def add(self, **kw):
        if self.n == len(self.tidx):
            self._grow()
        i = self.n
        for k, v in kw.items():
            getattr(self, k)[i] = v
        self.n += 1
        return i

    def path(self, i):
        out = []
        while i >= 0:
            out.append(i)
            i = self.parent[i]
        return out[::-1]


def arc_motion(x, y, th, v, dth, dt, S):
    """Eq. (6): constant-velocity arc (heading change dth over dt/2) followed by
    a straight segment (dt/2).  Vectorised over AUVs. Returns end state and an
    (A, S+1, 2) polyline."""
    d = (v * dt / 2)[:, None]
    s = np.linspace(0, 1, S + 1)[None, :]
    # arc part on the first half of the samples, straight on the second half
    u = np.clip(2 * s, 0, 1)
    w = np.clip(2 * s - 1, 0, 1)
    small = np.abs(dth) < 1e-6
    r = np.where(small[:, None], 0, d / np.where(small, 1, dth)[:, None])
    th0 = th[:, None]
    a = th0 + dth[:, None] * u
    ax = np.where(small[:, None], x[:, None] + d * u * np.cos(th0),
                  x[:, None] - r * np.sin(th0) + r * np.sin(a))
    ay = np.where(small[:, None], y[:, None] + d * u * np.sin(th0),
                  y[:, None] + r * np.cos(th0) - r * np.cos(a))
    th1 = th0 + dth[:, None]
    px = ax + d * w * np.cos(th1)
    py = ay + d * w * np.sin(th1)
    end = np.stack([px[:, -1], py[:, -1], th + dth], 1)
    return end, np.stack([px, py], -1)


def batch_fitness(R, pmap: SparseMap, h=H_DEFAULT, pairs=None):
    """Vectorised Eq. (2) for many robot configurations R (M, A, 2)."""
    M, A, _ = R.shape
    if pairs is None:
        pairs = np.triu_indices(A, 1)
    dx = R[:, None, :, 0] - pmap.x[None, :, None]
    dy = R[:, None, :, 1] - pmap.y[None, :, None]           # (M, K, A)
    d = D(np.hypot(dx, dy), h)
    pt = np.minimum(d[..., pairs[0]], d[..., pairs[1]]).mean(-1)          # (M, K)
    beta = np.sort(np.arctan2(dy, dx), axis=-1)
    gaps = np.diff(np.concatenate([beta, beta[..., :1] + 2 * np.pi], -1), axis=-1)
    c = C(np.abs(gaps - 2 * np.pi / A).mean(-1))
    return (pmap.p[None, :] * c * pt).sum(-1)


def batch_crossings(E, pairs):
    """E (M, A, S+1, 2) -> (M,) True if any two AUV edges cross or end < 1 m apart."""
    P = E[:, pairs[0]]
    Q = E[:, pairs[1]]                                       # (M, P, S+1, 2)
    a, b = P[:, :, :-1, None], P[:, :, 1:, None]
    c, d = Q[:, :, None, :-1], Q[:, :, None, 1:]

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    cross = ((o1 * o2 < -1e-9) & (o3 * o4 < -1e-9)).any(axis=(1, 2, 3))
    close = (np.hypot(*(P[:, :, -1] - Q[:, :, -1]).transpose(2, 0, 1)) < 1.0).any(1)
    return cross | close


class ExpansivePlanner:
    def __init__(self, cfg: PlannerConfig, obstacles=None):
        self.cfg = cfg
        self.obs = np.zeros((0, 3)) if obstacles is None else np.asarray(obstacles, float).reshape(-1, 3)

    def initial_states(self, shark_xy, origin=None, rng=None):
        """ISE/RRT: AUVs co-located with the shark; JSE: evenly on a circle of radius rho0."""
        A, c = self.cfg.n_auvs, self.cfg
        o = np.asarray(shark_xy if origin is None else origin, float)
        b = 2 * np.pi * np.arange(A) / A
        if c.variant == "JSE":
            return np.stack([o[0] + c.rho0 * np.cos(b), o[1] + c.rho0 * np.sin(b), b + np.pi / 2], 1)
        jit = np.zeros((A, 2)) if rng is None else rng.normal(0, 0.5, (A, 2))
        return np.column_stack([o[0] + jit[:, 0], o[1] + jit[:, 1], b])

    def obstacle_hit(self, E):
        if not len(self.obs):
            return np.zeros(len(E), bool)
        d = np.hypot(E[..., 0, None] - self.obs[:, 0], E[..., 1, None] - self.obs[:, 1])
        return (d < self.obs[:, 2] + self.cfg.obstacle_margin).any(axis=(1, 2, 3))

    # ------------------------------------------------------------------
    def plan(self, start_states, model: MotionModel, shark_state, keep_tree=True,
             pmaps=None) -> Plan:
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        A, S, B = c.n_auvs, c.edge_samples, c.batch
        n_steps = int(round(c.tmax / c.dt))
        times = np.arange(n_steps + 1) * c.dt
        if pmaps is None:
            pmaps = model.prob_maps(shark_state[0], shark_state[1], shark_state[2], times)
        w = 2.0 * times / c.tmax
        wsum = np.cumsum(w)
        pairs = np.triu_indices(A, 1)
        beta = 2 * np.pi * np.arange(A) / A
        ring_dir = np.stack([np.cos(beta), np.sin(beta)], 1)            # (A, 2)

        x0 = np.asarray(start_states, float)
        mx = np.array([p.mean()[0] for p in pmaps])
        my = np.array([p.mean()[1] for p in pmaps])
        lo = np.array([min(mx.min(), x0[:, 0].min()), min(my.min(), x0[:, 1].min())]) - c.sample_margin
        hi = np.array([max(mx.max(), x0[:, 0].max()), max(my.max(), x0[:, 1].max())]) + c.sample_margin

        T = Tree(A, cap=max(1024, c.max_nodes + B + 1), S=S)
        cen = np.array([x0[:, 0].mean(), x0[:, 1].mean(), 0.0])
        if c.variant == "JSE":
            # project the AUVs onto the formation ring around their centroid
            r = (x0[:, :2] - cen[:2]) @ np.eye(2)
            r_i = np.einsum("ad,ad->a", r, ring_dir)
            rho0 = float(np.clip(r_i.mean(), c.rho_min, c.rho_max))
            off0 = np.clip(r_i - rho0, -5, 5)
            # cluster heading: direction to the predicted shark motion
            cen[2] = np.arctan2(my[-1] - my[0], mx[-1] - mx[0]) if np.hypot(mx[-1] - mx[0], my[-1] - my[0]) > 1 else 0.0
        else:
            rho0, off0 = 0.0, np.zeros(A)
        f0 = batch_fitness(x0[None, :, :2], pmaps[0], c.h, pairs)[0]
        T.add(state=x0, center=cen, rho=rho0, off=off0, tidx=0, score=0.0, f=f0,
              edge=np.repeat(x0[:, None, :2], S + 1, axis=1))
        best_f = np.zeros(n_steps + 1)
        best_f[0] = f0
        best_end, best_F = -1, -np.inf
        t_start = time.perf_counter()
        t_first = None
        it = 0
        s_lin = np.linspace(0, 1, S + 1)[None, None, :, None]
        while time.perf_counter() - t_start < c.time_budget and T.n < c.max_nodes:
            n = T.n
            open_ = T.tidx[:n] < n_steps
            if not open_.any():
                break
            it += B
            # ---- NodeSelection (B random samples -> nearest open nodes) ----
            if c.variant == "JSE":
                q = rng.uniform(lo, hi, size=(B, 2))
                d = np.hypot(T.center[None, :n, 0] - q[:, None, 0], T.center[None, :n, 1] - q[:, None, 1])
            else:
                q = rng.uniform(lo, hi, size=(B, A, 2))
                dd = np.hypot(T.state[None, :n, :, 0] - q[:, None, :, 0], T.state[None, :n, :, 1] - q[:, None, :, 1])
                d = dd.mean(-1) if c.variant == "RRT" else dd.max(-1)
            cand = open_
            if c.expand_high_fitness and c.high_fitness is not None and c.variant != "RRT":
                hf = T.f[:n] >= c.high_fitness * best_f[T.tidx[:n]] - 1e-12
                if (open_ & hf).any():
                    cand = open_ & hf
            d[:, ~cand] = np.inf
            idx = d.argmin(1)                                                 # (B,)
            par = T.state[idx]                                                # (B, A, 3)
            k = T.tidx[idx] + 1

            # ---- NodeExpansion ---------------------------------------------
            extra = {}
            if c.variant in ("ISE", "RRT"):
                if c.variant == "ISE":
                    v = rng.uniform(c.v_min, c.v_max, (B, A))
                    dth = rng.uniform(-1, 1, (B, A)) * c.w_max * c.dt / 2
                else:
                    wmax = np.pi / 2
                    goal = np.arctan2(q[..., 1] - par[..., 1], q[..., 0] - par[..., 0])
                    err = (goal - par[..., 2] + np.pi) % (2 * np.pi) - np.pi
                    dth = np.clip(err, -wmax * c.dt / 2, wmax * c.dt / 2)
                    v = np.full((B, A), c.v_max)
                new, E = arc_motion(par[..., 0].ravel(), par[..., 1].ravel(), par[..., 2].ravel(),
                                    v.ravel(), dth.ravel(), c.dt, S)
                new = new.reshape(B, A, 3)
                E = E.reshape(B, A, S + 1, 2)
                bad = self.obstacle_hit(E) | batch_crossings(E, pairs)
            else:  # JSE
                cs = T.center[idx]
                vc = rng.uniform(c.v_min, c.v_max, B)
                dthc = rng.uniform(-1, 1, B) * c.w_max * c.dt / 2
                rho_old, off_old = T.rho[idx], T.off[idx]
                rho_new = np.clip(rho_old + rng.uniform(-c.jitter, c.jitter, B), c.rho_min, c.rho_max)
                off_new = np.clip(off_old + rng.uniform(-c.jitter, c.jitter, (B, A)), -5, 5)
                r_old = (rho_old[:, None] + off_old)[..., None, None] * ring_dir[None, :, None, :]
                for _ in range(2):   # second pass after velocity rescaling
                    cnew, cE = arc_motion(cs[:, 0], cs[:, 1], cs[:, 2], vc, dthc, c.dt, S)
                    r_new = (rho_new[:, None] + off_new)[..., None, None] * ring_dir[None, :, None, :]
                    E = cE[:, None] + (1 - s_lin) * r_old + s_lin * r_new          # (B, A, S+1, 2)
                    L = np.hypot(*np.diff(E, axis=2).transpose(3, 0, 1, 2)).sum(-1).max(-1)
                    fs = np.minimum(1.0, c.v_max * c.dt / np.maximum(L, 1e-9))
                    if np.all(fs >= 1 - 1e-9):
                        break
                    vc = vc * fs
                    rho_new = rho_old + (rho_new - rho_old) * fs
                    off_new = off_old + (off_new - off_old) * fs[:, None]
                hd = np.arctan2(E[:, :, -1, 1] - E[:, :, -2, 1], E[:, :, -1, 0] - E[:, :, -2, 0])
                new = np.concatenate([E[:, :, -1], hd[..., None]], -1)
                bad = self.obstacle_hit(E)

            # ---- fitness + HighFitness --------------------------------------
            f = np.zeros(B)
            for kk in np.unique(k):
                m = k == kk
                f[m] = batch_fitness(new[m, :, :2], pmaps[kk], c.h, pairs)
            ok = ~bad
            if c.high_fitness is not None and c.variant != "RRT":
                ref = T.f[idx] if c.high_fitness_mode == "parent" else best_f[k]
                ok &= f >= c.high_fitness * ref
            for b in np.nonzero(ok)[0]:
                kb = k[b]
                if c.high_fitness_mode == "best" and c.high_fitness is not None and c.variant != "RRT" \
                        and f[b] < c.high_fitness * best_f[kb]:
                    continue
                best_f[kb] = max(best_f[kb], f[b])
                score = T.score[idx[b]] + w[kb] * f[b]
                kw = dict(state=new[b], tidx=kb, score=score, f=f[b], parent=idx[b], edge=E[b])
                if c.variant == "JSE":
                    kw.update(center=cnew[b], rho=rho_new[b], off=off_new[b])
                j = T.add(**kw)
                if kb >= n_steps:
                    F = score / wsum[kb]
                    if F > best_F:
                        best_F, best_end = F, j
                        if t_first is None:
                            t_first = time.perf_counter() - t_start
        runtime = time.perf_counter() - t_start
        success = best_end >= 0
        if not success:  # fall back to the deepest, best-scoring node
            n = T.n
            kmax = T.tidx[:n].max()
            cand = np.nonzero(T.tidx[:n] == kmax)[0]
            best_end = int(cand[np.argmax(T.score[cand])])
            best_F = T.score[best_end] / max(wsum[kmax], 1e-12)
        ids = T.path(best_end)
        states = T.state[ids]
        dense = [T.edge[ids[0]][:, :1]] + [T.edge[j][:, 1:] for j in ids[1:]]
        paths = np.concatenate(dense, axis=1).transpose(1, 0, 2)
        return Plan(times=times[: len(ids)], states=states, paths=paths, fitness=float(best_F),
                    success=success, n_nodes=T.n, iterations=it, runtime=runtime,
                    time_to_first=t_first, tree=T if keep_tree else None)
