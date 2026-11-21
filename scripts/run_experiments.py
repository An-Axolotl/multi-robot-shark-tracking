"""Monte-Carlo experiments behind Figs. 8-10 and the run-time comparison.

    python scripts/run_experiments.py heatmap --runs 100
    python scripts/run_experiments.py auvs    --runs 20
    python scripts/run_experiments.py runtime --runs 10
    python scripts/run_experiments.py speed   --runs 20
    python scripts/run_experiments.py all     --runs 20

Each run is a 100 s closed-loop simulation (replan every 10 s, 30 s horizon) with
random obstacles; results are appended to results/*.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MODEL_NAMES = ["stationary", "cv", "markov"]
_models = {}


def models(speed=0.75):
    from sharkplan.motion import make_model
    key = speed
    if key not in _models:
        _models[key] = {n: make_model(n, speed=speed) for n in MODEL_NAMES}
    return _models[key]


def closed_loop_job(job):
    from sharkplan.planner import PlannerConfig
    from sharkplan.sim import default_obstacles, run_closed_loop
    variant, pred, act, A, hf, seed, budget, speed = (job[k] for k in
        ["variant", "predict", "actual", "n_auvs", "high_fitness", "seed", "budget", "speed"])
    M = models(speed)
    cfg = PlannerConfig(variant=variant, n_auvs=A, time_budget=budget,
                        high_fitness=0.8 if hf else None, seed=seed)
    obs = default_obstacles(np.random.default_rng(10_000 + seed))
    t = time.perf_counter()
    r = run_closed_loop(cfg, M[pred], M[act], obstacles=obs, seed=seed)
    return dict(job, score=r.score, n_fail=r.n_fail, plan_runtime=r.plan_runtime,
                wall=time.perf_counter() - t)


def runtime_job(job):
    """Open-loop: time until the first end-game node (time-to-solution)."""
    from sharkplan.planner import ExpansivePlanner, PlannerConfig
    from sharkplan.sim import default_obstacles
    M = models()
    cfg = PlannerConfig(variant=job["variant"], n_auvs=job["n_auvs"], time_budget=job["budget"],
                        max_nodes=200_000, seed=job["seed"])
    pl = ExpansivePlanner(cfg, default_obstacles(np.random.default_rng(10_000 + job["seed"])))
    x0 = pl.initial_states((0, 0), rng=np.random.default_rng(job["seed"]))
    p = pl.plan(x0, M["markov"], np.zeros(3), keep_tree=False)
    return dict(job, time_to_first=p.time_to_first if p.time_to_first is not None else np.nan,
                success=p.success, fitness=p.fitness, n_nodes=p.n_nodes, iterations=p.iterations)


def run(jobs, fn, out, procs):
    out = RES / out
    print(f"{len(jobs)} jobs -> {out}")
    rows = []
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["heatmap", "auvs", "runtime", "speed", "all"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    base = dict(budget=a.budget, speed=0.75, n_auvs=4, high_fitness=True)
    todo = ["heatmap", "auvs", "runtime", "speed"] if a.what == "all" else [a.what]
    if "heatmap" in todo:
        jobs = [dict(base, variant=v, predict=p, actual=q, seed=s)
                for v in ["ISE", "JSE"] for p, q in itertools.product(MODEL_NAMES, MODEL_NAMES)
                for s in range(a.runs)]
        run(jobs, closed_loop_job, "heatmap.csv", a.procs)
    if "auvs" in todo:
        jobs = [dict(base, variant=v, high_fitness=hf, n_auvs=A, predict="markov", actual="markov", seed=s)
                for v, hf in [("ISE", True), ("ISE", False), ("JSE", True), ("JSE", False), ("RRT", False)]
                for A in range(2, 11) for s in range(a.runs)]
        run(jobs, closed_loop_job, "auv_sweep.csv", a.procs)
    if "runtime" in todo:
        jobs = [dict(variant=v, n_auvs=A, budget=10.0, seed=s)
                for v in ["ISE", "JSE"] for A in range(2, 11) for s in range(a.runs)]
        run(jobs, runtime_job, "runtime.csv", max(1, a.procs // 2))
    if "speed" in todo:
        jobs = [dict(base, variant=v, high_fitness=hf, speed=sp, predict="markov", actual="markov", seed=s)
                for v in ["ISE", "JSE"] for hf in [True, False]
                for sp in [0.1, 0.3, 0.5, 0.75, 0.9, 1.2] for s in range(a.runs)]
        run(jobs, closed_loop_job, "speed_sweep.csv", a.procs)


if __name__ == "__main__":
    main()
