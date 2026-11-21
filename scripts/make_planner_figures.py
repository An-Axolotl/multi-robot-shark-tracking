"""Planner figures and animations (paper Figs. 5, 6, 7, slides 22-23).

    python scripts/make_planner_figures.py [--budget 3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from sharkplan.fitness import step_fitness, time_weights
from sharkplan.motion import SparseMap, make_model
from sharkplan.planner import ExpansivePlanner, PlannerConfig
from sharkplan.plotting import AUV_COLORS, draw_obstacles, draw_plan, draw_tree, finish, save
from sharkplan.sim import run_closed_loop

ROOT = Path(__file__).resolve().parents[1]
FIG, ANI = ROOT / "figures", ROOT / "animations"
ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--skip-anim", action="store_true")
args = ap.parse_args()

MODELS = {"stationary": make_model("stationary"), "cv": make_model("cv"), "markov": make_model("markov")}
OBS = np.array([[50, 45, 10], [-20, -40, 6], [30, -30, 7], [-50, -25, 6], [75, 15, 5]])
SHARK0 = np.array([0.0, 0.0, 0.0])


def open_loop(variant, model, tmax, origin=None, budget=None, seed=0, obs=OBS, A=4):
    cfg = PlannerConfig(variant=variant, n_auvs=A, tmax=tmax, time_budget=budget or args.budget,
                        seed=seed, max_nodes=40000)
    pl = ExpansivePlanner(cfg, obs)
    if origin is None:
        x0 = pl.initial_states(SHARK0[:2], rng=np.random.default_rng(seed))
    else:   # robots spaced evenly +/- 15 m around an origin (slides)
        b = 2 * np.pi * np.arange(A) / A
        x0 = np.column_stack([origin[0] + 15 * np.cos(b), origin[1] + 15 * np.sin(b), b])
    plan = pl.plan(x0, model, SHARK0)
    return plan, cfg


def shark_dots(ax, model, plan, cfg, t=None, color="b"):
    times = plan.times if t is None else [t]
    for pm in model.prob_maps(*SHARK0, times):
        x, y = pm.argmax()
        ax.plot(x, y, "o", color=color, ms=3 if t is None else 6)


# ------------------------------------------------------------ Fig. 5
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (k, title) in zip(axs, [("stationary", "Stationary"), ("cv", "Constant velocity"), ("markov", "Markov")]):
    plan, cfg = open_loop("ISE", MODELS[k], 40.0, seed=args.seed)
    draw_tree(ax, plan.tree); draw_obstacles(ax, OBS); draw_plan(ax, plan)
    pm = MODELS[k].prob_maps(*SHARK0, [cfg.tmax])[0]
    ax.scatter(pm.x, pm.y, c=pm.p, cmap="Greens", s=6, zorder=0, alpha=0.6)
    ax.plot(*pm.argmax(), "o", color="lime", mec="k", ms=7)
    finish(ax, f"{title}: fitness of {plan.fitness:.4f}")
fig.suptitle("Fig. 5: ISE planner under stationary, constant-velocity and Markov shark models")
save(fig, FIG / "fig05_ise_motion_models.png")

# ------------------------------------------------------------ slides 22/23: individual planner origins
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, org in zip(axs, [(-40, 40), (25, 40), (0, 0)]):
    plan, cfg = open_loop("ISE", MODELS["markov"], 40.0, origin=org, seed=args.seed)
    draw_tree(ax, plan.tree); draw_obstacles(ax, OBS); draw_plan(ax, plan)
    shark_dots(ax, MODELS["markov"], plan, cfg)
    finish(ax, f"Total end fitness: {plan.fitness:.2f}\nRobot origin {org}, shark start (0,0)")
fig.suptitle("Algorithm - Individual (ISE) from different robot origins (robots +/-15 m around origin)")
save(fig, FIG / "slide_ise_origins.png")

# ------------------------------------------------------------ Fig. 6: 100 s ISE plan, constant velocity
plan6, cfg6 = open_loop("ISE", MODELS["cv"], 100.0, budget=max(8.0, 3 * args.budget), seed=args.seed)
S = cfg6.edge_samples
pm6 = MODELS["cv"].prob_maps(*SHARK0, plan6.times)
f6 = np.array([step_fitness(s[:, :2], p) for s, p in zip(plan6.states, pm6)])
w6 = time_weights(plan6.times, cfg6.tmax)


def cum_fit(k):
    return float((w6[:k + 1] * f6[:k + 1]).sum() / max(w6[:k + 1].sum(), 1e-12))


fig, axs = plt.subplots(3, 2, figsize=(11, 10))
for ax, T in zip(axs.ravel(), [0, 20, 40, 60, 80, 100]):
    k = int(round(T / cfg6.dt))
    draw_tree(ax, plan6.tree, alpha=0.5); draw_obstacles(ax, OBS)
    draw_plan(ax, plan6, upto=k * S + 1)
    ax.plot(*pm6[min(k, len(pm6) - 1)].argmax(), "o", color="b", ms=6)
    finish(ax, f"T = {T} s   fitness of {cum_fit(k):.4f}")
fig.suptitle("Fig. 6: 100 s ISE plan, 4 AUVs, shark anticipated and moving at 0.75 m/s (+x)")
save(fig, FIG / "fig06_ise_100s_plan.png")

# ------------------------------------------------------------ Fig. 7: JSE closed loop with Markov shark
cfg7 = PlannerConfig(variant="JSE", time_budget=args.budget, seed=args.seed)
sim7 = run_closed_loop(cfg7, MODELS["markov"], MODELS["markov"], total_time=100, obstacles=OBS,
                       seed=args.seed, keep_plans=True)
cfgI = PlannerConfig(variant="ISE", time_budget=args.budget, seed=args.seed)
simI = run_closed_loop(cfgI, MODELS["markov"], MODELS["markov"], total_time=100, obstacles=OBS,
                       seed=args.seed, keep_plans=True)
np.savez(ROOT / "results" / "closed_loop_examples.npz", jse_auv=sim7.auv, jse_shark=sim7.shark,
         jse_f=sim7.f_actual, ise_auv=simI.auv, ise_shark=simI.shark, ise_f=simI.f_actual, t=sim7.times)


def closed_loop_frame(ax, sim, cfg, t_now, title):
    ax.clear()
    ci = min(int(t_now // 10), len(sim.plans) - 1)
    plan = sim.plans[ci]
    draw_tree(ax, plan.tree, alpha=0.5); draw_obstacles(ax, OBS)
    S = cfg.edge_samples
    m = int(round(t_now / cfg.dt)) * S + 1
    P = sim.auv_dense[:m]
    for a in range(P.shape[1]):
        ax.plot(P[:, a, 0], P[:, a, 1], "k", lw=1.4)
        ax.plot(P[-1, a, 0], P[-1, a, 1], "ko", ms=4)
    k = int(round(t_now / cfg.dt))
    ax.plot(sim.shark[:k + 1, 0], sim.shark[:k + 1, 1], color="cyan", lw=4, alpha=0.9)
    ax.plot(sim.shark[k, 0], sim.shark[k, 1], "o", color="b", ms=7)
    w = 2 * sim.times[1:k + 1]
    fit = (w * sim.f_actual[1:k + 1]).sum() / w.sum() if k > 0 else sim.f_actual[0]
    ax.set_title(f"{title}  t = {t_now:.0f} s   fitness of {fit:.4f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()


fig, ax = plt.subplots(figsize=(7, 5.5))
closed_loop_frame(ax, sim7, cfg7, 80, "JSE")
fig.suptitle("Fig. 7: executed JSE closed-loop plan, Markov shark, 4 AUVs", fontsize=10)
save(fig, FIG / "fig07_jse_closed_loop_t80.png")

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
closed_loop_frame(axs[0], simI, cfgI, 100, "ISE")
closed_loop_frame(axs[1], sim7, cfg7, 100, "JSE")
save(fig, FIG / "closed_loop_ise_vs_jse_t100.png")
print(f"closed loop score ISE {simI.score:.3f}  JSE {sim7.score:.3f}")

if not args.skip_anim:
    for sim, cfg, name in [(sim7, cfg7, "JSE"), (simI, cfgI, "ISE")]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ts = np.arange(0, 100 + 1e-9, cfg.dt)
        ani = animation.FuncAnimation(fig, lambda i: closed_loop_frame(ax, sim, cfg, ts[i], name), frames=len(ts))
        ani.save(ANI / f"closed_loop_{name.lower()}_markov.gif", writer=animation.PillowWriter(fps=6))
        plt.close(fig)
        print("wrote", ANI / f"closed_loop_{name.lower()}_markov.gif")

    # Fig. 6 as an animation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    def f6frame(k):
        ax.clear()
        draw_tree(ax, plan6.tree, alpha=0.4); draw_obstacles(ax, OBS)
        draw_plan(ax, plan6, upto=k * S + 1)
        ax.plot(*pm6[k].argmax(), "o", color="b", ms=6)
        finish(ax, f"ISE 100 s plan  T = {plan6.times[k]:.0f} s  fitness of {cum_fit(k):.4f}")
    ani = animation.FuncAnimation(fig, f6frame, frames=len(plan6.times))
    ani.save(ANI / "ise_100s_plan_cv.gif", writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", ANI / "ise_100s_plan_cv.gif")

    # tree growth animation (JSE, Markov, 30 s horizon)
    planG, cfgG = open_loop("JSE", MODELS["markov"], 30.0, seed=args.seed)
    T = planG.tree
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ns = np.unique(np.linspace(2, T.n, 60).astype(int))
    def gframe(i):
        ax.clear()
        n = ns[i]
        for a in range(T.A):
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(T.edge[1:n, a], colors=AUV_COLORS[a], lw=0.5, alpha=0.7))
        draw_obstacles(ax, OBS)
        if i == len(ns) - 1:
            draw_plan(ax, planG)
        ax.set_xlim(-60, 90); ax.set_ylim(-60, 60); ax.set_aspect("equal")
        ax.set_title(f"JSE tree growth: {n} nodes")
    ani = animation.FuncAnimation(fig, gframe, frames=len(ns))
    ani.save(ANI / "jse_tree_growth.gif", writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print("wrote", ANI / "jse_tree_growth.gif")
