"""Figures from results/*.csv (run scripts/run_experiments.py first).

    python scripts/make_result_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sharkplan.plotting import save

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
NAMES = {"stationary": "Stationary", "cv": "Constant Velocity", "markov": "Markov"}
ORDER = ["stationary", "cv", "markov"]


def heat(df, variant):
    d = df[df.variant == variant]
    return (d.groupby(["predict", "actual"]).score.mean().unstack()
            .reindex(index=ORDER[::-1], columns=ORDER)), d.groupby(["predict", "actual"]).size().min()


p = RES / "heatmap.csv"
if p.exists():
    df = pd.read_csv(p)
    # Fig. 8 style (ISE, Blues, no annotations in the paper - we annotate anyway)
    H, n = heat(df, "ISE")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(H.values, cmap="Blues")
    ax.set_xticks(range(3), [NAMES[c] for c in H.columns]); ax.set_yticks(range(3), [NAMES[i].replace(" ", "\n") for i in H.index])
    for i in range(3):
        for j in range(3):
            v = H.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v > np.nanmean(H.values) else "k")
    ax.set_xlabel("Actual Shark Motion"); ax.set_ylabel("Model For Predicting Shark")
    ax.set_title("Actual Shark Motion Type vs. Shark Prediction Motion Model")
    fig.colorbar(im, label="Expected Trajectory Fitness")
    fig.text(0.5, -0.02, f"Fig. 8: ISE planner, mean of {n} closed-loop runs per cell", ha="center")
    save(fig, FIG / "fig08_ise_model_heatmap.png")
    # Fig. 9 style (JSE, rocket_r, annotated)
    H, n = heat(df, "JSE")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sns.heatmap(H.rename(index=NAMES, columns=NAMES), annot=True, fmt=".2f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Actual Trajectory Fitness"})
    ax.set_xlabel("Actual Shark Model"); ax.set_ylabel("Predicted Shark Model")
    ax.set_title("Joint Planner: Actual v. Predicted Shark Motion Model")
    fig.text(0.5, -0.02, f"Fig. 9: JSE planner, mean of {n} closed-loop runs per cell", ha="center")
    save(fig, FIG / "fig09_jse_model_heatmap.png")

p = RES / "auv_sweep.csv"
if p.exists():
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (v, hf), lab, col in [(("ISE", True), "Individual Score", "tab:blue"),
                              (("ISE", False), "Individual w/o High Fitness", "tab:orange"),
                              (("JSE", False), "Joint w/o High Fitness", "tab:green"),
                              (("JSE", True), "Joint Score", "tab:red"),
                              (("RRT", False), "Adapted RRT", "tab:purple")]:
        d = df[(df.variant == v) & (df.high_fitness == hf)]
        if d.empty:
            continue
        g = d.groupby("n_auvs").agg(score=("score", "mean"), fail=("n_fail", "mean"))
        ax.plot(g.index, g.score, label=lab, color=col)
        to = g[g.fail >= 5]   # >= half of the 10 replans timed out -> mark X
        if len(to):
            ax.plot(to.index, to.score, "X", color="tab:brown", ms=9)
    ax.set_ylim(0, 1); ax.set_xlabel("Number of AUVs"); ax.set_ylabel("Score")
    ax.set_title("Planner type by AUV number v. Score", fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.03, "Fig. 10: closed-loop Markov/Markov score vs. number of AUVs (X = majority of plans timed out)", ha="center", fontsize=8)
    save(fig, FIG / "fig10_score_vs_auvs.png")

p = RES / "runtime.csv"
if p.exists():
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v, col in [("ISE", "tab:blue"), ("JSE", "tab:red")]:
        g = df[df.variant == v].groupby("n_auvs")
        m = g.time_to_first.median()
        ax.plot(m.index, m.values, "o-", color=col, label=f"{v} (median)")
        ax.fill_between(m.index, g.time_to_first.quantile(.25), g.time_to_first.quantile(.75), color=col, alpha=.2)
        sr = g.success.mean()
        for A, s in sr.items():
            if s < 1:
                ax.annotate(f"{s:.0%}", (A, m.get(A, np.nan)), fontsize=7, color=col)
    ax.set_yscale("log"); ax.set_xlabel("Number of AUVs"); ax.set_ylabel("time to first full-horizon plan (s)")
    ax.set_title("Planner run time vs. number of AUVs (30 s horizon, Markov)")
    ax.legend()
    save(fig, FIG / "runtime_vs_auvs.png")

p = RES / "speed_sweep.csv"
if p.exists():
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (v, hf), d in df.groupby(["variant", "high_fitness"]):
        g = d.groupby("speed").score.agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"], g["sem"], marker="o", capsize=3,
                    label=f"{v} {'with' if hf else 'w/o'} high fitness")
    ax.set_ylim(0, 1); ax.set_xlabel("shark speed bin of the Markov model (m/s)"); ax.set_ylabel("Score")
    ax.set_title("Score vs. shark speed (Markov predicted & actual)")
    ax.legend(fontsize=8)
    save(fig, FIG / "score_vs_shark_speed.png")

# summary table
rows = []
for f in ["heatmap", "auv_sweep", "speed_sweep"]:
    p = RES / f"{f}.csv"
    if p.exists():
        d = pd.read_csv(p)
        for k, g in d.groupby(["variant", "high_fitness"]):
            rows.append(dict(experiment=f, variant=k[0], high_fitness=k[1], mean_score=g.score.mean(), runs=len(g)))
if rows:
    s = pd.DataFrame(rows)
    s.to_csv(RES / "summary.csv", index=False)
    print(s.to_string(index=False))
