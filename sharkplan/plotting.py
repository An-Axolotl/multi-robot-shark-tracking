"""Plot helpers that mimic the paper's figure style."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

AUV_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
              "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]


def draw_obstacles(ax, obs):
    for x, y, r in np.asarray(obs).reshape(-1, 3):
        ax.add_patch(Circle((x, y), r, fill=False, ec="c", lw=1.4))


def draw_tree(ax, tree, max_tidx=None, lw=0.5, alpha=0.6):
    n = tree.n
    m = np.arange(1, n)
    if max_tidx is not None:
        m = m[tree.tidx[1:n] <= max_tidx]
    for a in range(tree.A):
        segs = tree.edge[m, a]
        ax.add_collection(LineCollection(segs, colors=AUV_COLORS[a % 10], lw=lw, alpha=alpha))


def draw_plan(ax, plan, upto=None, color="k", lw=1.5):
    P = plan.paths if upto is None else plan.paths[:upto]
    for a in range(P.shape[1]):
        ax.plot(P[:, a, 0], P[:, a, 1], color=color, lw=lw)
        ax.plot(P[-1, a, 0], P[-1, a, 1], "o", color=color, ms=4)


def finish(ax, title=None, equal=True):
    if title:
        ax.set_title(title)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
