"""Shark-data figures: motion data, transition probabilities, Markov expansion.

    python scripts/make_data_figures.py
Outputs to figures/ and animations/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm

from sharkplan.data import build_transitions, load_fixes, one_minute_steps
from sharkplan.fitness import D
from sharkplan.motion import SPEED_BINS, MarkovModel, histogram_kernel
from sharkplan.plotting import save

FIG = Path(__file__).resolve().parents[1] / "figures"
ANI = Path(__file__).resolve().parents[1] / "animations"
FIG.mkdir(exist_ok=True); ANI.mkdir(exist_ok=True)

fixes = load_fixes()
tr = build_transitions(fixes)
steps = one_minute_steps(fixes)
print(f"{fixes.TRANSMITTER.nunique()} sharks, {len(fixes)} fixes, {len(steps)} one-minute steps, "
      f"{len(tr)} heading-normalised transitions")

# ---------------------------------------------------------------- Fig. 1
l = np.linspace(0, 70, 701)
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(l, D(l))
ax.set_xlabel("l (meters)"); ax.set_ylabel("D(l)")
ax.set_title("Fig. 1: distance function D(l), h = 50 m, γ = 0.05")
save(fig, FIG / "fig01_distance_function.png")

# ---------------------------------------------------------------- Fig. 2
CELL = 5.0
def surface(ax, sub, title):
    H, c = histogram_kernel(sub.dx, sub.dy, CELL, 150)
    X, Y = np.meshgrid(c, c)
    surf = ax.plot_surface(X, Y, H, cmap="jet", vmin=0, vmax=0.025, linewidth=0, antialiased=True,
                           rstride=1, cstride=1)
    ax.set_zlim(0, 0.03); ax.set_xlim(-150, 100); ax.set_ylim(-150, 100)
    ax.set_xlabel("dx (meters)"); ax.set_ylabel("dy (meters)"); ax.set_zlabel("Probability")
    ax.set_title(f"{title} (~{len(sub)} pts)", fontsize=10, fontweight="bold")
    ax.view_init(28, -60)
    return surf

fig = plt.figure(figsize=(7, 11))
for i, ((lo, hi), lab) in enumerate([((-1, 0), "Depth of -1m to 0m"), ((2, 5), "Depth of 2m to 5m")]):
    ax = fig.add_subplot(2, 1, i + 1, projection="3d")
    s = surface(ax, tr.by_depth(lo, hi), lab)
    fig.colorbar(s, ax=ax, shrink=0.6, pad=0.1)
fig.suptitle("Fig. 2: 1-minute transitions, shark at (0,0) heading +x\n(a) surface  (b) near the sea floor")
save(fig, FIG / "fig02_depth_transitions_3d.png")

# ---------------------------------------------------------------- Fig. 3
cmap = plt.get_cmap("jet").copy()
cmap.set_under("#e6e6fa")   # lavender background as in the paper
for units, ext, fname in [("cells", 30, "fig03_velocity_transitions.png"),
                          ("meters", 150, "fig03b_velocity_transitions_meters.png")]:
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
    for ax, (lo, hi) in zip(axs.ravel(), SPEED_BINS):
        sub = tr.by_speed(lo, hi)
        H, c = histogram_kernel(sub.dx, sub.dy, CELL, 150)
        k = c / CELL if units == "cells" else c
        lim = 22.5 if units == "cells" else 112.5
        m = np.abs(k) <= lim
        Hs = H[np.ix_(m, m)]
        e = k[m]
        im = ax.imshow(np.where(Hs > 0, Hs, -1), origin="lower", cmap=cmap, vmin=1e-9, vmax=Hs.max(),
                       extent=[e[0], e[-1], e[0], e[-1]], interpolation="bilinear")
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
        ax.set_title(f"Starting Velocity from {lo:.1f} to {hi:.1f} m/s ({len(sub)} points)", fontsize=9,
                     fontweight="bold")
        u = "5 m cells" if units == "cells" else "meters"
        ax.set_xlabel(f"dx ({u})"); ax.set_ylabel(f"dy ({u})")
    fig.suptitle("Fig. 3: 1-minute transition probabilities conditional on starting velocity"
                 + (" (axes in 5 m grid cells, as in the paper)" if units == "cells" else ""))
    save(fig, FIG / fname)

# ---------------------------------------------------------------- slide: heading change histogram
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(-np.pi, np.pi, 73)
ax.hist(tr.turn, bins=bins, color="#4a90d9", ec="white", lw=0.4)
ax.set_xlabel("Heading change per minute (radians)"); ax.set_ylabel("Occurrences")
ax.set_title(f"Heading change in radians ({len(tr)} transitions)")
save(fig, FIG / "slide_heading_change_hist.png")

# ---------------------------------------------------------------- speed / depth histograms
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].hist(steps.speed.clip(0, 3), bins=60, color="#4a90d9")
axs[0].set_xlabel("1-minute speed (m/s)"); axs[0].set_ylabel("count"); axs[0].set_title("Shark speed")
for lo, hi in SPEED_BINS:
    axs[0].axvline(lo, color="k", lw=0.5, ls=":")
d = fixes.DEPTH.dropna()
axs[1].hist(d.clip(-2, 12), bins=56, color="#2a9d8f")
axs[1].set_xlabel("depth (m)"); axs[1].set_title(f"Depth-tagged fixes ({len(d)})")
save(fig, FIG / "data_speed_depth_hist.png")

# ---------------------------------------------------------------- tracks overview
fig, ax = plt.subplots(figsize=(7, 6.5))
ids = sorted(fixes.TRANSMITTER.unique())
cols = plt.get_cmap("tab20")(np.linspace(0, 1, len(ids)))
for c_, sid in zip(cols, ids):
    g = fixes[fixes.TRANSMITTER == sid]
    ax.plot(g.X, g.Y, ".", ms=0.6, color=c_, alpha=0.5, label=sid)
ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title(f"VPS positions of {len(ids)} juvenile white sharks")
ax.legend(markerscale=12, fontsize=6, ncol=2, loc="upper left", frameon=False)
save(fig, FIG / "data_tracks.png")

# ---------------------------------------------------------------- slide: Markov expansion
mm = MarkovModel(tr, speed=0.75, cell=2.0)
P, c = mm.heading_frame_grid(120.0, half=200)
X, Y = np.meshgrid(c, c)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
s = ax.plot_surface(X, Y, P, cmap="jet", linewidth=0, rstride=2, cstride=2)
ax.set_xlabel("dx (m)"); ax.set_ylabel("dy (m)"); ax.set_zlabel("p")
ax.set_title("Probability model of future positions (t = 2 min, 0.6-0.8 m/s, heading +x)")
fig.colorbar(s, shrink=0.6)
save(fig, FIG / "slide_markov_expansion_3d.png")

# animation of the Markov expansion
times = np.arange(0, 181, 5)
frames = []
for t in times:
    if t == 0:
        Pt = np.zeros((201, 201)); Pt[100, 100] = 1; cc = np.arange(-100, 101) * 2.0
    else:
        Pt, cc = mm.heading_frame_grid(float(t), half=200)
    frames.append((Pt, cc))
fig, ax = plt.subplots(figsize=(6, 4.6))
def draw(i):
    ax.clear()
    Pt, cc = frames[i]
    ax.imshow(np.sqrt(Pt), origin="lower", cmap="jet", extent=[cc[0], cc[-1], cc[0], cc[-1]])
    ax.set_xlim(-60, 200); ax.set_ylim(-100, 100)
    ax.set_title(f"Markov expansion of P(X_s,t), t = {times[i]} s"); ax.set_xlabel("dx (m)"); ax.set_ylabel("dy (m)")
ani = animation.FuncAnimation(fig, draw, frames=len(frames))
ani.save(ANI / "markov_expansion.gif", writer=animation.PillowWriter(fps=8))
plt.close(fig)
print("wrote", ANI / "markov_expansion.gif")
