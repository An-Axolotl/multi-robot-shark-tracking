"""Load the CSULB white-shark VPS array data and build heading-normalised
one-minute transitions (Section 4 of the paper).

Each transition uses three consecutive one-minute fixes p0, p1, p2 of the same
shark:  the *previous* step p0->p1 defines the shark heading and the
"starting velocity"; the *next* step p1->p2 is rotated into that heading frame
so that the shark always points along +x.  These (dx, dy) samples are what the
paper histograms into transition probabilities p(i,j,t | k,l,t-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
import os
DATA_DIR = Path(os.environ.get("SHARK_DATA", REPO / "data_src"))
DEFAULT_CSV = DATA_DIR / "SharkArray-01-animals.csv"


def load_fixes(csv: Path | str = DEFAULT_CSV, max_hpe: float | None = None) -> pd.DataFrame:
    """Raw VPS fixes -> clean per-shark time series on the 1-minute grid."""
    d = pd.read_csv(csv, index_col=0)
    d["t"] = pd.to_datetime(d["DATETIME"], format="%m/%d/%Y %H:%M")
    if max_hpe is not None:
        d = d[d["HPE"] <= max_hpe]
    d = d.sort_values(["TRANSMITTER", "t"]).drop_duplicates(["TRANSMITTER", "t"], keep="first")
    return d.reset_index(drop=True)[["TRANSMITTER", "t", "X", "Y", "D", "DEPTH", "HPE"]]


@dataclass
class Transitions:
    dx: np.ndarray          # next displacement, heading frame (m over 1 min)
    dy: np.ndarray
    speed0: np.ndarray      # starting velocity |p1-p0| / 60 s  (m/s)
    speed1: np.ndarray      # |p2-p1| / 60 s
    turn: np.ndarray        # heading change (rad, wrapped to [-pi, pi))
    depth: np.ndarray       # DEPTH at p1 (m, positive down, NaN if no depth tag)
    shark: np.ndarray       # transmitter id
    dt: float = 60.0

    def __len__(self):
        return len(self.dx)

    def mask(self, m: np.ndarray) -> "Transitions":
        return Transitions(self.dx[m], self.dy[m], self.speed0[m], self.speed1[m],
                           self.turn[m], self.depth[m], self.shark[m], self.dt)

    def by_speed(self, lo: float, hi: float) -> "Transitions":
        return self.mask((self.speed0 >= lo) & (self.speed0 < hi))

    def by_depth(self, lo: float, hi: float) -> "Transitions":
        return self.mask((self.depth >= lo) & (self.depth <= hi))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def build_transitions(fixes: pd.DataFrame | None = None, min_step_m: float = 0.0) -> Transitions:
    if fixes is None:
        fixes = load_fixes()
    out = []
    for sid, g in fixes.groupby("TRANSMITTER", sort=True):
        t = g["t"].values.astype("datetime64[s]").astype(np.int64)
        x, y, z = g["X"].values, g["Y"].values, g["DEPTH"].values
        ok = (t[1:-1] - t[:-2] == 60) & (t[2:] - t[1:-1] == 60)
        i = np.nonzero(ok)[0]
        if len(i) == 0:
            continue
        v0 = np.stack([x[i + 1] - x[i], y[i + 1] - y[i]], 1)
        v1 = np.stack([x[i + 2] - x[i + 1], y[i + 2] - y[i + 1]], 1)
        h0 = np.arctan2(v0[:, 1], v0[:, 0])
        h1 = np.arctan2(v1[:, 1], v1[:, 0])
        c, s = np.cos(-h0), np.sin(-h0)
        out.append(dict(
            dx=c * v1[:, 0] - s * v1[:, 1], dy=s * v1[:, 0] + c * v1[:, 1],
            speed0=np.hypot(*v0.T) / 60.0, speed1=np.hypot(*v1.T) / 60.0,
            turn=wrap(h1 - h0), depth=z[i + 1], shark=np.full(len(i), sid, dtype=object)))
    cat = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    tr = Transitions(**cat)
    if min_step_m > 0:
        tr = tr.mask(tr.speed0 * 60 >= min_step_m)
    return tr


def one_minute_steps(fixes: pd.DataFrame | None = None) -> pd.DataFrame:
    """All single 1-minute steps (two consecutive fixes): speed + depth."""
    if fixes is None:
        fixes = load_fixes()
    rows = []
    for sid, g in fixes.groupby("TRANSMITTER", sort=True):
        t = g["t"].values.astype("datetime64[s]").astype(np.int64)
        ok = np.nonzero(np.diff(t) == 60)[0]
        dx = np.diff(g["X"].values)[ok]
        dy = np.diff(g["Y"].values)[ok]
        rows.append(pd.DataFrame(dict(shark=sid, speed=np.hypot(dx, dy) / 60,
                                      depth=g["DEPTH"].values[ok])))
    return pd.concat(rows, ignore_index=True)
