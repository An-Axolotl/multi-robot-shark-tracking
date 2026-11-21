# Multi-Robot Shark Tracking: Expansive Planning and Trajectory Evaluation

Reconstruction of the code behind

> **Multi-Robot Expansive Planning and Trajectory Evaluation for Tracking and Localization of Marine Life**
> Kehlani A. Fay, Michael Giordano, Alberto Soto, Christopher M. Clark, Emily Spurgeon, James Anderson, Christopher G. Lowe.
> *38th ACM/SIGAPP Symposium on Applied Computing (SAC '23)*, Tallinn, Estonia, pp. 808-815, 2023.
> [doi:10.1145/3555776.3577749](https://doi.org/10.1145/3555776.3577749) · [slides](https://docs.google.com/presentation/d/1T-2BDdvBwcgFfQvNIiSkEVgcK2PtoIZC/edit?usp=sharing)

The original C# implementation was lost. This repository is a Python rebuild from the paper and the SAC '23
slides. It covers the shark motion models from the CSULB white-shark data, the fitness function, the ISE / JSE
expansive RRT planners, the adapted-RRT baseline, the closed-loop simulator, and every plot and animation. See
[Reconstruction notes](#reconstruction-notes) for where the paper was ambiguous and what was chosen.

<p align="center"><img src="animations/closed_loop_jse_markov.gif" width="48%"> <img src="animations/closed_loop_ise_markov.gif" width="48%"></p>
<p align="center"><em>Closed-loop JSE (left) and ISE (right) tracking a simulated shark that moves according to the
data-driven Markov model. Coloured trees are the current plan's RRT, black lines are the executed AUV paths,
cyan is the observed shark path, and cyan circles are obstacles.</em></p>

## Abstract

Traditional techniques for marine life tracking use stationary receivers that detect and obtain measurements from
tagged animals. Recently, such static systems have been replaced by multiple mobile robots, e.g., autonomous
underwater vehicles (AUVs), equipped with omni-directional hydrophones that can accurately localize marine life. In
this paper, the application of homogeneous multi-AUV systems to track and localize marine life is used as a
motivating example to develop new MRMP (Multi-Robot Motion Planning) algorithms. These algorithms generate
trajectories that maximize a new fitness function that incorporates 1) probabilistic motion models generated from
historical data of live sharks, and 2) ideal AUV formations for observing a shark from multiple sensor vantage
points. The two expansive RRT variants, named Independent State Expansion (ISE) planning and Joint State Expansion
(JSE) planning, differ in how new samples are randomly generated during the algorithm's random search. The fitness
function was developed to quantify how accurately the positioning of AUVs would trilaterate the target animal.
Through simulation, it was found that the Joint planner was 70% faster with respect to run time than Independent
planner, while both could produce similar mean fitness function values. The fitness for these variants was also
measured for simulations where different target motion models were used when calculating the fitness function,
highlighting the improved performance when using actual target motion models.

```bibtex
@inproceedings{fay2023multirobot,
  title     = {Multi-Robot Expansive Planning and Trajectory Evaluation for Tracking and Localization of Marine Life},
  author    = {Fay, Kehlani A. and Giordano, Michael and Soto, Alberto and Clark, Christopher M. and
               Spurgeon, Emily and Anderson, James and Lowe, Christopher G.},
  booktitle = {Proceedings of the 38th ACM/SIGAPP Symposium on Applied Computing},
  series    = {SAC '23},
  pages     = {808--815},
  year      = {2023},
  publisher = {ACM},
  doi       = {10.1145/3555776.3577749}
}
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/get_data.sh                                  # clones the shark data, links ./data_src
.venv/bin/python scripts/make_data_figures.py        # Figs. 1-3, data plots, Markov expansion gif
.venv/bin/python scripts/make_planner_figures.py     # Figs. 5-7, slide planner plots, animations
.venv/bin/python scripts/run_experiments.py all --runs 20   # Monte-Carlo runs -> results/*.csv
.venv/bin/python scripts/make_result_figures.py      # Figs. 8-10, shark speed
```

**Data.** The movement data (22 juvenile white sharks, CSULB Shark Lab VPS array, Carpinteria CA, 2020) is not
redistributed here. It lives in the private repository `ciherrera20/WhiteShark-DataExploration` (`data/`).
`scripts/get_data.sh` clones it with your SSH key, or you can set `SHARK_DATA=/path/to/data`. Only
`SharkArray-01-animals.csv` is used.

## Repository layout

| path | contents |
|---|---|
| `sharkplan/data.py` | load VPS fixes, build heading-normalised 1-minute transitions (dx, dy, starting speed, turn, depth) |
| `sharkplan/motion.py` | transition kernels conditioned on speed or depth; Markov / constant-velocity / stationary models; `P(X_s,t)` |
| `sharkplan/fitness.py` | Eqs. 1-4: `D(l)`, `C(θ̄)`, step fitness `f(T_t)`, trajectory fitness `F(T)` |
| `sharkplan/planner.py` | Algorithm 1: `ExpansivePlanner` with `variant="ISE" / "JSE" / "RRT"` |
| `sharkplan/sim.py` | closed-loop simulator (replan every 10 s, 30 s horizon, shark sampled from its model) |
| `scripts/` | figure, animation and experiment scripts |
| `figures/`, `animations/`, `results/` | generated outputs (committed) |
| `docs/slides/` | plots from the original SAC '23 presentation, for comparison |

## Method

**Shark motion models (paper Sec. 4).** For three consecutive one-minute fixes of the same shark, the step
p0→p1 sets the shark's heading and *starting velocity*. The next step p1→p2 is rotated into that heading frame. The
histogram of these displacements is the transition probability `p(i,j,t | k,l,t-1)`, and it is conditioned on
starting speed (Fig. 3) or depth (Fig. 2). The data has 22 sharks, 28,824 one-minute steps and 11,867 three-fix
transitions (the paper reports about 30,000 data points). The map `P(X_s,t)` is the kernel translated and rotated
onto the last shark measurement and propagated by Markovian expansion (Eq. 5, a 2-D convolution). Within a minute,
displacements are scaled linearly with `t/60`.

**Fitness (Eqs. 1-4).** `F(T) = Σ_t (2t/t_max) f(T_t)`, with
`f(T_t) = Σ_cells p_ij,t · C(θ̄_ij,t) · mean_pairs min(D(q_ij,u), D(q_ij,v))`.
`D` keeps every AUV between h/5 = 10 m and h = 50 m from each likely shark cell. `C = (1/(1+θ̄))³` rewards AUVs that
surround the cell evenly (θ̄ is the mean |bearing gap − 2π/A|). `F` is normalised by Σ(2t/t_max), so 1.0 is the
theoretical maximum.

**Planners (Sec. 5, Algorithm 1).** Nodes hold the joint state of all A AUVs (plus the cluster centre `X_C` for
JSE). In each iteration the planner samples a random point, picks the nearest open high-fitness node (ISE: the max over robots of
the distance; JSE: the centre distance; RRT: the mean distance), expands it, checks collisions and HighFitness, and
keeps the best node in the end-game region (t ≥ t_max).

* **ISE:** each AUV draws `v ∈ [v_min, v_max]` and `Δθ ∈ [−Δθ_max, Δθ_max]` and follows the arc plus straight
  segment of Eq. 6. Edges are checked against obstacles and against each other (no crossing paths).
* **JSE:** the arc motion is applied to the cluster centre. AUVs sit on a ring `x_i = x_c + ρ cos β_i`,
  `β_i = 2πi/A` (Eq. 7). The radius and per-AUV offsets jitter by ±0.5 m, and the centre speed is rescaled so the
  fastest AUV stays within `v_max`. No inter-AUV check is needed.
* **Adapted RRT baseline:** each robot steers toward a random configuration (ω_max = π/2), with no HighFitness test.

Iver3 limits are used: `v_max = 2.5 m/s`, `ω_max = π/4 rad/s`, expansion `Δt = 2 s`. The closed loop replans every
10 s with a 30 s horizon. At each replan the simulated shark is advanced by its *actual* model and handed to the
planner as a perfect measurement. A run's score is the time-weighted fitness of the executed AUV trajectories,
evaluated against the actual shark position.

## Figures

### Shark data and motion models

| reconstruction | original |
|---|---|
| ![](figures/fig02_depth_transitions_3d.png) **Fig. 2**: 1-minute transitions at the surface vs. 2-5 m depth | ![](docs/slides/slide12.png) |
| ![](figures/fig03_velocity_transitions.png) **Fig. 3**: transitions conditional on starting velocity. Bin counts match the paper (4084/2111/1313 vs. 4067/2108/1273). | ![](docs/slides/slide13.png) |
| ![](figures/slide_heading_change_hist.png) heading change per minute | (slide 13, left) |
| ![](figures/slide_markov_expansion_3d.png) Markov expansion `P(X_s, t=2 min)` | ![](docs/slides/slide14.png) |
| ![](figures/fig01_distance_function.png) **Fig. 1**: distance function `D(l)` | ![](docs/slides/slide19.png) |

Also: `figures/fig03b_velocity_transitions_meters.png` (Fig. 3 in metres), `figures/data_tracks.png`,
`figures/data_speed_depth_hist.png`, `animations/markov_expansion.gif`.

### Planners

| reconstruction | original |
|---|---|
| ![](figures/fig05_ise_motion_models.png) **Fig. 5**: ISE under stationary / constant-velocity / Markov models | ![](docs/slides/slide23.png) |
| ![](figures/slide_ise_origins.png) ISE from different robot origins | ![](docs/slides/slide24.png) |
| ![](figures/fig06_ise_100s_plan.png) **Fig. 6**: 100 s ISE plan, shark at 0.75 m/s | (paper Fig. 6) |
| ![](figures/fig07_jse_closed_loop_t80.png) **Fig. 7**: JSE closed loop, Markov shark, t = 80 s | (paper Fig. 7) |

Animations: `animations/ise_100s_plan_cv.gif`, `animations/jse_tree_growth.gif`,
`animations/closed_loop_{ise,jse}_markov.gif`.

### Results

Closed-loop runs (100 s each, replan every 10 s, 30 s horizon, 1 s planning budget, 4 random obstacles).
Data: `results/heatmap.csv` (100 runs per cell), `results/auv_sweep.csv` and `results/speed_sweep.csv`
(20 runs per point), summarised in `results/summary.csv`.

| reconstruction | original |
|---|---|
| ![](figures/fig08_ise_model_heatmap.png) **Fig. 8**: ISE, predicted vs. actual shark model | ![](docs/slides/slide25.png) |
| ![](figures/fig09_jse_model_heatmap.png) **Fig. 9**: JSE, predicted vs. actual shark model | ![](docs/slides/slide26.png) |
| ![](figures/fig10_score_vs_auvs.png) **Fig. 10**: score vs. number of AUVs | (paper Fig. 10) |
| ![](figures/score_vs_shark_speed.png) score vs. shark speed | (paper Sec. 6 text) |

**Comparison with the paper**

| finding in the paper | reconstruction |
|---|---|
| HighFitness strongly improves both planners (ISE 0.78 vs 0.03, JSE 0.82 vs 0.01) | Reproduced for ISE (0.26 vs 0.09 across the AUV sweep, 0.37 vs 0.09 across speeds). Smaller for JSE (0.59 vs 0.54), because the ring formation is already good without pruning. |
| The ISE planner repeatedly times out after 7 AUVs; JSE scales | Reproduced: most ISE replans time out from 7 AUVs, its score falls as AUVs are added, and JSE rises slowly toward a plateau. |
| Between 2 and 4 AUVs, ISE can beat JSE | Roughly equal at 2 AUVs (0.60 vs 0.62); JSE is higher from 3 AUVs. |
| Adapted RRT baseline scores about 0.022 | 0.06 |
| Matched models score best; a stationary shark predicted as moving scores lowest (JSE 0.35 / 0.39) | Reproduced for the moving-model rows (JSE 0.30 / 0.27). In ISE the stationary and constant-velocity diagonal entries are the best in their columns. |
| Absolute matched scores about 0.81-0.88 | 0.35-0.57. See note 7 below: Python speed and the unspecified details (notes 2, 3, 5) cap absolute fitness. |
| Scores roughly constant over shark speeds 0-1 m/s | Roughly constant from 0.3 to 0.9 m/s (JSE 0.52-0.57, ISE 0.34-0.38), lower at 1.2 m/s where transition data is sparse. |


## Reconstruction notes

The paper leaves several details open. These are the choices made here; each one is a parameter in
`PlannerConfig` or `MarkovModel`.

1. **Fig. 3 axis units.** The paper labels Fig. 3 "meters", but the peaks (≈7 at 0.6-0.8 m/s) only make sense as
   5 m grid cells: the data's median one-minute advance in that bin is 38 m. `fig03_velocity_transitions.png` uses
   cells to match the paper, and `fig03b_...` uses metres. The planner uses true metres (`MarkovModel(scale=1)`).
   `scale=0.2` reproduces the cells-as-metres reading, which gives tighter predictions and higher scores.
2. **HighFitness(n_new)** is not defined in the paper. The implementation accepts a node when
   `f(n_new) ≥ 0.8 · (best f found so far at that time step)` (`high_fitness_mode="best"`). A parent-relative test
   is available as `"parent"`. NodeSelection also **expands only from high-fitness nodes**: open nodes that still
   meet the same 0.8 · best test (`expand_high_fitness=True`). In a matched-model check (4 seeds × 3 models),
   adding this raised the mean closed-loop score from 0.45 to 0.55 for ISE and from 0.49 to 0.53 for JSE.
   "w/o High Fitness" runs disable both the pruning and the expansion rule.
3. **Angle term.** θ̄ uses the gaps between angularly adjacent AUV bearings, so a perfect formation scores
   `C = 1`. Eq. 4 is very sensitive: a perfect 30 m ring that is 5 m off-centre already drops to f ≈ 0.63.
4. **Arc + straight expansion (Eq. 6)** splits Δt evenly between the arc and the straight segment.
5. **Node selection.** Random samples are drawn in a box around the predicted shark path (20 m margin).
6. **Scores** are the time-weighted executed fitness against the actual shark over 100 s closed-loop runs, with 4
   random obstacles per run.
7. **Speed.** Python with numpy batches 64 expansions per round. Absolute plan quality depends on the time budget
   (1 s per replan here), so absolute fitness values are lower than the paper's C# numbers (~0.8). The relative
   trends are what the reconstruction targets: matched beats mismatched models, HighFitness beats no HighFitness,
   JSE is faster than ISE, and scores plateau as AUVs are added.

## License

MIT (code). The shark movement data belongs to the CSULB Shark Lab and is not included.
