# Object-Informed Manipulation MJX

GPU-accelerated planning for **non-prehensile manipulation**, built on
[JAX](https://jax.readthedocs.io/) and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

This repository implements **object-informed MPPI**: a hierarchical
formulation that splits long-horizon pushing into an *object-level* planner
(which decides what contact wrench the object needs) and a *robot-level*
planner (which decides how to realize it), coordinated by ADMM until the two
agree. It is built on top of a general library of sampling-based MPC
algorithms — MPPI, CEM, predictive sampling and others — which remain fully
available and serve as the interchangeable inner solvers of the ADMM
subproblems.

<p align="center">
  <img src="img/humanoid.gif" width="30%" />
  &nbsp;&nbsp;
  <img src="img/cube.gif" width="30%" />
</p>

## Contents

- [Algorithms](#algorithms)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Mathematical formulation](#mathematical-formulation)
- [Code layout](#code-layout)
- [Designing a task](#designing-a-task)
- [Domain randomization and risk](#domain-randomization-and-risk)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Algorithms

| Algorithm | Description | Import |
| --- | --- | --- |
| **[ADMM object-informed MPPI](#mathematical-formulation)** | **Hierarchical object/robot decomposition, coordinated to consensus on the contact wrench. Either subproblem accepts any sampler below.** | [`oim.algs.ADMM`](oim/algs/admm.py) |
| [Predictive sampling](https://arxiv.org/abs/2212.00541) | Take the lowest-cost rollout at each iteration. | [`oim.algs.PredictiveSampling`](oim/algs/predictive_sampling.py) |
| [MPPI](https://arxiv.org/abs/1707.02342) | Exponentially weighted average of the rollouts. | [`oim.algs.MPPI`](oim/algs/mppi.py) |
| [Cross Entropy Method](https://en.wikipedia.org/wiki/Cross-entropy_method) | Fit a Gaussian to the `n` best "elite" rollouts. | [`oim.algs.CEM`](oim/algs/cem.py) |
| [DIAL-MPC](https://arxiv.org/abs/2409.15610) | MPPI with dual-loop, annealed sampling covariance. | [`oim.algs.DIAL`](oim/algs/dial.py) |
| [MPPI-CMA](https://arxiv.org/pdf/2506.22087) | MPPI with an adaptive sampling distribution. | [`oim.algs.MppiCma`](oim/algs/mppi_cma.py) |
| [MTP](https://arxiv.org/abs/2505.01059) | Structured tensor sampling mixed with a local CEM update. | [`oim.algs.MTP`](oim/algs/mtp.py) |
| [CBO](https://en.wikipedia.org/wiki/Consensus_based_optimization) | Simulate an SDE that pulls samples toward a consensus point. | [`oim.algs.CBO`](oim/algs/cbo.py) |
| [Evosax](https://github.com/RobertTLange/evosax/) | Any of the 30+ evolution strategies in `evosax` (CMA-ES, DE, …). | [`oim.algs.Evosax`](oim/algs/evosax.py) |

## Setup

Requires Python ≥ 3.12 and CUDA 13. Managed entirely by
[uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/NikolaRaicevic2001/Object-Informed-Manipulation-MJX.git
cd Object-Informed-Manipulation-MJX
uv sync
```

| Command | What it does |
| --- | --- |
| `uv sync` | Create `.venv` and install everything pinned in `uv.lock` |
| `uv run <cmd>` | Run `<cmd>` in that environment (no activation needed) |
| `source .venv/bin/activate` | Activate it instead, if you prefer |
| `uv run pytest` | Run the test suite |
| `uv run ruff check .` | Lint |

## Quick start

The push-T-through-clutter task is the main demo: the object must reach an
SE(2) goal pose while avoiding three static obstacles, which requires
non-myopic reasoning — a greedy pusher gets stuck behind an obstacle.

It runs in **two worlds that share the same algorithm**. The ADMM loop, the
consensus space, the object-level subproblem, the object cost and the shared
robot cost terms are the same code with the same weights in both; only the
robot block's one-step rollout differs. Both worlds run from one entry
point, [`examples/pusht.py`](examples/pusht.py); pick the world with
`--world {3d,2d}` (default `3d`):

| | 3D (MuJoCo MJX) | 2D (analytic) |
| --- | --- | --- |
| Robot | 2-DOF point pusher, or 6-DoF xArm6 | disc robot, velocity-controlled |
| Physics | full MJX contact | closed-form single-point contact |
| Output | passive viewer, or `--headless` plot | plot on exit, gif with `--animate` |
| Use it for | final behaviour, real embodiments | isolating algorithm bugs |

The 2D world exists for **attribution**. An MJX rollout that misbehaves
could be failing in the contact solver, the constraint allocation, the
integrator, the actuator model, or the ADMM math; if the same behaviour
reproduces in 2D, it is the formulation. Everything there is `jax.numpy`,
so it jits and vmaps like the MJX path but also runs eagerly under
`--no-jit`, steppable in a debugger.

### Push-T in 3D

```bash
# ADMM object-informed MPPI, 2-DOF point pusher
uv run python examples/pusht.py admm

# ...on a 6-DoF UFACTORY xArm6 with a rigid pushing stick
uv run python examples/pusht.py --robot xarm6 admm

# Mix and match the inner solvers of the two ADMM subproblems
uv run python examples/pusht.py admm --robot-opt mppi --object-opt cbo

# No display: fixed step count, diagnostics plot + results JSON
uv run python examples/pusht.py --robot xarm6 admm --headless --steps 600

# Flat (non-hierarchical) baselines on the same task, for comparison
uv run python examples/pusht.py mppi
uv run python examples/pusht.py ps
```

Flag order matters: these go **before** the algorithm name.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--world {3d,2d}` | `3d` | Which world to run (table above) |
| `--robot {point,xarm6}` | `point` | Embodiment. `xarm6` always implies the cluttered scene |
| `--record` | off | Write an mp4 (needs `ffmpeg`). Combines with `--headless`, which renders offscreen — no display required |
| `--warp` | off | Use the experimental [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) backend instead of JAX for rollouts |

These go **after** `admm`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--robot-opt`, `--object-opt` | `mppi` | Inner solver per ADMM block: `mppi`, `cem`, `ps`, `cbo` |
| `--show-plans` | off | Overlay both blocks' predicted object trajectories — live, and in the recording (below) |
| `--headless` | off | No viewer; run `--steps` steps and save plot + results. Add `--record` for an mp4 too |
| `--steps` | 200 | Control steps. `--headless` only — the viewer path runs until you close the window |
| `--eval` | off | Run `--trials` seeds and save aggregate stats instead of one recorded run — see [Evaluating over trials](#evaluating-over-trials) |
| `--trials`, `--seed0` | 5, 0 | `--eval` only: how many seeds, and the first one (`seed0 .. seed0 + trials - 1`) |

#### Watching the two blocks negotiate

```bash
# live, in the viewer
uv run python examples/pusht.py --robot xarm6 admm --show-plans

# composited into the recorded video, and logged for offline review
uv run python examples/pusht.py --robot xarm6 --record admm \
    --headless --steps 300 --show-plans
```

The consensus variable is a *wrench*, which is hard to read as a number.
What it is negotiating over is object motion, and **both blocks predict a
trajectory of the same object** — so `--show-plans` draws the pair:

| Colour | Curve |
| --- | --- |
| **Amber** | What the object block intends — its decision $\mathbf{W}^o$ through the limit-surface dynamics, i.e. $x^{o*}$ |
| **Teal** | What the robot block would actually produce — its nominal $\mathbf{U}^r$ rolled out through MJX |

Where the curves lie on top of each other the blocks agree; where they peel
apart is the primal residual, made spatial. Each is drawn as a path with
periodic bars showing the object's predicted *heading* (orientation is half
the goal here, so a position-only path would hide a whole class of
disagreement), fading along the horizon so time direction reads off a still
frame.

It works in both output paths. The passive viewer's `user_scn` persists
between frames, so the overlay keeps a fixed slot in it; an offscreen
`mujoco.Renderer` rebuilds its scene on every `update_scene`, so there the
overlay is re-appended per frame, after that call. Under `--headless` the
two plans are also written to the states JSON as `object_plan` and
`robot_plan`, `(steps_run, H, 3)` each — so the divergence can be measured
offline rather than only watched.

Cost, measured rather than assumed: one extra nominal rollout per control
step, **6.2 ms against `optimize`'s 361 ms — 1.7%**, or +1.2% wall clock.
The plans are recomputed on demand rather than threaded through the ADMM
loop, so with the flag off nothing runs at all and timing is untouched.

On a machine with no display at all, set `MUJOCO_GL=egl` (or `osmesa` for
software rendering) so the offscreen renderer can get a GL context.

### Evaluating over trials

```bash
# 5 seeds of ADMM, plus flat MPPI/PS at the same sampler budget, for comparison
uv run python examples/pusht.py admm --eval --trials 5 --steps 200

# a flat baseline alone, same seeds
uv run python examples/pusht.py mppi --eval --trials 5 --steps 200

# 2D: ADMM only, no flat baseline (PushT2D has no plain running/terminal cost)
uv run python examples/pusht.py --world 2d --env corridor admm --eval
```

`--eval` runs `--trials` seeds of the chosen algorithm instead of one
recorded run, aggregates success rate, position/orientation error
(mean & std), frequency and execution time via
[`oim/utils/metrics.py`](oim/utils/metrics.py)'s `aggregate_metrics`, and
saves the result with [`oim/utils/results_eval.py`](oim/utils/results_eval.py)
to `oim/results/results_eval/`. On 3D, `admm --eval` also runs flat
MPPI/PS at the same sampler budget so ADMM's object-level hierarchy is
judged against a bigger-search baseline, not just itself.

### Push-T in 2D

```bash
# T-block through the same clutter as the MJX scene
uv run python examples/pusht.py --world 2d admm

# push *through* a tight opening rather than around obstacles
uv run python examples/pusht.py --world 2d --env corridor admm
uv run python examples/pusht.py --world 2d --env gate admm

# no jit anywhere -- breakpoint inside the physics or the ADMM math
uv run python examples/pusht.py --world 2d --no-jit admm --steps 3

# empty scene, to separate "can it push at all" from "can it route"
uv run python examples/pusht.py --world 2d --no-obstacles admm

# A/B the contact-action object parameterization against the default
uv run python examples/pusht.py --world 2d --contact-action admm
```

`--world 2d` only ever runs `admm` (see [Push-T in 3D](#push-t-in-3d)
above), but the subcommand is still required — `admm` is not implied.
`--env`, like every other top-level flag, must come **before** it.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--env {clutter,corridor,gate}` | `clutter` | Scenario (below) |
| `--samples` | 64 | Rollouts per sub-optimizer |
| `--horizon` | 15 | Consensus horizon $H$, in steps of `dt = 0.05` |
| `--contact-action` | off | Object block decides $[p, f_n, f_t]$ instead of the wrench — see [object action parameterization](#object-action-parameterization) |
| `--no-relocate` | off | Disable the global contact-point search (`--contact-action` only) |
| `--no-obstacles` | off | Strip the obstacles |
| `--no-jit` | off | Run eagerly, so physics and ADMM math are steppable |
| `--animate` | off | Also write a gif |
| `--no-plot` | off | Results JSON only, no figure |

Three scenarios, ordered by what they stress
([`oim/sim2d/scenarios.py`](oim/sim2d/scenarios.py)):

| `--env` | Stresses | Straight-line clearance |
| --- | --- | --- |
| `clutter` | routing *around* obstacles; mirrors the MJX scene exactly | 41 mm |
| `corridor` | pushing *through* a narrow horizontal channel | 15 mm |
| `gate` | passing a vertical slot, then rotating to the goal | 5 mm |

`clutter` never forces the object through an opening it barely fits — there
is always a way round — so a planner can look healthy on it while being
unable to commit to a passage. That is precisely the non-myopic behaviour
the object-level block exists to provide, hence the other two. The
clearances above are asserted in `tests/test_sim2d.py` rather than trusted.

New scenarios are plain [`oim/objects/`](oim/objects/) primitives — no MJCF
involved:

```python
from oim.objects import Circle, t_shape_footprint
from oim.sim2d import PushT2D, build_admm_2d, run_2d

task = PushT2D(
    footprint=t_shape_footprint(),
    goal=[0.5, 0.48, 0.785],
    obstacles=[Circle(center=[0.08, 0.32], radius=0.04)],
)
ctrl, params = build_admm_2d(task, n_admm=6)
log = run_2d(task, ctrl, params, robot_pos0=(0.0, -0.13), max_steps=200)
```

### Shared ADMM knobs

Same flag names and meanings in both worlds. In 3D these all belong to the
`admm` subcommand, so they go after it; 2D has no subcommands, so order is
free there.

| Flag | 3D | 2D | Meaning |
| --- | --- | --- | --- |
| `--rho` | 10.0 | 10.0 | Initial penalty $\rho$ |
| `--gamma` | 0.1 | 0.1 | Proximal weight $\gamma$ |
| `--steps` | 200 | 200 | Control steps |
| `--n-admm` | 8 | 6 | Max ADMM iterations per control step |
| `--seed` | 5 | 0 | RNG seed |

The last two differ only as CLI defaults — the underlying
`build_admm_2d(n_admm=8)` matches 3D, so pass `--n-admm 8 --seed 5` for a
like-for-like comparison. Everything not exposed as a flag (horizon,
$\epsilon_r$, $\epsilon_s$, noise annealing, wrench bounds) is already
identical; see [`oim/sim2d/run.py`](oim/sim2d/run.py).

### Outputs

Both entry points name every artifact of one run
`{family}_{variant}_{method}_{timestamp}`, so the plot, gif, video and
results JSON of a run pair up:

| Path | Written when |
| --- | --- |
| `oim/recordings/pusht{2,3}d_*.png` | Always (2D), `--headless` (3D) — trajectory + residual diagnostics |
| `oim/recordings/pusht2d_*.gif` | `--animate` |
| `oim/recordings/pusht3d_*.mp4` | `--record`, with or without `--headless` |
| `oim/results/pusht{2,3}d_*_metrics_*.json` | Always — hyperparameters, per-step ADMM residuals and $\rho$, goal errors, whether the goal was reached |
| `oim/results/pusht{2,3}d_*_states_*.json` | Always — the scene and the motion, enough to reconstruct the run without the video |

The states file splits what never moves from what does. The goal, the
obstacles and the object's footprint are written **once**; the object pose
and twist, robot position and velocity, applied control, and realized and
agreed wrench are written **every control step** — plus full `qpos`/`qvel`
in 3D, so a run can be replayed exactly rather than only plotted. Both
worlds use the same key names and frames, so a 2D and a 3D run can be
compared entry for entry. Each file carries a `schema` block describing its
own indexing and conventions.

### Config-driven launches

[`oim/run_launch.py`](oim/run_launch.py) covers the same ground as
`examples/pusht.py` (2D/3D, every algorithm, single runs and `--eval`) but
every hyperparameter comes from a YAML config, not a CLI default — a
single-run and an `--eval` run of the same algorithm always build the
identical thing, and comparing two algorithms means comparing the same
sampler budget, task physics and execution model, not whatever a CLI flag
happened to be set to that day. It has its own code (not a caller of
`examples/pusht.py`); the tradeoff is deliberate, same as
`examples/pusht.py` vs. the older `examples/ablation_pusht.py`.

`--env {name}` selects `oim/configs/{name}.yaml`, which supplies every
default below; only `clutter` exists today (one 3D scene, one 2D
scenario), but the flag is there now so a future second environment (a
different 3D layout, or a 2D scenario bundled with its own physics) is a
new YAML file, not a CLI change. Every other flag matches
`examples/pusht.py`'s exactly and still works as a **one-off override** —
omit it and the YAML value is used; pass it and that run alone uses the
override, nothing else does.

```bash
# 3D ADMM, interactive viewer, every arg spelled out (--env is the only
# one that isn't already in oim/configs/clutter.yaml's defaults)
uv run python -m oim.run_launch \
    --env clutter --world 3d --robot point --samples 64 --horizon 15 \
    admm --robot-opt mppi --object-opt mppi --n-admm 8 --rho 10.0 \
    --gamma 0.1 --seed 5 --steps 200

# 3D ADMM on the xArm6, headless, recorded, mixed sub-optimizers
uv run python -m oim.run_launch \
    --env clutter --world 3d --robot xarm6 --samples 64 --horizon 15 \
    --record admm --robot-opt cem --object-opt cbo --n-admm 8 --rho 10.0 \
    --gamma 0.1 --seed 5 --headless --steps 300 --show-plans

# 3D flat CEM baseline, 5-trial eval, every eval flag explicit
uv run python -m oim.run_launch \
    --env clutter --world 3d --robot point --samples 64 --horizon 15 \
    cem --eval --trials 5 --seed0 0 --steps 200

# 2D ADMM, every 2D-specific toggle explicit (--animate needs the plot,
# so it isn't combined with --no-plot here)
uv run python -m oim.run_launch \
    --env clutter --world 2d --samples 64 --horizon 15 \
    --contact-action --no-relocate --obstacles --animate \
    admm --n-admm 6 --rho 10.0 --gamma 0.1 --seed 0 --steps 200

# 2D ADMM, 5-trial eval (2D has no flat baseline to compare against)
uv run python -m oim.run_launch \
    --env clutter --world 2d admm --eval --trials 5 --seed0 0 --steps 200
```

`oim/configs/clutter.yaml` groups every value by what it configures —
`world3d` (robot, planning/execution timesteps), `world2d` (scenario,
object mass/friction), `sampler` (shared `num_samples`/`horizon`, plus a
block per sub-optimizer: `mppi`, `cem`, `ps`, `cbo`), `admm` (`n_admm`,
`rho`, `gamma`, the sub-optimizer choices, residual tolerances, noise
annealing), and `run` (seed, steps, trials, the goal-reached tolerances).
That last one exists because `oim/sim3d/run.py`'s `run_3d_admm` and
`run_3d_plain` otherwise default to *different* "reached" thresholds
(0.05/0.05 vs. 0.06/0.10) — `run_launch.py` passes the same config value
to every method it builds so a `--eval` comparison isn't quietly biased
toward whichever method got the more lenient threshold.

### Other entry points

| What | Where |
| --- | --- |
| Open-loop trajectory optimization | [`oim/open_loop.py`](oim/open_loop.py), demo in [`examples/cart_pole_trajectory_optimization.py`](examples/cart_pole_trajectory_optimization.py) |
| Asynchronous simulation — controller and simulator in separate processes, for a realistic picture of closed-loop latency | [`oim/sim3d/asynchronous.py`](oim/sim3d/asynchronous.py), demo in [`examples/cube_async.py`](examples/cube_async.py) |
| Headless MJX ADMM driver, returning the same log dict as `run_2d` | [`oim/sim3d/run.py`](oim/sim3d/run.py) |
| ADMM-vs-flat-baseline evaluation over trials — same task, same robot-level sampler budget, reports success rate / position + orientation error (mean & std) / frequency / execution time | [`oim/utils/metrics.py`](oim/utils/metrics.py), run via `examples/pusht.py`'s `--eval` (see [Evaluating over trials](#evaluating-over-trials)) or config-driven via [`oim/run_launch.py`](oim/run_launch.py), saved via [`oim/utils/results_eval.py`](oim/utils/results_eval.py) to `oim/results/results_eval/` |
| Other demos from the base library (pendulum, cart-pole, humanoid standup and mocap, cube rotation, walker, crane, …) | [`examples/`](examples/) |

## Mathematical formulation

We plan for a robot manipulating an unactuated rigid object through contact.
Rather than treating this as one monolithic contact-implicit problem, we model
it as **two subsystems coupled only through the contact wrench**, and reach
agreement on that wrench with ADMM.

### Subsystem models

The robot is input-affine, with $x^r_t$ its configuration and $u^r_t$ its
control:

```math
x^r_{t+1} = f_r(x^r_t) + g_r(x^r_t)\,u^r_t + h_r(x^r_t)\,w^r_t .
```

The object is unactuated and moves only through the contact wrench
$w^o_t \triangleq [f_x,\, f_y,\, \tau]^\top \in \mathbb{R}^{p^o}$, which the
object-level planner treats as a *decision variable* rather than as the outcome
of complementarity constraints:

```math
x^o_{t+1} = f_o(x^o_t) + h_o(x^o_t)\,w^o_t .
```

For quasi-static planar pushing the object's twist is proportional to the
applied wrench through the **ellipsoidal limit surface**, giving a closed-form
model with no simulator in the loop:

```math
\dot{x}^o = D\,w^o, \qquad
D \triangleq \operatorname{diag}\big(\mu m g,\ \mu m g,\ c\,r\,\mu m g\big)^{-1},
```

```math
x^o_{t+1} = x^o_t + \Delta t\, D\, w^o_t ,
```

with $\mu$ the friction coefficient, $m$ the object mass, and $c, r$ the
limit-surface pressure coefficient and characteristic radius. Note that
$D^{-1}$ is exactly the **friction-cone limit** — the largest wrench the
support surface can transmit — which is reused below as a natural normalizer.

### Joint problem

Over a horizon $H$, with $\mathbf{Z} \triangleq \{z_t\}$ the shared consensus
variable:

```math
\begin{aligned}
\min_{\mathbf{W}^o,\, \mathbf{U}^r,\, \mathbf{Z}} \quad
& \sum_{t=0}^{H-1}\Big(\ell_o(x^o_t) + \ell_r(x^r_t, u^r_t)\Big) + \ell_f(x^o_H) \\
\text{s.t.}\quad
& x^o_{t+1} = \text{object dynamics via } w^o_t, \\
& x^r_{t+1} = \text{robot dynamics via } u^r_t, \quad (x^r_t, u^r_t) \in \mathcal{C}, \\
& w^o_t = z_t, \qquad \hat{w}^o_t = z_t .
\end{aligned}
```

The realized wrench $\hat{w}^o_t$ is **not** a decision variable — it is
whatever the simulator reports once $\mathbf{U}^r$ has been rolled out.
Splitting $\mathbf{W}^o$ from $\mathbf{U}^r$ through $z_t$ is what lets the two
planners run independently.

### ADMM decomposition

This is the $N = 2$ case of global-variable-consensus ADMM. Extraction maps
$A^i$ pull each block's own estimate of the wrench:

```math
A^o(\mathbf{U}^o)_t = w^o_t,
\qquad
A^r(\mathbf{U}^r)_t = \hat{w}^o_t ,
```

$A^o$ reads the object planner's proposal straight off its decision variable,
while $A^r$ reads the wrench the robot's rolled-out motion *actually* imparts
on the object. Each ADMM iteration $l$ runs four steps.

**1 — Subproblem updates.** Each block minimizes its own cost plus a proximal
term and the consensus penalty, both evaluated inside the sampler's rollout
cost:

```math
\mathbf{U}^{o,(l+1)} = \arg\min_{\mathbf{U}^o}
\Big\{ J_o(\mathbf{U}^o)
+ \tfrac{\gamma}{2}\big\|\mathbf{U}^o - \mathbf{U}^{o,(l)}\big\|^2
+ \tfrac{\rho}{2}\sum_{t} \big\| A^o(\mathbf{U}^o)_t - z^{(l)}_t + y^{o,(l)}_t \big\|^2 \Big\}
```

```math
\mathbf{U}^{r,(l+1)} = \arg\min_{\mathbf{U}^r}
\Big\{ J_r(\mathbf{U}^r)
+ \tfrac{\gamma}{2}\big\|\mathbf{U}^r - \mathbf{U}^{r,(l)}\big\|^2
+ \tfrac{\rho}{2}\sum_{t} \big\| A^r(\mathbf{U}^r)_t - z^{(l)}_t + y^{r,(l)}_t \big\|^2 \Big\}
```

The proximal term $\gamma > 0$ acts as inertia between iterations. It prevents
the radical trajectory shifts that non-convex contact dynamics otherwise
induce, and supplies the strong convexity that non-convex ADMM convergence
results require.

**2 — Consensus update.** The wrench space is unconstrained, so the projection
$\Pi_\mathcal{Z}$ is the identity and the update is a plain average:

```math
z^{(l+1)}_t = \tfrac{1}{2}\Big( A^o(\mathbf{U}^{o,(l+1)})_t + y^{o,(l)}_t
+ A^r(\mathbf{U}^{r,(l+1)})_t + y^{r,(l)}_t \Big)
```

**3 — Dual update.** The scaled duals integrate the disagreement, so a wrench
the robot repeatedly fails to produce is penalized ever more heavily until both
planners settle on something mutually feasible:

```math
y^{i,(l+1)}_t = y^{i,(l)}_t + A^i(\mathbf{U}^{i,(l+1)})_t - z^{(l+1)}_t,
\qquad i \in \{o, r\}
```

**4 — Adaptive penalty and variance.** From the primal residual
$r^{(l+1)} = [A^o - z\,;\ A^r - z]$ and the dual residual
$d^{(l+1)} = \rho\,(z^{(l+1)} - z^{(l)})$:

```math
\rho \leftarrow
\begin{cases}
2\rho, & \|r^{(l+1)}\| > 10\,\|d^{(l+1)}\| \\
\rho/2, & \|d^{(l+1)}\| > 10\,\|r^{(l+1)}\| \\
\rho, & \text{otherwise}
\end{cases}
\qquad
\Sigma_u^{(l+1)} \leftarrow \max\big(\Sigma_{\min},\, \kappa\,\|r^{(l+1)}\|\big)
```

Annealing the exploration covariance with the residual is meant to make the
planners explore widely while they disagree and quieten as they converge.
**It is off by default** ($\kappa = 0$) in both worlds: the primal residual
does not converge on this task, so $\kappa\|r\|$ sits pinned at its upper
clip rather than annealing anything. Measured over a 600-step run at
identical seed and configuration, leaving it on ended at a position error of
4.65 versus 2.01 with it off.

### Algorithm

> **Given** state $x_0$, previous $(\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}^o, \mathbf{Y}^r)$, parameters $\rho, \gamma$
> 1. Warm-start all five by shifting the previous solution one step
> 2. **for** $l = 0, \dots, N_{\mathrm{ADMM}} - 1$:
> 3. &nbsp;&nbsp;&nbsp;&nbsp; Object update — sampling MPC with the proximal + consensus penalty
> 4. &nbsp;&nbsp;&nbsp;&nbsp; Robot update — sampling MPC on the real robot model, same penalties
> 5. &nbsp;&nbsp;&nbsp;&nbsp; Consensus update $z^{(l+1)}$
> 6. &nbsp;&nbsp;&nbsp;&nbsp; Dual updates $y^{o,(l+1)},\ y^{r,(l+1)}$
> 7. &nbsp;&nbsp;&nbsp;&nbsp; Adapt $\rho$; anneal $\Sigma_u$
> 8. &nbsp;&nbsp;&nbsp;&nbsp; **break** if $\|r\| \le \epsilon_r$ and $\|d\| \le \epsilon_s$
> 9. Apply $u^r_0$, shift, observe $x_1$

The loop is a `jax.lax.while_loop`, so the early exit survives `jax.jit` and
the whole control step compiles to a single kernel.

### Cost functions

All SE(2) tracking terms share one weighted squared distance
([`se2_distance_sq`](oim/objects/planar_pushing.py)), with the angle wrapped
to $(-\pi, \pi]$:

```math
d^2_{w}(x, g) = w_p \|p - p^g\|^2 + w_\theta\, \mathrm{wrap}(\theta - \theta^g)^2 .
```

**Object block** ([`PlanarPushingObject`](oim/objects/planar_pushing.py)) —
goal tracking, obstacle clearance, and effort:

```math
\ell_o(x^o_t, w_t) = d^2_{q}(x^o_t, g)
+ w_{\text{obs}} \sum_{j} \max\big(\delta - \mathrm{sdf}(b_j(x^o_t)),\ 0\big)^2
+ r_o \|w_t\|^2 ,
\qquad
\ell_f(x^o_H) = d^2_{q_f}(x^o_H, g) ,
```

where $b_j(x^o)$ are the object's footprint boundary samples in the world
frame and $\delta$ is the clearance margin. The clearance term is geometric,
not a simulator contact force: the object block has no simulator.

**Robot block** — the task's own cost, plus the ADMM consensus penalty
(added by the ADMM layer, via the same function the object block uses):

```math
J_r(x^r_t, u^r_t) = r_r \|u^r_t\|^2
+ \underbrace{d^2_{q}(x^o_t, g)}_{\text{goal}}
+ \underbrace{d^2_{q}(x^o_t, x^{o*}_t)}_{\ell_c,\ \text{coupling}}
+ \ell_r(x^r_t) ,
```

matching paper eq. 17: $x^{o*}_t$ is the object planner's own nominal
trajectory from this ADMM iteration. The wrench consensus is a dense,
per-step signal but an indirect one; goal-tracking is the sparse, direct
signal the task actually cares about. Both solvers carry it so that
tracking error cannot accumulate unchecked between the (coarser) points
where the two actually agree on a wrench. $\ell_r$ shares its first two
terms across the two worlds and differs only where the embodiment forces
it to:

```math
\ell_r = \underbrace{w_{ee}\max\big(\|p^{ee}_t - p^o_t\|^2 - r_0^2,\ 0\big)
+ w_{\text{align}}\, \psi_{\text{align}}}_{\text{both worlds}}
+ \underbrace{w_{\text{tilt}}\, \psi_{\text{tilt}}
+ w_{z}\,(z^{ee}_t - z^\ast)^2}_{\text{3D only}}
+ \underbrace{w_{\text{obs}} \sum_j \max\big(\delta - \mathrm{sdf}(p^{ee}_t),\ 0\big)^2}_{\text{2D only}} ,
```

```math
\psi_{\text{align}} = \max\big(\gamma_0 - \cos\angle(p^o_t - p^{ee}_t,\ p^{o*}_t - p^o_t),\ 0\big),
\qquad
\psi_{\text{tilt}}(R) = \sqrt{\varrho^2 + \varphi^2} .
```

The approach term pulls the pusher toward the object but goes slack inside a
radius $r_0$; $\psi_{\text{align}}$ keeps it *behind* the object relative to
the goal. Both use the same weights on both sides
($w_{ee}=20$, $r_0=0.05$, $w_{\text{align}}=5$, $\gamma_0 = \cos(\pi/6)$).
The 3D-only terms shape a 3D end-effector *pose* — $\psi_{\text{tilt}}$
penalizes roll/pitch away from vertical and $(z^{ee}-z^\ast)^2$ holds the
stick tip at the block's mid-height — neither of which a disc robot has.
Conversely the 2D cost needs an explicit robot–obstacle clearance term,
which the 3D task gets from MJX contact for free.

$w_{\text{tilt}}$, $w_z$ and the 2D robot-obstacle weight are not given
numeric values in the paper.

### Object action parameterization

By default the object block's decision variable *is* the consensus variable:
it samples $w^o_t$ directly, subject only to the box bound
$|w^o_t| \le D^{-1}$, and $A^o$ is the identity up to a fixed rescaling.
That bound caps the wrench's *magnitude* but says nothing about its
*direction*: the sampler may still propose a pure torque or a pulling force,
and the consensus can converge onto something no point contact could realize.

A task may instead decide a **contact action**
$a_t = [p_x,\, p_y,\, f_n,\, f_t]$ — where on the object to push, in its body
frame, and with what normal/tangential force — and derive the wrench through
the contact Jacobian:

```math
A^o(\mathbf{U}^o)_t = J_c(p_t)^\top f_t
= \begin{bmatrix} f \\ (p^{c}_t - p^o_t) \times f \end{bmatrix},
\qquad f = f_n\,\hat{n}(p_t) + f_t\,\hat{t}(p_t) ,
```

subject to a projection applied to *every* sample:

```math
p_t \in \partial\mathcal{O}, \qquad 0 \le f_n \le f_{\max},
\qquad |f_t| \le \mu_c f_n .
```

Every reachable $A^o$ is then realizable by construction. The consensus
variable is still the 3-vector wrench, so nothing else in the ADMM layer
changes; only $\dim(a) = 4 \ne \dim(z) = 3$.

Sampling has to respect that geometry too. Contact points are perturbed and
re-projected onto the boundary, then rejection-filtered on normal alignment
($\hat{n}$ within $\tau_n$ of the nominal's) — an unfiltered Gaussian step
along the boundary can hop to the opposite face, which reverses the wrench
and reads to the consensus update as violent disagreement rather than
exploration. That filter makes the proposal *local*, so a separate
CEM-style search over the whole boundary re-chooses the contact point each
control step; without it the block can slide the contact along one face but
never decide to push from somewhere else, which is what routing around an
obstacle requires.

Tasks opt in via `object_action_dim`, `object_action_bounds`,
`object_action_to_consensus`, `project_object_action`,
`sample_object_actions` and `initial_object_action` on
[`ConsensusTask`](oim/task_base.py).

**Neither task uses it by default.** Both `PushT` and `PushT2D` sample the
wrench directly, on the view that *where* the robot touches the object is
the robot block's concern — the object planner only needs to say what
motion it wants, and making it also choose a contact point duplicates the
robot block's job inside the wrong subproblem. The contact-action path
remains available (`PushT2D(contact_actions=True)`, or
`examples/pusht.py --world 2d --contact-action admm`) so the two can be compared, since
the constraint it buys — every proposal inside the friction cone — is
real.

### Implementation notes

Where the implementation departs from the formulation above:

- **Penalty normalization.** The penalty and residuals are divided by the
  friction-cone limit $D^{-1}$ before squaring. Unnormalized, contact forces of
  ~10 N give a penalty of ~10², which swamps the task costs (~1) and drives the
  robot to optimize wrench matching instead of reaching the object. This is a
  diagonal preconditioning of the consensus constraint applied identically to
  both blocks, so the fixed point is unchanged, and it makes $\rho$,
  $\epsilon_r$ and $\epsilon_s$ scale-free.
- **The penalty is not $\Delta t$-weighted, the task costs are.** Both blocks
  compute $\Delta t\,\ell + \tfrac{\rho}{2}\|\cdot\|^2$ per step. The two
  blocks agree, so the fixed point is well defined, but the *effective* weight
  of the consensus penalty relative to the task cost scales as $1/\Delta t$ —
  changing the planning timestep silently re-tunes $\rho$.
- **Residual norms are unnormalized by horizon.** $\|r\|$ is a Frobenius norm
  over the stacked $(2H, 3)$ residual, not an RMS, so it grows like
  $\sqrt{2H}$. At $H = 15$ the observed residuals are $O(1)$, so
  $\epsilon_r = \epsilon_s = 0.5$ on both sides; the $0.05$ the paper uses is
  provably unreachable at this horizon and the early exit would never fire.
- **Variance annealing is additive, and disabled.** Because any sampler can be
  plugged into either block and most do not expose a mutable covariance, the
  wrappers *add* a perturbation of scale
  $\mathrm{clip}(\kappa\|r\|,\ \sigma_{\min},\ \sigma_{\max})$ on top of
  whatever the injected optimizer proposes, rather than replacing $\Sigma_u$.
  The upper clip is required — $\kappa\|r\|$ is otherwise an unbounded
  positive feedback loop — and with $\|r\|$ not converging on this task the
  clip binds permanently, so the default is $\kappa = 0$.
- **The direct wrench is box-bounded at the friction-cone limit.** The object
  block's action is clamped to $\pm D^{-1}$, so it cannot propose a wrench the
  support surface could not transmit no matter where the robot pushed. This is
  the half of the friction-cone constraint that survives sampling the wrench
  directly rather than a contact action.
- **$\rho$ and $\|r\|$ persist across control steps.** `rho_init` is only the
  $t = 0$ value; the adapted $\rho$ and the last primal residual are carried in
  the policy parameters and are never reset, so both drift over a run.
- **Obstacle clearance is geometric.** The object block has no simulator, so
  $\ell_o$ uses a signed-distance hinge on the object footprint rather than the
  simulator contact force $\lambda_t$.
- **Dual anti-windup.** Duals are clipped to $\pm y_{\max}$. Note this is why
  the $z$-update keeps the dual terms: $\sum_i y^i = 0$ is an ADMM invariant
  that would make $z = \tfrac{1}{2}(A^o + A^r)$ equivalent, but clipping breaks
  the invariant, so the duals must be carried explicitly.
- **Warm-start tail.** The receding-horizon shift zero-fills the vacated slot
  for $z$, $y^o$, $y^r$ and for a direct-wrench object block. For a structured
  action space it repeats the last value instead, since the zero vector need
  not be a feasible action there (a zero contact point is the object's own
  origin, which is not on its boundary).
- **Horizons.** The formulation permits $H^c \le \min(H^o, H^r)$; the
  implementation uses $H^o = H^r = H^c$, enforced in the `ADMM` constructor.

## Code layout

```
oim/
├── alg_base.py           SamplingBasedController: warm-start, spline knots,
│                           parallel rollouts, domain randomization, risk
├── task_base.py          Task (cost + MuJoCo model)
│                         ConsensusTask (the ADMM contract: object dynamics,
│                           A^r extraction, optional action parameterization)
├── risk.py               AverageCost, WorstCase, CVaR, … across randomizations
├── open_loop.py          offline trajectory optimization + playback
│
├── algs/                 every sampler shares sample_knots / update_params
│   ├── admm.py           ADMM loop; ConsensusSpace, WrenchConsensus;
│   │                       ObjectSubproblem, RobotSubproblem;
│   │                       RobotRollout / MJXRollout  ← the 2D/3D seam
│   ├── mppi.py  cem.py  predictive_sampling.py  cbo.py
│   └── dial.py  mppi_cma.py  mtp.py  evosax.py
│
├── objects/              analytic, simulator-free — shared by 2D and 3D
│   ├── sdf.py            Shape/Circle/Box/Capsule/Polygon, sdf_and_grad
│   ├── planar_pushing.py limit-surface dynamics + object-level costs
│   └── contact.py        w = J_cᵀf, friction-cone projection, contact
│                           sampling, boundary search for where to push
│
├── sim2d/                analytic 2D world — no MuJoCo anywhere
│   ├── engine.py         Sim2DState/Sim2DModel, resolve_contact,
│   │                       Analytic2DRollout
│   ├── task.py           PushT2D (a ConsensusTask, no MuJoCo)
│   ├── scenarios.py      clutter / corridor / gate
│   └── run.py            build_admm_2d, run_2d
│
├── sim3d/                MuJoCo drivers
│   ├── deterministic.py  run_interactive: viewer, replanning, mp4 recording
│   ├── run.py            run_3d_admm: headless, logs what run_2d logs;
│   │                       run_3d_plain: same, for any non-ADMM controller
│   ├── plan_overlay.py   both blocks' predicted object paths, in the viewer
│   └── asynchronous.py   controller and simulator in separate processes
│
├── tasks/                MuJoCo tasks — pusht.py is the ADMM one
├── models/               MJCF scenes and meshes (xarm6, g1, pusht_clutter, …)
└── utils/                spline interpolation, video recording, results JSON,
                            metrics.py (ADMM-vs-baseline comparison)
```

### How a control step propagates

One call to `ADMM.optimize(state, params)`, top to bottom:

| Stage | Code | What crosses the boundary |
| --- | --- | --- |
| Warm-start | `ADMM.optimize` | previous $\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}$, shifted one step |
| Object block | `ObjectSubproblem` → `PlanarPushingObject.step` | samples $\mathbf{W}^o$, rolls out analytically, returns $A^o$ |
| Robot block | `RobotSubproblem` → `RobotRollout.step` | samples $\mathbf{U}^r$, rolls out in MJX **or** 2D, returns $A^r$ |
| Consensus + duals | `ConsensusSpace` | $z, y^o, y^r$ — normalized by `task.consensus_scale()` |
| Convergence | `jax.lax.while_loop` | $\|r\|, \|d\|$; adapt $\rho$; exit on $\epsilon_r, \epsilon_s$ |

Two structural invariants make that work:

**Any sampler fits either ADMM block.** Both subproblems only ever call
`sample_knots` / `update_params` on their injected optimizer, which is what
makes `--robot-opt` / `--object-opt` interchangeable.

**Only one thing is simulator-specific.** `RobotRollout` is the entire 2D/3D
seam — `ADMM`, `ConsensusSpace`, `ObjectSubproblem` and everything under
`objects/` are shared verbatim, so 2D is not a second implementation of the
algorithm. The consensus penalty is owned by `ConsensusSpace` and applied by
the ADMM layer to *both* blocks; tasks are deliberately prevented from adding
their own (`robot_running_cost` receives no $z$, $y$ or $\rho$), so the two
blocks cannot silently drift into scoring the consensus variable differently.

## Designing a task

An ordinary sampling-based MPC problem

```math
\min_{u} \ \sum_t \ell(x_t, u_t) + \phi(x_{T+1}) \quad \text{s.t.} \quad x_{t+1} = f(x_t, u_t)
```

needs only a MuJoCo model plus two cost methods:

```python
class MyTask(Task):
    def __init__(self):
        super().__init__(mj_model, ...)

    def running_cost(self, x: mjx.Data, u: jax.Array) -> jax.Array: ...
    def terminal_cost(self, x: mjx.Data) -> jax.Array: ...
```

To additionally support ADMM, mix in [`ConsensusTask`](oim/task_base.py) and
declare the object-level subproblem and the consensus variable:

```python
class MyTask(Task, ConsensusTask):
    consensus_dim              # dimension of z
    consensus_scale()          # characteristic magnitude of z (normalizer)
    object_dynamics()          # closed-form x^o_{t+1} = f^o(x^o_t, w_t)
    object_running_cost()      # l_o
    object_terminal_cost()     # l_f
    object_state_from_robot()  # pull x^o out of the robot's MJX state
    realized_consensus()       # A^r: extraction map, read from the rollout
    robot_running_cost()       # J_r = l_o + l_r + l_c  (no ADMM penalty!)
    robot_terminal_cost()
```

Six further hooks — `object_action_dim`, `object_action_bounds`,
`object_action_to_consensus`, `project_object_action`,
`sample_object_actions` and `initial_object_action` — are optional. They
default to sampling the consensus variable directly, box-bounded at
`consensus_scale()`; override them only for a structured object action space
like the [contact action](#object-action-parameterization).

To implement a new *sampler*, subclass
[`SamplingBasedController`](oim/alg_base.py) and provide `init_params`,
`sample_knots` and `update_params`; parallel rollouts, spline
parameterization, domain randomization and risk aggregation are handled for
you. Such a sampler is immediately usable as either ADMM subproblem solver.

## Domain randomization and risk

Rolling out across perturbed models in parallel is nearly free on GPU.
Override `domain_randomize_model` / `domain_randomize_data` on a task, then
pass `num_randomizations` to any controller. Costs are aggregated across
domains by a [`RiskStrategy`](oim/risk.py): `AverageCost` (default),
`WorstCase`, `BestCase`, `ExponentialWeightedAverage`, `ValueAtRisk`, or
`ConditionalValueAtRisk`.

```python
ctrl = PredictiveSampling(
    task, num_samples=32, noise_level=0.1,
    num_randomizations=16, risk_strategy=WorstCase(),
)
```

Rollouts then have shape `(num_randomizations, num_samples, num_time_steps, ...)`.

## Running on the real xArm6

`oim/real3d/` runs the same ADMM push-T controller on a physical UFACTORY
xArm6. The planner (`ADMM.optimize`), the task cost and the MJX rollouts are
reused verbatim from the simulation path; only the outer loop's I/O changes:

```
sim3d:  mjx_data <- mj_data ;      mj_data.ctrl = u ; mujoco.mj_step(...)
real3d: mjx_data <- ROS sensors ;  publish u to the arm's velocity controller
```

| File | Role |
| --- | --- |
| [`oim/real3d/interface.py`](oim/real3d/interface.py) | `RobotWorldInterface` (the I/O seam): `MujocoMockInterface` for laptop testing, `Ros2Interface` for hardware |
| [`oim/real3d/run_real.py`](oim/real3d/run_real.py) | the closed loop -- the hardware counterpart of `sim3d/run.py::_run` |
| [`examples/pusht_real.py`](examples/pusht_real.py) | entry point; same controller build as `examples/pusht.py` |

MJX is still used on hardware -- it is the planner's internal predictive model,
run on the GPU every control step. Because the planner is a plain jitted JAX
function, it is called in-process; unlike the Isaac-Gym OI-MPPI stack there is
no zerorpc planner server (that was a workaround for Isaac's per-process sim
context, which MJX does not need).

### Laptop dry-run (no hardware)

Drives a MuJoCo sim through the hardware interface, so the whole loop -- state
assembly, command mapping, logging -- is validated with no robot and no ROS:

```bash
python examples/pusht_real.py --mock --steps 200
# hold vs. stream the plan within each replanning period:
python examples/pusht_real.py --mock --command-mode stream --steps 200
```

CPU JIT is slow (~100 s/control step); use it to check plumbing, not behaviour.
The run writes `oim/results/pusht3d_xarm6_mock_admm_*_states_*.json`, the same
schema a simulation run produces, so the two can be compared entry-for-entry.

### Environment (read first)

Collapsing OI-MPPI's two planner-side processes into one means `pusht_real.py`
needs **one environment with both ROS 2 (`rclpy`, `tf2_ros`) and the CUDA JAX
stack (`jax[cuda13]`, `mujoco-mjx`, `oim`)**. The catch: ROS 2 Humble ships
`rclpy` for Python 3.10, but `oim`/JAX 0.8 need Python ≥ 3.12, so you cannot
just `source /opt/ros/humble/setup.bash` into the uv venv. Two workable paths:

- **pixi / RoboStack (recommended).** [`oim/real3d/pixi.toml`](oim/real3d/pixi.toml)
  pins `ros-humble-rclpy` + CUDA JAX + MJX in one env: `cd oim/real3d && pixi
  install && pixi shell`, add the repo to `PYTHONPATH`, run `pusht_real.py`
  there. This single-env resolve is the riskiest step (conda ROS + pypi
  CUDA-JAX) -- timebox it.
- **Two-process fallback (ready).** If that env won't resolve, run the ROS I/O
  and the planner in separate envs, bridged by a localhost socket -- OI-MPPI's
  split, minus zerorpc:

  ```bash
  # ROS env (pixi humble): holds all the rclpy I/O
  python -m oim.real3d.ros_bridge --port 5599
  # JAX env: planner talks to the bridge, no rclpy imported here
  python examples/pusht_real.py --socket --bridge-port 5599 --replan-rate 2.5
  ```

  `oim/real3d/ros_bridge.py` wraps the same `Ros2Interface`; `--socket`
  swaps `pusht_real.py` onto a `SocketInterface`. The `RobotWorldInterface`
  seam is what makes this a drop-in.

### Running on the robot

FoundationPose runs on the **perception laptop** as its own stack (its own
repo, Docker, conda `my` env -- camera + mask + pose node + TF broadcaster;
see its readme). It publishes the `fp_object_pose` TF, which `Ros2Interface`
reads by default (`object_frame`; use `"sam6d_object"` for SAM-6D).

Laptop and desktop are two machines, so connect their ROS 2 over the LAN
first: on **both**, in every terminal, `source oim/real3d/scripts/setup_dds_env.sh`
(matches `RMW_IMPLEMENTATION` + `ROS_DOMAIN_ID`; on a shared subnet CycloneDDS
discovers over multicast, so no XML is needed). Then check `ros2 topic list`
shows the other host. Only if that fails (multiple NICs / multicast blocked)
fall back to [`config/cyclonedds.xml`](oim/real3d/config/cyclonedds.xml).

On the **desktop**, three terminals:

| Terminal | Command |
| --- | --- |
| robot bringup (`keti_ws` docker) | `ros2 launch main real_xarm6.launch.py` |
| planner / driver (pixi env) | `python examples/pusht_real.py --dry-run --replan-rate 2.5 --steps 400` |
| RViz (optional) | `rviz2 -d <config>.rviz` |

The driver publishes to `velocity_controller/commands_nominal` by default,
which feeds the **CBF safety filter** (`commands_nominal` → CBF node →
`commands` → motors) -- so with the CBF node up, the arm moves (filtered).
`--dry-run` instead publishes to a dead `_dryrun` topic nothing subscribes to,
so the arm stays still while you inspect state / TF / planned velocities in
RViz; real no-motion also means the CBF node down or an e-stop. Other flags:
`--velocity-topic`, `--replan-rate`, `--command-mode {hold,stream}`,
`--control-rate`, `--warp`.

### Bring-up & calibration checklist

The mock validates everything except the physical setup. On the robot, verify:

- **World frame / the base TF link.** FoundationPose's TF chain
  `xarm_device → camera → fp_object_pose` is rooted at the robot base, but the
  planner wants the object in the MJX world (arm base at `XARM6_BASE_POS=(0.2,
  0.75)`, yaw −90°). `Ros2Interface` publishes this `world → xarm_device`
  transform itself (`_publish_base_tf`, from the same model constants), so no
  separate node is needed -- just set `base_z` to the real base height and
  confirm the yaw sign by jogging, then check in RViz that object/goal/obstacles
  land at their MJCF coordinates.
- **Joint mapping.** Jog each joint and confirm `joint{i}` ↔ `xarm6_joint{i}`
  and the sign convention (CW = +) match the model.
- **joint6.** Set the real wrist-roll joint to the angle it is welded at in the
  MJX scene (the planner commands it 0).
- **Object yaw.** Confirm the FoundationPose yaw sign matches the block's `T_z`
  hinge (`TODO(lab)` in `_lookup_object_se2`).
- **Physics params.** Match `PlanarPushingObject(mu, mass, limit_surface_radius)`
  in `oim/tasks/pusht.py` to the real block/table, or the planner's object
  prediction will drift from reality.
- **Start pose.** Check `XARM6_START_QPOS_DEG` puts the stick tip at the block
  and is collision-free.

### Real-time and speed

`ADMM.optimize` is ~0.36 s/step on the lab GPU, so the arm replans at ~2.5 Hz
versus 20 Hz in sim -- an 8× coarser control rate that is a real sim-to-real
factor, not just calibration. Staged plan:

1. **Single-thread + hold @ ~2.5 Hz** (default) -- get the first closed-loop push.
2. **`--command-mode stream`** -- if motion is jerky at the low replan rate.
3. **Speed up `optimize`** -- try `--warp`; lower `num_samples` /
   `num_randomizations`; make the *planning* model cheaper than the execution
   model (fewer `iterations`/`ls_iterations`, simpler collision geoms in the
   rollout MJCF, shorter horizon). Validate each against the mock.
4. **Async driver** -- a planner thread posting the latest plan to a high-rate
   streaming loop -- only if 2–4 are not enough.

## Citation

If you use this work, please cite:

```bibtex
@article{raicevic2026objectinformed,
  title   = {Object-Informed Model Predictive Path Integral Control
             for Non-Prehensile Robot Manipulation},
  author  = {Raicevic, Nikola and Kim, Hyomuk and Mulla, Shahid and
             Radhakrishnan, Bharath Raam and Yu, Chenbin and
             Lee, Ki Myung Brian and Atanasov, Nikolay},
  year    = {2026}
}
```

## Acknowledgements

This project is a fork of [**Hydrax**](https://github.com/vincekurtz/hydrax) by
Vince Kurtz, which provides the sampling-based MPC framework — the
controller/task abstractions, spline parameterization, parallel MJX rollouts,
domain randomization, and every non-ADMM algorithm listed above. The ADMM
object-informed manipulation layer is our addition. Hydrax is itself inspired
by [MJPC](https://github.com/google-deepmind/mujoco_mpc).

```bibtex
@misc{kurtz2024hydrax,
  title  = {Hydrax: Sampling-based model predictive control on GPU
            with JAX and MuJoCo MJX},
  author = {Kurtz, Vince},
  year   = {2024},
  note   = {https://github.com/vincekurtz/hydrax}
}
```

The xArm6 model derives from [UFACTORY](https://www.ufactory.cc/)'s published
URDF; the Unitree G1 model is from
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) (see
[`oim/models/g1/LICENSE`](oim/models/g1/LICENSE)). Motion-capture
references come from the
[LocoMuJoCo](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets)
dataset.

## License

MIT — see [LICENSE](LICENSE).
