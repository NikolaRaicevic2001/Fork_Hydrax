# Object-Informed Manipulation MJX

GPU-accelerated planning for **non-prehensile manipulation**, on
[JAX](https://jax.readthedocs.io/) and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

**Object-informed MPPI** splits long-horizon pushing into an *object-level*
planner (what contact wrench does the object need?) and a *robot-level*
planner (how do I produce it?), coordinated by ADMM until the two agree on
the wrench. Both blocks accept any sampler from the library below as their
inner solver.

<p align="center">
  <img src="img/humanoid.gif" width="30%" />
  &nbsp;&nbsp;
  <img src="img/cube.gif" width="30%" />
</p>

- [Setup](#setup) · [Running](#running) · [Method](#method) ·
  [Code layout](#code-layout) · [Extending](#extending) ·
  [Algorithms](#algorithms) · [Citation](#citation)

## Setup

Python ≥ 3.12, CUDA 13, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NikolaRaicevic2001/Object-Informed-Manipulation-MJX.git
cd Object-Informed-Manipulation-MJX && uv sync
```

| | |
| --- | --- |
| `uv run <cmd>` | Run in the environment (or `source .venv/bin/activate`) |
| `uv run pytest` | Tests |
| `uv run ruff check .` | Lint |

## Running

Push-T through clutter: drive an SE(2) goal past three static obstacles.
**Three programs, one per job**, so an expensive step never repeats for a
cheap one:

| Program | Runs | Writes | Reads |
| --- | --- | --- | --- |
| [`examples/pusht.py`](examples/pusht.py) | one experiment | `results/runs/*.json` | — |
| [`oim/run_launch.py`](oim/run_launch.py) | a sweep, one subprocess per cell | `results/sweeps/*.json` | the sweep config |
| [`oim/run_eval.py`](oim/run_eval.py) | nothing | `results/eval/*.{json,md}` | run files |

Run files hold only what was *observed*. Success, goal error, execution time
and frequency are derived by `run_eval`, so a new metric or a stricter
tolerance costs a second rather than a re-run.

### Single runs

```bash
# 3D xArm6: MPPI object block, 600 headless steps, mp4 + consensus overlay
uv run python examples/pusht.py --robot xarm6 --record admm \
    --object-opt mppi --headless --steps 600 --show-plans

# 2D gate: contact-action object block, animated, eager for a debugger
uv run python examples/pusht.py --world 2d --env gate --contact-action \
    --no-jit --animate admm --n-admm 12 --rho 20 --steps 40 --seed 3
```

World and scene flags go **before** the algorithm name, solver and run flags
**after** it.

| Flag | Default | |
| --- | --- | --- |
| ***before the algorithm*** | | |
| `--world {3d,2d}` | `3d` | `3d`: MJX contact, point or xArm6 robot, `admm`/`mppi`/`ps`. `2d`: analytic single-point contact, disc robot, `admm` only — same ADMM code, for separating algorithm bugs from simulator bugs |
| `--samples`, `--horizon` | 64, 15 | Rollouts per block, consensus horizon $H$ |
| `--robot {point,xarm6}` | `point` | *3D:* embodiment; `xarm6` implies clutter |
| `--record` | off | *3D:* mp4 (needs `ffmpeg`); with `--headless`, renders offscreen |
| `--warp` | off | *3D:* experimental [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) rollouts. Also disables JAX's GPU preallocation, since Warp allocates outside that pool and otherwise cannot build its CUDA graphs |
| `--env {clutter,corridor,gate}` | `clutter` | *2D:* scenario (below) |
| `--scene {clutter,gym2}` | `clutter` | *3D:* MJCF layout. `gym2` is the single-obstacle IsaacGym conversion and needs `--robot xarm6`. Recorded in the run identity, so `run_eval` can group on it |
| `--contact-action` | off | *2D:* object block decides $[p, f_n, f_t]$, not the wrench ([below](#object-action-parameterization)) |
| `--no-jit`, `--no-obstacles`, `--no-relocate`, `--animate`, `--no-plot` | off | *2D:* debug and output toggles |
| ***after the algorithm*** | | |
| `--robot-opt`, `--object-opt` | `mppi` | Inner solver per block: `mppi`/`cem`/`ps`/`cbo` |
| `--n-admm` | 8 | Max ADMM iterations per control step |
| `--rho`, `--gamma` | 10.0, 0.1 | Penalty $\rho$, proximal weight $\gamma$ |
| `--steps`, `--seed` | 200, 5 | Control steps, RNG seed |
| `--headless` | off | No viewer; run `--steps` and save a run file |
| `--show-plans` | off | *3D:* overlay both blocks' predicted object paths — **amber** what the object block intends ($x^{o*}$), **teal** what the robot block would produce; their gap is the primal residual, made spatial. Costs 1.7% of a control step |
| *not exposed* | | Shared by both worlds: $\epsilon_r = \epsilon_s = 0.5$, noise annealing off, wrench bounds $\pm D^{-1}$, success tolerance 5 cm / 0.05 rad |

| `--env` | Stresses | Clearance |
| --- | --- | --- |
| `clutter` | routing *around* obstacles; mirrors the MJX scene exactly. Always leaves a way round, so a planner can pass it while unable to commit to a passage | 41 mm |
| `corridor` | pushing *through* a narrow horizontal channel | 15 mm |
| `gate` | a vertical slot, then rotating to the goal | 5 mm |

| Output | When |
| --- | --- |
| `results/runs/*.json` | `--headless` — settings, scene, per-step states/controls/wrenches/residuals/timings. Same keys and frames in both worlds; each carries a `schema` block |
| `recordings/*.png` | always (2D), `--headless` (3D) — trajectory + residuals |
| `recordings/*.gif` / `*.mp4` | `--animate` / `--record` |

### Sweeps

```yaml
# oim/configs/run_launch_config.yaml
sweep:
  task: [{ world: 3d, robot: point }, { world: 3d, robot: xarm6 }]
  algorithm: [admm, mppi]
  horizon: [5, 15, 25]
  seed: [0, 1, 2, 3, 4]
fixed: { steps: 200, headless: true }
```

```bash
uv run python -m oim.run_launch                        # the whole product
uv run python -m oim.run_launch --dry-run              # print, run nothing
uv run python -m oim.run_launch --only algorithm=admm  # narrow it
uv run python -m oim.run_launch --warp --set steps=50  # override `fixed:`
```

| | |
| --- | --- |
| Axes | `task`, `algorithm`, `robot_opt`, `object_opt`, `horizon`, `samples`, `n_admm`, `seed` — anything `examples/pusht.py` takes as a flag. An empty list drops the axis, leaving `pusht.py`'s own default |
| `fixed:` | applied to every cell; override per sweep with `--set key=value` (`--warp` is shorthand for `--set warp=true`). Unknown keys are rejected before the first cell runs |
| Order | outermost axis first: every algorithm, horizon and seed of task 1 finishes before task 2 starts |
| Invalid cells | dropped: 2D has no flat baselines, and a flat baseline has no ADMM blocks to sweep |
| Isolation | one subprocess per cell, so a crash or OOM costs one cell, not the sweep, and each run gets the whole GPU and a clean allocator |
| GPU barrier | waits for the previous cell's memory to be released before starting the next, and reports free MiB per cell. `--min-free-mib N` additionally requires headroom (the xArm6 ADMM cell peaks near 12000) |
| On failure | logged and skipped (`--stop-on-error` to abort instead) |
| Manifest | `results/sweeps/sweep_*.json` — each cell's settings, exact command, exit status, duration |

### Evaluation

```bash
uv run python -m oim.run_eval                        # every run, auto-grouped

# pick the tasks, algorithms and horizons to summarise, one row each
uv run python -m oim.run_eval \
    --filter task=pusht3d_point,pusht3d_xarm6 \
    --filter algorithm=admm,mppi \
    --filter horizon=5,15,25 \
    --group-by task algorithm horizon --format markdown

uv run python -m oim.run_eval --pos-tol 0.02         # re-score, no re-running
```

| | |
| --- | --- |
| `--filter KEY=A,B` | keep runs whose field is one of these; repeatable. Values of one field are OR-ed, different fields AND-ed. Any field in the table below, or any recorded hyperparameter |
| `--group-by` | one table row per distinct combination. Omitted: identity **plus any hyperparameter that varies**, so a horizon sweep grows a horizon column by itself |
| seeds | always the trials *within* a row, never a row of their own |

| Column | Paper | |
| --- | --- | --- |
| `SR` | SR | fraction reaching both tolerances, re-derived from the final pose |
| `eps_d` ± | $\epsilon_d$ | position error, all trials |
| `eps_d^s` | $\epsilon_d^s$ | position error, successful trials only |
| `theta` ± | — | orientation error; not in the paper |
| `T` | $T$ | *simulated* time (`steps_run × dt`), machine-independent; a failed trial is credited the slowest time in the **sweep**, so methods stay comparable |
| `f (Hz)` | $\bar{f}$ | wall-clock planning rate, from the recorded `compute_time` |


## Method

A robot manipulates an unactuated rigid object through contact. Rather than
one monolithic contact-implicit problem, this is **two subsystems coupled
only through the contact wrench**, reconciled by ADMM.

### Subsystem models

| | Dynamics | |
| --- | --- | --- |
| Robot | $x^r_{t+1} = f_r(x^r_t) + g_r(x^r_t) u^r_t + h_r(x^r_t) w^r_t$ | input-affine |
| Object | $x^o_{t+1} = f_o(x^o_t) + h_o(x^o_t) w^o_t$ | unactuated; moves only via the wrench |

The object planner treats $w^o_t \triangleq [f_x, f_y, \tau]^\top$ as a
*decision variable* rather than the outcome of complementarity constraints.
For quasi-static planar pushing the **ellipsoidal limit surface** gives a
closed-form model with no simulator in the loop:

```math
\dot{x}^o = D\,w^o, \quad
D \triangleq \operatorname{diag}\big(\mu m g,\ \mu m g,\ c\,r\,\mu m g\big)^{-1},
\quad x^o_{t+1} = x^o_t + \Delta t\, D\, w^o_t
```

with $\mu$ friction, $m$ mass, $c, r$ the limit-surface pressure coefficient
and characteristic radius. $D^{-1}$ is the **friction-cone limit** — the
largest wrench the support can transmit — reused throughout as the natural
normalizer.

### Consensus problem

Over horizon $H$, with $\mathbf{Z} = \{z_t\}$ shared:

```math
\begin{aligned}
\min_{\mathbf{W}^o,\, \mathbf{U}^r,\, \mathbf{Z}} \quad
& \sum_{t=0}^{H-1}\Big(\ell_o(x^o_t) + \ell_r(x^r_t, u^r_t)\Big) + \ell_f(x^o_H) \\
\text{s.t.}\quad
& x^o_{t+1} \text{ via } w^o_t, \quad x^r_{t+1} \text{ via } u^r_t, \quad (x^r_t, u^r_t) \in \mathcal{C}, \\
& w^o_t = z_t, \qquad \hat{w}^o_t = z_t .
\end{aligned}
```

The realized wrench $\hat{w}^o_t$ is **not** a decision variable — it is
whatever the rollout produces once $\mathbf{U}^r$ is applied. Splitting
$\mathbf{W}^o$ from $\mathbf{U}^r$ through $z_t$ is what lets the two
planners run independently. Extraction maps pull each block's estimate:

```math
A^o(\mathbf{U}^o)_t = w^o_t \quad\text{(read off the decision variable)},
\qquad
A^r(\mathbf{U}^r)_t = \hat{w}^o_t \quad\text{(read off the rollout)} .
```

### ADMM iteration

The $N = 2$ case of global-variable-consensus ADMM. Each iteration $l$:

| # | Step | |
| --- | --- | --- |
| 1 | $\mathbf{U}^{i,(l+1)} = \arg\min \big\{ J_i + \tfrac{\gamma}{2}\lVert \mathbf{U}^i - \mathbf{U}^{i,(l)}\rVert^2 + \tfrac{\rho}{2}\sum_t \lVert A^i_t - z^{(l)}_t + y^{i,(l)}_t\rVert^2 \big\}$ | both blocks, $i \in \{o,r\}$; penalties evaluated inside the sampler's rollout cost |
| 2 | $z^{(l+1)}_t = \tfrac{1}{2}\big( A^o_t + y^{o,(l)}_t + A^r_t + y^{r,(l)}_t \big)$ | $\Pi_\mathcal{Z} = \mathrm{id}$, so a plain average |
| 3 | $y^{i,(l+1)}_t = y^{i,(l)}_t + A^i_t - z^{(l+1)}_t$ | duals integrate disagreement |
| 4 | $\rho \leftarrow 2\rho$ if $\lVert r\rVert > 10\lVert d\rVert$; $\rho/2$ if $\lVert d\rVert > 10\lVert r\rVert$ | from $r = [A^o - z; A^r - z]$, $d = \rho(z^{(l+1)} - z^{(l)})$ |

The proximal term $\gamma > 0$ is inertia between iterations: it prevents
the radical trajectory shifts non-convex contact dynamics induce, and
supplies the strong convexity non-convex ADMM convergence results require.

> **Given** $x_0$, previous $(\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}^o, \mathbf{Y}^r)$, parameters $\rho, \gamma$
> 1. Warm-start all five by shifting one step
> 2. **for** $l = 0 \dots N_{\mathrm{ADMM}}-1$: steps 1–4 above
> 3. &nbsp;&nbsp;&nbsp;&nbsp; **break** if $\lVert r\rVert \le \epsilon_r$ and $\lVert d\rVert \le \epsilon_s$
> 4. Apply $u^r_0$, shift, observe $x_1$

A `jax.lax.while_loop`, so the early exit survives `jit` and a control step
compiles to one kernel.

### Costs

All SE(2) tracking shares one weighted squared distance
([`se2_distance_sq`](oim/objects/planar_pushing.py)), angle wrapped to
$(-\pi,\pi]$: $d^2_w(x,g) = w_p\lVert p - p^g\rVert^2 + w_\theta \mathrm{wrap}(\theta-\theta^g)^2$.

```math
\ell_o(x^o_t, w_t) = d^2_{q}(x^o_t, g)
+ w_{\text{obs}} \sum_{j} \max\big(\delta - \mathrm{sdf}(b_j(x^o_t)),\ 0\big)^2
+ r_o \lVert w_t\rVert^2 ,
\qquad \ell_f = d^2_{q_f}(x^o_H, g)
```

$b_j(x^o)$ are the footprint's boundary samples in world frame, $\delta$ the
clearance margin. The clearance term is geometric, not a simulator contact
force — the object block has no simulator.

```math
J_r(x^r_t, u^r_t) = r_r \lVert u^r_t\rVert^2
+ \underbrace{d^2_{q}(x^o_t, g)}_{\text{goal}}
+ \underbrace{d^2_{q}(x^o_t, x^{o*}_t)}_{\ell_c\ \text{coupling}}
+ \ell_r(x^r_t)
```

with $x^{o*}_t$ the object planner's nominal trajectory from this iteration
(paper eq. 17). The wrench consensus is dense but indirect; goal tracking is
sparse but direct — both solvers carry it so tracking error cannot
accumulate between the coarser points where the blocks actually agree.

```math
\ell_r = \underbrace{w_{ee}\max\big(\lVert p^{ee}_t - p^o_t\rVert^2 - r_0^2,\ 0\big)
+ w_{\text{align}} \psi_{\text{align}}}_{\text{both worlds}}
+ \underbrace{w_{\text{tilt}} \psi_{\text{tilt}} + w_{z}(z^{ee}_t - z^\ast)^2}_{\text{3D only}}
+ \underbrace{w_{\text{obs}} \textstyle\sum_j \max(\delta - \mathrm{sdf}(p^{ee}_t), 0)^2}_{\text{2D only}}
```

```math
\psi_{\text{align}} = \max\big(\gamma_0 - \cos\angle(p^o_t - p^{ee}_t,\ p^{o*}_t - p^o_t),\ 0\big),
\qquad \psi_{\text{tilt}}(R) = \sqrt{\varrho^2 + \varphi^2}
```

The approach term pulls the pusher in but goes slack inside $r_0$;
$\psi_{\text{align}}$ keeps it *behind* the object relative to the goal.
Same weights on both sides ($w_{ee}=20$, $r_0=0.05$, $w_{\text{align}}=5$,
$\gamma_0=\cos(\pi/6)$). The 3D-only terms shape an end-effector *pose*
(stick vertical, at block mid-height), which a disc robot does not have;
conversely 2D needs an explicit robot–obstacle term that MJX contact
provides for free. $w_{\text{tilt}}$, $w_z$ and the 2D obstacle weight are
not given numeric values in the paper.

### Object action parameterization

By default the object block's decision variable *is* the consensus
variable: it samples $w^o_t$ directly, box-bounded at $|w^o_t| \le D^{-1}$.
That bounds the wrench's *magnitude* but not its *direction* — the sampler
may still propose a pure torque or a pulling force.

A task may instead decide a **contact action**
$a_t = [p_x, p_y, f_n, f_t]$ — where to push in the body frame, and with
what normal/tangential force — and derive the wrench through the contact
Jacobian, making every reachable $A^o$ realizable by construction:

```math
A^o_t = J_c(p_t)^\top f = \begin{bmatrix} f \\ (p^{c}_t - p^o_t) \times f \end{bmatrix},
\quad f = f_n\hat{n}(p_t) + f_t\hat{t}(p_t),
\quad p_t \in \partial\mathcal{O},\ 0 \le f_n \le f_{\max},\ |f_t| \le \mu_c f_n
```

$z$ is still the 3-vector wrench, so nothing else in the ADMM layer changes;
only $\dim(a) = 4 \ne \dim(z) = 3$. Sampling must respect the geometry:
points are perturbed, re-projected onto the boundary, then rejection-
filtered on normal alignment (an unfiltered step can hop to the opposite
face, reversing the wrench). That makes the proposal local, so a separate
CEM search over the whole boundary re-chooses the contact point each step —
without it the block can slide along one face but never decide to push from
elsewhere, which is what routing around an obstacle requires.

**Neither task uses it by default** (`--contact-action` opts in): where the
robot touches is the robot block's concern, and making the object planner
choose it duplicates that job in the wrong subproblem. Tasks opt in via
`object_action_dim`, `object_action_bounds`, `object_action_to_consensus`,
`project_object_action`, `sample_object_actions`, `initial_object_action`.

### Implementation notes

Where the implementation departs from the formulation above.

| | What, and why |
| --- | --- |
| **Penalty normalization** | Penalty and residuals are divided by $D^{-1}$ before squaring. Unnormalized, ~10 N forces give ~10² against task costs of ~1, and the robot optimizes wrench matching instead of reaching the object. Identical diagonal preconditioning on both blocks, so the fixed point is unchanged and $\rho, \epsilon_r, \epsilon_s$ become scale-free. |
| **Penalty is not $\Delta t$-weighted** | Both blocks compute $\Delta t\,\ell + \tfrac{\rho}{2}\lVert\cdot\rVert^2$. They agree, so the fixed point is well defined, but the penalty's effective weight scales as $1/\Delta t$ — changing the planning timestep silently re-tunes $\rho$. |
| **Residuals unnormalized by horizon** | $\lVert r\rVert$ is a Frobenius norm over $(2H,3)$, not an RMS, so it grows like $\sqrt{2H}$. At $H=15$ residuals are $O(1)$, so $\epsilon_r = \epsilon_s = 0.5$; the paper's $0.05$ is unreachable here and the early exit would never fire. |
| **Variance annealing additive, and off** | Most samplers expose no mutable covariance, so the wrappers *add* $\mathrm{clip}(\kappa\lVert r\rVert, \sigma_{\min}, \sigma_{\max})$ rather than replacing $\Sigma_u$. The upper clip is required ($\kappa\lVert r\rVert$ is otherwise a positive feedback loop), and since $\lVert r\rVert$ does not converge here the clip binds permanently — so $\kappa = 0$. Measured over 600 steps at identical seed: final position error 4.65 with annealing on, 2.01 off. |
| **Direct wrench box-bounded** | Clamped to $\pm D^{-1}$, so the block cannot propose a wrench the support could not transmit. This is the half of the friction-cone constraint that survives sampling the wrench rather than a contact action. |
| **$\rho$ and $\lVert r\rVert$ persist** | `rho_init` is only the $t=0$ value; both are carried in the policy parameters and never reset, so they drift over a run. |
| **Dual anti-windup** | Duals clipped to $\pm y_{\max}$. This is why the $z$-update keeps the dual terms: $\sum_i y^i = 0$ is an ADMM invariant that would make $z = \tfrac12(A^o + A^r)$ equivalent, but clipping breaks it. |
| **Warm-start tail** | The shift zero-fills the vacated slot for $z, y^o, y^r$ and a direct-wrench block; a structured action space repeats the last value instead, since zero need not be feasible there (a zero contact point is the object's origin, not on its boundary). |
| **Horizons** | The formulation permits $H^c \le \min(H^o, H^r)$; the implementation enforces $H^o = H^r = H^c$. |

## Code layout

```
oim/
├── alg_base.py           SamplingBasedController: warm-start, spline knots,
│                           parallel rollouts, domain randomization, risk
├── task_base.py          Task; ConsensusTask (the ADMM contract)
├── risk.py               AverageCost, WorstCase, CVaR, …
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
│   └── contact.py        w = J_cᵀf, friction-cone projection, sampling
│
├── sim2d/                analytic 2D world — no MuJoCo anywhere
│   ├── engine.py         Sim2DState/Sim2DModel, resolve_contact
│   ├── task.py           PushT2D          scenarios.py  clutter/corridor/gate
│   └── run.py            build_admm_2d, run_2d
│
├── sim3d/                MuJoCo drivers
│   ├── deterministic.py  run_interactive: viewer, replanning, recording
│   ├── run.py            run_3d_admm / run_3d_plain: headless + logging
│   ├── plan_overlay.py   both blocks' predicted paths, viewer and video
│   └── asynchronous.py   controller and simulator in separate processes
│
├── run_launch.py         sweep driver;  run_eval.py  post-hoc metrics
├── configs/              run_launch_config.yaml (the sweep definition)
├── tasks/  models/       MuJoCo tasks; MJCF scenes and meshes
└── utils/                spline, video, results.py (run files), metrics.py
```

One `ADMM.optimize(state, params)` call, top to bottom:

| Stage | Code | What crosses the boundary |
| --- | --- | --- |
| Warm-start | `ADMM.optimize` | previous $\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}$, shifted |
| Object block | `ObjectSubproblem` → `PlanarPushingObject.step` | samples $\mathbf{W}^o$, rolls out analytically, returns $A^o$ |
| Robot block | `RobotSubproblem` → `RobotRollout.step` | samples $\mathbf{U}^r$, rolls out in MJX **or** 2D, returns $A^r$ |
| Consensus + duals | `ConsensusSpace` | $z, y^o, y^r$, normalized by `consensus_scale()` |
| Convergence | `jax.lax.while_loop` | $\lVert r\rVert, \lVert d\rVert$; adapt $\rho$; exit test |

Two invariants make that work:

- **Any sampler fits either block.** Both subproblems only call
  `sample_knots`/`update_params`, which is what makes `--robot-opt` and
  `--object-opt` interchangeable.
- **Only `RobotRollout` is simulator-specific.** `ADMM`, `ConsensusSpace`,
  `ObjectSubproblem` and all of `objects/` are shared verbatim, so 2D is not
  a second implementation. The consensus penalty is owned by
  `ConsensusSpace` and applied by the ADMM layer to *both* blocks; tasks
  cannot add their own (`robot_running_cost` receives no $z, y, \rho$), so
  the blocks cannot drift into scoring $z$ differently.

## Extending

A plain sampling-based MPC problem
$\min_u \sum_t \ell(x_t,u_t) + \phi(x_{T+1})$ s.t. $x_{t+1} = f(x_t,u_t)$
needs a MuJoCo model plus `running_cost` and `terminal_cost`. For ADMM, mix
in [`ConsensusTask`](oim/task_base.py):

```python
class MyTask(Task, ConsensusTask):
    consensus_dim              # dimension of z
    consensus_scale()          # characteristic magnitude of z (normalizer)
    object_dynamics()          # closed-form x^o_{t+1} = f^o(x^o_t, w_t)
    object_running_cost()      # l_o          object_terminal_cost()   # l_f
    object_state_from_robot()  # pull x^o out of the robot state
    realized_consensus()       # A^r, read off the rollout
    robot_running_cost()       # J_r = l_o + l_r + l_c  (no ADMM penalty!)
    robot_terminal_cost()
```

A new **2D scenario** is plain [`oim/objects/`](oim/objects/) primitives —
no MJCF, no MuJoCo:

```python
task = PushT2D(footprint=t_shape_footprint(), goal=[0.5, 0.48, 0.785], obstacles=[Circle(center=[0.08, 0.32], radius=0.04)])
ctrl, params = build_admm_2d(task, n_admm=6)
log = run_2d(task, ctrl, params, robot_pos0=(0.0, -0.13), max_steps=200)
```

## Algorithms

Any of these serves as either ADMM subproblem solver, or runs standalone.

| Algorithm | Description | Import |
| --- | --- | --- |
| **[ADMM object-informed MPPI](#method)** | **Hierarchical object/robot decomposition, coordinated to consensus on the contact wrench.** | [`oim.algs.ADMM`](oim/algs/admm.py) |
| [Predictive sampling](https://arxiv.org/abs/2212.00541) | Take the lowest-cost rollout. | [`oim.algs.PredictiveSampling`](oim/algs/predictive_sampling.py) |
| [MPPI](https://arxiv.org/abs/1707.02342) | Exponentially weighted average of rollouts. | [`oim.algs.MPPI`](oim/algs/mppi.py) |
| [CEM](https://en.wikipedia.org/wiki/Cross-entropy_method) | Fit a Gaussian to the `n` elite rollouts. | [`oim.algs.CEM`](oim/algs/cem.py) |
| [DIAL-MPC](https://arxiv.org/abs/2409.15610) | MPPI with dual-loop annealed covariance. | [`oim.algs.DIAL`](oim/algs/dial.py) |
| [MPPI-CMA](https://arxiv.org/pdf/2506.22087) | MPPI with an adaptive sampling distribution. | [`oim.algs.MppiCma`](oim/algs/mppi_cma.py) |
| [MTP](https://arxiv.org/abs/2505.01059) | Structured tensor sampling + local CEM update. | [`oim.algs.MTP`](oim/algs/mtp.py) |
| [CBO](https://en.wikipedia.org/wiki/Consensus_based_optimization) | SDE pulling samples toward a consensus point. | [`oim.algs.CBO`](oim/algs/cbo.py) |
| [Evosax](https://github.com/RobertTLange/evosax/) | 30+ evolution strategies (CMA-ES, DE, …). | [`oim.algs.Evosax`](oim/algs/evosax.py) |

## Citation

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

A fork of [**Hydrax**](https://github.com/vincekurtz/hydrax) by Vince Kurtz,
which provides the sampling-based MPC framework — controller/task
abstractions, spline parameterization, parallel MJX rollouts, domain
randomization, and every non-ADMM algorithm above. The ADMM object-informed
layer is our addition. Hydrax is itself inspired by
[MJPC](https://github.com/google-deepmind/mujoco_mpc).

```bibtex
@misc{kurtz2024hydrax,
  title  = {Hydrax: Sampling-based model predictive control on GPU
            with JAX and MuJoCo MJX},
  author = {Kurtz, Vince},
  year   = {2024},
  note   = {https://github.com/vincekurtz/hydrax}
}
```

The xArm6 model derives from [UFACTORY](https://www.ufactory.cc/)'s
published URDF; the Unitree G1 model is from
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) (see
[`oim/models/g1/LICENSE`](oim/models/g1/LICENSE)). Motion-capture references
come from [LocoMuJoCo](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets).

## License

MIT — see [LICENSE](LICENSE).
