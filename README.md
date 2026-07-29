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
- [2D debugging world](#2d-debugging-world)
- [Designing a task](#designing-a-task)
- [Domain randomization and risk](#domain-randomization-and-risk)
- [Other utilities](#other-utilities)
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

Requires Python ≥ 3.12 and CUDA 13. Using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/NikolaRaicevic2001/Object-Informed-Manipulation-MJX.git
cd Object-Informed-Manipulation-MJX
uv sync
```

Then either prefix commands with `uv run`, or activate the environment with
`source .venv/bin/activate`. A conda environment is also provided
(`conda env create -f environment.yml && pip install -e .`).

Run the tests with `uv run pytest`.

## Quick start

The push-T-through-clutter task is the main demo: the object must reach an
SE(2) goal pose while avoiding three static obstacles, which requires
non-myopic reasoning — a greedy pusher gets stuck behind an obstacle.

It runs in **two worlds that share the same algorithm**. The ADMM loop, the
consensus space, and the object-level subproblem are the same objects in
both; only the robot block's one-step rollout differs. Pick the world by
which entry point you run:

| | 3D (MuJoCo MJX) | 2D (analytic) |
| --- | --- | --- |
| Entry point | [`examples/pusht.py`](examples/pusht.py) | [`examples/pusht2d.py`](examples/pusht2d.py) |
| Robot | 2-DOF point pusher, or 6-DoF xArm6 | disc robot, velocity-controlled |
| Physics | full MJX contact | closed-form single-point contact |
| Viewer | MuJoCo passive viewer | matplotlib plot on exit |
| Use it for | final behaviour, real embodiments | isolating algorithm bugs |

### 3D — MuJoCo MJX

```bash
# ADMM object-informed MPPI, 2-DOF point pusher
uv run python examples/pusht.py admm

# ...on a 6-DoF UFACTORY xArm6 with a rigid pushing stick
uv run python examples/pusht.py --robot xarm6 admm

# Mix and match the inner solvers of the two ADMM subproblems
uv run python examples/pusht.py admm --robot-opt mppi --object-opt cbo

# Record an mp4 into oim/recordings/ (requires ffmpeg)
uv run python examples/pusht.py admm --record
```

Flat (non-hierarchical) baselines on the same task, for comparison:

```bash
uv run python examples/pusht.py mppi
uv run python examples/pusht.py ps
```

ADMM knobs: `--n-admm` (max iterations per control step), `--rho` (initial
penalty $\rho$), `--gamma` (proximal weight $\gamma$), `--seed`.

### 2D — analytic

No MuJoCo, no GPU contact solver, and every intermediate is inspectable.
Writes a trajectory + diagnostics plot to
`oim/recordings/pusht2d_<env>.png` (and a gif with `--animate`).

```bash
# T-block through the same clutter, contact-action object block
uv run python examples/pusht2d.py

# push *through* a tight opening rather than around obstacles
uv run python examples/pusht2d.py --env corridor
uv run python examples/pusht2d.py --env gate

# A/B the object parameterization against the one the 3D tasks use
uv run python examples/pusht2d.py --direct-wrench

# no jit anywhere -- breakpoint inside the physics or the ADMM math
uv run python examples/pusht2d.py --no-jit --steps 3

# empty scene, to separate "can it push at all" from "can it route"
uv run python examples/pusht2d.py --no-obstacles

# write an animated gif alongside the summary plot
uv run python examples/pusht2d.py --animate
```

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

Shared knobs: `--n-admm`, `--rho`, `--gamma`, `--seed`, `--steps`,
`--samples`, `--horizon`, `--no-relocate`.

See [2D debugging world](#2d-debugging-world) for what is and is not shared
between the two, and [3D vs 2D](#3d-vs-2d) for where their formulations
currently differ.

Other demos inherited from the base library live in [`examples/`](examples/)
(pendulum, cart-pole, humanoid standup and mocap tracking, cube rotation,
walker, crane, …).

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
while $A^r$ reads the wrench the robot's motion *actually* imparts, from the
simulator's contact forces. Each ADMM iteration $l$ runs four steps.

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

**4 — Adaptive penalty and variance.** From the primal and dual residuals
$r^{(l+1)}$, $d^{(l+1)}$:

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

Annealing the exploration covariance with the residual makes the planners
explore widely while they disagree and quieten down as they converge,
suppressing the jitter a fixed $\Sigma_u$ injects near agreement.

### Algorithm

> **Given** state $x_0$, previous $(\mathbf{U}^o, \mathbf{U}^r, \mathbf{Z}, \mathbf{Y}^o, \mathbf{Y}^r)$, parameters $\rho, \gamma$
> 1. Warm-start all five by shifting the previous solution one step
> 2. **for** $l = 0, \dots, N_{\mathrm{ADMM}} - 1$:
> 3. &nbsp;&nbsp;&nbsp;&nbsp; Object update — sampling MPC with the proximal + consensus penalty
> 4. &nbsp;&nbsp;&nbsp;&nbsp; Robot update — sampling MPC on the real MJX model, same penalties
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

**Robot block** — the task's own cost, with **no** consensus penalty (the
ADMM layer adds that, via the same function the object block uses):

```math
J_r(x^r_t, u^r_t) = r_r \|u^r_t\|^2
+ \underbrace{d^2_{q}(x^o_t, g)}_{\text{goal}}
+ \underbrace{d^2_{q}(x^o_t, x^{o*}_t)}_{\ell_c,\ \text{coupling}}
+ \ell_r(x^r_t) ,
```

where $x^{o*}_t$ is the object planner's own nominal trajectory from this
ADMM iteration. The embodiment-specific part $\ell_r$ differs between the
two worlds:

```math
\ell^{\text{3D}}_r = w_{ee}\max\big(\|p^{ee}_t - p^o_t\|^2 - r_0^2,\ 0\big)
+ w_{\text{align}}\, \psi_{\text{align}}
+ w_{\text{tilt}}\, \psi_{\text{tilt}}
+ w_{z}\,(z^{ee}_t - z^\ast)^2 ,
```

```math
\ell^{\text{2D}}_r = w_{ee}\max\big(\|p^{r}_t - p^o_t\|^2 - r_0^2,\ 0\big)
+ w_{\text{obs}} \sum \max\big(\delta - \mathrm{sdf}(p^{r}_t),\ 0\big)^2 ,
```

```math
\psi_{\text{align}} = \max\big(\gamma_0 - \cos\angle(p^o_t - p^{ee}_t,\ p^{o*}_t - p^o_t),\ 0\big),
\qquad
\psi_{\text{tilt}}(R) = \sqrt{\varrho^2 + \varphi^2} .
```

The approach term is common: it pulls the pusher toward the object but goes
slack inside a radius $r_0$. Beyond that, $\psi_{\text{align}}$ keeps the
end-effector *behind* the object relative to the goal, $\psi_{\text{tilt}}$
penalizes roll/pitch away from vertical, and $(z^{ee} - z^\ast)^2$ holds the
stick tip at the block's mid-height — all three are 3D end-effector shaping
with no 2D analogue. Conversely the 2D cost needs an explicit robot–obstacle
clearance term, which the 3D task gets from MJX contact for free.

$w_{\text{tilt}}$, $w_z$ and the 2D robot-obstacle weight are not given
numeric values in the paper.

### Object action parameterization

By default the object block's decision variable *is* the consensus variable:
it samples $w^o_t$ directly, and $A^o$ is the identity (up to a fixed
rescaling). Nothing then constrains $w^o_t$ to be a wrench any point contact
could apply — the sampler may propose a pure torque, a pulling force, or a
wrench outside the friction cone, and the consensus can converge onto
something the robot could never realize.

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

Tasks opt in via `object_action_dim`, `object_action_to_consensus`,
`project_object_action`, `sample_object_actions` and
`initial_object_action` on [`ConsensusTask`](oim/task_base.py). Currently
`PushT2D` uses it and the MJX `PushT` does not.

### Implementation notes

Places where the code deliberately departs from the formulation above:

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
  $\sqrt{2H}$. At $H = 15$ the observed residuals are $O(1)$, which makes
  $\epsilon_r = \epsilon_s = 0.05$ unreachable — the early exit never fires and
  every control step runs all $N_{\mathrm{ADMM}}$ iterations. The 2D defaults
  use $0.5$ instead, which does fire.
- **Variance annealing is additive.** Because any sampler can be plugged into
  either block and most do not expose a mutable covariance, the wrappers *add*
  a perturbation of scale $\mathrm{clip}(\kappa\|r\|,\ \sigma_{\min},\ \sigma_{\max})$
  on top of whatever the injected optimizer proposes, rather than replacing
  $\Sigma_u$. The upper clip is required: $\kappa\|r\|$ is otherwise an
  unbounded positive feedback loop.
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

### 3D vs 2D

Both worlds run the same `ADMM`, `ConsensusSpace` and `ObjectSubproblem`.
Where they currently differ — deliberately, and **not yet reconciled**:

| | 3D (`PushT`, MJX) | 2D (`PushT2D`, analytic) |
| --- | --- | --- |
| Robot rollout | `MJXRollout` — `mjx.step` | `Analytic2DRollout` — `resolve_contact` |
| Object decision $A^o$ | wrench $w$ sampled directly, unbounded | contact action $[p, f_n, f_t]$, cone-projected |
| Realized $A^r$ | `"twist"`: $w = D^{-1}\dot{x}^o$ (default), or `"contact"`: $-\,q^{\text{frc}}_{\text{constraint}}$ (point pusher only) | contact solver's own $J_c^\top f$ |
| $\ell_r$ terms | approach + align + tilt + tip-height | approach + robot obstacle clearance |
| Robot obstacle avoidance | emergent, via MJX contact | explicit hinge cost in $\ell_r$ |
| Variance annealing | on, $\kappa = 0.1 \in [0.05, 0.5]$ | off — the task owns the proposal |
| $\epsilon_r,\ \epsilon_s$ | $0.05$ (never fires) | $0.5$ |
| Contact relocation | none | CEM search over the boundary each step |

The $\ell_r$ row is the substantive one: the 3D robot cost shapes a
*3D end-effector pose* (keep the stick vertical, at block mid-height, behind
the object relative to the goal), none of which has a 2D analogue, while the
2D cost carries an explicit robot-obstacle term that the 3D task gets from
the simulator for free.

Which of these are permanent consequences of the embodiment and which are
drift to be reconciled — plus five further formulation options not currently
implemented on either side — are tracked in
[`docs/consistency-review.md`](docs/consistency-review.md).

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
├── sim2d/                analytic 2D world  (see below)
│   ├── engine.py         Sim2DState/Sim2DModel, resolve_contact,
│   │                       Analytic2DRollout
│   ├── task.py           PushT2D (a ConsensusTask, no MuJoCo)
│   ├── scenarios.py      clutter / corridor / gate
│   └── run.py            build_admm_2d, run_2d
│
├── sim3d/                MuJoCo drivers
│   ├── deterministic.py  run_interactive: viewer, replanning, mp4 recording
│   └── asynchronous.py   controller and simulator in separate processes
│
├── tasks/                MuJoCo tasks — pusht.py is the ADMM one
├── models/               MJCF scenes and meshes (xarm6, g1, pusht_clutter, …)
└── utils/                spline interpolation, video recording
```

The two structural invariants worth knowing:

**Any sampler fits either ADMM block.** Both subproblems only ever call
`sample_knots` / `update_params` on their injected optimizer, which is what
makes `--robot-opt` / `--object-opt` interchangeable.

**Only one thing is simulator-specific.** `RobotRollout` is the entire 2D/3D
seam — `ADMM`, `ConsensusSpace`, `ObjectSubproblem` and everything under
`objects/` are shared verbatim. The consensus penalty is owned by
`ConsensusSpace` and applied by the ADMM layer to *both* blocks; tasks are
deliberately prevented from adding their own (`robot_running_cost` receives no
$z$, $y$ or $\rho$), so the two blocks cannot silently drift into scoring the
consensus variable differently.

## 2D debugging world

`oim/sim2d/` is an analytic planar world — a disc robot pushing a rigid
footprint, with single-point contact resolved in closed form against the
object's SDF. It is **not** a second implementation of the algorithm: the
`ADMM` loop, `ConsensusSpace`, `ObjectSubproblem` and
`PlanarPushingObject` are the same objects the MuJoCo tasks use. Only the
robot block's one-step rollout is swapped, through
[`RobotRollout`](oim/algs/admm.py):

| | 3D | 2D |
| --- | --- | --- |
| Rollout backend | `MJXRollout` (`mjx.step`) | `Analytic2DRollout` |
| State | `mjx.Data` | `Sim2DState` |
| Model | `mjx.Model` | `Sim2DModel` |
| Task | `PushT(Task, ConsensusTask)` | `PushT2D(ConsensusTask)` |
| ADMM loop, consensus, object block | **shared** | **shared** |

The point is attribution: an MJX rollout that misbehaves could be failing
in the contact solver, the constraint allocation, the integrator, the
actuator model, or the ADMM math. If a behaviour reproduces in 2D, it is
the formulation. Everything is `jax.numpy`, so it jits and vmaps like the
MJX path, but also runs eagerly:

```bash
uv run python examples/pusht2d.py --no-jit --steps 3   # steppable in a debugger
```

`Sim2DState` carries the realized wrench directly, so $A^r$ needs no
twist-inversion or contact-force extraction — `resolve_contact` computes it
explicitly. The 2D task also switches the object block onto the
[contact-action parameterization](#object-action-parameterization); see
[3D vs 2D](#3d-vs-2d) for the full list of formulation differences, which
are **not currently reconciled** between the two worlds.

Writing a new 2D scenario means constructing a `PushT2D` with a different
footprint, goal and obstacle list — all three are plain
[`oim/objects/`](oim/objects/) primitives, no MJCF involved:

```python
from oim.objects import Circle, Polygon, t_shape_footprint
from oim.sim2d import PushT2D, build_admm_2d, run_2d

task = PushT2D(
    footprint=t_shape_footprint(),
    goal=[0.5, 0.48, 0.785],
    obstacles=[Circle(center=[0.08, 0.32], radius=0.04)],
)
ctrl, params = build_admm_2d(task, n_admm=6)
log = run_2d(task, ctrl, params, robot_pos0=(0.0, -0.13), max_steps=200)
```

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
    realized_consensus()       # A^r: extraction map, read from the simulator
    robot_running_cost()       # J_r = l_o + l_r + l_c  (no ADMM penalty!)
    robot_terminal_cost()
```

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

## Other utilities

**Open-loop trajectory optimization.** The same controllers can optimize a
trajectory offline via [`oim/open_loop.py`](oim/open_loop.py) — see
[`examples/cart_pole_trajectory_optimization.py`](examples/cart_pole_trajectory_optimization.py).

**MuJoCo Warp (experimental).** Pass `impl="warp"` to a task constructor, or
`--warp` to the examples, to use
[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) instead of JAX
for rollouts.

**Asynchronous simulation.**
[`oim/sim3d/asynchronous.py`](oim/sim3d/asynchronous.py) runs
the controller and simulator in separate processes, for a more realistic
picture of closed-loop latency.

**Recording.** Pass `record_video=True` to `run_interactive` (or `--record` to
the examples) to write an mp4 into [`oim/recordings/`](oim/recordings/).
Requires `ffmpeg` on the `PATH`.

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
