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
- [Code map](#code-map)
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

The object-level cost tracks task-space progress and obstacle clearance:

```math
\ell_o(x^o_t) = w^o_d\, d^2(x^o_t, g) + w^o_f \big(\max(\lambda_t - f_0,\, 0)\big)^2 ,
```

```math
d^2(x^o, g) = \|p^o - p^g\|^2
+ w^o_\theta \left( \cos^{-1}\!\Big( \tfrac{\operatorname{tr}(R^{o\top} R^g) - 1}{2} \Big) \right)^2 .
```

The robot-level cost shapes the end-effector into a good pushing pose:

```math
\ell_r(x^r_t) = w_{ee} \max\big(\|p^{ee}_t - p^o_t\|^2 - r_0^2,\ 0\big)
+ w_{\text{align}}\, \psi_{\text{align}} + w_{\text{tilt}}\, \psi_{\text{tilt}} ,
```

```math
\psi_{\text{align}} = \max\big(\gamma_0 - \cos\angle(p^o_t - p^{ee}_t,\ p^{o*}_t - p^o_t),\ 0\big),
\qquad
\psi_{\text{tilt}}(R) = \sqrt{\varrho^2 + \varphi^2} .
```

The first term pulls the end-effector toward the object but goes slack inside a
radius $r_0$; $\psi_{\text{align}}$ keeps it *behind* the object relative to
the goal; $\psi_{\text{tilt}}$ penalizes roll/pitch away from vertical. The
robot problem additionally carries the coupling cost
$\ell_c(x^o_t, x^{o*}_t) = d^2(x^o_t, x^{o*}_t)$ against the object planner's
own nominal trajectory.

### Implementation notes

Places where the code deliberately departs from the formulation above:

- **Penalty normalization.** The penalty and residuals are divided by the
  friction-cone limit $D^{-1}$ before squaring. Unnormalized, contact forces of
  ~10 N give a penalty of ~10², which swamps the task costs (~1) and drives the
  robot to optimize wrench matching instead of reaching the object. This is a
  diagonal preconditioning of the consensus constraint applied identically to
  both blocks, so the fixed point is unchanged, and it makes $\rho$,
  $\epsilon_r$ and $\epsilon_s$ scale-free.
- **Variance annealing is additive.** Because any sampler can be plugged into
  either block and most do not expose a mutable covariance, the wrappers *add*
  a perturbation of scale $\mathrm{clip}(\kappa\|r\|,\ \sigma_{\min},\ \sigma_{\max})$
  on top of whatever the injected optimizer proposes, rather than replacing
  $\Sigma_u$. The upper clip is required: $\kappa\|r\|$ is otherwise an
  unbounded positive feedback loop.
- **Obstacle clearance is geometric.** The object block has no simulator, so
  $\ell_o$ uses a signed-distance hinge on the object footprint rather than the
  simulator contact force $\lambda_t$.
- **Dual anti-windup.** Duals are clipped to $\pm y_{\max}$.
- **Horizons.** The formulation permits $H^c \le \min(H^o, H^r)$; the
  implementation uses $H^o = H^r = H^c$, enforced in the `ADMM` constructor.

## Code map

| Concept | Code |
| --- | --- |
| ADMM loop, consensus/dual/penalty updates | [`oim/algs/admm.py`](oim/algs/admm.py) — `ADMM`, `ConsensusSpace`, `WrenchConsensus` |
| Object / robot subproblem wrappers | `ObjectSubproblem`, `RobotSubproblem` (same file) |
| Task-side contract for ADMM | [`oim/task_base.py`](oim/task_base.py) — `ConsensusTask` |
| Analytic object models (limit surface, SDF geometry) | [`oim/objects/`](oim/objects/) |
| Push-T task, both embodiments | [`oim/tasks/pusht.py`](oim/tasks/pusht.py) |
| Simulation driver, video recording | [`oim/simulation/deterministic.py`](oim/simulation/deterministic.py) |

Both subproblems only ever call `sample_knots` / `update_params` on their
injected optimizer, so **any** `SamplingBasedController` works in either slot —
that is what makes `--robot-opt` / `--object-opt` interchangeable.

The consensus penalty is owned by `ConsensusSpace` and applied by the ADMM
layer to *both* blocks. Tasks are deliberately prevented from adding their own
(`robot_running_cost` receives no $z$, $y$ or $\rho$), so the two blocks cannot
silently drift into scoring the consensus variable differently.

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
[`oim/simulation/asynchronous.py`](oim/simulation/asynchronous.py) runs
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
