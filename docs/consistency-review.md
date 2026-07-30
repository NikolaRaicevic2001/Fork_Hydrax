# Consistency review: 2D vs 3D ADMM

Working notes for reconciling the two worlds. Two purposes:

1. **Preserve what was not ported** from the earlier standalone 2D repo
   (`github.com/NikolaRaicevic2001/ADMM`), so that repo is no longer needed
   as a reference. Everything mechanical came across; what remains are
   *formulation* choices, recorded in Part 1 with exact semantics.
2. **Register the divergences** between this repo's own 2D and 3D paths, so
   the reconciliation pass has a single checklist (Parts 2 and 3).

Nothing here is implemented. Part 3 is permanent documentation; Parts 1 and
2 are meant to shrink to nothing as decisions are made.

---

## Part 1 — Mechanisms from the prior 2D repo, deliberately not ported

Each was a deliberate hold, not an oversight: all five change the
formulation rather than the plumbing.

### 1.1 Coupling-cost source — `ell_c_source: implied | live`

**What it was.** The robot block's coupling cost $\ell_c$ compares the
object's trajectory against the object planner's reference $x^{o*}$. The
question is *which* object trajectory:

```python
def _coupling_poses(self, pose0, wrenches, live_poses, dt):
    if self.ell_c_source == "live":
        return live_poses                                  # simulator's own
    return self._implied_poses(pose0, wrenches, dt)        # default

def _implied_poses(self, pose0, wrenches, dt):
    """Integrate the quasi-static limit surface under the realized wrench."""
    pose = tile(pose0)
    for t in range(H):
        pose = object_.propagate(pose, wrenches[:, t], dt)   # x <- x + dt D w
        poses[:, t] = pose
    return poses
```

Default was `implied`.

**What this repo does.** Always `live`. `PushT.robot_running_cost` reads
`self._block_pose(state)` and `PushT2D.robot_running_cost` reads
`state.object_pose` — in both cases the simulator's object pose.

**Why it might matter.** Under `implied`, $\ell_c$ asks *"does the wrench I
am producing move the object along the plan, according to the object
model?"* — evaluated in the **same** dynamics the object block uses. Under
`live` it asks *"does the simulated object follow the plan?"*, which folds
simulator-vs-limit-surface model error into the coupling cost. Since the
whole point of the decomposition is that the two blocks agree on a *wrench*,
`implied` arguably isolates consensus disagreement from model mismatch,
whereas `live` penalizes the robot for the object model being wrong.

This is the item most likely to change behaviour, because in 3D the MJX
object and the analytic limit surface genuinely disagree.

**To add.** `_implied_poses` needs the realized wrench per step, which the
robot rollout already computes as `consensus_val`; the coupling term would
move out of `robot_running_cost` (which sees one step at a time) into the
ADMM layer, or the task would need the implied pose passed in alongside
`obj_ref_t`. Not a local change.

### 1.2 Object motion during robot rollouts — `robot_rollout_mode: frozen | coupled`

**What it was.** During the robot block's MPPI rollouts, either hold the
object on the object planner's reference, or let it move:

```python
if freeze:                      # default
    for t in range(H):
        pose_t = tile(ref_poses[t])          # object pinned to x^{o*}_t
        _, new_robot, wrench = simulate_contact_step(..., freeze_object=True)
else:
    pose = tile(ref_poses[0])
    for t in range(H):
        new_pose, new_robot, wrench = simulate_contact_step(...)
        pose = new_pose                      # object evolves live
```

Default was `frozen`.

**What this repo does.** Always coupled — both `mjx.step` and
`Analytic2DRollout.step` advance the object.

**Why it might matter.** This is arguably the deepest of the five. In the
ADMM decomposition the *object block owns object dynamics*; the robot
block's job is only to realize a wrench. Letting the object move inside the
robot rollout means both blocks simulate the object, so the robot block can
"discover" object motion the object block never planned, and the two blocks
optimize against different object trajectories within a single iteration.
Freezing makes the robot subproblem exactly *"what controls produce the
required wrench, given the object goes where the object planner says?"*

**To add.** Cheap in 2D (skip the object update in `_substep`). Invasive in
3D: freezing inside MJX means overwriting the block's `qpos`/`qvel` every
step of the rollout scan, which is possible but changes the contact solve.

### 1.3 Seek override

**What it was.** Before the ADMM iterations, if the robot is far from the
chosen contact point, the entire nominal control sequence is *overwritten*
with a straight-line dash at it:

```python
p_world = pose0[:2] + rotate(pose0[2], p0)     # contact point, world frame
gap = p_world - robot_pos
seek_gap = contact_step_margin                          # analytical backend
seek_gap = max(contact_step_margin, 2 * robot_radius)   # mjx backend
if norm(gap) > seek_gap:
    speed = clip(norm(gap) / dt, seek_min_speed, seek_max_speed)
    u_nom = tile((gap / norm(gap)) * speed, H)
```

Defaults: `seek_min_speed: 0.4`, `seek_max_speed: 1.0`,
`contact_step_margin: 0.003`.

**What this repo does.** Nothing equivalent. Approach is handled entirely
by the soft term $w_{ee}\max(\|p^{r}-p^{o}\|^2 - r_0^2,\ 0)$ in $\ell_r$.

**Why it might matter.** Sampling-based MPC is weak during a long approach:
out of contact the realized wrench is identically zero, so the consensus
penalty is flat across *every* sample and only the weak approach term
distinguishes them. A deterministic seek reaches contact quickly and lets
the ADMM machinery do the part it is actually good at.

**Caveat.** It bypasses the optimizer for the approach phase, so it sits
outside the formulation — a controller heuristic, not part of the maths.
Worth deciding explicitly rather than by default.

### 1.4 Variance annealing on goal error

**What it was.** Sampling spread scaled by how far the *task* is from done:

```python
def _sigma_scale(self, pose):
    pos_err   = norm(pose[:2] - goal[:2])
    theta_err = abs(wrap_angle(pose[2] - goal[2]))
    normalized = max(pos_err / goal_pos_tol, theta_err / goal_theta_tol)
    return clip(normalized / sigma_anneal_band, min_sigma_scale, 1.0)
```

Defaults: `goal_pos_tol: 0.06`, `goal_theta_tol: 0.10`,
`sigma_anneal_band: 4.5`, `min_sigma_scale: 0.2`. The result *multiplies*
the samplers' own sigmas (`sigma_p`, `sigma_fn`, `sigma_ft`,
`sigma_robot`).

**What this repo does.** Anneals on the ADMM **primal residual** instead:
`noise_scale = clip(noise_kappa * primal_res, noise_min, noise_max)`, and
*adds* that perturbation on top of whatever the injected optimizer proposes
(because a general `SamplingBasedController` does not expose a mutable
covariance). 3D uses $\kappa = 0.1 \in [0.05, 0.5]$; 2D disables it.

**Why theirs may be safer.** Goal error is *exogenous*: it shrinks as the
task completes and cannot be inflated by the sampler. The primal residual
is *endogenous* — more noise produces more disagreement produces a larger
residual produces more noise. This repo already needs `noise_max` purely to
break that loop, which is itself evidence the signal is the wrong one.

Note the two also differ in *mechanism*: scaling the proposal (theirs) vs
adding an independent perturbation (ours). Adopting the goal-error signal
does not require adopting the scaling mechanism.

### 1.5 Clipping the realized wrench $A^r$

**What it was.** The MJX backend bounded the extracted wrench:

```python
self._wrench_clip = self.f_max * (1.0 + self.mu_c)      # 4.0 * 1.5 = 6.0
...
f_obj = -qfrc_constraint[[robot_x_dof, robot_y_dof]]
tau   = r[0] * f_obj[1] - r[1] * f_obj[0]
wrench = jnp.clip(jnp.array([f_obj[0], f_obj[1], tau]), -clip, +clip)
```

**What this repo does.** No clip on $A^r$ anywhere. The default `"twist"`
estimator is bounded incidentally by actuator speed
($0.6\ \mathrm{m/s} \times 7.848 \approx 4.7\ \mathrm{N}$), but
`_consensus_from_contact` reads `qfrc_constraint` directly and its own
docstring records spikes to **~16 N** between exactly-zero readings.

**Why it might matter.** An unbounded $A^r$ spike propagates into several
coupled quantities at once: into $z$ through the average, into the duals
(which then saturate at `max_dual`), and into the primal residual — which
drives both the $\rho$ adaptation *and*, in 3D, the exploration noise. One
contact-solver artefact can therefore disturb the whole iteration.

**Caveat — do not copy verbatim.** Their bound applies the same scalar to
all three components, i.e. newtons and newton-metres alike. With
$r \approx 0.06$ m the torque never approaches 6 N·m, so in practice it
bounded only the forces and the torque clip was inert. If adopted here,
clip **per dimension against `consensus_scale`** (the friction-cone limit
$[\mu m g,\ \mu m g,\ c r \mu m g]$), which this repo already computes and
which is dimensionally correct in every component.

---

## Part 2 — Divergences within this repo, to reconcile

Delete rows as they are decided.

| # | Item | 3D (`PushT`) | 2D (`PushT2D`) | Note |
| --- | --- | --- | --- | --- |
| ~~2.1~~ | ~~Object decision $A^o$~~ | — | — | **Resolved:** both now sample the wrench directly. The contact point is the robot block's concern; the object block only specifies the motion it wants. Contact-action remains opt-in for comparison. See 2.6 for the one property this gave up. |
| ~~2.2~~ | ~~$\epsilon_r,\ \epsilon_s$~~ | 0.5 | 0.5 | **Resolved:** 3D changed from 0.05 (provably unreachable at $H=15$) to 0.5, matching 2D. |
| ~~2.3~~ | ~~Variance annealing~~ | on, residual-driven | on, residual-driven | **Resolved:** 2D now uses the same `noise_min=0.05, noise_kappa=0.1, noise_max=0.5` as 3D via `build_admm_2d`. |
| 2.4 | Contact relocation | none | CEM search over the boundary each step | only meaningful if 2.1 is unified |
| 2.5 | $A^r$ estimator | `"twist"` (default) or `"contact"` | contact solver's own $J_c^\top f$ | mechanisms must differ; the *quantity* must not |
| ~~2.6~~ | ~~Bound on $A^o$~~ | $\pm$`consensus_scale()` | $\pm$`consensus_scale()` | **Resolved:** the default `object_action_bounds()` (`task_base.py`) now bounds direct-wrench mode at the friction-cone limit; see below. |

Item 2.4 is moot while 2.1 stands resolved as direct-wrench: contact
relocation only exists inside the contact-action parameterization.

**On 2.6 (resolved).** Resolving 2.1 in favour of the direct wrench dropped
the friction-cone constraint that the contact-action path enforced
structurally. That constraint had two separable parts, and only one of them
was really about the contact point:

* *where* the force acts — genuinely the robot's business, correctly
  discarded;
* *how large* the wrench may be — a property of the object and its support
  surface, and still true regardless of who applies it.

Before this fix, `object_action_bounds()` returned $\pm\infty$, and
`action_scale = 0.5\,D^{-1}$, so a sampled knot of 10 asked for five times
the largest wrench the table can transmit. That didn't fail silently —
$A^r$ couldn't match it, the residual stayed high and $\rho$ climbed — but
the iterations were spent chasing a target that was unreachable by
construction. `ConsensusTask.object_action_bounds()`'s default now returns
$\pm$`consensus_scale()` (the friction-cone limit), restoring the useful
half of the constraint without reintroducing the contact point. Both
`PushT` and `PushT2D`'s direct-wrench path pick this up automatically
(`PushT` via the base-class default, `PushT2D` by delegating to
`super().object_action_bounds()`); `PushT2D`'s contact-action mode keeps
its own box, since the action there isn't the wrench itself.

---

## Part 3 — Necessarily different; do not reconcile

These follow from the embodiment and should stay divergent. Kept here so
they are not mistaken for drift.

| Item | 3D | 2D | Why permanent |
| --- | --- | --- | --- |
| Rollout backend | `MJXRollout` | `Analytic2DRollout` | this *is* the seam |
| State / model types | `mjx.Data` / `mjx.Model` | `Sim2DState` / `Sim2DModel` | follows the backend |
| $\psi_{\text{tilt}}$, tip height | present | absent | 3D end-effector pose shaping; no 2D analogue |
| Robot–obstacle clearance | emergent from MJX contact | explicit hinge in $\ell_r$ | 2D has no contact solver for the robot |
| Actuator model | velocity servos, `kv`, `forcerange` | direct velocity with magnitude saturation | no articulated arm in 2D |

---

## What was fully ported (for the record)

So it is clear nothing else remains in the old repo: the wrench map
($w = J_c^\top f$), single-point contact resolution with substeps and safe
displacement, obstacle push-out, the analytic engine, the physics-engine
abstraction (as `RobotRollout`), the SDF interface, contact sampling with
normal-alignment rejection, the contact-point search, the generic MPPI
core (as task hooks), the limit-surface object model, the `clutter` /
`corridor` / `gate` scenarios, the capsule SDF, gif animation, and
norm-based speed limiting.

Not taken, and not recommended: the YAML config layer (this repo configures
in Python), their MJX bridge (superseded), and `plot_plan_comparison`
(which only means something if 1.1 is adopted).
