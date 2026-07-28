# ADMM implementation vs. the paper

Maps *Object-Informed Model Predictive Path Integral Control for
Non-Prehensile Robot Manipulation* onto the code, so the two can be kept in
sync. Equation and algorithm numbers refer to the paper.

Code: [`hydrax/algs/admm.py`](../hydrax/algs/admm.py),
[`hydrax/task_base.py`](../hydrax/task_base.py) (`ConsensusTask`),
[`hydrax/objects/`](../hydrax/objects/),
[`hydrax/tasks/pusht.py`](../hydrax/tasks/pusht.py).

## Algorithm 4, line by line

| Paper | Code | Status |
| --- | --- | --- |
| 1. Warm-start by shifting the previous solution | `ADMM.optimize` shifts `z`, `gamma_o`, `gamma_r` and the object mean by one step; the robot mean is re-interpolated onto shifted knot times | ✅ |
| 2. `for l = 0 … N_ADMM-1` | `jax.lax.while_loop` bounded by `n_admm` | ✅ |
| 3. Object update, eq. 24 | `ObjectSubproblem.optimize` | ✅ |
| 4. Robot update, eq. 25 | `RobotSubproblem.optimize` | ✅ |
| 5. Consensus update, eq. 26 | `ConsensusSpace.z_update` | ✅ |
| 6. Dual update, eq. 27 | `ConsensusSpace.dual_update` | ✅ + anti-windup clip (extension) |
| 7. Adaptive penalty (`ρ←2ρ` / `ρ←ρ/2` on the 10× residual ratio) | `jnp.where` ladder in `ADMM._admm_iteration` | ✅ |
| 8. Variance update `Σ_u ← max(Σ_min, κ‖r‖)` | additive residual-scaled perturbation | ⚠️ approximated, see below |
| 9. Break when residuals are below tolerance | `while_loop` condition on `eps_r` / `eps_s` | ✅ |
| 10. Apply `u^r_0`, shift, observe | `run_interactive` queries the control spline | ✅ |

## Decision variables and extraction maps (eq. 23)

* `A^o(U^o)_t = w^o_t` — read straight off the object planner's decision
  variable: `ws = knots * task.object_action_scale()`.
* `A^r(U^r)_t = ŵ^o_t` — read from the simulator:
  `PushT.realized_consensus` returns `-qfrc_constraint[pusher_dofs]` plus the
  induced torque. Verified numerically against `mj_contactForce` (they agree
  to floating point), and MJX agrees with MuJoCo-C.

Both blocks report the wrench **in the world frame, about the block's pose
origin, in N and N·m**, and both score it through the *same*
`ConsensusSpace.penalty_cost`. The task is deliberately forbidden from adding
its own penalty (`robot_running_cost` takes no `z`/`dual`/`rho`), which is
what stops the two blocks from silently drifting apart.

## Primal/robot/object subproblems

Eq. 24/25 are implemented as, per block,

```
J  +  (γ/2)·‖U − U^(l)‖²  +  (ρ/2)·Σ_t ‖A(U)_t − z_t + y_t‖²
```

`J_o` is eq. 16 and `J_r` is eq. 17 (`ℓ_o + ℓ_r + ℓ_c`, plus effort). Both
blocks dt-weight their running cost and leave the proximal/penalty terms
un-weighted, matching each other.

Horizons: the paper allows `H^c ≤ min(H^o, H^r)`. The implementation uses the
special case `H^o = H^r = H^c`, enforced in `ADMM.__init__`
(`robot_optimizer.ctrl_steps == object_optimizer.num_knots`).

## Object dynamics (eq. 4–5)

`x^o_{t+1} = x^o_t + Δt·D·w^o_t`, `D = diag(μmg, μmg, c·r·μmg)^{-1}`, in
`PlanarPushingObject.step`.

The paper writes the limit-surface relation for the **body** twist; the code
uses the **world** frame. In 2D these coincide: `D` is isotropic in
translation (`R D R^T = D`) and the torque/angular-velocity pair is a
rotation invariant. Using the world frame lets `A^o` and `A^r` be compared
directly without a frame conversion.

## Deviations, and why

1. **Penalty/residual normalization (not in the paper).** `WrenchConsensus`
   divides by a characteristic scale — the friction-cone limit `μmg`, i.e.
   exactly `1/D` — before squaring. Unnormalized, contact forces of ~10 N
   give a penalty of ~10², which dwarfs the task costs (~1); the robot then
   optimizes wrench matching to the exclusion of reaching the object. This is
   a diagonal preconditioning of the consensus constraint, applied
   identically to both blocks, so the ADMM fixed point is unchanged. It also
   makes `rho`, `eps_r` and `eps_s` scale-free.

2. **Exploration-noise annealing (Alg. 4, line 8) is approximated.** The
   paper *replaces* MPPI's `Σ_u`. Because any `SamplingBasedController` can
   be plugged in and most do not expose a mutable covariance (MPPI's
   `noise_level` is baked into its jit trace), the subproblem wrappers
   instead **add** an independent perturbation of scale
   `clip(κ‖r‖, noise_min, noise_max)` on top of whatever the injected
   optimizer proposes. Near consensus the total noise floor is the injected
   optimizer's own, not `Σ_min`. The upper clip is required: `κ‖r‖` is
   otherwise an unbounded positive feedback loop (more noise → more
   disagreement → larger residual → more noise), which was observed to
   diverge.

3. **Object obstacle cost (eq. 18) uses geometry, not contact force.** The
   paper penalizes the simulator's object–environment contact force
   `w_f(max(λ_t − f_0, 0))²`. The object block here has no simulator, so
   `PlanarPushingObject` uses a signed-distance hinge on the object's
   footprint instead. Same intent, computable in the analytic model.

4. **`ψ_tilt` (eq. 22) is omitted** from `ℓ_r`. The Push-T pusher is a planar
   point with no end-effector orientation DOF, so the term is identically
   zero. It should be restored for the xArm6.

5. **Dual anti-windup.** `dual_update` clips to `±max_dual`; the paper leaves
   the duals unbounded.

## Non-mathematical bug worth remembering

The pusher's velocity actuator (`kv = 100`, `m = 1`) is only stable under
explicit Euler for `dt < 2m/kv = 0.02 s`. The planner runs at `dt = 0.05 s`,
so with `integrator="Euler"` **every robot rollout diverged** — the planner
believed any control sent the pusher to infinity, and the resulting behaviour
looked like the robot "drifting away from the object". The clutter MJCF now
uses `integrator="implicitfast"`, which is unconditionally stable for
actuator velocity feedback. `tests/test_pusht.py::test_clutter_planning_model_is_stable`
guards against a regression.
