# xArm6 + pushing stick

A clean, self-contained MJCF for the UFACTORY xArm6 manipulator with a rigid
pushing-stick end-effector, used as the robot side of the ADMM object-informed
MPPI push-T task (see `oim/tasks/pusht.py`'s `robot="xarm6"` option and
`oim/models/xarm6_pusht_clutter/`).

**Status: wired into `oim`.** This directory used to be a staging area
(`robot_assets/xarm6/` at this repository's root, back when the repository
was still named `Forked_Hydrax`; kept outside `oim/` to avoid colliding
with concurrent ADMM restructuring work) and has since been moved here
wholesale, unedited except for the actuator-type changes documented below --
its internal layout (`xarm6.xml` + `scene.xml` + `assets/` + `README.md` +
`LICENSE`) already matched the convention used by `oim/models/pusht_clutter/`
and `oim/models/g1/`, so the move needed no further restructuring.

## Contents

- `xarm6.xml` -- the robot model: 5 actuated joints, meshes, materials,
  velocity actuators (see point 9 below). No floor, no lights, no
  `<option>` (timestep/integrator are set by whatever scene composes this
  in, e.g. `oim/models/xarm6_pusht_clutter/`).
- `scene.xml` -- `xarm6.xml` plus a floor/skybox/lights, for standalone
  viewing (`mujoco.viewer` or the `verify_model.py` renderer below).
- `assets/` -- the 7 STL meshes actually used (`link1.stl`...`link6.stl`,
  `tool_flange.stl`), copied and renamed from the source repo below.
- `verify_model.py` -- standalone load/sanity-check/render script, no
  oim imports. Re-run after any edit to `xarm6.xml`.
- `verify_renders/` -- output of the above; gitignore-able, kept for now
  for visual review.
- `source/` -- verbatim copy of the upstream files this was built from
  (provenance/diffing only; nothing in `xarm6.xml`/`scene.xml` loads from
  here).
- `LICENSE` -- verbatim copy of the source repo's MIT license.

## Source and why

Mesh geometry and the kinematic tree (body offsets/`euler`s) come from
[julio-innx/xArm6-Gym-Env](https://github.com/julio-innx/xArm6-Gym-Env)
(MIT license), commit `99efe022f91788cb1ad3887858fa00dc0890ab85`,
`gym_xarm6/envs/assets/{xarm6.xml,shared.xml,reach.xml,stls/*}`.

`mujoco_menagerie` (the standard curated MuJoCo model source) does not have
an xArm6 model (confirmed via
[google-deepmind/mujoco_menagerie#206](https://github.com/google-deepmind/mujoco_menagerie/issues/206),
an open, unfulfilled request), and no other reliable ready-made xArm6 MJCF
turned up. The vendor's own distribution is URDF-only (see
`Object-Informed-Manipulation/src/mppi_isaac/assets/urdf/xarm6/` -- a
*separate* sibling project in this workspace, not this repository, despite
the now near-identical names) and doesn't
load into MuJoCo directly -- its mesh paths assume a ROS-package resolver
mujoco doesn't use (see the `project_xarm6_urdf_to_mjcf_conversion` memory
for that dead end, kept for reference in case a from-URDF path is revisited).

`xArm6-Gym-Env` is a real, working conversion, not a placeholder: it's built
on a reskinned OpenAI-Gym Fetch-env scaffold (hence the `robot0:...`
naming, a 3-DOF free "gantry" base, and a mocap+weld IK rig in the
original), but its actual arm geometry checks out -- e.g. its
elbow-to-forearm link length (0.2895 m) matches the real vendor URDF's
joint2->joint3 offset almost exactly, and the assembled, articulated
renders below look correct. What did **not** check out was its per-link
mass (Fetch-scale, ~2.3x too heavy overall) -- see below.

## What was changed from the source, and why

1. **Removed**: the mocap body, the `weld` equality, and the 3 unlimited
   base slide joints (`robot0:slide0/1/2`, Fetch's virtual gantry for
   IK-driven base positioning). `xarm6_link_base` here is a normal
   fixed-base body with no joint at all.
2. **Removed**: the gripper (`gripper.xml`, `stls/gripper/`) -- not needed
   for a pushing task, not copied into `assets/`.
3. **Renamed**: `robot0:shoulder_pan_joint` etc. -> `xarm6_joint1`...`5`;
   mesh files `J1.stl`...`J6.stl`, `Tool.stl` -> `link1.stl`...`link6.stl`,
   `tool_flange.stl`. The source's mesh numbering is off-by-one from the
   real vendor's: `J1.stl` is the *stationary* base-housing shell (no joint
   on its own body in the source), matching the real vendor URDF's
   `link_base`; `Tool.stl` -- which the source treats as a separate
   "tool_link" beyond a `wrist_roll_link` mesh -- is, geometrically and by
   mass (its bounding box is a compact ~7.5x7.5x2.8 cm disc), the real
   vendor's **link6** (the wrist mounting flange), not an extra part beyond
   it. That mapping is what let the real vendor's stick-mounting numbers
   below be reused directly, with no cross-frame transform.
4. **Fixed, not actuated: joint6 (wrist roll).** The real vendor's own
   `xarm6_stick.urdf` (in the sibling `Object-Informed-Manipulation` repo)
   locks this exact joint (`type="fixed"`) for the stick end-effector
   variant specifically, since rolling a symmetric stick about its own axis
   does nothing for pushing. Matched here rather than re-adding a 6th,
   functionally-meaningless actuator: `model.nu == 5`.
5. **Replaced per-link mass** with the real vendor values from
   `xarm_description`'s `xarm6_robot.urdf` (`link_base=2.7`, `link1=2.16`,
   `link2=1.71`, `link3=1.384`, `link4=1.115`, `link5=1.275`,
   `link6=0.1096`, all kg). The source repo's masses are Fetch-inherited
   and implausible for this robot (e.g. its "elbow_flex" link alone is
   4.221 kg; summed arm-only mass is ~24 kg vs. the real ~10.5 kg this
   file now totals -- `verify_model.py` prints this every run).
6. **Inertia intentionally left unset** (no `<inertial>` element; each
   geom instead carries an explicit `mass`, and MuJoCo's compiler infers
   the inertia tensor from that geom's actual mesh shape at compile time).
   This was a deliberate choice over hand-copying the vendor URDF's
   inertia tensors: those are expressed in the *real* URDF's per-link
   frames, which are rotated differently from this source's per-link
   frames (different `euler` reassignments at each body), and transforming
   a full 6-value inertia tensor across that mismatch by hand is exactly
   the kind of silent, hard-to-verify error this conversion was trying to
   avoid. Letting MuJoCo derive shape-consistent inertia from the mesh
   that's actually being simulated sidesteps the frame problem entirely,
   at the cost of not being a byte-exact match to the vendor's measured
   inertia tensors. Reasonable for a first pass; worth revisiting with a
   proper frame-transform (or a CAD cross-check) before trusting fine
   dynamics.
7. **Added: the pushing stick.** Not present in the source at all (its
   `Tool.stl`/`tool_link` is bare, meant for a gripper mount). Geometry
   (radius 0.005 m, length 0.173 m, mounted at `pos="0 0 -0.015"` relative
   to link6) is copied directly from the real vendor's `xarm6_stick.urdf`
   (`xarm6_ee_stick`/`xarm6_ee_tip` links) -- this transform is
   parent-child directly off link6 in both the vendor source and here, so
   no cross-frame conversion was needed. A `site` named `xarm6_tip` marks
   the stick's far end, mirroring `xarm6_ee_tip`, for future cost
   functions to reference (matching the `"pusher"` site convention already
   used by `oim/models/pusht/pusht.xml`).
8. **Stick mass (0.05 kg) is an estimate, not sourced** -- the vendor URDF
   gives the stick no `<inertial>` at all (it's welded, so URDF tooling
   doesn't require one). 0.05 kg is a middle-of-the-road guess for a thin
   5mm-radius, 173mm rod (aluminum ~0.037 kg, steel ~0.107 kg at that
   size). Flagged here so it isn't mistaken for a sourced value.
9. **Actuators**: `<velocity>` (servo) per joint, not `<motor>` (torque) or
   `<position>` -- `ctrl` is a target joint **velocity**, in radians/sec
   (unaffected by `<compiler angle="degree">`, which only converts
   angle-valued XML attributes like `<joint range>`, not actuator
   `ctrlrange`). Chosen over `<position>` (used briefly, then superseded)
   to match the ADMM pusht-clutter task's existing velocity-actuated
   point-mass pusher and the paper's own velocity-based action-space
   formulation, once this model became the robot side of that
   integration. `forcerange` caps output torque at the real vendor
   per-joint effort limits in N*m (`joint1/2=50, joint3/4/5=32`, from
   `xarm6_robot.urdf`'s `<limit effort="...">`). `ctrlrange`/`kv` are
   *not* the real vendor max-speed spec (unsourced, like the stick mass)
   -- confirmed empirically (not just by formula) that a velocity
   actuator's steady-state tracking is `qvel_ss = ctrl * kv/(kv +
   joint_damping)`, so with this file's existing `damping="50"` (point 10
   below) fixed, `ctrlrange=+-1.0 rad/s` with `kv=300` (50 N*m joints) /
   `kv=200` (32 N*m joints) gives ~80-86% steady-state tracking without
   exceeding `forcerange` except at the extreme edge of the range (where
   clipping just slows the approach, not an error) -- untuned beyond
   that, revisit once this drives an actual task/controller.
10. **Joint damping/armature** (50 / 1, uniform across joints) is carried
    over from the source's `<default>` block, unchanged. Untuned like
    everything above -- flagged as a good target for follow-up tuning
    once this is driven by an actual controller, not before.

## Bugs found via interactive testing, fixed (2026-07-27)

Manual testing with `python3 -m mujoco.viewer --mjcf=scene.xml` surfaced
two real bugs the automated checks in `verify_model.py` didn't catch
(that script never drives individual actuators or checks `ncon`):

- **Adjacent-link self-collision was locking every joint.** Each link's
  full-detail visual mesh is also used directly as its collision geom
  (see point 6/the "Explicitly not done yet" list below), and adjacent
  links' meshes are designed to touch flush at their joint's mating
  surface -- so MuJoCo registered a spurious contact at zero distance
  between every parent/child link pair, which does *not* get
  auto-excluded just because the bodies are joint-connected. Confirmed
  directly: commanding `xarm6_joint1`'s motor at its max torque (50 N*m)
  produced a `qfrc_constraint` of -41.6 N*m from the base/link1 contact
  alone, almost fully canceling the applied torque (`motor1` looked
  completely dead in the viewer). Fixed with a `<contact><exclude .../>`
  entry per adjacent body pair. Re-verified: `ncon == 0` at zero config,
  and the same max-torque test on `xarm6_joint1` now produces 45.3 rad/s^2
  of initial acceleration and rotates the joint 56 degrees in one second,
  instead of essentially nothing.
- **The stick clipped through `scene.xml`'s floor at rest.** Zero-config
  tip height is z=-0.0185 (visible in `verify_model.py`'s own printed
  output, just not connected to the floor's z=0 plane before this).
  Fixed by dropping the floor to `pos="0 0 -0.15"` in `scene.xml` --
  enough to clear the resting pose, but *not* a real table-mount/workspace
  decision (the arm can reach as low as z=-0.49 relative to its base at
  other joint configurations well within its declared limits); that needs
  a real joint-limit-aware height choice once this is composed into an
  actual task scene with a table, not eyeballed here.

**Note, since superseded twice**: with the original `<motor>` (torque)
actuators, no gravity-compensating controller, and no joint `stiffness`,
the arm sagged under gravity at `ctrl=0` (joints 2/3 drifted ~0.05-0.11
rad over 1s at zero control) -- physically correct for an uncontrolled
torque joint (a real xArm6 with power cut would do the same), not a bug.
Actuators were then switched to `<position>`, which *did* hold `ctrl=0`
against gravity by design (drift dropped to ~0.006 rad over 1s) -- but
that was itself superseded by the `<velocity>` actuators now in this file
(point 9 above), which are back to *not* actively holding a pose at
`ctrl=0` (only damping velocity, same as the original `<motor>` behavior)
since that's what the ADMM integration's action-space needs. Kept here as
the record of why each actuator type's `ctrl=0` behavior looks the way it
does, not as a live caveat about the current model.

## Verification done so far

`verify_model.py` (run standalone, no oim): loads `scene.xml` with no
path tricks, confirms `nu == 5`, prints per-body mass (totals ~10.5 kg,
matching the real vendor's summed arm-only mass), prints joint ranges,
and renders the zero pose plus two random-in-range poses to
`verify_renders/*.png`. Visual review of those renders confirms the
kinematic chain is coherent (links connect end-to-end with no gaps or
mismatched offsets) and the stick is correctly attached and protruding
from link6 at the tip.

Manually driven via `python3 -m mujoco.viewer --mjcf=scene.xml` (per-joint
motor sliders), which is what surfaced the two bugs fixed above -- worth
repeating after any future edit to `xarm6.xml`, since neither bug showed
up in `verify_model.py`'s own checks (it never drives an actuator or reads
`ncon`).

## Explicitly not done yet (next steps, not started)

- No integration into a `oim.Task`/`ConsensusTask` -- this is pure
  MJCF, no Python task wrapper.
- No scene composition with a pushable object, table, or obstacles (unlike
  `pusht_clutter/scene.xml`, this scene is just the arm on a bare floor).
  In particular, `scene.xml`'s current floor offset (`-0.15`) is only
  chosen to clear the resting pose, not a real reachable-workspace/table
  decision -- see the self-collision/floor bugfix section above.
- No actuator gain (`kp`/`kv`) tuning against an actual controller --
  current values (point 9 above) are a first-pass guess, not validated
  against real dynamics or a task's actual control frequency.
- Inertia is compiler-inferred (see point 6 above), not a verified match
  to the vendor's measured tensors.
- Collision is single-convex-hull-per-mesh (MuJoCo's default for a `mesh`
  geom) -- standard practice for arm links (matches Menagerie's own
  Panda/UR5e/xArm7 models). Adjacent-link self-collision is now excluded
  (see bugfix section above); **non-adjacent** self-collision (e.g. the
  wrist folding back far enough to hit the base or link1) is untested and
  not excluded -- worth a targeted check if large joint-range motions are
  ever driven through this model (an MPPI controller sampling widely
  could hit this).
