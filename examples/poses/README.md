# Start and goal poses

One file per `examples/` script, named after it. Each holds five start and
five goal poses for that scene's object, as world-frame SE(2)
`[x, y, theta]` — the same convention `oim/utils/scenes.py` uses for
`SceneSpec.goal`, and the same one the block's `T_x`/`T_y`/`T_z` joints
read, so a start pose is written straight into `qpos`.

```bash
uv run python examples/shelf_gap.py --start 3 --goal 2 admm --headless
uv run python examples/shelf_gap.py --goal 2 admm --headless   # start random
uv run python examples/shelf_gap.py admm --headless            # both random
```

| | |
| --- | --- |
| Pose `"1"` | The scene's original start/goal — the default before these files existed, so old runs stay comparable |
| Poses `"2"`–`"5"` | Perturbations of it, within 11 cm and ±0.45 rad, at least 4.5 cm apart |
| Clearance | Every pose is ≥ 4.5 cm clear of every obstacle, both ways: object boundary against each obstacle, and each obstacle's boundary against the object polygon |
| Reach | Every pose is inside the arm's usable annulus about its own base |
| Which was used | Recorded in the run file as `start_index` / `goal_index`, so a random draw is reproducible |

Poses are deliberately a *neighbourhood* of the nominal rather than spread
across the table: all five variants are then the same task under jitter,
which is what makes them a fair test of robustness. Varying the seed only
redraws the sampler's noise; varying the pose changes the problem.

`tests/test_poses.py` re-derives the clearance and reach of every pose in
every file, so a hand-edited pose that grazes an obstacle fails there
rather than in a run. Add a pose by editing the YAML and running that test.
