# Smoke runs

Twenty short runs — the five tabletop scenes x {`admm`, flat `mppi`} x two
seeds — kept so `oim/run_eval.py` has a full task x method grid to render
against without a GPU:

```bash
uv run python -m oim.run_eval --runs-dir oim/results/runs_smoke
uv run python -m oim.run_eval --runs-dir oim/results/runs_smoke --format latex
```

**These are not results.** They ran 25 control steps, which is roughly the
time the arm needs just to reach the block from its rest pose, so every one
of them scores SR 0 with the object still at its start pose. They exist to
exercise the table, not to compare methods.

They live here rather than in `results/runs/` for that reason: `run_eval`
averages every trial of a (task, method) cell together, so mixing 25-step
runs into real 300-step ones would silently drag every number down. The
`averaged over: steps (25, 300)` line above the table is the only warning
you would get.

Regenerate with, for each scene and seed:

```bash
uv run python examples/pusht.py --robot xarm6 --scene SCENE --no-plot \
    --warp ALGORITHM --steps 25 --seed SEED --headless
```
