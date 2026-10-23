# Reproducibility and experiment checklist

## What can be reproduced now?

- Deterministic synthetic sensor/reward checks from configs/demo.json.
- Unit-test results on a recorded Python/NumPy environment.
- The declared method implementation, with explicitly supplied new settings.

Exact reproduction of the published success ratios and travel times is **not
currently established**. Missing original reward constants, runtime settings,
scene details and run/checkpoint provenance cannot be reconstructed by renaming
files or matching an algorithm label.

No trained policy is supplied for the new package. Historic Result/ assets and
legacy checkpoints must not be represented as outputs of the new implementation.

## Before a new experiment

1. Save the complete config, including all reward and adapter fields.
2. Record repository commit and local changes:

   ~~~bash
   git rev-parse HEAD
   git diff
   ~~~

3. Record the Python version, hardware, OS, dependency versions, Unity build and
   checkpoint identity. A version list is not a portable environment guarantee:

   ~~~bash
   python --version
   python -m pip freeze
   ~~~

4. Archive the exact scene, waypoint/start/goal definitions, randomization
   settings, seeds and depth-model revision or file hashes.
5. Verify every sensor and action channel with a short non-learning simulator
   check before interpreting any training result.
6. Use distinct output directories for algorithms, scene/waypoint, seed,
   depth model and SBES condition.

## During and after the run

The runner saves the supplied config, basic runtime version information, model
weights and JSONL logs. It does not automatically archive simulator assets,
dependency lockfiles or uncommitted code; record those separately.

Keep episode-level outcomes and failure reasons. When aggregating:

- Report success numerator and denominator, not only percentages.
- State whether travel-time summaries include or exclude failed episodes.
- Separate training curves from held-out evaluation.
- Report uncertainty across independently seeded runs.
- Do not infer a reward-component effect without a matched ablation.
- Do not infer real-time onboard execution from simulation timing.

Cumulative episode counts apply within one run; an iteration is a collection/
update cycle. Configured seeds improve traceability but do not guarantee
bitwise-identical results across devices, package versions or Unity builds.

## Validation status

The NumPy-based reward, observation, replay, config and demo tests have been
run locally. PyTorch update tests are supplied but were skipped in the editing
environment because PyTorch was absent. No Unity integration, depth loading,
training campaign, GPU performance or physical-robot experiment was run.

Run the full suite in the intended environment before relying on the numerical
optimizer paths:

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
~~~

Policy checkpoints support evaluation only, not exact optimizer/replay/RNG
resume. Learning curves produced after restarting are a new run.
