# Implementation and numerical conventions

This is a **new implementation for future experiments**, based on the current
book/chapter.tex. It is not the original experimental implementation and does
not reproduce, validate, or replace the chapter's published results. No training
or Unity execution was performed when adding it.

The legacy entry points and checkpoint files remain unchanged. Use
python -m bluerov_nav.train, not ppo_gym.py or sac_env.py, for this path.
Legacy full-object pickle checkpoints are not compatible with the new models.

## Implemented components

| Chapter component | New implementation |
|---|---|
| Five reward components | reward.py: original movement, repetition, obstacle, horizontal-goal and vertical-goal branches |
| Observation history | environment.py: newest-first depth, goal, SBES and previous-action histories; default length 5 |
| Feature fusion | networks.py: 512 image + 96 goal + 32 range + 64 action features = 704 |
| A2C | Advantage-weighted log likelihood, fitted value targets and configurable L2 penalty |
| PPO | Clipped likelihood ratio with configurable minibatches/epochs |
| TRPO | KL Hessian-vector products, conjugate gradient, and backtracking checking improvement and KL |
| SAC | Reparameterized entropy-regularized actor, twin Q critics, replay and target critics |
| TD3 | Twin critics, target-policy smoothing, delayed actor updates, replay and target networks |
| Plot/log convention | Explicit iteration, environment_transitions and cumulative_episodes fields |

## Reward fidelity and missing values

- The positive movement bonus is paid below the displacement threshold. This
  reproduces the written condition; it does not fix or validate the reviewer's
  stagnation concern.
- The repetition penalty is independent of low motion.
- Vertical shaping uses the change of **signed vertical goal offset divided by
  simulation decision period**, not the commanded vertical action.
- A collision replaces the near-obstacle penalty; they do not accumulate.
- Horizontal goal acceptance uses distance <= goal_radius, preserving the
  original branch at equality.
- Vertical and surface collision tests share delta_v.
- Reward weights and thresholds are REQUIRED. There are no guessed chapter
  defaults. The ambiguous coefficient row is not silently assigned to GAE or
  target-network updates.

configs/navigation.template.json is intentionally **not runnable unchanged**.
Replace relevant null entries with explicitly chosen values and save a separate
experiment config. Choices must be reported as new-experiment choices. Numbers
in tests are synthetic fixtures, not recommended training parameters.

## Explicit engineering choices, not historical claims

The chapter fixes feature-group sizes but does not fully prescribe all layers,
normalization, action-density treatment, termination or logging conventions.

- Three convolutional layers and a 512-unit output hidden layer; unbounded value
  and Q outputs. These are not claimed to recover historical layer settings.
- A tanh-transformed Gaussian with the correct log-Jacobian for stochastic
  policies, not a clipped Gaussian with an uncorrected density. TRPO uses the KL
  of the pre-transform Gaussians, equivalent under the shared bijection.
- Depth maps are normalized to [0,1] by the backend and mapped to [-1,1].
  Goal values remain metric/radian values. No evolving running normalizer is
  applied to replay data.
- A configured position window measures endpoint displacement. Movement bonus
  is disabled until the window is full. Reset clears this history.
- Repetition means maximum componentwise difference from the previous executed
  normalized action is <= configured tolerance. The first action is not repeated.
- Time limits bootstrap value estimates, but GAE traces do not cross episode
  boundaries. True terminal transitions do not bootstrap.
- Goal termination is an explicit configuration choice. Collision always
  terminates. The configured step limit is applied exactly at that count.
- New evaluation success means horizontal goal acceptance without simultaneous
  clearance violation on the final frame, not positive reward. This is an explicit
  NEW metric, not a reconstruction of the published counts.
- Simulation time uses transition counts and the simulation decision period,
  not Unity wall-clock speed or engine time scale.
- Optional Unity visibility randomization uses an explicit reset payload.
  It does not recreate the chapter's textures/scenes/randomization campaign;
  those require the original Unity scene setup.

No changes to the chapter are made by this implementation.

## Configure and launch

Use the project's compatible Python/Unity ML-Agents environment. Core numerical
code needs NumPy and PyTorch; tests additionally need pytest. The optional Unity
backend needs the repository's ML-Agents/gym-unity dependencies and Transformers
for the supplied Hugging Face depth loader. There is no import-time simulator,
checkpoint download, remote logging or training.

1. Copy configs/navigation.template.json to your experiment config.
2. Select the algorithm and fill reward, adapter, optimizer and sensor settings.
3. Select an exact local depth checkpoint; neither its identity nor Unity sensor
   indices are guessed.
4. Check configuration without loading PyTorch, models or Unity:

       python -m bluerov_nav.train --config configs/my_experiment.json --check-config

5. Only when ready, explicitly opt into training:

       python -m bluerov_nav.train --config configs/my_experiment.json --train --output runs/ppo_demo

The output directory must not exist, protecting historical outputs. Use a new
directory for every run, waypoint and seed. Repeated campaigns are separate
launches; no five-run result is synthesized automatically.

Evaluate a trusted new-format checkpoint with deterministic mean actions:

    python -m bluerov_nav.train --config configs/my_experiment.json --evaluate runs/ppo_demo/policy-0200.pt --episodes 5 --output runs/ppo_demo_evaluation

Checkpoints support evaluation, not exact training resume: optimizer, replay and
environment state are not restored. Config, version information, episode-level
metrics and iteration metrics are stored alongside checkpoints. JSON null means
no completed episode in an iteration, not zero reward or zero success.

Replay holds complete image histories and next histories: roughly 0.8 MiB per
transition at history 5 before overhead. Choose a capacity that fits the machine.

## Unity telemetry contract

bluerov_nav.unity:make_env requires an executable plus explicit RGB/ray indices,
channel lists, metric scales, side-channel indices and action multipliers. The
template's side-channel UUID is inherited from this repository, not independently
verified against a Unity executable.

The backend keeps RGB and ray arrays separate, performs one depth prediction per
observation, converts heading to radians, and checks shapes. The simulator must
send fresh telemetry on reset and each transition. Missing telemetry raises an
error; no extra hidden action is executed to obtain it.

For another simulator, env_factory may be a Python module:callable accepting
(env_settings, history_length, seed) and returning the NavigationEnv reset/step
protocol. A NavigationEnv backend emits:

    depth:                 normalized [0,1], shape (128,160)
    goal:                  [horizontal distance, signed vertical offset, heading radians]
    position:              metric 3-vector in a consistent coordinate system
    range:                 scalar SBES metric distance
    horizontal_clearance:  nonnegative metric distance
    vertical_clearance:    nonnegative metric distance
    surface_clearance:     nonnegative metric distance

The adapter returns (observation, reward, terminated, truncated, info).
Setting use_sbes=false zeros ONLY the policy range feature; simulator clearance
inputs remain available for reward computation.

## Tests and remaining validation

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit

NumPy tests cover reward boundaries, independent repetition, shared surface
thresholds, history, reset, derivative timing, timeout, success flags, replay
ownership/next-SBES storage, and required configuration. PyTorch tests exercise
all five update paths and skip when PyTorch is absent. Full Unity integration,
depth-checkpoint compatibility, timing and training performance require tests
in the intended environment.
