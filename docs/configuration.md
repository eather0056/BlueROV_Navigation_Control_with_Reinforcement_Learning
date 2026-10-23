# Configuration reference

All commands in this guide run from the repository root. Start with
configs/demo.json for interface tests or configs/navigation.template.json for
Unity. Demo numbers are arbitrary engineering fixtures, not recovered study
parameters.

## General settings

| Field | Meaning |
|---|---|
| algorithm | a2c, ppo, trpo, sac or td3 |
| seed | Seed for Python, NumPy, PyTorch and the configured backend |
| device | PyTorch device; cpu is the portable starting point |
| history_length | Frames per modality; reported study value is 5 |
| iterations | Number of collection/update cycles |
| batch_steps | Environment transitions per training iteration |
| checkpoint_interval | Save model weights every this many iterations and at the end |
| env_factory | Importable module:callable accepting settings, history length and seed |

## Optimizer settings

Common: learning_rate and gamma.

| Algorithm | Additional required settings |
|---|---|
| A2C | gae_lambda, l2_reg |
| PPO | gae_lambda, l2_reg, clip, epochs, minibatch_size |
| TRPO | gae_lambda, l2_reg, max_kl, damping |
| SAC | target_tau, entropy_temperature, replay_capacity, replay_batch, warmup_steps, gradient_steps |
| TD3 | target_tau, replay_capacity, replay_batch, warmup_steps, gradient_steps, target_noise, noise_clip, policy_delay, exploration_noise |

GAE weighting and target-network averaging are different settings. Do not
interpret the original shared coefficient row as a verified mapping to both.
For SAC the entropy temperature is fixed by configuration; automatic tuning is
not implemented. gradient_steps counts updates per eligible environment
transition. policy_delay counts critic updates.

## Reward settings: env.reward

| Field | Meaning |
|---|---|
| delta_h | Horizontal collision-clearance threshold |
| delta_v | Shared vertical and surface-clearance threshold |
| goal_radius | Horizontal acceptance radius |
| crash_penalty | Positive magnitude of the negative collision term |
| success_bonus | Positive horizontal goal bonus |
| s0 | Near-obstacle distance weight |
| s1, s2 | Heading-error weights outside/inside the goal region |
| s3 | Signed vertical-goal-rate weight |
| s4 | Vertical-offset magnitude weight inside the goal region |
| stuck_threshold | Low-displacement threshold |
| movement_bonus | Positive incentive below the displacement threshold |
| repeat_penalty | Positive magnitude of the negative repeated-action term |

Distances are metric, heading is radians, and vertical-goal rate is offset change
per simulation second. Reward magnitudes must be positive; s0–s4 may be zero.
No field is silently supplied from legacy values.

## Environment adapter: env.adapter

- decision_period: simulated seconds per action, not wall-clock time.
- displacement_window: number of stored positions (at least 2). Bonus is
  disabled until the window is full.
- repeat_tolerance: maximum componentwise difference from the last normalized
  action classified as repetition; zero means exact equality.
- terminate_on_goal: explicit boolean; collision always terminates.
- max_steps: exact transition limit. Hitting it produces truncation.
- use_sbes: false zeros the policy's acoustic feature, not reward clearances.

These choices are not all specified by the publication. Record chosen values
as part of each new experiment rather than treating them as recovered facts.

## Unity settings

See [Unity setup](unity_setup.md) for the sensor contract. Paths, channel mappings,
action scaling, depth checkpoint identity and simulator timing must match the
actual executable. --check-config checks declarations, not simulator behaviour.
