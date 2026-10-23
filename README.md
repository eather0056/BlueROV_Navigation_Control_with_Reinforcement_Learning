# BlueROV Navigation with Deep Reinforcement Learning

Multimodal reinforcement learning for **simulated underwater waypoint navigation**.
The policy combines monocular depth, single-beam echo-sounder (SBES) range,
relative goal information, and action history. The new implementation supports
**A2C, PPO, TRPO, SAC and TD3**.

**Start here:** run the synthetic setup check below, then configure a Unity
environment. No training, downloads or remote logging starts when you import
the package.

> **Reproducibility scope:** this repository contains a new implementation and
> preserved legacy research code. The new implementation has not reproduced the
> paper's numerical results. Original run-specific reward values, scene settings
> and checkpoint provenance are incomplete. Do not substitute demo settings for
> historical experimental settings. See [Reproducibility](docs/reproducibility.md).

## 1. Install and check the setup

Run commands from the repository root:

~~~bash
git clone https://github.com/eather0056/BlueROV_Nav_DRL.git
cd BlueROV_Nav_DRL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
~~~

On Windows, activate with .venv\Scripts\activate instead. The package declares
Python 3.8 or newer; the **Unity integration has its own version constraints**
and must match your ML-Agents build. Do not assume every modern Python version
is compatible with an older Unity release.

Check configuration and exercise synthetic observations without PyTorch,
Unity, pretrained weights, or training:

~~~bash
python -m bluerov_nav.train --config configs/demo.json --check-config
python -m bluerov_nav.demo --config configs/demo.json --steps 8
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
~~~

The demo reports history shapes, reward components and a timeout at transition 8.
It is a deterministic **synthetic interface check**, not an underwater simulator
or navigation benchmark. PyTorch-dependent tests skip if PyTorch is absent.
The environment-variable prefix disables unrelated globally installed pytest
plugins; it can be omitted in a clean virtual environment.

For neural-network training and update tests:

~~~bash
python -m pip install -e ".[train,test]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
~~~

Use a PyTorch build appropriate to your machine if you need GPU support.
The dependency declarations are not a verified historical environment lockfile.

## 2. Optional synthetic training check

The following commands **do perform training**, but only on the tiny synthetic
demo. They require PyTorch and are not research experiments:

~~~bash
python -m bluerov_nav.train --config configs/demo.json \
  --train --output runs/demo_ppo

python -m bluerov_nav.train --config configs/demo.json \
  --evaluate runs/demo_ppo/policy-0002.pt \
  --episodes 5 --output runs/demo_ppo_eval
~~~

Every output directory must be new. This prevents accidental overwriting of
previous results. The installed console command bluerov-nav is equivalent to
python -m bluerov_nav.train.

## 3. Configure a Unity navigation experiment

You need:

- A compatible Unity executable and its scene/agent configuration.
- Matching ML-Agents and gym-unity Python dependencies.
- An exact local depth-model checkpoint and its image processor.
- Verified RGB, SBES/ray and side-channel telemetry mappings.
- Explicit reward coefficients, thresholds and training settings.

See [Unity setup and sensor contract](docs/unity_setup.md) before running.
The executable and depth weights are **not bundled by this new implementation**.
Submodule declarations alone do not establish that the original simulator is
available or compatible.

~~~bash
cp configs/navigation.template.json configs/my_navigation.json
~~~

Edit the copied file. Relevant null entries are intentional: required settings
are not inferred from an unrelated implementation. See the
[configuration reference](docs/configuration.md) for each group.

~~~bash
python -m bluerov_nav.train --config configs/my_navigation.json --check-config
python -m bluerov_nav.train --config configs/my_navigation.json \
  --train --output runs/ppo_wp1_seed1
~~~

Configuration checking validates required settings; it does not launch Unity,
verify a checkpoint's compatibility, or prove the sensor mapping is correct.

Choose the algorithm by setting algorithm to a2c, ppo, trpo, sac or td3.
PPO/A2C/TRPO use on-policy rollouts; SAC/TD3 use replay and action-value critics.
Each algorithm has explicitly required optimizer settings.

## 4. Evaluate a checkpoint

Use the matching experiment configuration and a **trusted new-format** checkpoint:

~~~bash
python -m bluerov_nav.train --config configs/my_navigation.json \
  --evaluate runs/ppo_wp1_seed1/policy-0200.pt \
  --episodes 5 --output runs/ppo_wp1_seed1_eval
~~~

The checkpoint number depends on iterations and checkpoint_interval. Evaluation
uses deterministic mean actions. Legacy pickled models are not interchangeable
with new checkpoints. Exact training resume is not implemented.

## 5. Outputs and metric definitions

| File | Contents |
|---|---|
| config.json | Exact supplied experiment configuration |
| provenance.txt | Implementation scope, NumPy/PyTorch versions and device |
| metrics.jsonl | Training iteration, transitions, cumulative episodes and losses |
| episodes.jsonl | Completed training episodes, returns, step counts and outcomes |
| policy-NNNN.pt | Model weights for evaluation |
| evaluation.jsonl | Per-episode deterministic evaluation records |

- **iteration** is the training-update iteration, not an environment transition.
- **cumulative_episodes** never resets between iterations within a training run.
- **success_ratio** in an iteration uses episodes completed in that iteration.
  JSON null means no completed episodes, not zero success.
- Evaluation success is horizontal goal acceptance without simultaneous clearance
  violation on the final frame. It is not inferred from positive reward.
- **simulation_time** is step count multiplied by the configured simulation
  decision period, not measured wall-clock runtime.
- New metrics are explicit implementation conventions, not reconstructed
  definitions of the original published aggregate results.

For multi-seed or multi-waypoint studies, use separate saved configurations and
output directories. Keep all failures and denominators when aggregating results.
See the [experiment checklist](docs/reproducibility.md).

## Repository organization

~~~text
bluerov_nav/                  Recommended package for new experiments
  reward.py                   Five-component reward and branch conditions
  environment.py              Observation history and episode adapter
  networks.py                 Multimodal actor, value and twin-Q networks
  algorithms.py               A2C/PPO/TRPO/SAC/TD3 update routines
  replay.py                   Bounded multimodal replay
  config.py                   Configuration validation
  train.py                    Explicit training/evaluation CLI
  unity.py                    Optional Unity and depth-model adapter
  demo.py                     Synthetic setup check
configs/
  demo.json                   Runnable synthetic example, not paper settings
  navigation.template.json    Required fields for a real simulation experiment
tests/unit/                   Reward, adapter, configuration and update tests
docs/
  configuration.md
  unity_setup.md
  implementation.md
  reproducibility.md
  legacy.md
pyproject.toml                Package metadata and optional dependencies
requirements-dev.txt          Editable development installation
requirements.txt              Preserved legacy dependencies
core/, models/, utils/        Legacy research implementation
*_gym.py, *_env.py            Legacy launch scripts
DPT/, assets/, Result/        Preserved models, assets and demonstrations
~~~

Legacy scripts remain at their existing paths to avoid breaking their imports
and asset references. They do not invoke the new package. Historical migration
details are documented in [Legacy code](docs/legacy.md).

## Troubleshooting

| Problem | What to check |
|---|---|
| Missing numerical reward setting | Fill the required field; do not copy demo values as historical values |
| Cannot import bluerov_nav | Run from the repository root or install with pip install -e . |
| Cannot import torch | Install the train extra in the active environment |
| Cannot import mlagents_envs/gym_unity | Install versions compatible with your Unity build |
| Missing local depth checkpoint | Set env.unity.depth.model_id to a complete local model directory compatible with the selected loader |
| No fresh side-channel telemetry | Verify UUID, sender, field order and reset/step transmission |
| RGB/ray shape error | Correct sensor indices; RGB is not a substitute for SBES |
| Output directory already exists | Choose a new run directory |
| Replay runs out of memory | Reduce replay capacity; full depth histories are memory intensive |
| Different results across machines | Record versions, hardware and seeds; GPU determinism is not guaranteed |

## Research and acknowledgments

Related publication: **Optimizing Underwater Robot Navigation: A Study of DRL
Algorithms and Multi-Modal Sensor Fusion**, Md Ether Deowan, Md Shamin Yeasher
Yousha, Tihan Mahmud Hossain, Shahriar Hassan and Ricard Marxer, ICRA 2025,
pp. 11270–11277.

The existing [demonstrations](Result/) and
[video](https://youtu.be/2yoDlwFw2qQ) are historical research artifacts, not outputs
generated by the new implementation.

Thanks to Ricard Marxer, the University of Toulon, and the ROS, Unity, ML-Agents
and reinforcement-learning communities. See [LICENSE](LICENSE) for the
repository's Apache-2.0 license; third-party assets and weights may have
separate terms.
