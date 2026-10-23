# Legacy code and naming migration

The recommended package for new experiments is bluerov_nav. The existing
core/, models/, utils/, DPT/, assets/, Result/ and top-level training scripts
are retained at their original paths. Moving them would break relative imports,
weight paths, simulator paths or historical checkpoint loading.

## Legacy entry points

| Path | Historical purpose |
|---|---|
| ppo_gym.py | PPO training |
| trpo_gym.py | TRPO training |
| a2c_env.py | A2C training |
| sac_env.py | Legacy SAC-labelled training path |
| ppo_gym_test.py, test_multi.py | Legacy evaluation |
| requirements.txt | Legacy dependency list |
| ex.slurm, ex_wandb.slurm | Site-specific cluster jobs |

These scripts are not wrappers around the new implementation and may initialize
Unity, load checkpoints or contact experiment-tracking services at startup.
Inspect their local paths and configuration before running them. Do not load
untrusted pickle checkpoints.

Machine-specific cluster accounts, VPN instructions and absolute workstation
paths are intentionally not part of the new quick start. The original tracked
README remains available through Git history.

## Names used by the new implementation

| Previous development name | Current name |
|---|---|
| chapter_impl | bluerov_nav |
| ChapterEnv | NavigationEnv |
| configs/chapter.template.json | configs/navigation.template.json |
| tests/chapter | tests/unit |
| docs/chapter_implementation.md | docs/implementation.md |
| requirements-chapter.txt | requirements-dev.txt and pyproject.toml |

Update custom environment factory strings and imports to bluerov_nav. There is
no automatic historical-checkpoint conversion. The old package name is not
retained as a second implementation.
