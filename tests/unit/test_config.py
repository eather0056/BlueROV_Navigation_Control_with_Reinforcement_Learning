from dataclasses import asdict
import json
from pathlib import Path
import pytest
from bluerov_nav.config import validate
from bluerov_nav.reward import RewardConfig


def config(algorithm):
    return dict(algorithm=algorithm, seed=1, history_length=5, iterations=200,
                batch_steps=2048, checkpoint_interval=10, env_factory="tests.fake:make_env",
                env={"reward": asdict(RewardConfig(.5, .2, .6, 10, 20, 2, 3, 4, 5, 6, .1, 1, 2))},
                optimizer=dict(learning_rate=3e-5, gamma=.99, gae_lambda=.95,
                               l2_reg=0, clip=.2, epochs=2, minibatch_size=2,
                               max_kl=.01, damping=.1, target_tau=.005,
                               entropy_temperature=.2, replay_capacity=4, replay_batch=2,
                               warmup_steps=2, gradient_steps=1, target_noise=.2,
                               noise_clip=.5, policy_delay=2, exploration_noise=.1))


@pytest.mark.parametrize("algorithm", ["a2c", "ppo", "trpo", "sac", "td3"])
def test_algorithm_config(algorithm):
    assert validate(config(algorithm))["algorithm"] == algorithm


@pytest.mark.parametrize("field,value", [("gae_lambda", None), ("gamma", 2),
                                       ("epochs", 1.5), ("clip", 1)])
def test_invalid_optimizer(field, value):
    c = config("ppo")
    c["optimizer"][field] = value
    with pytest.raises(ValueError):
        validate(c)


def test_template_intentionally_requires_original_settings():
    template = Path(__file__).resolve().parents[2] / "configs" / "navigation.template.json"
    with pytest.raises(ValueError, match="Specify numeric reward setting"):
        validate(json.loads(template.read_text()))
