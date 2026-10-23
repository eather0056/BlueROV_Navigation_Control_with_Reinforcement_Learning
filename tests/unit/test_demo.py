import json
from pathlib import Path
import numpy as np
from bluerov_nav.config import validate
from bluerov_nav.demo import make_env


def test_demo_config_and_deterministic_rollout():
    path = Path(__file__).resolve().parents[2] / "configs" / "demo.json"
    config = validate(json.loads(path.read_text()))
    env = make_env(config["env"], config["history_length"], config["seed"])
    results = []
    try:
        for _ in range(2):
            assert env.reset()["depth"].shape == (5, 128, 160)
            rewards = []
            for step in range(8):
                _, reward, terminated, truncated, _ = env.step([.2, 0])
                assert not terminated
                assert truncated == (step == 7)
                rewards.append(reward)
            results.append(rewards)
        np.testing.assert_array_equal(*results)
    finally:
        env.close()
