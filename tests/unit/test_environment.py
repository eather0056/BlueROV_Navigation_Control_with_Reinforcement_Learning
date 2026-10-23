from dataclasses import asdict
import numpy as np
import pytest
from bluerov_nav.environment import NavigationEnv
from bluerov_nav.reward import RewardConfig
from bluerov_nav.replay import Replay


class FakeBackend:
    def reset(self):
        self.i = 0
        return self.raw()

    def raw(self):
        return {"depth": np.full((128, 160), .5), "goal": [3, 1 - .1 * self.i, 0],
                "position": [self.i, 0, 0], "range": 2 + self.i,
                "horizontal_clearance": 2, "vertical_clearance": 2, "surface_clearance": 2}

    def step(self, action):
        self.i += 1
        return self.raw(), False

    def close(self):
        pass


def make_env(**kwargs):
    settings = dict(decision_period=.5, displacement_window=3, repeat_tolerance=0,
                    terminate_on_goal=True, max_steps=2)
    settings.update(kwargs)
    config = asdict(RewardConfig(.5, .2, .6, 10, 20, 2, 3, 4, 5, 6, .1, 1, 2))
    return NavigationEnv(FakeBackend(), config, **settings)


def test_history_derivative_and_timeout():
    env = make_env()
    obs = env.reset()
    assert obs["depth"].shape == (5, 128, 160)
    nxt, _, term, trunc, info = env.step([.2, -.1])
    assert not term and not trunc
    assert info["reward_components"]["vertical_goal"] == pytest.approx(1, abs=1e-6)
    np.testing.assert_allclose(nxt["actions"][0], [.2, -.1])
    np.testing.assert_allclose(nxt["range"][:2, 0], [3, 2])
    _, _, term, trunc, info = env.step([.2, -.1])
    assert trunc and not term and info["simulation_time"] == 1
    assert info["reward_components"]["repetition"] == -2
    with pytest.raises(RuntimeError):
        env.step([0, 0])
    env.reset()
    _, _, _, _, info = env.step([.2, -.1])
    assert info["reward_components"]["repetition"] == 0


def test_no_sbes_and_invalid_action():
    env = make_env(use_sbes=False)
    assert not env.reset()["range"].any()
    with pytest.raises(ValueError):
        env.step([0, 2])


def test_goal_collision_not_success():
    env = make_env()
    env.reset()
    raw = env.backend.raw
    def collide():
        x = raw()
        x["goal"][0] = 0
        x["surface_clearance"] = .1
        return x
    env.backend.raw = collide
    _, _, term, trunc, info = env.step([0, 0])
    assert term and not trunc and not info["success"]
    assert info["goal_reached"] and info["collision"]


def test_replay_copies_next_range():
    env = make_env()
    obs = env.reset()
    nxt, reward, term, _, _ = env.step([0, 0])
    replay = Replay(2, np.random.default_rng(1))
    replay.add(obs, [0, 0], reward, nxt, term)
    nxt["range"][:] = 999
    batch = replay.sample(1)
    assert batch[3]["range"][0, 0, 0] == 3
