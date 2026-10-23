"""Deterministic synthetic telemetry for setup checks, NOT an underwater simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import validate
from .environment import NavigationEnv


class DemoBackend:
    def __init__(self):
        self.goal = np.array([3.0, 0.5, 0.0])

    def reset(self):
        self.position = np.zeros(3)
        self.yaw = 0.0
        return self.observe()

    def observe(self):
        delta = self.goal - self.position
        heading = (np.arctan2(delta[2], delta[0]) - self.yaw + np.pi) % (2 * np.pi) - np.pi
        depth = np.broadcast_to(np.linspace(0, 1, 160, dtype=np.float32), (128, 160)).copy()
        return {"depth": depth, "goal": [np.hypot(delta[0], delta[2]), delta[1], heading],
                "position": self.position.copy(), "range": 5.0,
                "horizontal_clearance": 5.0, "vertical_clearance": 5.0,
                "surface_clearance": 5.0}

    def step(self, action):
        self.yaw += float(action[1]) * .05
        self.position += [.1 * np.cos(self.yaw), .05 * float(action[0]), .1 * np.sin(self.yaw)]
        return self.observe(), False

    def close(self):
        pass


def make_env(settings, history_length, seed):
    return NavigationEnv(DemoBackend(), settings["reward"],
                         history_length=history_length, **settings["adapter"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.json")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = validate(json.loads(Path(args.config).read_text()))
    if config["env_factory"] != "bluerov_nav.demo:make_env":
        parser.error("This check accepts only the synthetic demo factory")
    env = make_env(config["env"], config["history_length"], config["seed"])
    try:
        obs = env.reset()
        print(json.dumps({"kind": "synthetic_setup_check", "shapes": {k: list(v.shape) for k, v in obs.items()}}))
        for i in range(args.steps):
            _, reward, term, trunc, info = env.step([.2, 0])
            print(json.dumps({"transition": i + 1, "reward": reward,
                              "terminated": term, "truncated": trunc, **info}))
            if term or trunc:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
