"""Sensor-contract adapter. Simulator/model loading is intentionally separate."""
from collections import deque
import numpy as np
from .reward import RewardConfig, RewardState, compute_reward


KEYS = ("depth", "goal", "range", "actions")


class NavigationEnv:
    """Wrap a backend emitting metric telemetry and a normalized depth image.

    backend.reset() -> telemetry
    backend.step(normalized_action) -> telemetry, simulator_terminated
    See docs/bluerov_navementation.md for the explicit telemetry contract.
    All implementation choices omitted by the chapter are required settings.
    """
    def __init__(self, backend, reward, *, decision_period, displacement_window,
                 repeat_tolerance, terminate_on_goal, max_steps, history_length=5,
                 use_sbes=True):
        if decision_period <= 0 or not np.isfinite(decision_period):
            raise ValueError("decision_period must be positive simulation seconds")
        if not isinstance(displacement_window, int) or displacement_window < 2:
            raise ValueError("displacement_window must be an integer >= 2")
        if repeat_tolerance < 0 or not np.isfinite(repeat_tolerance):
            raise ValueError("repeat_tolerance must be finite and nonnegative")
        if not isinstance(terminate_on_goal, bool):
            raise ValueError("terminate_on_goal must be an explicit boolean")
        if not isinstance(max_steps, int) or max_steps < 1 or history_length < 1:
            raise ValueError("positive step limit and history length required")
        self.backend, self.reward_config = backend, RewardConfig(**reward)
        self.dt, self.window, self.tolerance = decision_period, displacement_window, repeat_tolerance
        self.terminate_on_goal, self.max_steps = terminate_on_goal, max_steps
        self.k, self.use_sbes = history_length, use_sbes

    def _validate(self, raw):
        raw = dict(raw)
        for key, shape in (("depth", (128, 160)), ("goal", (3,)), ("position", (3,))):
            raw[key] = np.array(raw[key], dtype=np.float32, copy=True)
            if raw[key].shape != shape or not np.isfinite(raw[key]).all():
                raise ValueError(f"{key} must have finite shape {shape}")
        if raw["depth"].min() < 0 or raw["depth"].max() > 1:
            raise ValueError("backend depth must be normalized to [0, 1]")
        if raw["goal"][0] < 0 or abs(raw["goal"][2]) > np.pi + 1e-5:
            raise ValueError("goal requires nonnegative horizontal distance and heading in radians")
        for key in ("range", "horizontal_clearance", "vertical_clearance", "surface_clearance"):
            raw[key] = float(raw[key])
            if not np.isfinite(raw[key]) or raw[key] < 0:
                raise ValueError(f"{key} must be a finite nonnegative metric distance")
        return raw

    def _frame(self, raw, action):
        # Fixed normalization avoids changing replay observations during training.
        return {"depth": 2 * raw["depth"] - 1, "goal": raw["goal"].copy(),
                "range": np.array([raw["range"] if self.use_sbes else 0], dtype=np.float32),
                "actions": np.asarray(action, dtype=np.float32).copy()}

    def _obs(self):
        return {key: np.stack([f[key] for f in self.frames]) for key in KEYS}

    def reset(self):
        self.current = self._validate(self.backend.reset())
        self.positions = deque([self.current["position"].copy()], maxlen=self.window)
        self.previous_action, self.steps, self.finished = None, 0, False
        frame = self._frame(self.current, np.zeros(2))
        self.frames = deque([frame] * self.k, maxlen=self.k)
        return self._obs()

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset() is required after episode end")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError("action must be [vertical, yaw] in [-1, 1]^2")
        raw, simulator_end = self.backend.step(action.copy())
        raw = self._validate(raw)
        self.positions.append(raw["position"].copy())
        ready = len(self.positions) == self.window
        displacement = float(np.linalg.norm(self.positions[-1] - self.positions[0]))
        repeated = (self.previous_action is not None and
                    np.max(abs(action - self.previous_action)) <= self.tolerance)
        state = RewardState(
            raw["horizontal_clearance"], raw["vertical_clearance"], raw["surface_clearance"],
            float(raw["goal"][0]), float(raw["goal"][1]), float(raw["goal"][2]),
            float((raw["goal"][1] - self.current["goal"][1]) / self.dt),
            displacement, repeated, ready)
        reward, parts, collision, goal = compute_reward(state, self.reward_config)
        self.steps += 1
        terminated = bool(collision or simulator_end or (goal and self.terminate_on_goal))
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.finished = terminated or truncated
        self.frames.appendleft(self._frame(raw, action))
        self.current, self.previous_action = raw, action.copy()
        info = {"reward_components": parts, "collision": collision, "goal_reached": goal,
                "success": bool(goal and not collision),
                "episode_steps": self.steps, "simulation_time": self.steps * self.dt}
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        self.backend.close()
