"""Validate chapter facts separately from required new-experiment choices."""
import math
from .reward import RewardConfig


def validate(config):
    if config.get("algorithm") not in {"a2c", "ppo", "trpo", "sac", "td3"}:
        raise ValueError("algorithm must be a2c, ppo, trpo, sac or td3")
    for name in ("history_length", "iterations", "batch_steps", "checkpoint_interval"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(config.get("env_factory"), str) or ":" not in config["env_factory"]:
        raise ValueError("env_factory must be module:callable")
    if not isinstance(config.get("env"), dict):
        raise ValueError("env settings are required")
    RewardConfig(**config["env"].get("reward", {}))
    if config["env_factory"] == "bluerov_nav.unity:make_env":
        adapter = config["env"].get("adapter", {})
        for name in ("decision_period", "displacement_window", "repeat_tolerance",
                     "terminate_on_goal", "max_steps"):
            if adapter.get(name) is None:
                raise ValueError(f"Set env.adapter.{name} explicitly")
        from .environment import NavigationEnv
        NavigationEnv(None, config["env"]["reward"], history_length=config["history_length"], **adapter)
        unity = config["env"].get("unity", {})
        required_unity = ("executable", "worker_id", "no_graphics", "time_scale",
                          "side_channel_uuid", "depth_factory", "depth", "rgb_observation",
                          "ray_observation", "sbes_indices", "horizontal_ray_indices",
                          "vertical_ray_indices", "ray_scale", "goal_indices",
                          "position_indices", "surface_index", "heading_units", "action_scale")
        for name in required_unity:
            if unity.get(name) is None:
                raise ValueError(f"Set env.unity.{name} explicitly")
        for name in ("sbes_indices", "horizontal_ray_indices", "vertical_ray_indices",
                     "goal_indices", "position_indices"):
            indexes = unity[name]
            if not isinstance(indexes, list) or not indexes or any(type(i) is not int or i < 0 for i in indexes):
                raise ValueError(f"env.unity.{name} must list nonnegative channel indices")
        if len(unity["goal_indices"]) != 3 or len(unity["position_indices"]) != 3:
            raise ValueError("goal_indices and position_indices must each contain 3 entries")
        if unity["rgb_observation"] == unity["ray_observation"]:
            raise ValueError("RGB and ray observations must be separate")
        if unity["heading_units"] not in {"degrees", "radians"}:
            raise ValueError("heading_units must be degrees or radians")
        if unity["ray_scale"] <= 0 or unity["time_scale"] <= 0:
            raise ValueError("ray_scale and time_scale must be positive")
        if unity["depth_factory"] == "bluerov_nav.unity:make_hf_depth" and not unity["depth"].get("model_id"):
            raise ValueError("Set env.unity.depth.model_id to the exact checkpoint")
        if unity.get("randomize_visibility"):
            if unity.get("reset_payload") is None or unity.get("visibility_range") is None or unity.get("visibility_payload_index") is None:
                raise ValueError("Visibility randomization requires payload, range and payload index")
    s = config.get("optimizer", {})
    required = ["learning_rate", "gamma"]
    if config["algorithm"] in {"ppo", "a2c", "trpo"}:
        required += ["gae_lambda", "l2_reg"]
    if config["algorithm"] == "ppo":
        required += ["clip", "epochs", "minibatch_size"]
    if config["algorithm"] == "trpo":
        required += ["max_kl", "damping"]
    if config["algorithm"] in {"sac", "td3"}:
        required += ["target_tau", "replay_capacity", "replay_batch", "warmup_steps",
                     "gradient_steps"]
    if config["algorithm"] == "sac":
        required += ["entropy_temperature"]
    if config["algorithm"] == "td3":
        required += ["target_noise", "noise_clip", "policy_delay", "exploration_noise"]
    integer = {"epochs", "minibatch_size", "replay_capacity", "replay_batch",
               "warmup_steps", "gradient_steps", "policy_delay"}
    zero_allowed = {"l2_reg", "damping", "target_noise", "noise_clip",
                    "exploration_noise", "gae_lambda"}
    for name in required:
        value = s.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Set optimizer.{name} explicitly; it is not inferred from the chapter")
        if value < 0 or (value == 0 and name not in zero_allowed):
            raise ValueError(f"Invalid optimizer.{name}")
        if name in integer and type(value) is not int:
            raise ValueError(f"optimizer.{name} must be an integer")
    if not 0 < s["gamma"] < 1:
        raise ValueError("gamma must be in (0,1)")
    for name in ("gae_lambda", "target_tau"):
        if name in required and s[name] > 1:
            raise ValueError(f"{name} must not exceed 1")
    if "clip" in required and s["clip"] >= 1:
        raise ValueError("PPO clip must be in (0,1)")
    if "replay_capacity" in required and s["replay_capacity"] < s["replay_batch"]:
        raise ValueError("replay_capacity must be >= replay_batch")
    return config
