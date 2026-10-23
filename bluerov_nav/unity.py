"""Optional Unity backend with explicit sensor/channel mappings and local depth weights."""
import importlib
import uuid
import numpy as np
from .environment import NavigationEnv


def make_hf_depth(settings):
    """Load only explicitly selected, already-local HF depth weights by default."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = settings["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("Select the exact depth model/checkpoint")
    local = settings.get("local_files_only", True)
    device = torch.device(settings.get("device", "cpu"))
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=local).to(device).eval()

    def predict(rgb):
        inputs = processor(images=rgb, return_tensors="pt", do_rescale=False).to(device)
        with torch.no_grad():
            depth = model(**inputs).predicted_depth
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(128, 160),
                                                    mode="bicubic", align_corners=False)[0, 0]
            span = depth.max() - depth.min()
            depth = ((depth - depth.min()) / span.clamp_min(1e-8)).clamp(0, 1)
        return depth.cpu().numpy()
    return predict


class UnityBackend:
    def __init__(self, settings, seed):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from gym_unity.envs import UnityToGymWrapper
        self.settings = settings

        class TelemetryChannel(SideChannel):
            def __init__(self):
                super().__init__(uuid.UUID(settings["side_channel_uuid"]))
                self.values = None

            def on_message_received(self, msg):
                self.values = np.asarray(msg.read_float32_list(), dtype=np.float64)

            def configure(self, values):
                message = OutgoingMessage()
                message.write_float32_list(values)
                self.queue_message_to_send(message)

        self.channel = TelemetryChannel()
        module, factory = settings["depth_factory"].split(":")
        self.depth = getattr(importlib.import_module(module), factory)(settings["depth"])
        self.rng = np.random.default_rng(seed)
        engine = EngineConfigurationChannel()
        unity = UnityEnvironment(settings["executable"], seed=seed,
                                 worker_id=settings["worker_id"],
                                 no_graphics=settings["no_graphics"],
                                 side_channels=[engine, self.channel])
        try:
            engine.set_configuration_parameters(time_scale=settings["time_scale"])
            self.env = UnityToGymWrapper(unity, allow_multiple_obs=True)
        except Exception:
            unity.close()
            raise

    def _decode(self, observations):
        s = self.settings
        rgb = np.asarray(observations[s["rgb_observation"]], dtype=np.float32)
        rays = np.asarray(observations[s["ray_observation"]])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rays.ndim != 1:
            raise ValueError("RGB and ray observation mappings do not match Unity outputs")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Unity RGB must be finite [0,1]")
        telemetry = self.channel.values
        if telemetry is None:
            raise RuntimeError("No fresh side-channel telemetry; verify Unity protocol")
        g = telemetry[s["goal_indices"]].copy()
        if s["heading_units"] == "degrees":
            g[2] = np.deg2rad(g[2])
        elif s["heading_units"] != "radians":
            raise ValueError("heading_units must be degrees or radians")
        g[2] = (g[2] + np.pi) % (2 * np.pi) - np.pi
        # All scales/index sets are explicit; never substitute RGB for range data.
        distance = lambda indices: float(np.min(rays[indices]) * s["ray_scale"])
        return {"depth": self.depth(rgb), "goal": g,
                "position": telemetry[s["position_indices"]],
                "range": distance(s["sbes_indices"]),
                "horizontal_clearance": distance(s["horizontal_ray_indices"]),
                "vertical_clearance": distance(s["vertical_ray_indices"]),
                "surface_clearance": abs(float(telemetry[s["surface_index"]]))}

    def reset(self):
        s = self.settings
        # Optional explicit visibility protocol, matching the side-channel payload.
        payload = s.get("reset_payload")
        if payload is not None:
            payload = list(payload)
            if s.get("randomize_visibility", False):
                low, high = s["visibility_range"]
                if not 0 < low <= high:
                    raise ValueError("visibility_range must be positive and ordered")
                payload[s["visibility_payload_index"]] = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
            self.channel.configure(payload)
        self.channel.values = None
        observations = self.env.reset()
        return self._decode(observations)

    def step(self, action):
        self.channel.values = None
        scale = np.asarray(self.settings["action_scale"], dtype=np.float32)
        if scale.shape != (2,) or not np.isfinite(scale).all():
            raise ValueError("action_scale must specify finite [vertical, yaw] multipliers")
        observations, _, done, _ = self.env.step(np.asarray(action) * scale)
        return self._decode(observations), bool(done)

    def close(self):
        self.env.close()


def make_env(settings, history_length, seed):
    adapter = settings["adapter"]
    # Validate adapter/reward before opening a simulator or loading a model.
    probe = NavigationEnv(None, settings["reward"], history_length=history_length, **adapter)
    probe.backend = UnityBackend(settings["unity"], seed)
    return probe
