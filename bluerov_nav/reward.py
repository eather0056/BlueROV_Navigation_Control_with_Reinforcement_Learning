"""Literal reward branches from the chapter, independent of Unity and PyTorch."""
from dataclasses import dataclass, asdict
from math import isfinite


@dataclass(frozen=True)
class RewardConfig:
    delta_h: float
    delta_v: float
    goal_radius: float
    crash_penalty: float
    success_bonus: float
    s0: float
    s1: float
    s2: float
    s3: float
    s4: float
    stuck_threshold: float
    movement_bonus: float
    repeat_penalty: float

    def __post_init__(self):
        positive = {"delta_h", "delta_v", "goal_radius", "crash_penalty",
                    "success_bonus", "stuck_threshold", "movement_bonus", "repeat_penalty"}
        for name, value in asdict(self).items():
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise ValueError(f"Specify numeric reward setting {name}; no chapter default exists")
            if not isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid reward setting {name}: {value}")


@dataclass(frozen=True)
class RewardState:
    horizontal_clearance: float
    vertical_clearance: float
    surface_clearance: float
    horizontal_goal: float
    vertical_goal: float
    heading_radians: float
    vertical_goal_rate: float
    displacement: float
    repeated_action: bool
    displacement_ready: bool = True


def compute_reward(state, config):
    """Return all five signed components and geometry flags.

    Collision takes precedence ONLY within the obstacle component. Goal and
    repetition terms are independent, as in the chapter. Do not gate repetition
    on stagnation, substitute vertical action for distance rate, or pay both
    near-wall and collision penalties. This preserves the original equations,
    including the positive low-displacement incentive.
    """
    for name, value in asdict(state).items():
        if name not in {"repeated_action", "displacement_ready"} and not isfinite(value):
            raise ValueError(f"Nonfinite reward input {name}")
    for name in ("horizontal_clearance", "vertical_clearance", "surface_clearance",
                 "horizontal_goal", "displacement"):
        if getattr(state, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    c, x = config, state
    collision = (x.horizontal_clearance < c.delta_h or
                 x.vertical_clearance < c.delta_v or x.surface_clearance < c.delta_v)
    if collision:
        obstacle = -c.crash_penalty
    elif x.horizontal_clearance < 2 * c.delta_h:
        obstacle = -c.s0 * (2 * c.delta_h - x.horizontal_clearance)
    else:
        obstacle = 0.0
    goal = x.horizontal_goal <= c.goal_radius
    horizontal = (c.success_bonus - c.s2 * abs(x.heading_radians) if goal
                  else -c.s1 * abs(x.heading_radians))
    if goal:
        vertical = -c.s4 * abs(x.vertical_goal)
    elif x.vertical_goal_rate <= 0:
        vertical = c.s3 * abs(x.vertical_goal_rate)
    else:
        vertical = -c.s3 * abs(x.vertical_goal_rate)
    components = {
        "obstacle": obstacle, "horizontal_goal": horizontal, "vertical_goal": vertical,
        "movement": (c.movement_bonus if x.displacement_ready and
                     x.displacement < c.stuck_threshold else 0.0),
        "repetition": -c.repeat_penalty if x.repeated_action else 0.0,
    }
    return sum(components.values()), components, collision, goal
