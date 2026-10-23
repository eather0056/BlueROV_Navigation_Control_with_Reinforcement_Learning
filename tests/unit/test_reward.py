from dataclasses import replace
import pytest
from bluerov_nav.reward import RewardConfig, RewardState, compute_reward


# Arbitrary unit-test fixtures, NOT reported experiment parameters.
CONFIG = RewardConfig(.5, .2, .6, 10, 20, 2, 3, 4, 5, 6, .1, 1, 2)
STATE = RewardState(2, 2, 2, 3, .4, .2, -.3, .5, False)


def test_goal_and_vertical_branches():
    _, c, collision, goal = compute_reward(STATE, CONFIG)
    assert not collision and not goal
    assert c["horizontal_goal"] == pytest.approx(-.6)
    assert c["vertical_goal"] == pytest.approx(1.5)
    _, c, _, goal = compute_reward(replace(STATE, horizontal_goal=.6), CONFIG)
    assert goal
    assert c["horizontal_goal"] == pytest.approx(19.2)
    assert c["vertical_goal"] == pytest.approx(-2.4)
    _, c, _, _ = compute_reward(replace(STATE, vertical_goal_rate=.3), CONFIG)
    assert c["vertical_goal"] == pytest.approx(-1.5)


@pytest.mark.parametrize("distance,expected", [(.49, -10), (.5, -1), (.75, -.5), (1, 0)])
def test_obstacle_boundary(distance, expected):
    _, c, _, _ = compute_reward(replace(STATE, horizontal_clearance=distance), CONFIG)
    assert c["obstacle"] == pytest.approx(expected)


def test_collision_overrides_near_wall_penalty():
    _, c, collision, _ = compute_reward(replace(STATE, horizontal_clearance=.75, surface_clearance=.1), CONFIG)
    assert collision and c["obstacle"] == -10


def test_repetition_not_gated_by_low_motion():
    total, c, _, _ = compute_reward(replace(STATE, repeated_action=True), CONFIG)
    assert c["repetition"] == -2 and c["movement"] == 0
    assert total == pytest.approx(sum(c.values()))
    _, c, _, _ = compute_reward(replace(STATE, displacement=.05), CONFIG)
    assert c["movement"] == 1
    _, c, _, _ = compute_reward(replace(STATE, displacement=.05, displacement_ready=False), CONFIG)
    assert c["movement"] == 0


def test_no_guessed_parameters():
    with pytest.raises(ValueError, match="Specify numeric"):
        replace(CONFIG, s0=None)
    with pytest.raises(ValueError):
        replace(CONFIG, delta_v=-1)


def test_nonfinite_input():
    with pytest.raises(ValueError):
        compute_reward(replace(STATE, heading_radians=float("nan")), CONFIG)
