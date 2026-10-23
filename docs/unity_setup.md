# Unity setup and integration

The synthetic demo does not need Unity. This page applies only to actual Unity
navigation experiments.

## Required external assets

1. A Unity build with camera, acoustic observations and telemetry side channel.
2. Python ML-Agents and gym-unity versions matching that build.
3. An exact local depth-model checkpoint and processor configuration.
4. Scene/start/goal definitions and the domain-randomization setup for your runs.

The historical README used Python 3.8, ML-Agents release_18 and gym-unity 0.27.0.
That is legacy setup information, not a newly tested compatibility guarantee.
Do not install a different current ML-Agents version and assume its protocol
matches an older executable. Use the environment associated with your Unity build.
The new package does not automatically initialize submodules or build Unity.

## Depth loading

Install the optional loader dependencies:

~~~bash
python -m pip install -e ".[train,depth]"
~~~

Set env.unity.depth_factory to bluerov_nav.unity:make_hf_depth and set
env.unity.depth.model_id to the exact local model directory. The supplied loader
uses Transformers AutoImageProcessor and AutoModelForDepthEstimation, with local
files only by default. A standalone legacy .pt file is not automatically
compatible with that loader. A compatible MiDaS/Depth Anything model export or
a separate explicitly selected loader is required.

A custom depth factory accepts a settings dictionary and returns a callable:
RGB H×W×3 float image in [0,1] -> finite normalized depth array 128×160 in [0,1].
The model must run inference without updating its weights.

## Sensor and action mapping

Explicitly configure:

- rgb_observation and ray_observation: separate Unity observation-list indices.
- sbes_indices: channels used for the scalar range feature.
- horizontal_ray_indices and vertical_ray_indices: reward clearance channels.
- ray_scale: converts normalized ray values to metres.
- goal_indices: horizontal distance, signed vertical offset, heading, in that order.
- position_indices: consistent metric 3D position ordering.
- surface_index: telemetry position relative to the surface; absolute value is used.
- heading_units: degrees or radians.
- action_scale: multipliers for normalized [vertical, yaw] commands.
- side_channel_uuid: UUID used by your Unity telemetry sender.

The UUID supplied in the template comes from the legacy repository; it is not
evidence that a particular executable uses that protocol. Indices are intentionally
left empty. Shape validation prevents RGB rows being mistaken for acoustic returns.

The side channel must provide fresh telemetry during reset and every step.
If it does not, the adapter fails explicitly rather than silently issuing an
extra action. Configure forward motion and coordinate conventions in the simulator.

## Timing and randomization

time_scale controls engine acceleration. It is not decision_period, which is
the simulated time between actions and determines the vertical-goal derivative
and reported simulation time.

Optional visibility randomization requires:

- randomize_visibility=true;
- an explicit reset_payload list;
- visibility_payload_index identifying its visibility slot;
- visibility_range=[minimum, maximum], both positive.

Visibility is sampled log-uniformly. The rest of the payload is passed unchanged.
This does not recreate randomized materials, textures, obstacle dynamics or
the original waypoint campaign; those must be configured in Unity.

The template disables this optional randomization until a verified payload is
provided. It is not a ready-made reproduction configuration.
