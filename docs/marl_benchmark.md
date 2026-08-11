# MARL benchmark

This benchmark evaluates only the six selected checkpoints in
`train_weights/`. It reuses the existing `MultiF1TenthEnvironment`,
state builder, CARES policy classes/loaders, raw MARL action path, vehicle
model, sensors, Gazebo world, and synchronous stepping sequence.

## Resolved experiment

The checked-in declaration is
`f1tenth_controllers/config/marl_benchmark.json`. It maps:

- MATD3: `MATD3_f1tenth_checkpoint_Seed_100.pth`
- MAPPO: `MAPPO_f1tenth_checkpoint_Seed_42.pth`
- MASAC: `MASAC_f1tenth_checkpoint_Seed_42.pth`
- ITD3: `ITD3_agent_f1tenth_checkpoint_Seed_100.pth`
- IPPO: `IPPO_agent_f1tenth_checkpoint_Seed_42.pth`
- ISAC: `ISAC_agent_f1tenth_checkpoint_Seed_456.pth`

The file sizes and SHA256 values in that declaration are mandatory. A
missing or mismatched file stops preflight; the loader never searches old
training runs or substitutes a different checkpoint.

The selected world and waypoint identifier are both
`test_track_02_350`. The centreline is counter-clockwise and has a
waypoint-polyline length of approximately 179.143 m. Waypoint 10 is the
fixed start. Time trials use its centreline pose; races use equal lateral
offsets of +0.30 m and -0.30 m.

`test_track_01_350` is deliberately excluded: its current
`waypoints.py` entry maps to `TEST_TRACK_02_WAYPOINTS`. Both
`test_track_01_*.sdf` and `test_track_02_*.sdf` worlds exist; the
previous `.dsf` spelling was a typo.

The Gazebo smoke test observed 0.617 m between the two settled vehicle
centres, minimum LiDAR returns of 0.547 m and 0.555 m, and no initial
collision signal.

## Build and checkpoint preflight

From the ROS workspace:

```bash
colcon build --symlink-install --packages-select \
  f1tenth_environments f1tenth_controllers
source install/setup.bash
ros2 run f1tenth_controllers marl_benchmark preflight
```

Preflight loads all six actors on CPU and verifies checkpoint structure,
actor identity, 11 observation inputs, 2 outputs, finite actions, exact
repeatability, and raw actions in [-1, 1]. It uses CARES evaluation
inference and applies no deployment smoothing, speed controller, action
denormalization, or other post-processing.

## ROS transport isolation

Use the declared `ROS_DOMAIN_ID=77` for both the simulator launch and
benchmark command. This isolates DDS discovery from unrelated ROS/VS Code
participants. It does not change Gazebo physics or simulator time. The
benchmark refuses another domain.

If ROS processes were forcibly killed and Fast DDS reports shared-memory
port-lock errors, stop all benchmark simulator processes before running:

```bash
fastdds shm clean
```

The benchmark also applies a five-wall-second timeout to its own Gazebo
world-control and set-pose service calls. Training retains its original
unbounded service wait.

## Pilot

Pilot results use a distinct configuration ID and a declared 30
simulator-second timeout. They cannot be mixed with a full-campaign
directory. The short timeout is for reset, observation, action, clock,
collision, and result-pipeline smoke testing; it is not a lap-performance
result.

For six one-car pilot trials, use two terminals:

```bash
# Terminal 1
source install/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py \
  track:=test_track_02_350 num_opponents:=0 marl_env:=true
```

```bash
# Terminal 2
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --pilot --result-dir /path/to/pilot_results
```

Stop the first launch, then run the 15-pair race pilot:

```bash
# Terminal 1
source install/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py \
  track:=test_track_02_350 num_opponents:=1 marl_env:=true
```

```bash
# Terminal 2
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --pilot --result-dir /path/to/pilot_results
```

## Scripted Gazebo integration checks

The separate scripted command adapts the repository's unchanged
`PurePursuit` class to the same synchronous environment and lap/race runner.
It is verification tooling, not a trained-policy action-processing stage.

With zero opponents launched as for a time trial, run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark_scripted time-trial \
  --result-dir /path/to/scripted_results
```

Then relaunch with one opponent and run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark_scripted head-to-head \
  --result-dir /path/to/scripted_results
```

The race uses parallel +0.30 m and -0.30 m waypoint paths at 0.8 m/s and
0.5 m/s. It verifies a faster scripted car, interpolated finish ordering,
and along-track lead without changing the world or vehicle physics.

## Full campaign

Use a new, empty result directory. Launch zero opponents as above. In the
benchmark terminal, use the same setup and transport settings as the pilot:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --result-dir /path/to/full_results
```

This schedules 10 trials for each of the six algorithms. Then restart the
same world with one opponent and run in the benchmark terminal:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --result-dir /path/to/full_results
```

The full race schedule contains 120 heats: 15 unordered pairings, two
seeds, both left/right assignments, and both primary/opponent simulator
slot assignments. The opponent speed multiplier is 1.0 only in the
benchmark configuration; the training default remains 0.9.

Commands are resumable. Existing deterministic trial/heat IDs are skipped,
and duplicate result writes are rejected. A directory with a different
manifest is rejected.

## Timing and outcomes

Official time comes from Gazebo `/clock`. The start is the simulator time
immediately before the first joint command publication after reset and
fresh sensor readiness. Finish time is interpolated between synchronized
samples.

A valid lap must pass 25%, 50%, and 75% virtual sectors in order, accumulate
a full forward lap, cross in the correct direction, and avoid projection
jumps above the declared 1.0 m projection slack plus the 5.0 m/s training
physical speed limit multiplied by the actual simulator-time sample delta.
Backward crossings and repeated finish crossings are not accepted.

Race actions are computed sequentially in one process from the same
observation dictionary, then passed together to one environment step.
Finish leads use interpolated along-track progress, not Euclidean distance.

Collision attribution uses only existing training signals. A separated
single-car collision is attributed to that car; a supported rear-end event
may be attributed to the closing trailing car. Side contact and otherwise
ambiguous events remain `indeterminate`. The LiDAR collision threshold
does not identify which physical body was contacted.

## Artifacts and summaries

Each result directory can contain:

- `run_manifest.json`
- `time_trial_trials.csv`
- `head_to_head_trials.csv`
- `events.jsonl`
- regenerated JSON summaries and PNG plots

Generate summaries with:

```bash
ros2 run f1tenth_controllers marl_benchmark_summary \
  --result-dir /path/to/results
```

Time-trial summaries report completion rate and DNF counts separately from
completed-lap mean, median, sample standard deviation, and a two-sided 95%
Student-t confidence interval. DNFs never receive artificial lap times.
Head-to-head summaries retain outcome and pairing counts, win rates, and
winning along-track lead statistics in metres.

Result directories, checkpoints, ROS build products, and temporary files
must remain uncommitted.
