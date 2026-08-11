# MARL benchmark

This benchmark evaluates user-supplied MARL checkpoints with the existing
`MultiF1TenthEnvironment`, state builder, CARES policy implementations and
loaders, raw MARL action path, vehicle model, sensors, Gazebo world, and
synchronous stepping sequence. Checkpoint files are inputs, not identities
hard-coded in the repository.

## Supported checkpoints

Put each selected checkpoint directly in `train_weights/`. Its filename must
start with one supported algorithm name followed by an underscore:

```text
<ALGORITHM>_<name>.pth
```

Supported prefixes are `MATD3`, `MAPPO`, `MASAC`, `ITD3`, `IPPO`, and `ISAC`.
The prefix comparison is case-insensitive. Everything after the first
underscore is descriptive, so a seed is optional. Valid examples include:

```text
ISAC_candidate.pth
MATD3_experiment_7.pth
IPPO_agent_f1tenth_checkpoint_Seed_42.pth
```

`ISAC.pth` is invalid because it has no underscore and descriptive suffix.
`TD3_candidate.pth` is invalid because TD3 is not one of the supported MARL
algorithms.

Discovery is deliberately strict:

- Only direct `train_weights/*.pth` children are considered. Historical run
  directories are never searched.
- At most one checkpoint may be present for each algorithm. Multiple files
  with the same algorithm prefix stop the command instead of selecting one.
- A single checkpoint is sufficient for preflight and time trials.
- Head-to-head evaluation requires at least two different algorithms. Put all
  checkpoints participating in that race campaign in the folder together, or
  select two from the folder with repeated `--algorithm` options.

The checked-in
`f1tenth_controllers/config/marl_benchmark.json` contains the experiment and
simulation protocol only. It does not contain checkpoint filenames, hashes,
file sizes, or training seeds. At invocation time the benchmark computes each
selected file's SHA256 and size. The runtime-resolved configuration ID is recomputed from the selected set.
Those values, the detected algorithm,
`f1tenth` actor identity, loader architecture, dimensions, and checkpoint
structure are written to `run_manifest.json`.

A filename prefix does not override checkpoint validation. For example, a
file named `ISAC_candidate.pth` whose structure is TD3-like fails preflight
with the detected structural mismatch.

## Resolved experiment

The selected world and waypoint identifier are both `test_track_02_350`. The
centreline is counter-clockwise and has a waypoint-polyline length of
approximately 179.143 m. Waypoint 10 is the fixed start. Time trials use its
centreline pose; races use equal lateral offsets of +0.30 m and -0.30 m.

`test_track_01_350` is deliberately excluded: its current `waypoints.py` entry
maps to `TEST_TRACK_02_WAYPOINTS`. Both `test_track_01_*.sdf` and
`test_track_02_*.sdf` worlds exist; the previous `.dsf` spelling was a typo.

The Gazebo smoke test observed 0.617 m between the two settled vehicle
centres, minimum LiDAR returns of 0.547 m and 0.555 m, and no initial collision
signal.

## Build and checkpoint preflight

Run commands from the ROS workspace containing this repository:

```bash
cd /path/to/ros2_ws
source /opt/ros/humble/setup.bash
source /path/to/gz/install/setup.bash
colcon build --symlink-install --packages-select \
  f1tenth_environments f1tenth_controllers
source install/setup.bash
```

Preflight every checkpoint currently in `train_weights/`:

```bash
ros2 run f1tenth_controllers marl_benchmark preflight
```

If several checkpoints are present, preflight only a named algorithm with:

```bash
ros2 run f1tenth_controllers marl_benchmark preflight --algorithm ISAC
```

Preflight loads selected actors on CPU and verifies checkpoint structure and
family, actor identity, 11 observation inputs, 2 outputs, finite actions,
exact repeated-observation determinism, and raw actions in `[-1, 1]`. It uses
CARES evaluation inference and applies no deployment smoothing, speed
controller, action denormalization, or other post-processing.

Do not start Gazebo experiments if preflight fails. Correct the filename only
when the prefix is wrong; do not rename a structurally incompatible file to
bypass the check.

## Campaign identity and result directories

Use a new result directory for each exact checkpoint set and experiment
configuration. The manifest binds the directory to:

- checkpoint filenames, hashes, sizes, and loader metadata;
- experiment configuration and generated trial/heat IDs;
- track geometry and resolved starting poses;
- Git revisions and dirty states.

Commands are resumable only while those inputs remain identical. Completed
deterministic trial/heat IDs are skipped and duplicate result writes are
rejected. Adding, removing, replacing, or selecting a different checkpoint
changes the manifest, so use a new result directory.

For both time trials and races in one campaign, keep the same checkpoint set
in `train_weights/` and use the same result directory for both commands. With
only one uploaded checkpoint, run the time-trial workflow only.

## ROS and Gazebo isolation

Use `ROS_DOMAIN_ID=77` for both the simulator launch and benchmark command.
This isolates DDS discovery from unrelated ROS/VS Code participants. It does
not change Gazebo physics or simulator time. The benchmark refuses another
domain.

Before a run, make sure only one Gazebo server supplies `/clock`:

```bash
ros2 topic info /clock -v
```

The publisher count must be one. Multiple simulator clocks can produce
backward command/observation timestamps and invalidate a run. If forcibly
killed ROS processes leave Fast DDS shared-memory lock errors, stop all
benchmark simulator processes and then run:

```bash
fastdds shm clean
```

The benchmark applies a five-wall-second timeout to its own Gazebo
world-control and set-pose service calls. Training retains its original
unbounded service wait.

## Pilot

Pilot results use a distinct configuration ID and a declared 30
simulator-second timeout. They cannot be mixed with a full-campaign directory.
The pilot checks reset, observation, action, clock, collision, and result
plumbing; it is not a lap-performance result.

Start the time-trial simulator in terminal 1:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py \
  track:=test_track_02_350 num_opponents:=0 marl_env:=true
```

Run one pilot trial per discovered algorithm in terminal 2:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --pilot --result-dir /path/to/pilot_results
```

Stop terminal 1, relaunch it with `num_opponents:=1`, and then run one pilot
heat per discovered unordered algorithm pairing:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py \
  track:=test_track_02_350 num_opponents:=1 marl_env:=true
```

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --pilot --result-dir /path/to/pilot_results
```

With one checkpoint, the first command runs one pilot trial and the race
command intentionally refuses to start. With `N` checkpoints, the pilot has
`N` time trials and `N*(N-1)/2` race heats.

## Full campaign

Launch zero opponents as in the pilot, then run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --result-dir /path/to/full_results
```

This schedules 10 trials for every discovered or explicitly selected
algorithm. To evaluate only one checkpoint while several files are present:

```bash
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --algorithm ISAC --result-dir /path/to/isac_results
```

For head-to-head, stop and relaunch Gazebo with one opponent, keep the exact
same checkpoint selection, and run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --result-dir /path/to/full_results
```

To race only two algorithms from a folder containing more files, repeat the
selection flag and use a result directory dedicated to that pair:

```bash
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --algorithm ISAC --algorithm MATD3 \
  --result-dir /path/to/isac_vs_matd3_results
```

For `N` selected algorithms, the full race schedule contains
`8*N*(N-1)/2` heats: every unordered pairing, two protocol seeds, both
left/right assignments, and both primary/opponent simulator-slot assignments.
Six algorithms therefore produce 120 heats. The opponent speed multiplier is
1.0 only in the benchmark configuration; the training default remains 0.9.

## Optional Gazebo visualization

The benchmark server is launched headlessly. Attach the Gazebo GUI from a
third terminal with the same Gazebo installation sourced:

```bash
source /path/to/gz/install/setup.bash
gz sim -g
```

The benchmark terminal prints the active time-trial algorithm or race pairing
before movement starts. In Gazebo, the model names are `f1tenth` for the
primary simulator slot and `opponent_1` for the opponent slot. Detailed
left/right and simulator-slot assignments are recorded in
`head_to_head_trials.csv`.

Rendering can increase wall-clock duration. Official results use Gazebo
simulator time, but headless execution remains preferable for the final
campaign after visual validation.

## Scripted Gazebo integration checks

The separate scripted command adapts the repository's unchanged `PurePursuit`
class to the same synchronous environment and lap/race runner. It is
verification tooling, not a trained-policy action-processing stage.

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
0.5 m/s. It verifies a faster scripted car, interpolated finish ordering, and
along-track lead without changing the world or vehicle physics.

## Timing and outcomes

Official time comes from Gazebo `/clock`. The start is the simulator time
immediately before the first joint command publication after reset and fresh
sensor readiness. Finish time is interpolated between synchronized samples.

A valid lap must pass 25%, 50%, and 75% virtual sectors in order, accumulate a
full forward lap, cross in the correct direction, and avoid projection jumps
above the declared 1.0 m projection slack plus the 5.0 m/s training physical
speed limit multiplied by the actual simulator-time sample delta. Backward
crossings and repeated finish crossings are not accepted.

Race actions are computed sequentially in one process from the same
observation dictionary, then passed together to one environment step. Finish
leads use interpolated along-track progress, not Euclidean distance.

Collision attribution uses only existing training signals. A separated
single-car collision is attributed to that car; a supported rear-end event
may be attributed to the closing trailing car. Side contact and otherwise
ambiguous events remain `indeterminate`. The LiDAR collision threshold does
not identify which physical body was contacted.

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

Result directories, checkpoints, ROS build products, and temporary files must
remain uncommitted.
