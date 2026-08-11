# MARL benchmark

This benchmark evaluates user-supplied MARL checkpoints with the existing
`MultiF1TenthEnvironment`, state builder, CARES policy implementations and
loaders, raw MARL action path, vehicle model, sensors, Gazebo world, and
synchronous stepping sequence. Checkpoint files are inputs, not identities
hard-coded in the repository.

## Supported checkpoints

Put each selected checkpoint directly in `train_weights/`. Its filename must
start with the CARES algorithm name followed by an underscore:

```text
<ALGORITHM>_<name>.pth
```

The prefix comparison is case-insensitive. The benchmark does not keep a fixed
algorithm whitelist: it takes the text before the first underscore as the
algorithm name and asks the existing CARES configuration/factory loader to
load it. An unknown CARES algorithm therefore fails preflight with the loader's
exact configuration error, not during filename discovery. Everything after the first
underscore is descriptive, so a seed is optional. Valid examples include:

```text
ISAC_candidate.pth
MATD3_experiment_7.pth
IPPO_agent_f1tenth_checkpoint_Seed_42.pth
```

`ISAC.pth` is invalid because it has no underscore and descriptive suffix.
`MADDPG_candidate.pth` and `TD3_candidate.pth` are accepted by
discovery; whether they can be evaluated is decided by the existing CARES
loader and MARL observation/actor-loading path.

Discovery is deliberately strict:

- Only direct `train_weights/*.pth` children are considered. Historical run
  directories are never searched.
- Multiple checkpoints with the same algorithm prefix are supported. Each file
  is a distinct competitor identified by its filename stem; for example,
  `MASAC_seed_42.pth` and `MASAC_seed_100.pth` are independently reported and
  are also paired against each other.
- A single checkpoint is sufficient for preflight and time trials.
- Head-to-head evaluation requires at least two distinct checkpoint files. Put all
  checkpoints participating in that race campaign in the folder together, or
  select variants with repeated `--checkpoint` options.

The checked-in
`f1tenth_controllers/config/marl_benchmark.json` contains the experiment and
simulation protocol only. It does not contain checkpoint filenames, hashes,
file sizes, or training seeds. At invocation time the benchmark computes each
selected file's SHA256 and size. The runtime-resolved configuration ID is
recomputed from the selected set.
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
centreline pose; races use the same centreline lane with a 1.0 m
centre-to-centre longitudinal
separation: the declared lead car is +0.5 m ahead of the common start gate and
the chaser is -0.5 m behind it.

`test_track_01_350` is deliberately excluded: its current `waypoints.py` entry
maps to `TEST_TRACK_02_WAYPOINTS`. Both `test_track_01_*.sdf` and
`test_track_02_*.sdf` worlds exist; the previous `.dsf` spelling was a typo.

The previous side-by-side smoke measurements no longer describe the active
protocol. The 1.0 m longitudinal gap is intentionally larger than the vehicle
wheelbase and is validated by unit tests; visually confirm settled placement in
the scripted pilot before the final campaign.

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

If several checkpoints are present, preflight every variant of one algorithm
with:

```bash
ros2 run f1tenth_controllers marl_benchmark preflight --algorithm MASAC
```

To preflight one exact variant, use its filename without `.pth`:

```bash
ros2 run f1tenth_controllers marl_benchmark preflight \
  --checkpoint MASAC_seed_42
```

`--algorithm` and `--checkpoint` are mutually exclusive. Preflight loads
selected actors on CPU and verifies checkpoint structure and
family, actor identity, 11 observation inputs, 2 outputs, finite actions,
exact repeated-observation determinism, and raw actions in `[-1, 1]`. It uses
CARES evaluation inference and applies no deployment smoothing, speed
controller, action denormalization, or other post-processing.

Do not start Gazebo experiments if preflight fails. Correct the filename only
when the prefix is wrong; do not rename a structurally incompatible file to
bypass the check.

## Campaign identity and result directories

By default, results are written outside the repository under
`~/f1tenth_benchmark_results/f1tenth_marl_benchmark_<manifest-id>`. Override
the base with `F1TENTH_BENCHMARK_RESULTS_DIR`, or the exact directory with
`--result-dir`. Use a new result directory for each exact checkpoint set and
experiment configuration. The manifest binds the directory to:

- checkpoint filenames, hashes, sizes, and loader metadata;
- experiment configuration and generated trial/heat IDs;
- track geometry and resolved starting poses;
- Git revisions and dirty states.

Commands are resumable only while those inputs remain identical. Completed
deterministic trial/heat IDs are skipped and duplicate result writes are
rejected. Adding, removing, replacing, or selecting a different checkpoint
changes the manifest, so use a new result directory.

For both time trials and races in one campaign, keep the same checkpoint set
in `train_weights/`. If `--result-dir` is omitted, identical manifests resolve
to the same default directory; if it is supplied, pass the same directory to
both commands. With only one uploaded checkpoint, run the time-trial workflow only.

## Docker: make results visible on the host PC

When the benchmark runs inside Docker or a VS Code Dev Container, `~` and
`Path.home()` refer to the container home. The default path is therefore
`/home/anyone/f1tenth_benchmark_results` inside the container; it is not a
hidden Ubuntu folder and it will not appear in the host Files application
unless that path is bind-mounted. A folder is hidden on Ubuntu only when its
name starts with a dot.

Before adding a bind mount, copy any existing container-only results to the
host. Run these commands in a desktop host terminal, not the VS Code container
terminal:

```bash
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Image}}"
mkdir -p "$HOME/f1tenth_benchmark_results"
docker cp <container_name>:/home/anyone/f1tenth_benchmark_results/. \
  "$HOME/f1tenth_benchmark_results/"
```

Replace `<container_name>` with the F1TENTH/VS Code container name shown by
`docker ps`. Copy existing results before mounting because a bind mount hides
the old container directory while the mount is active.

For future runs, bind the host folder to the benchmark default container path.
For Docker Compose, add this entry under the F1TENTH service `volumes` section:

```yaml
services:
  <f1tenth-service>:
    volumes:
      - ${HOME}/f1tenth_benchmark_results:/home/anyone/f1tenth_benchmark_results
```

For `.devcontainer/devcontainer.json`, add or extend `mounts`:

```json
{
  "mounts": [
    "source=${localEnv:HOME}/f1tenth_benchmark_results,target=/home/anyone/f1tenth_benchmark_results,type=bind"
  ]
}
```

For a direct `docker run`, include:

```bash
--mount type=bind,source="$HOME/f1tenth_benchmark_results",target=/home/anyone/f1tenth_benchmark_results
```

Create the host directory before rebuilding or restarting the container. Once
the mount is active, the normal benchmark command needs no `--result-dir`; its
manifest-specific directory will appear directly under the host
`~/f1tenth_benchmark_results`. Confirm from the host with:

```bash
ls -lah "$HOME/f1tenth_benchmark_results"
```

If a different mounted container path is preferred, point the benchmark to it
with `F1TENTH_BENCHMARK_RESULTS_DIR` or `--result-dir`.

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

Run one pilot trial per discovered checkpoint in terminal 2:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --pilot
```

Stop terminal 1, relaunch it with `num_opponents:=1`, and then run one pilot
heat per discovered unordered checkpoint pairing:

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
  --pilot
```

With one checkpoint, the first command runs one pilot trial and the race
command intentionally refuses to start. With `N` selected checkpoints, the pilot has
`N` time trials and `N*(N-1)/2` race heats.

## Full campaign

Launch zero opponents as in the pilot, then run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=0
ros2 run f1tenth_controllers marl_benchmark time-trials
```

This schedules `trials_per_algorithm` trials for every discovered or selected
checkpoint (the configuration key is retained for compatibility but now means
trials per checkpoint). If fewer seeds are listed, the benchmark keeps the
listed values and deterministically appends consecutive unused integers until
the requested trial count is reached. Extra listed seeds are ignored.

To evaluate only one checkpoint while several files are present:

```bash
ros2 run f1tenth_controllers marl_benchmark time-trials \
  --checkpoint ISAC_candidate
```

For head-to-head, stop and relaunch Gazebo with one opponent, keep the exact
same checkpoint selection, and run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark head-to-head
```

To race only two exact checkpoints from a folder containing more files,
repeat the
selection flag and use a result directory dedicated to that pair:

```bash
ros2 run f1tenth_controllers marl_benchmark head-to-head \
  --checkpoint MASAC_seed_42 --checkpoint MASAC_seed_100
```

For `N` selected checkpoints, the full race schedule contains
`8*N*(N-1)/2` heats: every unordered pairing, two protocol seeds, both
lead/chaser assignments, and both primary/opponent simulator-slot assignments.
Six checkpoints therefore produce 120 heats. The opponent speed multiplier is
1.0 only in the benchmark configuration; the training default remains 0.9.

## Optional Gazebo visualization

The benchmark server is launched headlessly. Attach the Gazebo GUI from a
third terminal with the same Gazebo installation sourced:

```bash
source /path/to/gz/install/setup.bash
gz sim -g
```

The benchmark terminal prints the active checkpoint ID, its algorithm, or
race pairing
before movement starts. In Gazebo, the model names are `f1tenth` for the
primary simulator slot and `opponent_1` for the opponent slot. Detailed
lead/chaser and simulator-slot assignments are recorded in
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
ros2 run f1tenth_controllers marl_benchmark_scripted time-trial
```

Then relaunch with one opponent and run:

```bash
source install/setup.bash
export ROS_DOMAIN_ID=77
export F1TENTH_NUM_OPPONENTS=1
ros2 run f1tenth_controllers marl_benchmark_scripted head-to-head
```

The race uses the same centreline path with the configured 1.0 m longitudinal
gap at 0.8 m/s and 0.5 m/s. It verifies a faster scripted car, interpolated finish ordering, and
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

Every runner prints its resolved output directory before starting. Each result
directory can contain:

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
winning along-track lead statistics in metres. Summary keys and plot labels use
checkpoint IDs, so two seeds with the same algorithm prefix are never merged.

Generated results are intentionally preserved for reproducibility; the runner
does not delete completed data. They live outside the repository by default,
and both `/results/` and `/train_weights/` are ignored inside this
repository to prevent accidental commits.

Result directories, checkpoints, ROS build products, and temporary files must
remain uncommitted.
