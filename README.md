# Autonomous F1tenth Gym v2.0
Using reinforcement learning techniques to drive the f1tenth vehicle platform - head-to-head and time trial racing!

![F1tenth Car](media/f1tenth-min.png)

# Docker Installation (Development)
These instructions will setup the f1tenth environment inside of a Docker container pre-installing all required dependencies.

Follow the instructions at the Docker site to install Docker https://docs.docker.com/engine/install/ubuntu/

## Build the Image
The first step is to pull the code base and build the Docker image to the local computer. 

We will create a folder to store the code locally - this way the code is always accesible outside of the container.

```bash
mkdir -p ~/f1tenth_docker
```

Clone the `autonomous_f1tenth` repository (dev/v2 branch):
```bash
cd ~/f1tenth_docker
git clone --branch dev/v2 https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Clone the `f1tenth` repository as a sibling workspace package (outside this repository).
```bash
cd ~/f1tenth_docker
git clone --recurse-submodules https://github.com/UoA-CARES/f1tenth.git
```

The development image in `Dockerfile.sim` will setup all the external dependencies automatically for you. The command below will build the Docker image for you. 

```bash
cd ~/f1tenth_docker/autonomous_f1tenth

docker build -t f1tenth:dev \
  -f Dockerfile.sim \
  --no-cache \
  --build-arg USER_NAME=anyone \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .
```

You will need to add below to the `~/.bashrc` to enable screen sharing between Docker and the host.

``` bash
echo "xhost +local:docker" >> ~/.bashrc
source ~/.bashrc
```

## Run a Container
The second step runs a container from the image for you to work in - you can remove and re-create the containers, only need to create the image once. 

To run the Docker image and mount the code effectively run the command below

```bash
docker run -dit \
  --name f1_dev \
  --network host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --ipc=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/f1tenth_docker:/home/anyone/ros2_ws/src" \
  f1tenth:dev
```

To finalise the installation process you need to do these steps manually (inside the Docker container). To enter the Docker container run this below:

```bash
docker exec -it f1_dev bash
```

The command above puts you in an active shell within the Docker container - you are now operating within the container not the host machine.

Install dependencies using `rosdep`
```bash
cd ~/ros2_ws
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

Colcon build the package
```bash
cd ~/ros2_ws
colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

To run the training instances you can use the Quick Start instructions below - from within the Docker container using the command below on the new terminal windows first.

```bash
docker exec -it f1_dev bash
```

## Delete Container
If the container gets broken by installations or other issues

```bash
docker rm f1_dev -f
```

Then rebuild the container using the instructions above.

# Source Installation Instructions
Follow these steps to set up the Autonomous F1tenth Gym v2.0 on your system. The instructions below will guide you through installing all required dependencies, cloning the necessary repositories, and building the workspace to get started with simulation and reinforcement learning.

## Dependencies
Before proceeding, ensure your system meets the following software requirements. These dependencies are essential for running the simulation environment and reinforcement learning workflows. Follow the provided links for installation instructions and version details.

## Gazebo Garden  (source)
We source build Gazebo Garden, and use a forked `gz-sim` that fixes some issues with Ackermann control in the base version. Follow the instructions for installing Gazebo Garden from source [here](https://gazebosim.org/docs/garden/install_ubuntu_src) then follow the instructions below to replace 'gz-sim' with our custom version.

```
cd ~/workspace/src
rm -rdf gz-sim
git clone https://github.com/UoA-CARES/gz-sim.git
cd ~/workspace
colcon build --merge-install
echo "source ~/workspace/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## ROS 2.0 Humble
Follow the instructions to install ROS 2.0 Humble from the main installation instructions: [Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html).

## CARES Reinforcement Learning
The CARES RL package provides the primary set of training algorithms and tools for reinforcement learning with the F1tenth Gym. While the gym environment is compatible with other custom RL frameworks or scripts, CARES RL offers a streamlined interface and ready-to-use implementations for most users. Note that CARES RL is not a strict installation dependency for the gym itself, but is recommended for standard training workflows.

Follow the instructions to install the CARES Reinforcement Learning package from the main installation instructions: [CARES RL v3.1.0](https://github.com/UoA-CARES/cares_reinforcement_learning/tree/V3.1.0).

### Car positions in `step`/`reset` info (for racing-line video)

Every `step()` and `reset()` call (single-agent `F1tenthEnvironment` and
multi-agent `MultiF1TenthEnvironment` alike) includes each car's world-frame
XY position - subscribed from the same simulator odometry already used for
race-position tracking - in the returned `info` dict:

- `info["position_xy"]`: the agent's own `(x, y)` in world metres.
- `info["positions_xy"]`: `{car_name: (x, y), ...}` for every car in the race.
- `info["<other_car>_position_xy"]`: the same per-opponent, flattened -
  mirrors the existing `distance_to_<other_car>` convention.

These are world-frame (not track-relative like `agent_track_position`), so a
consumer can plot each car's raw racing line directly and step through it
frame-by-frame to build a video of the race, the way `drone_gym`'s
`episode_positions` feeds its own video generation. This repo only exposes
the positions; plotting/video assembly is intentionally left to the RL side.

## Package Installation
Follow these instructions to run/test this repository on your local machine. Ensure you have installed the dependencies outlined above.

These instructions assume you are using '~/ros2_ws/src' as your ROS2 workspace. Please adjust those commands as required if you are using a different workspace folder.

Make the `~/ros2_ws/src` directory

```bash
mkdir -p ~/ros2_ws/src
```

Clone the autonomous_f1tenth repository (dev/v2 branch):
```bash
cd ~/ros2_ws/src
git clone --branch dev/v2 https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Clone the `f1tenth` repository as a sibling workspace package (outside this repository).
```bash
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/UoA-CARES/f1tenth.git
```

Install dependencies using `rosdep`
```bash
cd ~/ros2_ws
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

Colcon build the package
```bash
cd ~/ros2_ws
colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

# Quick Start

To run and view RL training, use up to four separate terminals (one command per terminal):

![Gazebo Viz](media/f1tenth-gz.gif)

**1. Launch the simulation environment:**

This starts the simulation with your chosen track and number of opponents. The main arguments are:

- `track`: The name of the track/world to load (e.g., `multi_track`, `track_1`, etc.)
- `num_opponents`: The number of opponent cars to spawn (the agent car is always present, so total cars = 1 + num_opponents)

Example (agent car + 2 opponents = 3 cars total):
```
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py track:=multi_track num_opponents:=2 marl_env:=false
```

MARL example (agent car + 3 learned opponents = 4 cars total):
```
ros2 launch f1tenth_bringup sim_environment_bringup.launch.py track:=multi_track_01 num_opponents:=3 marl_env:=true
```

**2. Start RL training:**

The RL agent (e.g., CARES RL) will instantiate the environment using EnvironmentFactory and pass all RL/environment parameters via the config argument. 

Single-agent training:
```
cares-rl train cli f1tenth --task CarRace SAC
```

Multi-agent training:
```
F1TENTH_TRACK=multi_track_01 F1TENTH_NUM_OPPONENTS=3 cares-rl train cli multi_f1tenth --task MultiCarRace MATD3
```

For MARL, use the `multi_f1tenth` gym with `MultiCarRace`. Using `multi_f1tenth --task CarRace` creates the single-car environment and MARL algorithms will fail because the environment has no multi-agent `agents` metadata. `F1TENTH_NUM_OPPONENTS` must match the simulation launch argument. MARL training resets choose one loaded sub-track per episode and randomly assign all agent identities to the available start slots, so the primary car can start in any race position. Set `F1TENTH_ACTIVE_TRACK=<track_key>` to lock training to one sub-track. `F1TENTH_OPPONENT_START_GAP` and `F1TENTH_OPPONENT_GAP` tune the start-slot spacing; defaults use the base waypoint followed by +8, +12, and +16 waypoints. Evaluation resets retain their deterministic ordering.

For reproducible evaluation of user-supplied MARL checkpoints, see
[the MARL benchmark runbook](docs/marl_benchmark.md).

**3. (Optional) View simulation in Gazebo:**

```
gz sim -g
```

**4. (Optional) Visualization with RViz**

To visualize the simulation and topics, launch RViz with the provided configuration:

```bash
ros2 launch f1tenth_bringup rviz.launch.py
```

The RViz configuration file is located at:
```
f1tenth_bringup/rviz/f1tenth.rviz
```
You can customize this file to suit your visualization needs.


## Running Multiple Instances in Parallel

To run multiple independent training or simulation instances on the same machine, set unique ROS and Gazebo communication domains for each instance (set of terminals for each run):

```bash
export ROS_DOMAIN_ID=42   
export GZ_PARTITION=42
```

You can use any integer value (e.g., 42, 43, 44, ...) as long as each parallel instance uses a different value. This ensures that ROS 2 and Gazebo messages do not interfere between runs.
