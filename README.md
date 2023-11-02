# Autonomous F1tenth
Using reinforcement learning techniques to drive the f1tenth

### This repository uses
| Dependencies | Version |
| ----------- | ----------- |
| Gazebo | [Garden](https://gazebosim.org/docs/garden/install_ubuntu_src) |
| ROS2 | [Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html) |
| CARES RL |[Link](https://github.com/UoA-CARES/cares_reinforcement_learning) |

We source build Gazebo Garden, and use a forked `gz-sim` – it's best just to use the docker

# Installation Instructions
Follow these instructions to run/test this repository on your local machine.

## Locally
Ensure you have installed the dependencies outlined above.

Install colcon
```
sudo apt install python3-colcon-common-extensions
```

To use the forked gz-sim run the following command before building gazebo
```
rm -rdf gz-sim
git clone https://github.com/UoA-CARES/gz-sim.git
```

Clone the repository
```
git clone --recurse-submodules https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Install the dependencies using `rosdep`
```
cd autonomous_f1tenth/
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y
```

If running on the physical car, install additional dependency
```
sudo apt-get install -y ros-humble-urg-node
```

## Using Docker (Recommended)

Use this command to run the docker, it will automatically pull the image if not found locally:
```
docker run --rm -it --network host --gpus all -e DISPLAY -e GZ_PARTITION=12 -e ROS_DOMAIN_ID=12 -v "$PWD/rl_logs:/ws/rl_logs" caresrl/autonomous_f1tenth:latest bash
```
**Note: it is important to have a different GZ_PARITION/ROS_DOMAIN_ID for every container you plan on running**

# Runnning the Simulations
There are several **Reinforcement Learning** Environments that are available. Refer to the wiki for more detailed information.
| Environment      | Task | Observation |
| ----------- | ----------- | ----------- |
| CarGoal      | Drive to a goal position       | Odometry, Goal Position |
| CarWall   | Drive to a goal position inside a walled area | Odometry, Lidar, Goal Position |
| CarBlock   | Drive to a goal position, with static obstacles randomly placed | Odometry, Lidar, Goal Position |
| CarTrack   | Drive around a track | Orientation, Lidar |
| CarBeat   | Overtake a Car | _Still in progress_|

This repository provides two functions:
1. **Training** a reinforcement learning agent on a particular _environment_
2. **Testing** (evaluating) your trained agent on a particular _environment_

To control **aspects** of the **training/testing** – eg. environment, hyperparameters, model paths – you edit the `src/reinforcement_learning/config/train.yaml` and `src/reinforcement_learning/config/test.yaml` files respectively.

An example of the yaml that controls training is shown below:
```
train:
  ros__parameters:
    environment: 'CarBeat' # CarGoal, CarWall, CarBlock, CarTrack, CarBeat
    max_steps_exploration: 5000
    track: 'track_2' # track_1, track_2, track_3 -> only applies for CarTrack
    max_steps_exploration: 1000
    max_steps_training: 1000000
    reward_range: 1.0
    collision_range: 0.2
    observation_mode: 'no_position'
    # gamma: 0.95
    # tau: 0.005
    # g: 5
    # batch_size: 32
    # buffer_size: 1000000
    # actor_lr: 0.0001
    # critic_lr: 0.001
    # max_steps: 10
    # step_length: 0.25
    # seed: 123
```

Once you have configured your desired experiment you can run the following to launch your training/testing:
```bash
colcon build
. install/setup.bash
ros2 launch reinforcement_learning train.launch.py # or test.launch.py if you were testing your agents
```
Refer to this [link](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html) for more information on `colcon build` in `ros2`
