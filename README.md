# Autonomous F1tenth
Using reinforcement learning techniques to drive the f1tenth vehicle platform.

### Dependencies
| Dependencies | Version |
| ----------- | ----------- |
| Gazebo | [Garden](https://gazebosim.org/docs/garden/install_ubuntu_src) |
| ROS2 | [Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html) |
| CARES RL | [Link](https://github.com/UoA-CARES/cares_reinforcement_learning) |

We source build Gazebo Garden, and use a forked `gz-sim`. To use the forked `gz-sim` run the following command before building Gazebo

```
cd ~/gz/src
rm -rdf gz-sim
git clone https://github.com/UoA-CARES/gz-sim.git
cd ~/gz
colcon build --merge-install
echo "source ~/gz/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

# Package Installation Instructions
Follow these instructions to run/test this repository on your local machine.

Ensure you have installed the dependencies outlined above.

Clone the autonomous_f1tenth repository.
```
git clone https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Clone the `f1tenth` repository as a sibling workspace package (outside this repository).

```
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/UoA-CARES/f1tenth.git
```

Install dependencies using `rosdep`

```
cd ~/ros2_ws
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

Colcon build the package

```
cd ~/ros2_ws
colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

# Run RL Training Example
To run and view training, use three separate terminals (one command per terminal).

First, run the f1tenth simulation environment with the current task. The example below runs CarRace with three follow-the-gap opponents.
```
ros2 launch autonomous_bringup environment_bringup.launch.py environment:=CarRace num_opponents:=3
```

Run the learning algorithm using the command below. Ensure the task matches the environment started above.
```
cares-rl train cli f1tenth --task CarRace SAC
```

To view the training of the cars run Gazebo.
```
gz sim -g
```
