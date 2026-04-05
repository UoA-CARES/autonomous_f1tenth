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
cd ~/workspace/src
rm -rdf gz-sim
git clone https://github.com/UoA-CARES/gz-sim.git
cd ~/workspace
colcon build --merge-install
```

If running on the physical car, install additional dependency
```
sudo apt-get install -y ros-humble-urg-node
```

# Installation Instructions
Follow these instructions to run/test this repository on your local machine.

### Locally
Ensure you have installed the dependencies outlined above.

Clone the repository
```
git clone https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Install dependencies using `rosdep`

```
cd autonomous_f1tenth/
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

#

```
cares-rl train cli f1tenth --task CarRace SAC
ros2 launch environments environment_bringup.launch.py environment:=CarRace num_opponents:=3
gz sim -g
```
