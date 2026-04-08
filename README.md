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

Clone the automous_f1tenth repository
```
git clone https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Clone f1tenth repo into the src directory. 

```
cd ~/autonomous_f1tenth/src
git clone --recurse-submodules https://github.com/UoA-CARES/f1tenth.git
```

Install dependencies using `rosdep`

```
cd ~/autonomous_f1tenth
rosdep update -y
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

Colcon build the package

```
cd ~/autonomous_f1tenth
colcon build --symlink-install
echo "source ~/autonomous_f1tenth/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

# Run RL Training Example
To run and view training you need three sperate terminals for each command.

First run the f1tenth simulation envrionment with the current task - the example below runs CarRace with three follow the gap opponents. 
```
ros2 launch environments environment_bringup.launch.py environment:=CarRace num_opponents:=3
```

Run the learning algorithms through the command below - make sure the task matches the envrionment that you ran above.
```
cares-rl train cli f1tenth --task CarRace SAC
```

To view the training of the cars run Gazebo.
```
gz sim -g
```
