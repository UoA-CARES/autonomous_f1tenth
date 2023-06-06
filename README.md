# Autonomous F1tenth
Using reinforcement learning techniques to drive the f1tenth

# Installation Instructions
Follow these instructions to run/test this repository on your local machine.

## Using Docker (Recommended)
`git clone` the repository and all submodules
```bash
git clone --recurse-submodules https://github.com/UoA-CARES/autonomous_f1tenth.git
```

Build the docker container
```bash
. ./build.sh
```

Spin up the container using:
```bash
docker run --rm -it --network host --gpus all -e DISPLAY -e GZ_PARTITION=<partition num> -e ROS_DOMAIN_ID=<domain_id> -v "$PWD/data:/ws/data" -v "$PWD/models:/ws/models" -v "$PWD/figures:/ws/figures" autonomous_f1tenth:latest bash

# By convention, GZ_PARTITION and ROS_DOMAIN_ID are the same
```
**Note: it is important to have a different GZ_PARITION for every container you plan on running**

# Runnning the Simulations
Once inside the container:

Source the underlay:
```bash
. install/setup.bash
```

Use ros2 to launch the simulation:
```bash
# Example: If I wanted to launch the cargoal training
ros2 launch reinforcement_learning car_goal.launch.py
```
