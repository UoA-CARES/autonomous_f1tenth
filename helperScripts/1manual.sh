#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  return
}

# trap cleanup SIGINT SIGTERM
trap cleanup SIGINT SIGTERM

cd ~/autonomous_f1tenth/

export GZ_PARTITION=50
export ROS_DOMAIN_ID=50

. install/setup.bash

colcon build
ros2 launch reinforcement_learning sanity_check.launch.py &

wait
