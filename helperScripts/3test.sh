#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  return
}

# trap cleanup SIGINT SIGTERM
trap cleanup SIGINT SIGTERM

cd ~/autonomous_f1tenth/

export GZ_PARTITION=150
export ROS_DOMAIN_ID=150

. install/setup.bash

colcon build
ros2 launch reinforcement_learning test.launch.py &

wait
