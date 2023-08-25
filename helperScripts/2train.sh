#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  return
}

# trap cleanup SIGINT SIGTERM
trap cleanup SIGINT SIGTERM

cd ~/autonomous_f1tenth/

export GZ_PARTITION=100
export ROS_DOMAIN_ID=100

. install/setup.bash

colcon build
ros2 launch reinforcement_learning train.launch.py &

gz sim -g &

wait
