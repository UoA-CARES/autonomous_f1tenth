#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  return
}

# trap cleanup SIGINT SIGTERM
trap cleanup SIGINT SIGTERM

cd ~/autonomous_f1tenth/

if [ -n "$1" ]; then
  export GZ_PARTITION=$1
  export ROS_DOMAIN_ID=$1
else
  export GZ_PARTITION=50
  export ROS_DOMAIN_ID=50
fi

. install/setup.bash

colcon build
ros2 launch reinforcement_learning sanity_check.launch.py &

gz sim -g &

wait
