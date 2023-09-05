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
  export GZ_PARTITION=100
  export ROS_DOMAIN_ID=100
fi

. install/setup.bash

colcon build
ros2 launch reinforcement_learning train.launch.py &

if [ "$1" == "y" ] || [ "$1" == "Y" ]; then
  gz sim -g &
fi

wait
