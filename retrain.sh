#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  exit 1
}

trap cleanup SIGINT SIGTERM

# Time limit for training
if [ -z "$1" ]; then
  echo "Please enter the time limit for training, usage: ./retrain.sh <time_limit> [<partition_number>]"
  echo "Eg: 10s, 6h -> usage: ./retrain.sh 10s"
  exit 1
fi

# Set the partition number
if [ -n "$2" ]; then
	export GZ_PARTITION="$2"
else
	export GZ_PARTITION=150
fi

. install/setup.bash

# Rerun every $1 time
while true; do
	colcon build
	timeout "$1" gz sim -g &
	timeout "$1" ros2 launch reinforcement_learning train.launch.py

	# Set the new paths for actor_path and critic_path
	new_actor_path=""
	new_critic_path=""

	# Use sed to update the actor_path and critic_path in the YAML file
	sed -i "s#actor_path: '.*'#actor_path: '$new_actor_path'#g" src/reinforcement_learning/config/train.yaml
	sed -i "s#critic_path: '.*'#critic_path: '$new_critic_path'#g" src/reinforcement_learning/config/train.yaml
done

