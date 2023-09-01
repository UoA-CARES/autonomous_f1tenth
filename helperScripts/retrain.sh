#!/bin/bash

# Ctrl + C to exit the program
function cleanup() {
  return
}

# trap cleanup SIGINT SIGTERM
trap cleanup SIGINT SIGTERM

cd ..

# Time limit for training
if [ -z "$1" ]; then
  echo "Please enter the time limit for training"
  echo "Usage: ./retrain.sh <time_limit> [<partition_number>]"
  echo "Eg: 10s, 6h"
  echo "Usage: ./retrain.sh 10s"
  return
fi

# Set the partition number
if [ -n "$2" ]; then
	export GZ_PARTITION="$2"
fi

. install/setup.bash

# Rerun every $1 time
while true; do
	colcon build
	timeout --signal=SIGKILL "$1" ros2 launch reinforcement_learning train.launch.py

  # Get the latest folder in rl_logs
	latest_folder=$(ls -lt ./rl_logs | grep '^d' | head -n 1 | awk '{print $NF}')

	# Set the new paths for actor_path and critic_path
	new_actor_path="rl_logs/$latest_folder/models/actor_checkpoint.pht"
	new_critic_path="rl_logs/$latest_folder/models/critic_checkpoint.pht"

  # Check if the new paths exist
	if [ ! -e "$new_actor_path" ] || [ ! -e "$new_critic_path" ]; then
    echo "Error: $new_actor_path or $new_critic_path does not exist"
    return
  fi

	# Use sed to update the actor_path and critic_path in the YAML file
	sed -i "s#actor_path: '.*'#actor_path: '$new_actor_path'#g" src/reinforcement_learning/config/train.yaml
	sed -i "s#critic_path: '.*'#critic_path: '$new_critic_path'#g" src/reinforcement_learning/config/train.yaml
done
