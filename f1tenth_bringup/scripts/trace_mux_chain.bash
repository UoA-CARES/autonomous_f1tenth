#!/usr/bin/env bash
set -euo pipefail

source ~/ros2_ws/install/setup.bash

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
echo

echo "Nodes:"
ros2 node list | sort | grep -E 'ackermann_mux|joy_teleop|ackermann_to_vesc|vesc_driver' || true

echo
echo "Mux params:"
ros2 param get /ackermann_mux topics.joystick.topic || true
ros2 param get /ackermann_mux topics.joystick.timeout || true
ros2 param get /ackermann_mux topics.joystick.priority || true

echo
echo "Topic endpoints: /teleop"
ros2 topic info /teleop -v || true

echo
echo "Topic endpoints: /ackermann_cmd"
ros2 topic info /ackermann_cmd -v || true

echo
echo "Samples; hold deadman and move throttle/steering while this runs:"
echo "--- /teleop ---"
timeout 3s ros2 topic echo --once /teleop || true
echo "--- /ackermann_cmd ---"
timeout 3s ros2 topic echo --once /ackermann_cmd || true
