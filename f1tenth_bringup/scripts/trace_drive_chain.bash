#!/usr/bin/env bash
set -u

_domain="${1:-${ROS_DOMAIN_ID:-0}}"
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${_script_dir}/use_car_domain.bash" "${_domain}" >/dev/null

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo

echo "Nodes expected on the car:"
timeout 5s ros2 node list | grep -E 'ackermann_mux|ackermann_to_vesc|vesc_driver' || true
echo

for topic in \
  /teleop \
  /ackermann_cmd \
  /commands/motor/speed \
  /commands/servo/position \
  /sensors/core \
  /sensors/servo_position_command; do
  echo "=== ${topic} ==="
  timeout 5s ros2 topic info -v "${topic}" || true
  echo
done

echo "Ackermann mux parameters:"
timeout 5s ros2 param get /ackermann_mux topics.joystick.topic || echo "topics.joystick.topic unavailable or timed out"
timeout 5s ros2 param get /ackermann_mux topics.joystick.priority || echo "topics.joystick.priority unavailable or timed out"
timeout 5s ros2 param get /ackermann_mux topics.joystick.timeout || echo "topics.joystick.timeout unavailable or timed out"
echo

echo "VESC parameters:"
timeout 5s ros2 param get /vesc_driver_node port || echo "vesc port unavailable or timed out"
timeout 5s ros2 param get /ackermann_to_vesc_node speed_to_erpm_gain || echo "speed_to_erpm_gain unavailable or timed out"
timeout 5s ros2 param get /ackermann_to_vesc_node steering_angle_to_servo_gain || echo "steering_angle_to_servo_gain unavailable or timed out"

echo
echo "Message samples; hold deadman and move throttle while this runs:"
for topic in /teleop /ackermann_cmd /commands/motor/speed /commands/servo/position; do
  echo "--- ${topic} sample ---"
  timeout 3s ros2 topic echo --once "${topic}" || true
  echo
done
