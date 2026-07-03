#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_setup_dir="${_script_dir}"
while [ "${_setup_dir}" != "/" ]; do
  if [ -f "${_setup_dir}/install/setup.bash" ]; then
    # colcon setup files may reference unset variables, so keep this script
    # strict while sourcing them safely.
    _restore_nounset=0
    case "$-" in
      *u*)
        _restore_nounset=1
        set +u
        ;;
    esac
    source "${_setup_dir}/install/setup.bash"
    if [ "${_restore_nounset}" = "1" ]; then
      set -u
    fi
    break
  fi
  _setup_dir="$(dirname "${_setup_dir}")"
done

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 command not found; source your workspace install/setup.bash first." >&2
  exit 1
fi

echo "Workspace setup: ${_setup_dir}/install/setup.bash"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}"
echo

echo "Nodes:"
ros2 node list | sort | grep -E 'ackermann_mux|joy_teleop|ackermann_to_vesc|vesc_driver' || true

echo
echo "Mux node info:"
ros2 node info /ackermann_mux || true

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
