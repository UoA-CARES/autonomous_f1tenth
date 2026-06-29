# Source this in a shell before using ros2 CLI tools with a car domain.
# Usage: source /home/anyone/ros2_ws/install/f1tenth_bringup/share/f1tenth_bringup/scripts/use_car_domain.bash 2

_domain="${1:-${ROS_DOMAIN_ID:-0}}"
_rmw="${2:-rmw_cyclonedds_cpp}"

if [ -f /home/anyone/ros2_ws/install/setup.bash ]; then
  source /home/anyone/ros2_ws/install/setup.bash
fi

export ROS_DOMAIN_ID="${_domain}"
export RMW_IMPLEMENTATION="${_rmw}"
export ROS_LOCALHOST_ONLY="0"

# ros2cli uses a daemon for graph queries. If it was started on another
# domain/RMW, topic lists can look intermittent even when the nodes are fine.
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

printf 'ROS_DOMAIN_ID=%s\n' "$ROS_DOMAIN_ID"
printf 'RMW_IMPLEMENTATION=%s\n' "$RMW_IMPLEMENTATION"
printf 'ROS_LOCALHOST_ONLY=%s\n' "$ROS_LOCALHOST_ONLY"
