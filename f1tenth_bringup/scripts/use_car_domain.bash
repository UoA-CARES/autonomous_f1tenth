# Source this in a shell before using ros2 CLI tools with a car domain.
# Usage: source ~/autonomous_f1tenth/f1tenth_bringup/scripts/use_car_domain.bash 2

_domain="${1:-${ROS_DOMAIN_ID:-0}}"
_requested_rmw="${2:-rmw_fastrtps_cpp}"
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${_requested_rmw}" in
  fastrcps|fastrcps_cpp|rmw_fastrcps|rmw_fastrcps_cpp|fastdds|fastdds_cpp|fastrtps|fastrtps_cpp)
    _requested_rmw="rmw_fastrtps_cpp"
    ;;
  cyclone|cyclone_cpp|cyclonedds|cyclonedds_cpp)
    _requested_rmw="rmw_cyclonedds_cpp"
    ;;
esac

_find_setup_dir="${_script_dir}"
while [ "${_find_setup_dir}" != "/" ]; do
  if [ -f "${_find_setup_dir}/install/setup.bash" ]; then
    _restore_nounset=0
    case "$-" in
      *u*)
        _restore_nounset=1
        set +u
        ;;
    esac

    source "${_find_setup_dir}/install/setup.bash"

    if [ "${_restore_nounset}" = "1" ]; then
      set -u
    fi
    break
  fi
  _find_setup_dir="$(dirname "${_find_setup_dir}")"
done

if ros2 pkg prefix "${_requested_rmw}" >/dev/null 2>&1; then
  _rmw="${_requested_rmw}"
else
  printf 'Requested RMW %s is not available; falling back to rmw_fastrtps_cpp\n' "${_requested_rmw}"
  _rmw="rmw_fastrtps_cpp"
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
