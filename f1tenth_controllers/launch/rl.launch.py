from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

algorithm = "TD3"


def _prepend_pythonpath(path: Path) -> str:
    existing = __import__("os").environ.get("PYTHONPATH", "")
    return f"{path}:{existing}" if existing else str(path)


def _set_optional_discovery_server(context):
    discovery_server = LaunchConfiguration("ros_discovery_server").perform(context)
    if not discovery_server:
        return []
    return [SetEnvironmentVariable("ROS_DISCOVERY_SERVER", discovery_server)]


def _set_cares_pythonpath(context):
    cares_python_path = LaunchConfiguration("cares_python_path").perform(context)
    if not cares_python_path:
        return []
    return [
        SetEnvironmentVariable(
            "PYTHONPATH",
            _prepend_pythonpath(Path(cares_python_path).expanduser()),
        )
    ]


def generate_launch_description():
    car_name_arg = DeclareLaunchArgument("car_name", default_value="f1tenth")
    algorithm_arg = DeclareLaunchArgument("algorithm", default_value=algorithm)
    rmw_implementation_arg = DeclareLaunchArgument(
        "rmw_implementation",
        default_value="rmw_fastrtps_cpp",
        description="ROS middleware implementation used by all RL nodes.",
    )
    ros_domain_id_arg = DeclareLaunchArgument(
        "ros_domain_id",
        default_value="0",
        description="ROS domain used by all RL nodes.",
    )
    ros_discovery_server_arg = DeclareLaunchArgument(
        "ros_discovery_server",
        default_value="",
        description="Optional Fast DDS discovery server, for example 172.22.1.87:11811.",
    )
    cares_python_path_arg = DeclareLaunchArgument(
        "cares_python_path",
        default_value=".",
        description="Path containing the cares_reinforcement_learning package to add to PYTHONPATH.",
    )
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="overtaking_models/" + algorithm + "_checkpoint.pth",
    )
    max_speed_arg = DeclareLaunchArgument("max_speed", default_value="3.0")
    max_turn_arg = DeclareLaunchArgument("max_turn", default_value="0.434")
    min_speed_arg = DeclareLaunchArgument("min_speed", default_value="0.5")
    min_turn_arg = DeclareLaunchArgument("min_turn", default_value="-0.434")
    odom_mode_arg = DeclareLaunchArgument("odom_mode", default_value="velocity_only")
    lidar_mode_arg = DeclareLaunchArgument("lidar_mode", default_value="processed")
    forward_half_angle_arg = DeclareLaunchArgument(
        "forward_half_angle", default_value="45.0"
    )
    n_forward_arg = DeclareLaunchArgument("n_forward", default_value="5")
    wheelbase_arg = DeclareLaunchArgument("wheelbase", default_value="0.325")
    deadman_topic_arg = DeclareLaunchArgument(
        "deadman_topic", default_value="/rl_deadman"
    )
    deadman_timeout_arg = DeclareLaunchArgument(
        "deadman_timeout_sec", default_value="0.25"
    )
    command_timeout_arg = DeclareLaunchArgument(
        "command_timeout_sec", default_value="0.25"
    )

    main = Node(
        package="f1tenth_controllers",
        executable="rl_policy",
        output="screen",
        name="rl_policy",
        parameters=[
            {
                "car_name": LaunchConfiguration("car_name"),
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "max_speed": LaunchConfiguration("max_speed"),
                "max_turn": LaunchConfiguration("max_turn"),
                "min_speed": LaunchConfiguration("min_speed"),
                "min_turn": LaunchConfiguration("min_turn"),
                "odom_mode": LaunchConfiguration("odom_mode"),
                "lidar_mode": LaunchConfiguration("lidar_mode"),
                "forward_half_angle": LaunchConfiguration("forward_half_angle"),
                "n_forward": LaunchConfiguration("n_forward"),
                "wheelbase": LaunchConfiguration("wheelbase"),
            }
        ],
    )


    deadman = Node(
        package="f1tenth_controllers",
        executable="rl_deadman",
        output="screen",
        name="rl_deadman",
        parameters=[
            {
                "car_name": LaunchConfiguration("car_name"),
                "deadman_topic": LaunchConfiguration("deadman_topic"),
                "deadman_timeout_sec": LaunchConfiguration("deadman_timeout_sec"),
                "command_timeout_sec": LaunchConfiguration("command_timeout_sec"),
            }
        ],
    )

    vel_recorder = Node(
        package="f1tenth_recorders",
        executable="vel_recorder",
        name="vel_recorder",
        output="screen",
        parameters=[{"onSim": False}],
    )

    lidar_recorder = Node(
        package="f1tenth_recorders",
        executable="lidar_recorder",
        name="lidar_recorder",
        output="screen",
    )

    return LaunchDescription(
        [
            car_name_arg,
            algorithm_arg,
            rmw_implementation_arg,
            ros_domain_id_arg,
            ros_discovery_server_arg,
            cares_python_path_arg,
            checkpoint_path_arg,
            max_speed_arg,
            max_turn_arg,
            min_speed_arg,
            min_turn_arg,
            odom_mode_arg,
            lidar_mode_arg,
            forward_half_angle_arg,
            n_forward_arg,
            wheelbase_arg,
            deadman_topic_arg,
            deadman_timeout_arg,
            command_timeout_arg,
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID",
                LaunchConfiguration("ros_domain_id"),
            ),
            OpaqueFunction(function=_set_optional_discovery_server),
            OpaqueFunction(function=_set_cares_pythonpath),
            main,
            deadman,
            vel_recorder,
            lidar_recorder,
        ]
    )
