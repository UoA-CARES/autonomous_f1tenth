from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

algorithm = "TD3"
DEFAULT_MAX_SPEED = "3.0"
DEFAULT_MIN_SPEED = "0.5"
DEFAULT_MAX_TURN = "0.434"
DEFAULT_MIN_TURN = "-0.434"


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
        default_value="",
    )
    controlled_agent_id_arg = DeclareLaunchArgument(
        "controlled_agent_id",
        default_value="",
        description="MARL agent id whose actor controls the physical car. Defaults to car_name.",
    )
    marl_agent_ids_arg = DeclareLaunchArgument(
        "marl_agent_ids",
        default_value="",
        description="Comma-separated MARL agent ids used during training. Defaults to controlled_agent_id.",
    )
    marl_teams_arg = DeclareLaunchArgument(
        "marl_teams",
        default_value="",
        description="Semicolon-separated teams, e.g. ego:f1tenth;opp:opponent_1,opponent_2.",
    )
    marl_parameter_sharing_scope_arg = DeclareLaunchArgument(
        "marl_parameter_sharing_scope",
        default_value="",
        description="Optional MARL sharing scope, e.g. individual, shared, team_critic, team_all.",
    )
    marl_use_agent_id_arg = DeclareLaunchArgument(
        "marl_use_agent_id",
        default_value="",
        description="Optional independent-MARL identity conditioning flag from training.",
    )
    marl_use_team_id_arg = DeclareLaunchArgument(
        "marl_use_team_id",
        default_value="",
        description="Optional independent-MARL team identity conditioning flag from training.",
    )
    marl_action_scaling_arg = DeclareLaunchArgument(
        "marl_action_scaling",
        default_value="legacy_direct",
        description=(
            "MARL actor action scaling: legacy_direct matches existing CARES F1Tenth "
            "checkpoints; normalized is for checkpoints trained with [-1, 1] "
            "action denormalisation."
        ),
    )
    max_speed_arg = DeclareLaunchArgument("max_speed", default_value=DEFAULT_MAX_SPEED)
    training_max_speed_arg = DeclareLaunchArgument(
        "training_max_speed", default_value="5.0"
    )
    max_turn_arg = DeclareLaunchArgument("max_turn", default_value=DEFAULT_MAX_TURN)
    min_speed_arg = DeclareLaunchArgument("min_speed", default_value=DEFAULT_MIN_SPEED)
    min_turn_arg = DeclareLaunchArgument("min_turn", default_value=DEFAULT_MIN_TURN)
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
        "deadman_timeout_sec", default_value="0.1"
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
                "controlled_agent_id": LaunchConfiguration("controlled_agent_id"),
                "marl_agent_ids": LaunchConfiguration("marl_agent_ids"),
                "marl_teams": LaunchConfiguration("marl_teams"),
                "marl_parameter_sharing_scope": LaunchConfiguration(
                    "marl_parameter_sharing_scope"
                ),
                "marl_use_agent_id": LaunchConfiguration("marl_use_agent_id"),
                "marl_use_team_id": LaunchConfiguration("marl_use_team_id"),
                "marl_action_scaling": LaunchConfiguration("marl_action_scaling"),
                "max_speed": LaunchConfiguration("max_speed"),
                "training_max_speed": LaunchConfiguration("training_max_speed"),
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
            controlled_agent_id_arg,
            marl_agent_ids_arg,
            marl_teams_arg,
            marl_parameter_sharing_scope_arg,
            marl_use_agent_id_arg,
            marl_use_team_id_arg,
            marl_action_scaling_arg,
            max_speed_arg,
            training_max_speed_arg,
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
