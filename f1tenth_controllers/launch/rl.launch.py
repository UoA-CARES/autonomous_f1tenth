from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    car_name_arg = DeclareLaunchArgument("car_name", default_value="f1tenth")
    algorithm_arg = DeclareLaunchArgument("algorithm", default_value="TD3")
    rmw_implementation_arg = DeclareLaunchArgument(
        "rmw_implementation",
        default_value="rmw_fastrtps_cpp",
        description="ROS middleware implementation used by all RL nodes.",
    )
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="overtaking_models/SAC_checkpoint.pth",
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
    deadman_button_arg = DeclareLaunchArgument("deadman_button", default_value="5")
    joy_topic_arg = DeclareLaunchArgument("joy_topic", default_value="/joy")
    joy_timeout_arg = DeclareLaunchArgument(
        "joy_timeout_sec", default_value="0.25"
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
                "deadman_button": LaunchConfiguration("deadman_button"),
                "joy_topic": LaunchConfiguration("joy_topic"),
                "joy_timeout_sec": LaunchConfiguration("joy_timeout_sec"),
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
            deadman_button_arg,
            joy_topic_arg,
            joy_timeout_arg,
            command_timeout_arg,
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            main,
            deadman,
            vel_recorder,
            lidar_recorder,
        ]
    )
