from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    car_name_arg = DeclareLaunchArgument("car_name", default_value="f1tenth")
    algorithm_arg = DeclareLaunchArgument("algorithm", default_value="TD3")
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="overtaking_models/350000_checkpoint.pth",
    )
    max_speed_arg = DeclareLaunchArgument("max_speed", default_value="2")
    max_turn_arg = DeclareLaunchArgument("max_turn", default_value="0.45")
    min_speed_arg = DeclareLaunchArgument("min_speed", default_value="0")
    min_turn_arg = DeclareLaunchArgument("min_turn", default_value="-0.45")

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
            checkpoint_path_arg,
            max_speed_arg,
            max_turn_arg,
            min_speed_arg,
            min_turn_arg,
            main,
            vel_recorder,
            lidar_recorder,
        ]
    )
