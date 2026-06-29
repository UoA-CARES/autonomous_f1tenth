from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    algorithm_arg = DeclareLaunchArgument("algorithm", default_value="astar")
    map_arg = DeclareLaunchArgument("map", default_value="stateMap.pgm")
    yaml_path_arg = DeclareLaunchArgument("yaml_path", default_value="stateMap.yaml")

    alg = Node(
        package="f1tenth_controllers",
        executable="planner",
        output="screen",
        parameters=[
            {"alg": LaunchConfiguration("algorithm")},
            {"map": LaunchConfiguration("map")},
            {"yaml_path": LaunchConfiguration("yaml_path")},
        ],
    )
    return LaunchDescription(
        [
            algorithm_arg,
            map_arg,
            yaml_path_arg,
            SetParameter(name="use_sim_time", value=True),
            alg,
        ]
    )
