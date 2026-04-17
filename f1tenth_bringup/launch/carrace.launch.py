import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch(context):
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_gazebo = get_package_share_directory("f1tenth_gazebo")
    pkg_f1tenth_bringup = get_package_share_directory("f1tenth_bringup")

    track = LaunchConfiguration("track").perform(context)
    car_name = LaunchConfiguration("car_name").perform(context)
    num_opponents = int(LaunchConfiguration("num_opponents").perform(context))

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-s -r {pkg_gazebo}/worlds/{track}.sdf",
        }.items(),
    )

    # Main car at (3.0, 3.0), default
    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, "simulation_bringup.launch.py")
        ),
        launch_arguments={
            "name": car_name,
            "world": "empty",
            "x": "3.0",
            "y": "3.0",
            "z": "3.0",
            "R": "0.0",
            "P": "0.0",
            "Y": "0.0",
        }.items(),
    )

    # Opponents: spread along x axis
    opponent_entities = []
    for opponent_index in range(num_opponents):
        opponent_car_name = f"f{opponent_index + 2}tenth"
        x_pos = 3.0 + 2.0 * (opponent_index + 1)  # e.g., 5.0, 7.0, ...
        y_pos = 3.0
        opponent_entities.extend(
            [
                IncludeLaunchDescription(
                    launch_description_source=PythonLaunchDescriptionSource(
                        os.path.join(
                            pkg_f1tenth_bringup, "simulation_bringup.launch.py"
                        )
                    ),
                    launch_arguments={
                        "name": opponent_car_name,
                        "world": "empty",
                        "x": str(x_pos),
                        "y": str(y_pos),
                        "z": "3.0",
                        "R": "0.0",
                        "P": "0.0",
                        "Y": "0.0",
                    }.items(),
                ),
                Node(
                    package="f1tenth_controllers",
                    executable="ftg_policy",
                    output="screen",
                    parameters=[
                        {"car_name": opponent_car_name, "track_name": track},
                    ],
                ),
            ]
        )

    return [gz_sim, f1tenth, *opponent_entities]


def generate_launch_description():

    track_arg = DeclareLaunchArgument("track", default_value="track_1")

    car_name = DeclareLaunchArgument("car_name", default_value="f1tenth")

    num_opponents = DeclareLaunchArgument("num_opponents", default_value="1")

    service_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/world/empty/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/empty/create@ros_gz_interfaces/srv/SpawnEntity",
            "/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity",
            "/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/world/empty/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock",
        ],
        remappings=[
            ("/world/empty/clock", "/clock"),
        ],
    )

    return LaunchDescription(
        [
            track_arg,
            OpaqueFunction(function=launch),
            service_bridge,
            car_name,
            num_opponents,
        ]
    )
