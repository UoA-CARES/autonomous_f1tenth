import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def spawn_cars(context, *args, **kwargs):
    pkg_bringup = get_package_share_directory("f1tenth_bringup")

    num_opponents = int(LaunchConfiguration("num_opponents").perform(context))
    marl_env = LaunchConfiguration("marl_env").perform(context).lower() == "true"
    num_cars = 1 + num_opponents

    car_nodes = []
    for i in range(num_cars):
        if (i == 0):
            car_name = "f1tenth"
        else:
            car_name = f"opponent_{i}"
        x_pos = 3.0 + 2.0 * i
        y_pos = 3.0
        # Agent car (first car) gets no controller, opponents get FTG
        if marl_env:
            controller = ""
        else:
            controller = "" if i == 0 else "ftg"
        car_nodes.append(
            IncludeLaunchDescription(
                launch_description_source=PythonLaunchDescriptionSource(
                    os.path.join(pkg_bringup, "sim_car_bringup.launch.py")
                ),
                launch_arguments={
                    "name": car_name,
                    "world": "empty",
                    "x": str(x_pos),
                    "y": str(y_pos),
                    "z": "3.0",
                    "R": "0.0",
                    "P": "0.0",
                    "Y": "0.0",
                    "controller": controller,
                }.items(),
            )
        )
    return car_nodes


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")

    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_gazebo = get_package_share_directory("f1tenth_gazebo")

    # config_path and config_params are not needed for launch argument defaults
    track_arg = DeclareLaunchArgument(
        "track",
        default_value="multi_track_01",
    )
    num_opponents_arg = DeclareLaunchArgument(
        "num_opponents",
        default_value="0",
        description="Number of opponent cars (agent car is always present)",
    )
    marl_env_arg = DeclareLaunchArgument(
        "marl_env",
        default_value="true",
        description="Whether to use MARL environment (no FTG controllers)",
    )

    def make_gz_sim(context):
        track = LaunchConfiguration("track").perform(context)
        world_path = os.path.join(pkg_gazebo, "worlds", f"{track}.sdf")
        return [
            IncludeLaunchDescription(
                launch_description_source=PythonLaunchDescriptionSource(
                    os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
                ),
                launch_arguments={"gz_args": f"-s -r {world_path}"}.items(),
            )
        ]

    gz_sim = OpaqueFunction(function=make_gz_sim)

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
            num_opponents_arg,
            marl_env_arg,
            SetEnvironmentVariable(
                name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
            ),
            SetParameter(name="use_sim_time", value=True),
            gz_sim,
            service_bridge,
            OpaqueFunction(function=spawn_cars),
        ]
    )
