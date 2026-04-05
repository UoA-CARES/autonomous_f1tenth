import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import Node

import xacro


# xacro.process_file(xacro_file, mappings={"robot_name": {name}}).toxml()
def spawn_func(context, *args, **kwargs):

    description_pkg_path = os.path.join(
        get_package_share_directory("f1tenth_description")
    )
    xacro_file = os.path.join(description_pkg_path, "urdf", "robot.urdf.xacro")

    world = LaunchConfiguration("world").perform(context)
    name = LaunchConfiguration("name").perform(context)

    x = LaunchConfiguration("x").perform(context)
    y = LaunchConfiguration("y").perform(context)
    z = LaunchConfiguration("z").perform(context)

    R = LaunchConfiguration("R").perform(context)
    P = LaunchConfiguration("P").perform(context)
    Y = LaunchConfiguration("Y").perform(context)

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": xacro.process_file(
                        xacro_file, mappings={"robot_name": name}
                    ).toxml(),
                    "frame_prefix": name,
                }
            ],
            remappings=[
                ("/tf", f"/{name}/tf"),
                ("/tf_static", f"/{name}/tf_static"),
            ],
            namespace=name,
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world",
                world,
                "-name",
                name,
                "-topic",
                f"/{name}/robot_description",
                "-x",
                x,
                "-y",
                y,
                "-z",
                z,
                "-R",
                R,
                "-P",
                P,
                "-Y",
                Y,
            ],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                f"/{name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
                # f'/{name}/camera@sensor_msgs/msg/Image@gz.msgs.Image',
                # f'/{name}/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f"/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                f"/model/{name}/odometry_with_covariance@nav_msgs/msg/Odometry@gz.msgs.OdometryWithCovariance",
                # f'/model/{name}/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                f"/world/{world}/model/{name}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
                f"/model/{name}/pose@geometry_msgs/msg/Pose@gz.msgs.Pose",
                f"/{name}/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            ],
            remappings=[
                (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                (f"/model/{name}/pose", f"/{name}/pose"),
                (f"/model/{name}/odometry", f"/{name}/odometry"),
                (
                    f"/model/{name}/odometry_with_covariance",
                    f"/{name}/odometry_with_covariance",
                ),
                (f"/world/{world}/model/{name}/joint_state", f"/{name}/joint_states"),
                # (f'/model/{name}/tf', '/tf'),
            ],
        ),
        # Node(
        #     package='robot_localization',
        #     executable='ekf_node',
        #     name='ekf_filter_node',
        #     output='screen',
        #     parameters=[os.path.join(description_pkg_path, 'config/ekf.yaml'),
        #                 {"use_sim_time": True}],
        # ),
        # Node(
        #     package='slam_toolbox',
        #     executable='async_slam_toolbox_node',
        #     name='slam_toolbox',
        #     output='screen',
        #     parameters=[ os.path.join(description_pkg_path, 'config', 'slam_toolbox.yaml'),
        #                 {"use_sim_time": True}],
        # )
    ]


def generate_launch_description():
    world_arg = DeclareLaunchArgument(name="world", description="name of world")

    name_arg = DeclareLaunchArgument(name="name", description="name of robot spawned")

    x = DeclareLaunchArgument(
        name="x", description="x position of robot", default_value="3.0"
    )

    y = DeclareLaunchArgument(
        name="y", description="y position of robot", default_value="3.0"
    )

    z = DeclareLaunchArgument(
        name="z", description="z position of robot", default_value="3.0"
    )

    R = DeclareLaunchArgument(
        name="R", description="roll of robot", default_value="0.0"
    )

    P = DeclareLaunchArgument(
        name="P", description="pitch of robot", default_value="0.0"
    )

    Y = DeclareLaunchArgument(name="Y", description="yaw of robot", default_value="0.0")

    return LaunchDescription(
        [world_arg, name_arg, x, y, z, R, P, Y, OpaqueFunction(function=spawn_func)]
    )
