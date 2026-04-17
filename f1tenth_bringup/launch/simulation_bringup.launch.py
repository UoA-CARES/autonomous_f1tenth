import os
import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def spawn_func(context, *args, **kwargs):

    description_pkg_path = os.path.join(
        get_package_share_directory("f1tenth_description")
    )
    xacro_file = os.path.join(description_pkg_path, "urdf", "robot.urdf.xacro")

    world = LaunchConfiguration("world").perform(context)
    name = LaunchConfiguration("name").perform(context)
    enable_camera = LaunchConfiguration("enable_camera").perform(context)

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
                        xacro_file,
                        mappings={
                            "robot_name": name,
                            "enable_camera": enable_camera,
                        },
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
                # Only unidirectional bridges (ros->gz or gz->ros as needed)
                f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist[ros2_to_gz]",
                f"/{name}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan[gz_to_ros2]",
                f"/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry[gz_to_ros2]",
                f"/model/{name}/pose@geometry_msgs/msg/Pose@gz.msgs.Pose[gz_to_ros2]",
                f"/{name}/imu@sensor_msgs/msg/Imu@gz.msgs.IMU[gz_to_ros2]",
            ],
            remappings=[
                (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                (f"/model/{name}/pose", f"/{name}/pose"),
                (f"/model/{name}/odometry", f"/{name}/odometry"),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(name="world", description="name of world"),
            DeclareLaunchArgument(name="name", description="name of robot spawned"),
            DeclareLaunchArgument(
                name="enable_camera",
                description="enable depth camera sensor",
                default_value="false",
            ),
            DeclareLaunchArgument(
                name="x", description="x position of robot", default_value="3.0"
            ),
            DeclareLaunchArgument(
                name="y", description="y position of robot", default_value="3.0"
            ),
            DeclareLaunchArgument(
                name="z", description="z position of robot", default_value="3.0"
            ),
            DeclareLaunchArgument(
                name="R", description="roll of robot", default_value="0.0"
            ),
            DeclareLaunchArgument(
                name="P", description="pitch of robot", default_value="0.0"
            ),
            DeclareLaunchArgument(
                name="Y", description="yaw of robot", default_value="0.0"
            ),
            OpaqueFunction(function=spawn_func),
        ]
    )
