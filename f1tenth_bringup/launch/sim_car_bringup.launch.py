import os
import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def spawn_car_func(context, *args, **kwargs):
    description_pkg_path = os.path.join(
        get_package_share_directory("f1tenth_description")
    )
    xacro_file = os.path.join(description_pkg_path, "urdf", "robot.urdf.xacro")

    name = LaunchConfiguration("name").perform(context)
    world = LaunchConfiguration("world").perform(context)
    enable_camera = LaunchConfiguration("enable_camera").perform(context)
    x = LaunchConfiguration("x").perform(context)
    y = LaunchConfiguration("y").perform(context)
    z = LaunchConfiguration("z").perform(context)
    R = LaunchConfiguration("R").perform(context)
    P = LaunchConfiguration("P").perform(context)
    Y = LaunchConfiguration("Y").perform(context)
    controller = LaunchConfiguration("controller").perform(context)

    nodes = [
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
                f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                f"/{name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                f"/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                f"/model/{name}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose",
                f"/{name}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            ],
            remappings=[
                (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                (f"/model/{name}/pose", f"/{name}/pose"),
                (f"/model/{name}/odometry", f"/{name}/odometry"),
            ],
        ),
    ]
    # Launch controller if specified
    if controller == "ftg":
        nodes.append(
            Node(
                package="f1tenth_controllers",
                executable="ftg_policy",
                output="screen",
                parameters=[{"car_name": name}],
                namespace=name,
            )
        )
    # Future: add more controller types here
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(name="name", description="name of robot spawned"),
            DeclareLaunchArgument(name="world", description="name of world"),
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
                name="z", description="z position of robot", default_value="0.0"
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
            DeclareLaunchArgument(
                name="controller",
                description="controller type (e.g. ftg, rl, etc.)",
                default_value="",
            ),
            OpaqueFunction(function=spawn_car_func),
        ]
    )
