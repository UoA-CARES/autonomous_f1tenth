import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("f1tenth_bringup")
    description_share = get_package_share_directory("f1tenth_description")

    default_vesc_config = os.path.join(bringup_share, "config", "vesc.yaml")
    default_sensors_config = os.path.join(bringup_share, "config", "sensors.yaml")
    default_mux_config = os.path.join(bringup_share, "config", "mux.yaml")
    default_ekf_config = os.path.join(bringup_share, "config", "ekf_real.yaml")
    default_joy_config = os.path.join(bringup_share, "config", "joy.yaml")
    robot_xacro = os.path.join(description_share, "urdf", "robot.urdf.xacro")

    car_name = LaunchConfiguration("car_name")
    vesc_config = LaunchConfiguration("vesc_config")
    sensors_config = LaunchConfiguration("sensors_config")
    mux_config = LaunchConfiguration("mux_config")
    ekf_config = LaunchConfiguration("ekf_config")
    joy_config = LaunchConfiguration("joy_config")

    launch_arguments = [
        DeclareLaunchArgument(
            "car_name",
            default_value="f1tenth",
            description="Robot name used for controller-facing topics.",
        ),
        DeclareLaunchArgument(
            "vesc_config",
            default_value=default_vesc_config,
            description="VESC driver and conversion parameter file.",
        ),
        DeclareLaunchArgument(
            "sensors_config",
            default_value=default_sensors_config,
            description="Hardware sensor parameter file.",
        ),
        DeclareLaunchArgument(
            "mux_config",
            default_value=default_mux_config,
            description="Ackermann command multiplexer parameter file.",
        ),
        DeclareLaunchArgument(
            "ekf_config",
            default_value=default_ekf_config,
            description="Robot localization EKF parameter file.",
        ),
        DeclareLaunchArgument(
            "joy_config",
            default_value=default_joy_config,
            description="Joy and joy_teleop parameter file.",
        ),
        DeclareLaunchArgument(
            "launch_joy",
            default_value="true",
            description="Launch joystick teleoperation with the hardware stack.",
        ),
        DeclareLaunchArgument(
            "launch_lidar",
            default_value="true",
            description="Launch the URG lidar driver.",
        ),
        DeclareLaunchArgument(
            "launch_ekf",
            default_value="true",
            description="Launch robot_localization EKF.",
        ),
    ]

    robot_description = xacro.process_file(
        robot_xacro,
        mappings={"robot_name": "f1tenth", "enable_camera": "false"},
    ).toxml()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    vesc_driver = Node(
        package="vesc_driver",
        executable="vesc_driver_node",
        name="vesc_driver_node",
        output="screen",
        parameters=[vesc_config],
    )

    ackermann_to_vesc = Node(
        package="vesc_ackermann",
        executable="ackermann_to_vesc_node",
        name="ackermann_to_vesc_node",
        output="screen",
        parameters=[vesc_config],
    )

    vesc_to_odom = Node(
        package="vesc_ackermann",
        executable="vesc_to_odom_node",
        name="vesc_to_odom_node",
        output="screen",
        parameters=[vesc_config],
        remappings=[("odom", "vesc/odom")],
    )

    ackermann_mux = Node(
        package="ackermann_mux",
        executable="ackermann_mux",
        name="ackermann_mux",
        output="screen",
        parameters=[mux_config],
    )

    lidar = Node(
        package="urg_node",
        executable="urg_node_driver",
        name="urg_node",
        output="screen",
        parameters=[sensors_config],
        remappings=[("scan", ["/", car_name, "/scan"])],
        condition=IfCondition(LaunchConfiguration("launch_lidar")),
    )

    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config],
        remappings=[("odometry/filtered", ["/", car_name, "/odometry"])],
        condition=IfCondition(LaunchConfiguration("launch_ekf")),
    )

    joy_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "joy_bringup.launch.py")
        ),
        launch_arguments={"joy_config": joy_config}.items(),
        condition=IfCondition(LaunchConfiguration("launch_joy")),
    )

    return LaunchDescription(
        launch_arguments
        + [
            robot_state_publisher,
            vesc_driver,
            ackermann_to_vesc,
            vesc_to_odom,
            ackermann_mux,
            lidar,
            ekf,
            joy_bringup,
        ]
    )
