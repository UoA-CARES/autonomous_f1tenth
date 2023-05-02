from launch import LaunchDescription 
from launch_ros.actions import Node 
import launch
import os
import json
import xacro

from ament_index_python.packages import get_package_share_directory

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, \
    SetEnvironmentVariable

from launch_ros.actions import Node

def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_environments = get_package_share_directory('environments')


    # TODO: remove the hardcoding of topic name
    car_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/model/f1tenth/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            f'/model/f1tenth/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        ]
    )
        # f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        # f'/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        # f'/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        # # f'/model/{name}/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
        # f'/world/{world}/model/{name}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model',
        # f'/model/{name}/pose@geometry_msgs/msg/Pose@gz.msgs.Pose',

    car_goal =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, 'cargoal.launch.py')),
        launch_arguments={
            'car_name': 'f1tenth',
        }.items() #TODO: this doesn't do anything
    )
    
    # Launch the Environment
    car_goal_main = Node(
            package='environments',
            executable='CarGoal',
            output='screen',
            emulate_tty=True,
            arguments={
                'hello': 'wporld'
            }.items()
    )

    # F1tenth Spawning
    # TODO: move into own launch file - f1tenth_simulation.launch.py
    xacro_file = os.path.join(pkg_f1tenth_description, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()
    
    f1tenth_ros_gz = Node(
            package='ros_gz_sim', executable='create',
            arguments=[
                '-name', 'f1tenth',
                '-string', robot_description,
            ],
            output='screen'
        )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        car_bridge,
        car_goal,
        car_goal_main,
        f1tenth_ros_gz,
])