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
    pkg_environments = get_package_share_directory('environments')

    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/world/empty/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/empty/create@ros_gz_interfaces/srv/SpawnEntity',
            f'/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity',
            f'/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ]
    )

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {pkg_environments}/worlds/wall.sdf',
        }.items()
    )

    #TODO: dynamically change car name
    #TODO: This doesn't work yet
    #TODO: Create CarWall Reset
    reset = Node(
            package='environments',
            executable='CarGoalReset',
            output='screen',
            arguments=['-car_name', 'f1tenth']
    )

    return LaunchDescription([
        gz_sim,
        service_bridge,
        reset,
])