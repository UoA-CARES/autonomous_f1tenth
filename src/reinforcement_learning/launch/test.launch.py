from launch import LaunchDescription 
from launch_ros.actions import Node 
import launch
import os

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

    car_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/model/car/pose@geometry_msgs/msg/Pose@gz.msgs.Pose',
        ]
    )

    f1tenth = Node(
        package='ros_gz_bridge',
        executable='create',
        output='screen',
        arguments=[
                '-world', world,
                '-name', name,
                '-topic', topic,
        ],
    )
    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': '-r empty.sdf',
        }.items()
    )

    # TODO: add RL environment spin up here

    car_goal = Node(
            package='environments',
            executable='CarGoal',
            output='screen',
            emulate_tty=True,
            arguments={
                'hello': 'wporld'
            }.items()
    )

    
    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        gz_sim,
        service_bridge,
        car_bridge,
        car_goal,
        # training
])