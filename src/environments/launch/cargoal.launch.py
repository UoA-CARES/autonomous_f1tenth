import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
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
            f'/world/empty/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock'
        ],
        remappings=[
            (f'/world/empty/clock', f'/clock'),
        ],
    )

    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': 'f1tenth',
            'world': 'empty'
        }.items()
    )

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/empty.sdf',
        }.items()
    )

    #TODO: dynamically change car name
    #TODO: This doesn't work yet
    reset = Node(
            package='environments',
            executable='CarGoalReset',
            output='screen',
            emulate_tty=True,
    )

    return LaunchDescription([
        gz_sim,
        f1tenth,
        service_bridge,
        reset,
])