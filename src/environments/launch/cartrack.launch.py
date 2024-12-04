import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch(context, *args, **kwargs):
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_environments = get_package_share_directory('environments')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')

    track = LaunchConfiguration('track').perform(context)
    car_name = LaunchConfiguration('car_name').perform(context)
    
    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/{track}.sdf',
        }.items()
    )

    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': car_name,
            'world': 'empty'
        }.items()
    )


    return[gz_sim, f1tenth]

def generate_launch_description():

    track_arg = DeclareLaunchArgument(
        'track',
        default_value='track_1'
    )

    car_name = DeclareLaunchArgument(
        'car_name',
        default_value='f1tenth'
    )

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

    stepping_service = Node(
            package='environments',
            executable='SteppingService',
            output='screen',
            emulate_tty=True,
    )

    reset = Node(
            package='environments',
            executable='CarTrackReset',
            output='screen',
            emulate_tty=True,
    )

    return LaunchDescription([
        track_arg,
        OpaqueFunction(function=launch),
        service_bridge,
        reset,
        stepping_service,
        car_name,
])