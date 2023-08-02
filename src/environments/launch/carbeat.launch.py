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

    track = LaunchConfiguration('track').perform(context)
    
    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/{track}.sdf',
        }.items()
    )

    return[gz_sim]

def generate_launch_description():
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')

    track_arg = DeclareLaunchArgument(
        'track',
        default_value='track_1'
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

    f1tenth_one = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': 'f1tenth_one',
            'world': 'empty',
            'x': '-5',
            'y': '-5',
            'z': '1',
        }.items()
    )

    f1tenth_two = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': 'f1tenth_two',
            'world': 'empty',
            'x': '5',
            'y': '5',
            'z': '0.4',
        }.items()
    )

    #TODO: dynamically change car name
    #TODO: This doesn't work yet
    #TODO: Create CarWall Reset
    reset = Node(
            package='environments',
            executable='CarBeatReset',
            output='screen',
    )

    ld = LaunchDescription([
        track_arg,
        OpaqueFunction(function=launch),
        service_bridge,
        reset,
        f1tenth_one,
        f1tenth_two,
    ])
    
    return ld 