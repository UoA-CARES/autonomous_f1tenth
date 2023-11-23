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
    car_one = LaunchConfiguration('car_one').perform(context)
    car_two = LaunchConfiguration('car_two').perform(context)

    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-s -r {pkg_environments}/worlds/{track}.sdf',
        }.items()
    )

    f1tenth_one = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': car_one,
            'world': 'empty',
            'x': '0',
            'y': '0',
            'z': '5',
        }.items()
    )

    f1tenth_two = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': car_two,
            'world': 'empty',
            'x': '0',
            'y': '1',
            'z': '5',
        }.items()
    )

    controller = Node(
        package='controllers',
        executable='ftg_policy',
        output='screen',
        parameters=[
            {'car_name': car_two, 'track_name': track},
        ],
    )
    return[gz_sim, controller, f1tenth_one, f1tenth_two]

def generate_launch_description():
    track_arg = DeclareLaunchArgument(
        'track',
        default_value='track_1'
    )

    car_one = DeclareLaunchArgument(
        'car_one',
        default_value='f1tenth_one'
    )

    car_two = DeclareLaunchArgument(
        'car_two',
        default_value='f1tenth_two'
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

    reset = Node(
            package='environments',
            executable='CarBeatReset',
            output='screen',
    )

    stepping_service = Node(
            package='environments',
            executable='SteppingService',
            output='screen',
            emulate_tty=True,
    )

    ld = LaunchDescription([
        track_arg,
        car_one,
        car_two,
        OpaqueFunction(function=launch),
        service_bridge,
        reset,
        stepping_service
    ])
    
    return ld 