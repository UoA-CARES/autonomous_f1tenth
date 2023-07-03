import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
    pkg_environments = get_package_share_directory('environments')

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, 'cartrack.launch.py')),
        launch_arguments={
            'car_name': 'f1tenth',
        }.items() #TODO: this doesn't do anything
    )
    
    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': 'f1tenth',
            'world': 'empty'
        }.items()
    )

    config = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'car_wall.yaml'
    )

    # Launch the Environment
    main = Node(
            package='reinforcement_learning',
            executable='car_track_training',
            parameters=[
                config
            ],
            name='car_track_training',
            output='screen',
            emulate_tty=True, # Allows python print to show
    )

    return LaunchDescription([
        #TODO: Find a way to remove this
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        SetParameter(name='use_sim_time', value=True),
        environment,
        f1tenth,
        main
])