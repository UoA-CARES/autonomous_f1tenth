import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import TextSubstitution
import yaml

env_launch = {
    'CarGoal': 'cargoal',
    'CarWall': 'carwall',
    'CarBlock': 'carblock',
    'CarTrack': 'cartrack'
}

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
    pkg_environments = get_package_share_directory('environments')

    config_path = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'sanity_check.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config['sanity_check']['ros__parameters']['environment']

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, f'{env_launch[env]}.launch.py')),
        launch_arguments={
            'track': TextSubstitution(text=str(config['sanity_check']['ros__parameters']['track'])),
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

    # Launch the Environment
    main = Node(
            package='reinforcement_learning',
            executable='sanity_check',
            parameters=[
                config_path
            ],
            name='sanity_check',
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