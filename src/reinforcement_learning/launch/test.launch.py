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
    'CarTrack': 'cartrack',
    'CarBeat': 'carbeat',
    'CarOvertake': 'carovertake',
    'TwoCar': 'twocar'
}

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_environments = get_package_share_directory('environments')

    config_path = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'test.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config['test']['ros__parameters']['environment']

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, f'{env_launch[env]}.launch.py')),
        launch_arguments={
            'track': TextSubstitution(text=str(config['test']['ros__parameters']['track'])),
            'car_name': TextSubstitution(text=str(config['test']['ros__parameters']['car_name']) if 'car_name' in config['test']['ros__parameters'] else 'f1tenth'),
            'car_one': TextSubstitution(text=str(config['test']['ros__parameters']['car_name']) if 'car_name' in config['test']['ros__parameters'] else 'f1tenth'),
            'car_two': TextSubstitution(text=str(config['test']['ros__parameters']['ftg_car_name']) if 'ftg_car_name' in config['test']['ros__parameters'] else 'ftg_car'),
        }.items() #TODO: this doesn't do anything
    )

    # Launch the Environment
    main = Node(
            package='reinforcement_learning',
            executable='test',
            parameters=[
                config_path
            ],
            name='test',
            output='screen',
            emulate_tty=True, # Allows python print to show
    )
    if env == 'TwoCar':
        car2_config_path = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'test_copy.yaml'
    )
        main2 = Node(
                package='reinforcement_learning',
                executable='test',
                parameters=[
                    car2_config_path
                ],
                name='test',
                output='screen',
                emulate_tty=True, # Allows python print to show
            )
        return LaunchDescription([
        #TODO: Find a way to remove this
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        SetParameter(name='use_sim_time', value=True),
        environment,
        main,
        main2
        ])

    return LaunchDescription([
        #TODO: Find a way to remove this
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        SetParameter(name='use_sim_time', value=True),
        environment,
        main
])