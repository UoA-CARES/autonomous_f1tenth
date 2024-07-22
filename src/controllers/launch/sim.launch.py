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
}

alg_launch = {
    'ftg': 'ftg',
    'rl': 'rl',
    'random': 'random',
    'turn_drive': 'turn_drive',
    'mpc': 'mpc',
}

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_environments = get_package_share_directory('environments')
    pkg_controllers = get_package_share_directory('controllers')

    config_path = os.path.join(
        pkg_controllers,
        'sim.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config['sim']['ros__parameters']['environment']
    alg = config['sim']['ros__parameters']['algorithm']
    tracking = config['sim']['ros__parameters']['tracking']

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, f'{env_launch[env]}.launch.py')),
        launch_arguments={
            'track': TextSubstitution(text=str(config['sim']['ros__parameters']['track'])),
            'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth'),
            'car_one': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth'),
            #'car_two': TextSubstitution(text=str(config['sim']['ros__parameters']['ftg_car_name']) if 'ftg_car_name' in config['sim']['ros__parameters'] else 'ftg_car'),
        }.items() #TODO: this doesn't do anything
    )


    # Launch the Environment
    sim = Node(
            package='controllers',
            executable='sim',
            parameters=[
                config_path
            ],
            name='sim',
            output='screen',
            emulate_tty=True, # Allows python print to show
    )


    if tracking:
        alg = Node(
        package='controllers',
        executable='track',
        output='screen',
        parameters=[{'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth')},
                    {'alg': TextSubstitution(text=str(alg))}],
        )
    else:
        if (f'{alg}' != 'rl'):
            alg = Node(
                package='controllers',
                executable=f'{alg}_policy',
                output='screen',
                parameters=[{'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth')}],
            )
        else:
            alg = IncludeLaunchDescription(
                launch_description_source = PythonLaunchDescriptionSource(
                    os.path.join(pkg_controllers, f'{alg_launch[alg]}.launch.py')),
                launch_arguments={
                    'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth'),
                }.items()
            )
    

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        SetParameter(name='use_sim_time', value=True),
        environment,
        alg,
        sim,
])