import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
import yaml

def generate_launch_description(): #TODO: include the train launch file here
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_f1tenth_bringup = get_package_share_directory('f1tenth_bringup')
    pkg_environments = get_package_share_directory('environments')

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_environments, 'cargoal.launch.py')),
        launch_arguments={
            'car_name': 'f1tenth',
        }.items() #TODO: this doesn't do anything
    )
    
    f1tenth = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_f1tenth_bringup, 'simulation_bringup.launch.py')),
        launch_arguments={
            'name': 'f1tenth',
            'world': 'empty',
        }.items()
    )
    
    config_path = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'car_goal.yaml'
    )

    config = yaml.load(open(config_path))
    mode = config['meta']['ros__parameters']['mode']

    # Launch the Environment
    main = Node(
            package='reinforcement_learning',
            executable='car_goal_training' if mode != 'evaluation' else 'car_goal_testing',
            parameters=[
                config_path
            ],
            name='car_goal_training' if mode != 'evaluation' else 'car_goal_testing',
            output='screen',
            emulate_tty=True, # Allows python print to show
    )

    return LaunchDescription([
        #TODO: Find a way to remove this
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        environment,
        f1tenth,
        main
])