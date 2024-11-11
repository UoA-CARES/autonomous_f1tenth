import os
from ament_index_python import get_package_share_directory
import launch_ros
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
    pkg_slam = get_package_share_directory('slam_toolbox')

    config_path = os.path.join(
        pkg_controllers,
        'sim.yaml'
    )

    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config['sim']['ros__parameters']['environment']
    alg = config['sim']['ros__parameters']['algorithm']
    ispreplanned = config['sim']['ros__parameters']['preplan']


    match alg:
        case 'rl':
            tracking = False
        case 'ftg':
            tracking = False
        case 'random':
            tracking = True
        case 'turn_drive':
            tracking = True
        case 'mpc':
            tracking = True
        case 'pure_pursuit':
            tracking = True


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
        state_machine = Node(
            package='controllers',
            executable= 'state_machine',
            output='screen',
            emulate_tty=True,
            parameters = [
                #ispreplanned
            ]
        )

        alg = Node(
            package='controllers',
            executable='track',
            output='screen',
            parameters=[{'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth')},
                        {'alg': TextSubstitution(text=str(alg))}, {'path_file_path': TextSubstitution(text=str(config['sim']['ros__parameters']['path_file_path']))}],
        )

        if ispreplanned:

            return LaunchDescription([
            SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
            SetParameter(name='use_sim_time', value=True),
            environment,
            alg,
            sim,
            state_machine,
            ])
        
        else:
            
            # TF TREE: map ------------ odom --------------- (baselink--lidar_link)
            # https://www.youtube.com/watch?v=ZaiA3hWaRzE
            # Launch static transform node to provide TF info for baselink to lidar_link (sticking them together cause it is not known somehow)
            lidar_to_base_tf_node = launch_ros.actions.Node( 
                package='tf2_ros', 
                executable='static_transform_publisher', 
                arguments=['0', '0', '0', '0', '0', '0', "f1tenthbase_link", "f1tenthhokuyo_10lx_lidar_link"], 
                output='screen'
            )
            # Launch odom localization node to provide TF info from odom reading to base_link
            odom_to_base_tf_node = launch_ros.actions.Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[os.path.join(pkg_f1tenth_description, 'config/ekf.yaml'), {'use_sim_time': True}]
            )

            slam_node = IncludeLaunchDescription(
            launch_description_source = os.path.join(pkg_slam,f'launch/online_async_launch.py'),
            launch_arguments = {
                'use_sim_time': 'True',
                'slam_params_file':"./src/f1tenth/f1tenth_description/config/slam_toolbox.yaml"
            }.items() 
            )

            ftg_node = Node(
                    package='controllers',
                    executable='ftg_policy',
                    output='screen',
                    parameters=[{'car_name': TextSubstitution(text=str(config['sim']['ros__parameters']['car_name']) if 'car_name' in config['sim']['ros__parameters'] else 'f1tenth')}],
            )
            

            return LaunchDescription([
                SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
                SetParameter(name='use_sim_time', value=True),
                environment,
                alg,
                sim,
                lidar_to_base_tf_node,
                odom_to_base_tf_node,
                slam_node,
                state_machine,
                ftg_node
            ])

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

