import os
from ament_index_python import get_package_share_directory
import launch_ros
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import TextSubstitution
import yaml

def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    pkg_environments = get_package_share_directory('environments')
    pkg_controllers = get_package_share_directory('controllers')
    pkg_slam = get_package_share_directory('slam_toolbox')

    config_path = os.path.join(pkg_controllers, 'sim.yaml')
    config = yaml.load(open(config_path), Loader=yaml.Loader)
    env = config['sim']['ros__parameters']['environment']
    alg = config['sim']['ros__parameters']['algorithm']
    startStage = config['sim']['ros__parameters']['start_stage']
    car_name = config['sim']['ros__parameters'].get('car_name', 'f1tenth')

    match alg:
        case 'rl' | 'ftg':
            tracking = False
        case 'random' | 'turn_drive' | 'mpc' | 'pure_pursuit':
            tracking = True

    environment =  IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(os.path.join(pkg_environments, f'{env.lower()}.launch.py')),
        launch_arguments={'track': TextSubstitution(text=str(config['sim']['ros__parameters']['track'])),
            'car_name': car_name,
            'car_one': car_name,
            #'car_two': TextSubstitution(text=str(config['sim']['ros__parameters']['ftg_car_name']) if 'ftg_car_name' in config['sim']['ros__parameters'] else 'ftg_car'),
        }.items()
    )

    sim = Node(
        package='controllers',
        executable='sim',
        parameters=[config_path],
        name='sim',
        output='screen',
        emulate_tty=True,)

    if tracking:
        state_machine = Node(
            package='controllers',
            executable= 'state_machine',
            output='screen',
            emulate_tty=True,
            parameters = [{'startStage': TextSubstitution(text=str(config['sim']['ros__parameters']['start_stage']))}]
        )
        alg = Node(
            package='controllers',
            executable='track',
            output='screen',
            parameters=[{'car_name': car_name},
                {'alg': TextSubstitution(text=str(alg))}, 
                {'path_file_path': TextSubstitution(text=str(config['sim']['ros__parameters']['path_file_path']))}],
        )

        if startStage == 'track':
            return LaunchDescription([
                SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
                SetParameter(name='use_sim_time', value=True),
                environment,
                alg,
                sim,
                state_machine,
            ])
        else:
            lidar_to_base_tf_node = Node( 
                package='tf2_ros', 
                executable='static_transform_publisher', 
                arguments=['0', '0', '0', '0', '0', '0', "f1tenthbase_link", "f1tenthhokuyo_10lx_lidar_link"], 
                output='screen'
            )
            odom_to_base_tf_node = Node(
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
                parameters=[{'car_name': car_name}],
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

    elif (f'{alg}' != 'rl'):
        alg = Node(
            package='controllers',
            executable=f'{alg}_policy',
            output='screen',
            parameters=[{'car_name': car_name}],
        )
    else:
        alg = IncludeLaunchDescription(
            launch_description_source = PythonLaunchDescriptionSource(os.path.join(pkg_controllers, f'{alg}.launch.py')),
            launch_arguments={'car_name': car_name,}.items()
        )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        SetParameter(name='use_sim_time', value=True),
        environment,
        alg,
        sim,
    ])

