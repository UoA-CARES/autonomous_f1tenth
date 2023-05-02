from launch import LaunchDescription 
from launch_ros.actions import Node 
import launch
import os
import json
import xacro

from ament_index_python.packages import get_package_share_directory

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, \
    SetEnvironmentVariable

from launch_ros.actions import Node

def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_f1tenth_description = get_package_share_directory('f1tenth_description')
    
    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/world/empty/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/empty/create@ros_gz_interfaces/srv/SpawnEntity',
            f'/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity',
            f'/world/empty/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ]
    )

    car_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/model/f1tenth/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        ]
    )
        # f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        # f'/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        # f'/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        # # f'/model/{name}/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
        # f'/world/{world}/model/{name}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model',
        # f'/model/{name}/pose@geometry_msgs/msg/Pose@gz.msgs.Pose',
    gz_sim = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': '-r empty.sdf',
        }.items()
    )
    
    # Launch the Environment
    car_goal = Node(
            package='environments',
            executable='CarGoal',
            output='screen',
            emulate_tty=True,
            arguments={
                'hello': 'wporld'
            }.items()
    )

    # F1tenth Spawning
    # TODO: move into own launch file - f1tenth_simulation.launch.py

    xacro_file = os.path.join(pkg_f1tenth_description, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    spawn_args = {'entity_factory': {'name': 'car', 'sdf_filename': f'{pkg_f1tenth_description}/urdf/robot.urdf.xacro'}}
    spawn_args = json.dumps(spawn_args)

    f1tenth_ros = ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/world/empty/create', 'ros_gz_interfaces/srv/SpawnEntity', spawn_args],
            output='screen')
    
    f1tenth_gz = ExecuteProcess(
            cmd=['gz', 'service', '-s', '/world/empty/create', '--reqtype', 'gz.msgs.EntityFactory', '--reptype', 'gz.msgs.Boolean', '--timeout', '1000', '--req', f'sdf: \"{robot_description}\", name: \"cool_car\"'],
            output='screen')
    
    f1tenth_ros_gz = Node(
            package='ros_gz_sim', executable='create',
            arguments=[
                '-world', 'empty',
                '-name', 'f1tenth',
                '-string', robot_description,
            ],
            output='screen'
        )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=pkg_f1tenth_description[:-19]),
        gz_sim,
        service_bridge,
        car_bridge,
        car_goal,
        f1tenth_ros_gz,
        # OpaqueFunction(function=spawn_func)
])