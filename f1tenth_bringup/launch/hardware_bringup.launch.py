import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, PushRosNamespace
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution, LaunchConfiguration

def generate_launch_description():
	joy_config = os.path.join(
		get_package_share_directory('f1tenth_bringup'),
		'config',
		'joy.yaml'
	)
	vesc_config = os.path.join(
		get_package_share_directory('f1tenth_bringup'),
		'config',
		'vesc.yaml'
	)
	sensors_config = os.path.join(
		get_package_share_directory('f1tenth_bringup'),
		'config',
		'sensors.yaml'
	)
	mux_config = os.path.join(
		get_package_share_directory('f1tenth_bringup'),
		'config',
		'mux.yaml'
	)

	joy_la = DeclareLaunchArgument(
		'joy_config',
		default_value=joy_config,
		description='Descriptions for joy and joy_teleop configs')
	vesc_la = DeclareLaunchArgument(
		'vesc_config',
		default_value=vesc_config,
		description='Descriptions for vesc configs')
	sensors_la = DeclareLaunchArgument(
		'sensors_config',
		default_value=sensors_config,
		description='Descriptions for sensor configs')
	mux_la = DeclareLaunchArgument(
		'mux_config',
		default_value=mux_config,
		description='Descriptions for ackermann mux configs')

	ld = LaunchDescription([joy_la, vesc_la, sensors_la, mux_la])

	joy_node = Node(
		package='joy',
		executable='joy_node',
		name='joy',
		parameters=[LaunchConfiguration('joy_config')]
	)
	joy_teleop_node = Node(
