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

	joy_la = DeclareLaunchArgument(
		'joy_config',
		default_value=joy_config,
		description='Descriptions for joy and joy_teleop configs')

	ld = LaunchDescription([joy_la])

	joy_node = Node(
		package='joy',
		executable='joy_node',
		name='joy',
		parameters=[LaunchConfiguration('joy_config')]
	)

	# finalize
	ld.add_action(joy_node)

	return ld
