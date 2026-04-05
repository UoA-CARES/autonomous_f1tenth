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
		package='joy_teleop',
		executable='joy_teleop',
		name='joy_teleop',
		parameters=[LaunchConfiguration('joy_config')]
	)
	ackermann_to_vesc_node = Node(
		package='vesc_ackermann',
		executable='ackermann_to_vesc_node',
		name='ackermann_to_vesc_node',
		parameters=[LaunchConfiguration('vesc_config')]
	)
	vesc_to_odom_node = Node(
		package='vesc_ackermann',
		executable='vesc_to_odom_node',
		name='vesc_to_odom_node',
		parameters=[LaunchConfiguration('vesc_config')],
		remappings=[('odom', 'f1tenth/odometry')]
	)
	vesc_driver_node = Node(
		package='vesc_driver',
		executable='vesc_driver_node',
		name='vesc_driver_node',
		parameters=[LaunchConfiguration('vesc_config')]
	)
	urg_node = Node(
		package='urg_node',
		executable='urg_node_driver',
		name='urg_node',
		parameters=[LaunchConfiguration('sensors_config')],
		remappings=[('scan', 'f1tenth/scan')]
	)
	ackermann_mux_node = Node(
		package='ackermann_mux',
		executable='ackermann_mux',
		name='ackermann_mux',
		parameters=[LaunchConfiguration('mux_config')],
		remappings=[('ackermann_cmd_out', 'ackermann_cmd')]
	)
	# ekf_filter_node = Node(
    #    package='robot_localization',
    #    executable='ekf_node',
    #    name='ekf_filter_node',
    #    output='screen',
    #    parameters=[os.path.join ( get_package_share_directory('f1tenth_bringup'), 'config', 'ekf_real.yaml')],
	# #    remappings=[('odometry/filtered','f1tenth/odometry')]
    # )

	# finalize
	# ld.add_action(joy_node)
	ld.add_action(joy_teleop_node)
	ld.add_action(ackermann_to_vesc_node)
	ld.add_action(vesc_to_odom_node)
	ld.add_action(vesc_driver_node)
	ld.add_action(ackermann_mux_node)
	# ld.add_action(ekf_filter_node)
	ld.add_action(urg_node)

	return ld