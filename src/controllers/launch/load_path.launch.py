from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution


def generate_launch_description():
    load_path = Node(
        package='controllers',
        executable='load_path',
        output='screen',
        parameters=[{'car_name': 'f1tenth'}],
        
    )


    return LaunchDescription([
        load_path
])