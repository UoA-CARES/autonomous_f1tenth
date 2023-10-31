from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

def generate_launch_description():
    car_name = LaunchConfiguration('car_name')

    car_name_arg = DeclareLaunchArgument(
        'car_name',
        default_value='ftg_car'
    )
 
    main = Node(
        package='controllers',
        executable='ftg_policy',
        output='screen',
        parameters=[{'car_name': car_name}],
    )

    return LaunchDescription([
        car_name_arg,
        main
        ])