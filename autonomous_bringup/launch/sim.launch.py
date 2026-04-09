import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    pkg_f1tenth_description = get_package_share_directory("f1tenth_description")
    pkg_bringup = get_package_share_directory("autonomous_bringup")
    pkg_controllers = get_package_share_directory("f1tenth_controllers")
    pkg_slam = get_package_share_directory("slam_toolbox")

    config_path = os.path.join(pkg_bringup, "config", "sim.yaml")
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    env = config["sim"]["ros__parameters"]["environment"]
    alg = config["sim"]["ros__parameters"]["algorithm"]
    start_stage = config["sim"]["ros__parameters"]["start_stage"]
    car_name = config["sim"]["ros__parameters"].get("car_name", "f1tenth")
    track = config["sim"]["ros__parameters"]["track"]
    path_file_path = config["sim"]["ros__parameters"]["path_file_path"]

    env_launch = PythonLaunchDescriptionSource(
        os.path.join(pkg_bringup, f"{env.lower()}.launch.py")
    )
    alg_launch = PythonLaunchDescriptionSource(
        os.path.join(pkg_controllers, f"{alg}.launch.py")
    )
    env_var = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=pkg_f1tenth_description[:-19]
    )

    match alg:
        case "rl" | "ftg":
            tracking = False
        case "random" | "turn_drive" | "mpc" | "pure_pursuit":
            tracking = True

    environment = IncludeLaunchDescription(
        env_launch,
        launch_arguments={
            "track": track,
            "car_name": car_name,
            "car_one": car_name,
        }.items(),
    )

    sim = Node(
        package="f1tenth_controllers",
        executable="sim",
        parameters=[config_path],
        name="sim",
        output="screen",
        emulate_tty=True,
    )

    if tracking:
        state_machine = Node(
            package="f1tenth_controllers",
            executable="state_machine",
            output="screen",
            emulate_tty=True,
            parameters=[{"startStage": start_stage}],
        )
        alg = Node(
            package="f1tenth_controllers",
            executable="track",
            output="screen",
            parameters=[
                {"car_name": car_name},
                {"alg": alg},
                {"path_file_path": path_file_path},
            ],
        )

        if start_stage == "track":
            return LaunchDescription(
                [
                    env_var,
                    SetParameter(name="use_sim_time", value=True),
                    environment,
                    alg,
                    sim,
                    state_machine,
                ]
            )
        else:
            lidar_to_base_tf_node = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "f1tenthbase_link",
                    "f1tenthhokuyo_10lx_lidar_link",
                ],
                output="screen",
            )
            odom_to_base_tf_node = Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[
                    os.path.join(pkg_f1tenth_description, "config/ekf.yaml"),
                    {"use_sim_time": True},
                ],
            )
            slam_node = IncludeLaunchDescription(
                launch_description_source=PythonLaunchDescriptionSource(
                    os.path.join(pkg_slam, "launch", "online_async_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "True",
                    "slam_params_file": os.path.join(
                        pkg_f1tenth_description,
                        "config",
                        "slam_toolbox.yaml",
                    ),
                }.items(),
            )
            ftg_node = Node(
                package="f1tenth_controllers",
                executable="ftg_policy",
                output="screen",
                parameters=[{"car_name": car_name}],
            )

            return LaunchDescription(
                [
                    env_var,
                    SetParameter(name="use_sim_time", value=True),
                    environment,
                    alg,
                    sim,
                    lidar_to_base_tf_node,
                    odom_to_base_tf_node,
                    slam_node,
                    state_machine,
                    ftg_node,
                ]
            )

    elif f"{alg}" != "rl":
        alg = Node(
            package="f1tenth_controllers",
            executable=f"{alg}_policy",
            output="screen",
            parameters=[{"car_name": car_name}],
        )
    else:
        alg = IncludeLaunchDescription(
            alg_launch, launch_arguments={"car_name": car_name}
        )

    return LaunchDescription(
        [env_var, SetParameter(name="use_sim_time", value=True), environment, alg, sim]
    )
