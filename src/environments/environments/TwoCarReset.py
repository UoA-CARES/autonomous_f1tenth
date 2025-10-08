import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from environments.F1tenthReset import F1tenthReset
from environment_interfaces.srv import Reset
from f1tenth_control.SimulationServices import SimulationServices
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point

from ament_index_python import get_package_share_directory

from .util import get_quaternion_from_euler

class TwoCarReset(F1TenthReset):
    def __init__(self):
        super().__init__('two_car')

def main():
    rclpy.init()
    pkg_environments = get_package_share_directory('environments')

    reset_service = TwoCarReset()
    pkg_environments = get_package_share_directory('environments')

    services = SimulationServices('empty')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf", pose=[1, 1, 1], name='goal')

    reset_service.get_logger().info('Environment Spawning Complete')

    executor = MultiThreadedExecutor()
    executor.add_node(reset_service)
    
    executor.spin()

    # rclpy.spin(reset_service)
    reset_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()