import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from environment_interfaces.srv import Reset
from f1tenth_control.SimulationServices import SimulationServices
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point

from ament_index_python import get_package_share_directory

from .util.util import get_quaternion_from_euler

class CarWallReset(Node):
    def __init__(self):
        super().__init__('car_wall_reset')

        srv_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv = self.create_service(Reset, 'car_wall_reset', callback=self.service_callback, callback_group=srv_cb_group)

        set_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/empty/set_pose',
            callback_group=set_pose_cb_group
        )

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')
 

    def service_callback(self, request, response):

        # self.get_logger().info(f'Reset Service Request Received: relocating goal to x={request.x} y={request.y}')

        goal_req = self.create_request('goal', x=request.gx, y=request.gy, z=1)
        car_req = self.create_request('f1tenth')

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        #TODO: Call async and wait for both to execute
        self.set_pose_client.call(goal_req)
        self.set_pose_client.call(car_req)

        # self.get_logger().info('Successfully Reset')
        response.success = True

        return response

    def create_request(self, name, x=0, y=0, z=0):
        req = SetEntityPose.Request()

        req.entity = Entity()
        req.entity.name = name
        req.entity.type = 2 # M
        
        req.pose = Pose()
        req.pose.position = Point()
        
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)

        return req

def main():
    rclpy.init()
    pkg_environments = get_package_share_directory('environments')

    reset_service = CarWallReset()
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