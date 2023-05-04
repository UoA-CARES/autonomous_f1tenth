import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from environment_interfaces.srv import Reset
from simulation.simulation_services import SimulationServices

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point

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

        self.get_logger().info(f'Reset Service Request Received: relocating goal to x={request.x} y={request.y}')

        goal_req = self.create_request('goal', x=request.x, y=request.y, z=1)
        car_req = self.create_request('f1tenth')

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        #TODO: Call async and wait for both to execute
        self.set_pose_client.call(goal_req)
        self.set_pose_client.call(car_req)

        self.get_logger().info('Successfully Reset')
        response.success = True

        return response

    def create_request(self, name, x=0, y=0, z=0.5):
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

    reset_service = CarWallReset()

    executor = MultiThreadedExecutor()
    executor.add_node(reset_service)
    
    executor.spin()

    # rclpy.spin(reset_service)
    reset_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()