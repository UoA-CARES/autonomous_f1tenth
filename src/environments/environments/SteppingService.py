from rclpy.node import Node

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from ros_gz_interfaces.srv import ControlWorld
from std_srvs.srv import SetBool


class SteppingService(Node):
    def __init__(self):
        super().__init__('stepping_service')

        srv_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv = self.create_service(SetBool, 'stepping_service', callback=self.service_callback, callback_group=srv_cb_group)

        set_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self.world_control_client = self.create_client(
            ControlWorld,
            f'world/empty/control',
            callback_group=set_pose_cb_group
        )

        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')
 

    def service_callback(self, request, response):

        req = self.create_request(pause=request.data)

        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')
        
        self.world_control_client.call(req)
        
        response.success = True

        return response

    def create_request(self, pause=False):
        req = ControlWorld.Request()

        req.world_control.pause = pause

        return req

def main():
    rclpy.init()
    stepping_service = SteppingService()

    executor = MultiThreadedExecutor()
    executor.add_node(stepping_service)
    
    executor.spin()

    stepping_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()