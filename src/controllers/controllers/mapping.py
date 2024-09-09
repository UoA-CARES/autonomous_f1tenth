import sys

from slam_toolbox.srv import SaveMap
import rclpy
from rclpy.node import Node
import time


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(SaveMap, '/slam_toolbox/save_map')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SaveMap.Request()

    def send_request(self, name):
        self.req.name.data = name
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    time.sleep(5)
    response = minimal_client.send_request('myMap')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()