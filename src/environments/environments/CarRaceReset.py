import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from environment_interfaces.srv import Reset
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point
from .util import get_quaternion_from_euler

class CarRaceReset(Node):
    def __init__(self):
        super().__init__('car_race_reset')
        srv_cb_group = MutuallyExclusiveCallbackGroup()
        set_pose_cb_group = MutuallyExclusiveCallbackGroup()

        self.srv = self.create_service(Reset, 'car_race_reset', callback=self.service_callback, callback_group=srv_cb_group)
        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/empty/set_pose',
            callback_group=set_pose_cb_group
        )

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

    def service_callback(self, request, response):
        car_req = self.create_request(request.car_name, x=request.cx, y=request.cy, z=0, yaw=request.cyaw)
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        self.set_pose_client.call(car_req)
        response.success = True
        return response

    def create_request(self, name, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        req = SetEntityPose.Request()
        req.entity = Entity()
        req.pose = Pose()
        req.pose.position = Point()
        orientation = get_quaternion_from_euler(roll, pitch, yaw)

        req.entity.name = name
        req.entity.type = 2 
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.x = orientation[0] 
        req.pose.orientation.y = orientation[1] 
        req.pose.orientation.z = orientation[2] 
        req.pose.orientation.w = orientation[3] 
        return req

def main():
    rclpy.init()
    reset_service = CarRaceReset()
    reset_service.get_logger().info('Environment Spawning Complete')

    executor = MultiThreadedExecutor()
    executor.add_node(reset_service)
    executor.spin()
    reset_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()