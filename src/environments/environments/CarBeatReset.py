import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from environment_interfaces.srv import CarBeatReset as CarBeatResetSrv
from f1tenth_control.SimulationServices import SimulationServices
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point

from ament_index_python import get_package_share_directory

from .util import get_quaternion_from_euler

class CarBeatReset(Node):
    def __init__(self):
        super().__init__('car_beat_reset')

        srv_cb_group = MutuallyExclusiveCallbackGroup()
        self.srv = self.create_service(CarBeatResetSrv, 'car_beat_reset', callback=self.service_callback, callback_group=srv_cb_group)

        set_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/empty/set_pose',
            callback_group=set_pose_cb_group
        )

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')
 

    def service_callback(self, request, response):

        # self.get_logger().info(f'Reset Service Request Received: relocating goal to x={request.cx_one} y={request.cy_one}')

        goal_req = self.create_request('goal', x=request.gx, y=request.gy, z=1)
        car_req_one = self.create_request(request.car_one, x=request.cx_one, y=request.cy_one, z=0.1, yaw=request.cyaw_one)
        car_req_two = self.create_request(request.car_two, x=request.cx_two, y=request.cy_two, z=0.1, yaw=request.cyaw_two)

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        #TODO: Call async and wait for both to execute
        if (request.flag == "goal_only"):
            self.set_pose_client.call(goal_req)
        else:
            self.set_pose_client.call(goal_req)
            self.set_pose_client.call(car_req_one)
            self.set_pose_client.call(car_req_two)


        # self.get_logger().info('Successfully Reset')
        response.success = True

        return response

    def create_request(self, name, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        req = SetEntityPose.Request()

        req.entity = Entity()
        req.entity.name = name
        req.entity.type = 2 # M
        
        req.pose = Pose()
        req.pose.position = Point()
        
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)

        orientation = get_quaternion_from_euler(roll, pitch, yaw)
        req.pose.orientation.x = orientation[0] 
        req.pose.orientation.y = orientation[1] 
        req.pose.orientation.z = orientation[2] 
        req.pose.orientation.w = orientation[3] 

        return req

def main():
    rclpy.init()
    pkg_environments = get_package_share_directory('environments')

    reset_service = CarBeatReset()
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