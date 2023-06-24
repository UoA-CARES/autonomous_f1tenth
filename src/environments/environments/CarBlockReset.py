import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import random

from environment_interfaces.srv import Reset
from f1tenth_control.SimulationServices import SimulationServices
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point

from ament_index_python import get_package_share_directory

from .util import get_quaternion_from_euler, generate_position

class CarBlockReset(Node):
    def __init__(self):
        super().__init__('car_block_reset')

        self.srv = self.create_service(
            Reset, 
            'car_block_reset', 
            callback=self.service_callback, 
            callback_group= MutuallyExclusiveCallbackGroup()
        )

        self.set_pose_client = self.create_client(
            SetEntityPose,
            f'world/empty/set_pose',
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')
 

    def service_callback(self, request, response):
        
        # self.get_logger().info(f'Reset Service Request Received: relocating goal to x={request.x} y={request.y}')

        # Move the goal to new position & car back to origin
        goal_req = self.create_request('goal', x=request.x, y=request.y, z=1)
        car_req = self.create_request('f1tenth')

        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_pose service not available, waiting again...')

        #TODO: Call async and wait for both to execute
        self.set_pose_client.call(goal_req)
        self.set_pose_client.call(car_req)
        
        self.reset_obstacles()

        sm, md, lg = np.random.randint(low=2, high=5, size=(3,))
        
        self.set_obstacles(small=sm, medium=md, large=lg)
        

        response.success = True

        return response

    def reset_obstacles(self):

        for i in range(1, 4):
            self.set_pose_client.call(self.create_request(f'small_{i}',z=-10))
        
        for i in range(1, 4):
            self.set_pose_client.call(self.create_request(f'medium_{i}',z=-10))
        
        for i in range(1, 4):
            self.set_pose_client.call(self.create_request(f'large_{i}',z=-10))


    def set_obstacles(self, small, medium, large):
        
        for i in range(small):
            x, y = generate_position(inner_bound=5, outer_bound=8)
            self.set_pose_client.call(self.create_request(f'small_{i + 1}',x=x, y=y, ya=random.randint(0, 360)))
        
        for i in range(medium):
            x, y = generate_position(inner_bound=6, outer_bound=8)
            self.set_pose_client.call(self.create_request(f'medium_{i + 1}',x=x, y=y, ya=random.randint(0, 360)))
        
        for i in range(large):
            x, y = generate_position(inner_bound=6, outer_bound=8)
            self.set_pose_client.call(self.create_request(f'large_{i + 1}',x=x, y=y, ya=random.randint(0, 360)))

    def create_request(self, name, x=0, y=0, z=0, r=0, p=0, ya=0):
        req = SetEntityPose.Request()

        req.entity = Entity()
        req.entity.name = name
        req.entity.type = 2 # M
        
        req.pose = Pose()
        req.pose.position = Point()
        
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)

        q_x, q_y, q_z, q_w = get_quaternion_from_euler(r, p, ya)

        req.pose.orientation.x = q_x
        req.pose.orientation.y = q_y
        req.pose.orientation.z = q_z
        req.pose.orientation.w = q_w

        return req

def main():
    rclpy.init()
    pkg_environments = get_package_share_directory('environments')

    reset_service = CarBlockReset()
    pkg_environments = get_package_share_directory('environments')

    services = SimulationServices('empty')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/goal.sdf", pose=[1, 1, 1], name='goal')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_small.sdf", pose=[1, 1, -10], name='small_1')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_small.sdf", pose=[1, 1, -10], name='small_2')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_small.sdf", pose=[1, 1, -10], name='small_3')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_small.sdf", pose=[1, 1, -10], name='small_4')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_small.sdf", pose=[1, 1, -10], name='small_5')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle.sdf", pose=[1, 1, -10], name='medium_1')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle.sdf", pose=[1, 1, -10], name='medium_2')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle.sdf", pose=[1, 1, -10], name='medium_3')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle.sdf", pose=[1, 1, -10], name='medium_4')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle.sdf", pose=[1, 1, -10], name='medium_5')

    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_large.sdf", pose=[1, 1, -10], name='large_1')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_large.sdf", pose=[1, 1, -10], name='large_2')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_large.sdf", pose=[1, 1, -10], name='large_3')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_large.sdf", pose=[1, 1, -10], name='large_4')
    services.spawn(sdf_filename=f"{pkg_environments}/sdf/obstacle_large.sdf", pose=[1, 1, -10], name='large_5')

    reset_service.get_logger().info('Environment Spawning Complete')

    executor = MultiThreadedExecutor()
    executor.add_node(reset_service)
    
    executor.spin()

    # rclpy.spin(reset_service)
    reset_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()