import rclpy

from rclpy.task import Future

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from ament_index_python.packages import get_package_share_directory

from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity, SetEntityPose
from ros_gz_interfaces.msg import Entity, EntityFactory

from f1tenth_control.environment import Environment

import numpy as np

import time
import math
import uuid
import random

MODEL = 2


class RLEnv:
    """
    Simulation Environment Wrapper, used to interact with simulation according to OpenAI environment API
    """

    def __init__(self, reward_range=1, max_steps=15, collision_range=0.5, sim_speed_mult=1, delay_between_ep=0.5,
                 obstacle_num=3):
        """
        Initializes a reinforcement learning environment with the specified parameters.

        :param reward_range: The range of rewards that can be given for each step. Default is 1.
        :type reward_range: int or float
        :param max_steps: The maximum number of steps allowed in each episode. Default is 15.
        :type max_steps: int
        :param collision_range: The range at which a collision is detected. Default is 0.5.
        :type collision_range: int or float
        :param sim_speed_mult: The speed multiplier for the simulation. Default is 1.
        :type sim_speed_mult: int or float
        :param delay_between_ep: The delay in seconds between each episode. Default is 0.5.
        :type delay_between_ep: int or float
        :param obstacle_num: The number of obstacles to generate. Default is 3.
        :type obstacle_num: int
        """

        # delay between each step
        self.delay_between_ep = delay_between_ep / sim_speed_mult
        self.sim_hz = 1000 * sim_speed_mult

        self.env: Environment = Environment()
        self.env.config_world(
            world_file='empty.sdf',
            world_name='empty'
        ).build()

        time.sleep(2.5)

        self.entity = self.env.create_entity((0, 0))

        time.sleep(2.5)

        self.goal = {'name': None, 'position': None, 'visualise': False}
        self.goal_file_path = f"{get_package_share_directory('reinforcement_learning')}/sdf/goal.sdf"

        self.obstacles = []
        self.obstacle_paths = [
            f"{get_package_share_directory('reinforcement_learning')}/sdf/obstacle_small.sdf",
            f"{get_package_share_directory('reinforcement_learning')}/sdf/obstacle.sdf",
            f"{get_package_share_directory('reinforcement_learning')}/sdf/obstacle_large.sdf"
        ]

        self.obstacle_file_path = f"{get_package_share_directory('reinforcement_learning')}/sdf/obstacle.sdf"

        self.REWARD_RANGE = reward_range
        self.COLLISION_RANGE = collision_range

        self.max_step = max_steps
        self.step_counter = 0

        # Spawn Service --------------------------------------
        self.spawn_client = self.env.create_client(
            SpawnEntity,
            f'world/{self.env.world_name}/create',
        )
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.env.get_logger().info('spawn service not available, waiting again...')

        self.spawn_future = Future()

        # Delete Service --------------------------------------
        self.delete_client = self.env.create_client(
            DeleteEntity,
            f'world/{self.env.world_name}/remove',
        )
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.env.get_logger().info('delete service not available, waiting again...')

        self.delete_future = Future()

        # Set Pose Service --------------------------------------
        self.set_pose_client = self.env.create_client(
            SetEntityPose,
            f'world/{self.env.world_name}/set_pose',
        )
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.env.get_logger().info('delete service not available, waiting again...')

        self.delete_future = Future()

        # self.generate_obstacles(obstacle_num)
        # self.spawn(sdf_filename=f"{get_package_share_directory('reinforcement_learning')}/sdf/wall.sdf", name="wall")

    def log(self, msg):
        self.env.get_logger().info(msg)

    def spawn(self, *, sdf=None, sdf_filename=None, name='cool_car', pose=None, orientation=None):
        """
        Spawns Entity inside simulation

        :param sdf: sdf or urdf in string form
        :param sdf_filename: the path to your sdf or urdf file
        :param name: desired name of the entity
        :param pose: [x_pos, y_pos, z_pos]: desired x, y, and z position for entity to spawn

        :return: nothing
        """
        if sdf and sdf_filename:
            raise Exception('You passed both an sdf, and a path; pass only ONE')

        if pose is None:
            pose = [0, 0, 0]

        if orientation is None:
            orientation = [0, 0, 0, 0]

        if len(pose) != 3:
            raise Exception(f'Expected Pose length is 3, you gave {len(pose)}')

        request = SpawnEntity.Request()

        request.entity_factory = EntityFactory()

        if sdf:
            request.entity_factory.sdf = sdf
        elif sdf_filename:
            request.entity_factory.sdf_filename = sdf_filename

        x, y, z = pose
        request.entity_factory.pose.position.x = float(x)
        request.entity_factory.pose.position.y = float(y)
        request.entity_factory.pose.position.z = float(z)

        q_x, q_y, q_z, q_w = orientation
        request.entity_factory.pose.orientation.x = float(q_x)
        request.entity_factory.pose.orientation.y = float(q_y)
        request.entity_factory.pose.orientation.z = float(q_z)
        request.entity_factory.pose.orientation.w = float(q_w)

        request.entity_factory.name = name

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.env.get_logger().info('spawn service not available, waiting again...')

        self.spawn_future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self.env, self.spawn_future)

        return self.spawn_future.result()

    def delete_entity(self, name, entity_type):
        """
        Deletes an entity from gazebo simulation

        :param name: name of the entity
        :param entity_type: int: type of entity using the following scheme
            NONE      = 0
            LIGHT     = 1
            MODEL     = 2
            LINK      = 3
            VISUAL    = 4
            COLLISION = 5
            SENSOR    = 6
            JOINT     = 7
        :return: none
        """

        request = DeleteEntity.Request()
        request.entity = Entity()

        request.entity.name = name
        request.entity.type = entity_type

        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.env.get_logger().info('delete service not available, waiting again...')

        self.delete_future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self.env, self.delete_future)

        return self.delete_future.result()

    def reset(self):
        """
        Calls the reset method on all entities inside the simulation
        :return:
            state: [
                car_x_pos,
                car_y_pos,
                car_z_orientation,
                car_w_orientation,
                goal_x_pos,
                goal_y_pos
                ]:
                the state of the environment, after reset
            info: additional information about the environment
        """
        self.env.reset()

        self.step_counter = 0

        if self.goal['visualise']:
            self.delete_entity(
                name=self.goal['id'],
                entity_type=MODEL
            )
        time.sleep(self.delay_between_ep)

        odom, lidar = self.entity.get_data()

        odom = self.process_odom(odom)
        ranges, _ = self.process_lidar(lidar)

        self.goal = self.generate_goal(True)
        goal_position = self.goal['position']

        state = np.concatenate((odom, ranges, goal_position))

        return state, self.get_info()

    def step(self, action):
        """
        Make the car take an action, and observe the result after delay_between_ep seconds
        :param action: [lin_vel, ang_vel]: an array that contains the desired linear and angular velocity
        :return: nothing
        """
        self.step_counter += 1

        old_odom, _ = self.entity.get_data()
        old_odom = self.process_odom(old_odom)

        lin_vel, ang_vel = action

        lin_vel = float(lin_vel)
        ang_vel = float(ang_vel)

        self.entity.set_velocity(lin_vel, ang_vel)

        from timeit import default_timer as timer

        start = timer()
        has_collided = False

        while timer() - start < self.delay_between_ep and not has_collided:
            _, lidar = self.entity.get_data()
            ranges, _ = self.process_lidar(lidar)
            has_collided = has_collided or self.has_collided(ranges)

        odom, lidar = self.entity.get_data()

        odom = self.process_odom(odom)
        ranges, _ = self.process_lidar(lidar)

        reward = self.compute_reward(old_odom, odom, ranges)

        state = np.concatenate((odom, self.goal['position'], ranges))

        return state, reward, self.is_terminated(state) or has_collided, self.is_truncated(), self.get_info()

    def is_terminated(self, state):
        """
        Check if the environment has reached a terminal condition
        :param state: state of the current environment
        :return: bool: whether the environment has reached a terminal state
        """
        return self.reached_goal(state) or self.has_collided(state[8:])

    def reached_goal(self, state):
        current_car_position = state[:2]
        return math.dist(self.goal['position'], current_car_position) < self.REWARD_RANGE

    def has_collided(self, lidar_ranges):

        return any(ray < self.COLLISION_RANGE for ray in lidar_ranges)

    def is_truncated(self):
        """
        Checks if the current episode has been running too long. Prevents episode from running forever
        :return: bool: if the episode has gone on too long
        """
        return self.step_counter > self.max_step

    def compute_reward(self, old_odom, new_odom, new_lidar):
        """
        Computes the reward that the agent earned during the two states. The reward in this sim, is based the agent's
        progress. If the agent moved toward the goal, it earns positive reward, away earns negative reward. Reaching the
        goal earns the agent an extra 100
        :param old_state: [
                car_x_pos,
                car_y_pos,
                car_z_orientation,
                car_w_orientation,
                goal_x_pos,
                goal_y_pos
                ]:
                the earlier state
        :param new_state: [
                car_x_pos,
                car_y_pos,
                car_z_orientation,
                car_w_orientation,
                goal_x_pos,
                goal_y_pos
                ]:
                the later state
        :return: float: the reward earned by the agent
        """
        current_distance = math.dist(self.goal['position'], new_odom[:2])

        old_distance = math.dist(self.goal['position'], old_odom[:2])

        delta_distance = old_distance - current_distance

        reward = 0

        if current_distance < self.REWARD_RANGE:
            reward += 100

        if self.has_collided(new_lidar):
            reward -= 100

        reward += delta_distance

        return reward

    def get_info(self):
        return None

    def generate_goal(self, visualise: bool = False):
        """
        Randomly generates a goal position within 20 x 20 meter box
        :param visualise: spawn the goal
        :return:
            name: the name of the goal spawned into the sim
            [x, y]: position of the new goal
        """
        name = uuid.uuid4().hex
        x_pos = random.uniform(-5, 5)
        x_pos = x_pos + 3 if x_pos >= 0 else x_pos - 3
        y_pos = random.uniform(-5, 5)
        y_pos = y_pos + 3 if y_pos >= 0 else y_pos - 3

        pos = [x_pos, y_pos]
        if visualise:
            self.spawn(sdf_filename=self.goal_file_path,
                       name=name,
                       pose=[pos[0], pos[1], 1]
                       )

        return {'id': name, 'position': pos, 'visualise': visualise}

    def generate_obstacles(self, num):
        for _ in range(num):
            obs_file_path = random.choice(self.obstacle_paths)
            self.obstacles.append(self.generate_obstacle(obs_file_path))

    def generate_obstacle(self, obs_file_path):
        name = uuid.uuid4().hex
        x_pos = random.uniform(-5, 5)
        x_pos = x_pos + 3 if x_pos >= 0 else x_pos - 3
        y_pos = random.uniform(-5, 5)
        y_pos = y_pos + 3 if y_pos >= 0 else y_pos - 3

        # x, y = np.random.uniform(-10, 10, (2,))
        yaw = np.random.uniform(0, np.pi, (1,))
        orientation = self.get_quaternion_from_euler(0, 0, yaw)

        self.spawn(sdf_filename=obs_file_path,
                   name=name,
                   pose=[x_pos, y_pos, 0],
                   orientation=orientation
                   )

        return {'id': name, 'position': [x_pos, y_pos, 0]}

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Implementation from: https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/

        Convert an Euler angle to a quaternion.

        Input
          :param roll: The roll (rotation around x-axis) angle in radians.
          :param pitch: The pitch (rotation around y-axis) angle in radians.
          :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
          :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return [qx, qy, qz, qw]

    def process_lidar(self, lidar: LaserScan):
        """
            Transforms the raw lidar data into a more digestible format. Here, we only use the ranges and intensities
            :param lidar: Raw LaserScan data
            :return: the processed lidar data
            """

        ranges = lidar.ranges
        ranges = np.nan_to_num(ranges, posinf=float(10))

        intensities = lidar.intensities
        return ranges, intensities

    def process_odom(self, odom: Odometry):
        """
        Transforms the raw odometer data into a more digestible format. Here, we only use the following:
            Position:
                x and y
            Quaternion:
                z and w
            Velocity:
                linear and angular
        :param odom: Raw odometer data
        :return: the processed odometer data
        """
        pose = odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        twist = odom.twist.twist
        lin_vel = twist.linear
        ang_vel = twist.angular

        return np.array([position.x, position.y, orientation.z, orientation.w, lin_vel.x, ang_vel.z])
