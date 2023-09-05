import math
import numpy as np
import random
import rclpy
from rclpy import Future
from launch_ros.actions import Node
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry

from environment_interfaces.srv import CarBeatReset
from .termination import has_collided, has_flipped_over
from .util import process_odom, reduce_lidar
from .track_reset import track_info
from .goal_positions import goal_positions
from .waypoints import waypoints

class CarBeatEnvironment(Node):

    def __init__(self, car_one_name, car_two_name, reward_range=1, max_steps=50, collision_range=0.2, step_length=0.5, track='track_1', observation_mode= 'full', laps_to_run=1, max_goals=500):
        super().__init__('car_beat_environment')

        # Environment Details ----------------------------------------
        self.NAME = car_one_name
        self.OTHER_CAR_NAME = car_two_name
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length
        self.MAX_ACTIONS = np.asarray([3, 3.14])
        self.MIN_ACTIONS = np.asarray([0, -3.14])
        self.MAX_STEPS_PER_GOAL = max_steps
        self.OBSERVATION_MODE = observation_mode
        
        self.MAX_GOALS = max_goals
        match observation_mode:
            case 'full':
                self.OBSERVATION_SIZE = 8 + 10 
            case 'no_position':
                self.OBSERVATION_SIZE = 6 + 10
            case 'lidar_only':
                self.OBSERVATION_SIZE = 2 + 10
            case _:
                raise ValueError(f'Invalid observation mode: {observation_mode}')

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range
        self.ACTION_NUM = 2

        self.step_counter = 0

        self.track = track

        self.laps_to_run = laps_to_run

        # Goal/Track Info -----------------------------------------------
        self.goals_reached = 0
        self.start_goal_index = 0

        self.ftg_goals_reached = 0
        self.ftg_start_goal_index = 0
        
        self.steps_since_last_goal = 0

        if track != 'multi_track':
            self.all_goals = goal_positions[track]
            self.car_waypoints = waypoints[track]
        else:
            austin_gp = goal_positions['austin_track']
            budapest_gp = goal_positions['budapest_track']
            budapest_gp = [[x + 200, y] for x, y in budapest_gp]
            hockenheim_gp = goal_positions['hockenheim_track']
            hockenheim_gp = [[x + 300, y] for x, y in hockenheim_gp]
            
            self.all_car_goals = {
                'austin_track': austin_gp,
                'budapest_track': budapest_gp,
                'hockenheim_track': hockenheim_gp 
            }

            austin_wp = waypoints['austin_track']

            budapest_wp = []
            for x, y, yaw, index in waypoints['budapest_track']:
                x += 200
                budapest_wp.append((x, y, yaw, index))
            
            hockenheim_wp = []
            for x, y, yaw, index in waypoints['hockenheim_track']:
                x += 300
                hockenheim_wp.append((x, y, yaw, index))
            
            self.all_car_waypoints = {
                'austin_track': austin_wp,
                'budapest_track': budapest_wp,
                'hockenheim_track': hockenheim_wp
            }

            self.current_track = 'austin_track'

            self.all_goals = self.all_car_goals[self.current_track]
            self.car_waypoints = self.all_car_waypoints[self.current_track]

        # Pub/Sub ----------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            10
        )
        
        self.reset_pub = self.create_publisher(
            Empty,
            f'/reset',
            10
        )

        self.odom_sub_one = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.lidar_sub_one = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.odom_sub_two = Subscriber(
            self,
            Odometry,
            f'/{self.OTHER_CAR_NAME}/odometry',
        )

        self.lidar_sub_two = Subscriber(
            self,
            LaserScan,
            f'/{self.OTHER_CAR_NAME}/scan',
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub_one, self.lidar_sub_one, self.odom_sub_two, self.lidar_sub_two],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        # Reset Client -----------------------------------------------
        self.reset_client = self.create_client(
            CarBeatReset,
            'car_beat_reset'
        )

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def reset(self):
        self.step_counter = 0

        self.steps_since_last_goal = 0
        self.goals_reached = 0
        self.ftg_goals_reached = 30

        self.set_velocity(0, 0)

        if self.track == 'multi_track': 
            self.current_track = random.choice(list(self.all_car_goals.keys()))
            self.all_goals = self.all_car_goals[self.current_track]
            self.car_waypoints = self.all_car_waypoints[self.current_track]

        # New random starting point for the cars
        car_x, car_y, car_yaw, index = random.choice(self.car_waypoints)
        ftg_x, ftg_y, ftg_yaw, ftg_index = self.car_waypoints[(index + 30) % len(self.car_waypoints)]

        self.start_goal_index = index
        self.ftg_start_goal_index = index

        self.goal_position = self.all_goals[self.start_goal_index]
        self.ftg_goal_position = self.all_goals[self.ftg_start_goal_index]

        self.sleep()

        goal_x, goal_y = self.goal_position

        self.call_reset_service(
            car_x=car_x,
            car_y=car_y,
            car_Y=car_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            ftg_x=ftg_x,
            ftg_y=ftg_y,
            ftg_Y=ftg_yaw
        )

        state, _ = self.get_observation()

        info = {}

        return state, info

    def step(self, action):
        self.step_counter += 1

        _, full_state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()

        next_state, full_next_state  = self.get_observation()
        reward = self.compute_reward(full_state, full_next_state)
        terminated = self.is_terminated(full_next_state)
        truncated = self.steps_since_last_goal >= 10
        info = {}

        return next_state, reward, terminated, truncated, info

    def message_filter_callback(self, odom_one: Odometry, lidar_one: LaserScan, odom_two: Odometry, lidar_two: LaserScan):
        self.observation_future.set_result({'odom_one': odom_one, 'lidar_one': lidar_one, 'odom_two': odom_two, 'lidar_two': lidar_two})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom_one'], data['lidar_one'], data['odom_two'], data['lidar_two'] 

    def set_velocity(self, linear, angular):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(linear)

        self.cmd_vel_pub.publish(velocity_msg)

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)

    def is_terminated(self, state):
        
        return has_collided(state[8:19], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6]) \
            or self.goals_reached >= self.MAX_GOALS

    '''
    init ai car goal count 0
    init opponent goal count diffrenct between ai goal abs to oppent goal abs

    for each goal passed
    ai car goal count ++
    oppoent goal count ++

    if ai car goal count > oppent goal count:
        passed
    '''

    def call_reset_service(self, 
                           car_x, 
                           car_y, 
                           car_Y, 
                           goal_x, 
                           goal_y, 
                           ftg_x, 
                           ftg_y, 
                           ftg_Y
                           ):
        """
        Reset the car and goal position
        """

        request = CarBeatReset.Request()
        
        request.gx = goal_x
        request.gy = goal_y

        request.cx_one = car_x
        request.cy_one = car_y
        request.cyaw_one = car_Y

        request.cx_two = ftg_x
        request.cy_two = ftg_y
        request.cyaw_two = ftg_Y

        request.flag = "car_and_goal"

        # Publish to reset Topic to reset other nodes
        empty_msg = Empty()
        self.reset_pub.publish(empty_msg)

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, x, y):
        """
        Reset the goal position
        """

        request = CarBeatReset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom_one, lidar_one, odom_two, lidar_two = self.get_data()

        odom_one = process_odom(odom_one)
        odom_two = process_odom(odom_two)

        lidar_one = reduce_lidar(lidar_one)
        lidar_two = reduce_lidar(lidar_two)

        match self.OBSERVATION_MODE:
            case 'full':
                state = odom_one + lidar_one
            case 'no_position':
                state = odom_one[2:] + lidar_one
            case 'lidar_only':
                state = odom_one[-2:] + lidar_one
            case _:
                ValueError(f'Invalid observation mode: {self.OBSERVATION_MODE}')

        full_state = odom_one + lidar_one + odom_two + lidar_two + self.goal_position

        return state, full_state
 
    def compute_reward(self, state, next_state):

        reward = 0

        goal_position = self.goal_position

        current_distance = math.dist(goal_position, next_state[:2])
        
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.goals_reached} Reached')
            reward += 2
            self.goals_reached += 1

            # Updating Goal Position
            new_x, new_y = self.all_goals[(self.start_goal_index + self.goals_reached) % len(self.all_goals)]
            self.goal_position = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.steps_since_last_goal = 0

        ftg_current_distance = math.dist(self.all_goals[self.ftg_start_goal_index + self.ftg_goals_reached - 30], next_state[18:20])
        
        # Keeping track of FTG car goal number
        if ftg_current_distance < self.REWARD_RANGE:
            self.ftg_goals_reached += 1
        
        # If RL car has overtaken FTG car
        if self.goals_reached >= (self.ftg_goals_reached + 5):
            reward  += 200

        if has_collided(next_state[8:19], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        return reward
