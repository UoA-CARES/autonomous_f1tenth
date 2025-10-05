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
import numpy as np
from environment_interfaces.srv import CarBeatReset
from std_srvs.srv import SetBool
from .util import process_odom, avg_lidar, create_lidar_msg, get_all_goals_and_waypoints_in_multi_tracks, ackermann_to_twist, has_collided, has_flipped_over

from .goal_positions import goal_positions
from .waypoints import waypoints
import yaml

class CarBeatEnvironment(Node):

    """
    CarBeat Reinforcement Learning Environment:

        Task:
            Agent learns to drive a track and overtake a car that is driving at a constant speed.
            The second car is using the Follow The Gap algorithm.

        Observation:
            full:
                Car Position (x, y)
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            no_position:
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            lidar_only:
                Car Velocity
                Car Angular Velocity
                Lidar Data

            No. of lidar points is configurable

        Action:
            It's linear and angular velocity (Twist)
        
        Reward:
            +2 if it comes within REWARD_RANGE units of a goal
            +200 if it overtakes the Follow The Gap car
            -25 if it collides with a wall

        Termination Conditions:
            When the agent collides with a wall or the Follow The Gap car
        
        Truncation Condition:
            When the number of steps surpasses MAX_GOALS
    """
    def __init__(self,
                 rl_car_name,
                 ftg_car_name,
                 reward_range=1,
                 max_steps=50,
                 collision_range=0.2,
                 step_length=0.5,
                 track='multi_track',
                 observation_mode='lidar_only',
                 max_goals=500,
                 num_lidar_points=10,
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__('car_beat_environment')

        # Environment Details ----------------------------------------
                
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        self.NAME = rl_car_name
        self.OTHER_CAR_NAME = ftg_car_name
        self.MAX_STEPS = max_steps
        self.STEP_LENGTH = step_length
        self.MAX_STEPS_PER_GOAL = max_steps
        self.OBSERVATION_MODE = observation_mode
        self.NUM_SPAWNS = 0
        self.LIDAR_NUM = num_lidar_points
        
        self.MAX_GOALS = max_goals
        match observation_mode:
            case 'full':
                self.OBSERVATION_SIZE = 8 + 10 
            case 'no_position':
                self.OBSERVATION_SIZE = 6 + 10
            case 'lidar_only':
                self.OBSERVATION_SIZE = 2 + num_lidar_points
            case _:
                raise ValueError(f'Invalid observation mode: {observation_mode}')

        self.COLLISION_RANGE = collision_range
        self.REWARD_RANGE = reward_range

        self.STEP_COUNTER = 0

        self.TRACK = track

        # Goal/Track Info -----------------------------------------------
        self.GOALS_REACHED = 0
        self.START_GOAL_INDEX = 0

        self.FTG_GOALS_REACHED = 0
        self.FTG_START_GOAL_INDEX = 0
        self.FTG_OFFSET = 0
        self.STEPS_SINCE_LAST_GOAL = 0

        if 'multi_track' not in track:
            self.ALL_GOALS = goal_positions[track]
            self.CAR_WAYPOINTS = waypoints[track]
        else:
            self.ALL_CAR_GOALS, self.ALL_CAR_WAYPOINTS = get_all_goals_and_waypoints_in_multi_tracks(track)
            self.CURRENT_TRACK = list(self.ALL_CAR_GOALS.keys())[0]

            self.ALL_GOALS = self.ALL_CAR_GOALS[self.CURRENT_TRACK]
            self.CAR_WAYPOINTS = self.ALL_CAR_WAYPOINTS[self.CURRENT_TRACK]

        # Pub/Sub ----------------------------------------------------
        self.CMD_VEL_PUB = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            10
        )
        
        self.RESET_PUB = self.create_publisher(
            Empty,
            f'/reset',
            10
        )

        self.ODOM_SUB_ONE = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.LIDAR_SUB_ONE = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.ODOM_SUB_TWO = Subscriber(
            self,
            Odometry,
            f'/{self.OTHER_CAR_NAME}/odometry',
        )

        self.LIDAR_SUB_TWO = Subscriber(
            self,
            LaserScan,
            f'/{self.OTHER_CAR_NAME}/scan',
        )

        self.PROCESSED_PUBLISHER = self.create_publisher(
            LaserScan,
            f'/{self.NAME}/processed_scan',
            10
        )

        self.MESSAGE_FILTER = ApproximateTimeSynchronizer(
            [self.ODOM_SUB_ONE, self.LIDAR_SUB_ONE, self.ODOM_SUB_TWO, self.LIDAR_SUB_TWO],
            10,
            0.1,
        )

        self.MESSAGE_FILTER.registerCallback(self.message_filter_callback)

        self.OBSERVATION_FUTURE = Future()

        # Reset Client -----------------------------------------------
        self.RESET_CLIENT = self.create_client(
            CarBeatReset,
            'car_beat_reset'
        )

        while not self.RESET_CLIENT.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')

        # Stepping Client ---------------------------------------------
        self.STEPPING_CLIENT = self.create_client(
            SetBool,
            'stepping_service'
        )

        while not self.STEPPING_CLIENT.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('stepping service not available, waiting again...')

        self.TIMER = self.create_timer(step_length, self.TIMER_CB)
        self.TIMER_FUTURE = Future()

    def reset(self):
        self.STEP_COUNTER = 0
        self.NUM_SPAWNS = 0
        self.STEPS_SINCE_LAST_GOAL = 0
        self.GOALS_REACHED = 0
        self.FTG_OFFSET = np.random.randint(8, 12)
        self.FTG_GOALS_REACHED = 0

        self.set_velocity(0, 0)

        if 'multi_track' in self.track:
            self.CURRENT_TRACK = random.choice(list(self.ALL_CAR_GOALS.keys()))
            self.ALL_GOALS = self.ALL_CAR_GOALS[self.CURRENT_TRACK]
            self.CAR_WAYPOINTS = self.ALL_CAR_WAYPOINTS[self.CURRENT_TRACK]

        # New random starting point for the cars
        car_x, car_y, car_yaw, index = random.choice(self.CAR_WAYPOINTS)
        ftg_x, ftg_y, ftg_yaw, ftg_index = self.CAR_WAYPOINTS[(index + self.FTG_OFFSET) % len(self.CAR_WAYPOINTS)]
        
        self.START_GOAL_INDEX = index
        self.FTG_START_GOAL_INDEX = ftg_index

        self.GOAL_POSITION = self.ALL_GOALS[self.START_GOAL_INDEX]
        self.FTG_GOAL_POSITION = self.ALL_GOALS[self.FTG_START_GOAL_INDEX]

        self.sleep()

        goal_x, goal_y = self.GOAL_POSITION

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

        self.call_step(pause=False)
        state, _ = self.get_observation()
        self.call_step(pause=True)
        info = {}

        return state, info

    def step(self, action):
        self.STEP_COUNTER += 1

        self.call_step(pause=False)
        _, full_state = self.get_observation()

        lin_vel, ang_vel = action
        self.set_velocity(lin_vel, ang_vel)

        self.sleep()


        next_state, full_next_state  = self.get_observation()
        self.call_step(pause=True)
        
        reward = self.compute_reward(full_state, full_next_state)
        terminated = self.is_terminated(full_next_state)
        truncated = self.STEPS_SINCE_LAST_GOAL >= self.MAX_STEPS_PER_GOAL
        info = {}

        return next_state, reward, terminated, truncated, info

    def message_filter_callback(self, odom_one: Odometry, lidar_one: LaserScan, odom_two: Odometry, lidar_two: LaserScan):
        self.OBSERVATION_FUTURE.set_result({'odom_one': odom_one, 'lidar_one': lidar_one, 'odom_two': odom_two, 'lidar_two': lidar_two})

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.OBSERVATION_FUTURE)
        future = self.OBSERVATION_FUTURE
        self.OBSERVATION_FUTURE = Future()
        data = future.result()
        return data['odom_one'], data['lidar_one'], data['odom_two'], data['lidar_two'] 

    def is_terminated(self, state):
        
        return has_collided(state[8:19], self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6]) \
            or self.GOALS_REACHED >= self.MAX_GOALS
    

    def get_observation(self):

        # Get Position and Orientation of F1tenth
        odom_one, lidar_one, odom_two, lidar_two = self.get_data()
        
        num_points = self.LIDAR_NUM

        odom_one = process_odom(odom_one)
        odom_two = process_odom(odom_two)

        lidar_one_range = avg_lidar(lidar_one, self.LIDAR_NUM)
        lidar_two_range = avg_lidar(lidar_two, self.LIDAR_NUM)

        match self.OBSERVATION_MODE:
            case 'full':
                state = odom_one + lidar_one_range
            case 'no_position':
                state = odom_one[2:] + lidar_one_range
            case 'lidar_only':
                state = odom_one[-2:] + lidar_one_range
            case _:
                ValueError(f'Invalid observation mode: {self.OBSERVATION_MODE}')

        scan = create_lidar_msg(lidar_one, num_points, lidar_one_range)

        self.PROCESSED_PUBLISHER.publish(scan)

        full_state = odom_one + lidar_one_range + odom_two + lidar_two_range + self.GOAL_POSITION

        return state, full_state
 
    def compute_reward(self, state, next_state):

        reward = 0

        goal_position = self.GOAL_POSITION

        current_distance = math.dist(goal_position, next_state[:2])
        prev_distance = math.dist(goal_position, state[:2])

        reward += prev_distance - current_distance

        self.STEPS_SINCE_LAST_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            print(f'Goal #{self.GOALS_REACHED} Reached')
            reward += 2
            self.GOALS_REACHED += 1

            # Updating Goal Position
            new_x, new_y = self.ALL_GOALS[(self.START_GOAL_INDEX + self.GOALS_REACHED) % len(self.ALL_GOALS)]
            self.GOAL_POSITION = [new_x, new_y]

            self.update_goal_service(new_x, new_y)

            self.STEPS_SINCE_LAST_GOAL = 0

        ftg_current_distance = math.dist(self.ALL_GOALS[(self.FTG_START_GOAL_INDEX + self.FTG_GOALS_REACHED) % len(self.ALL_GOALS)], next_state[8 + self.LIDAR_NUM:8 + self.LIDAR_NUM + 2])

        # Keeping track of FTG car goal number
        if ftg_current_distance < self.REWARD_RANGE:
            self.FTG_GOALS_REACHED += 1
        
        # If RL car has overtaken FTG car
        if self.GOALS_REACHED >= (self.FTG_GOALS_REACHED + self.FTG_OFFSET + 3):
            print(f'RL Car has overtaken FTG Car')
            reward  += 200

            # Ensure overtaking won't happen again
            self.FTG_GOALS_REACHED += 500
            

        if has_collided(next_state[8:8 + self.LIDAR_NUM], self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 25  # TODO: find optimal value for this

        return reward
    
    def call_reset_service(self, car_x, car_y, car_Y, goal_x, goal_y, ftg_x, ftg_y, ftg_Y):
        """
        Reset the car and goal position
        """

        request = CarBeatReset.Request()
        
        request.gx = goal_x
        request.gy = goal_y

        request.car_one = self.NAME
        request.cx_one = car_x
        request.cy_one = car_y
        request.cyaw_one = car_Y

        request.car_two = self.OTHER_CAR_NAME
        request.cx_two = ftg_x
        request.cy_two = ftg_y
        request.cyaw_two = ftg_Y

        request.flag = "car_and_goal"

        # Publish to reset Topic to reset other nodes
        empty_msg = Empty()
        self.RESET_PUB.publish(empty_msg)

        future = self.RESET_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()
    

    
