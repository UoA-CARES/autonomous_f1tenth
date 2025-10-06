import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy import Future
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from environment_interfaces.srv import Reset
from .util import ackermann_to_twist
import yaml


class F1tenthEnvironment(Node):
    '''
    Repository Parent Environment:
        
        The responsibilities of this class is the following:
            - handle the topic subscriptions/publishers
            - fetching of car data (raw)
            - define the interface for environments to implement
    '''
    def __init__(self,
                 env_name,
                 car_name,
                 reward_range=0.5, 
                 max_steps=3000, 
                 collision_range=0.2,
                 step_length,
                 lidar_points = 10,
                 track='track_1',
                 observation_mode='lidar_only',
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__(env_name + '_environment')

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")

        #####################################################################################################################
        # Init params ----------------------------------------------
        self.NAME = car_name
        self.REWARD_RANGE = reward_range
        self.MAX_STEPS = max_steps
        self.COLLISION_RANGE = collision_range
        self.STEP_LENGTH = step_length
        self.LIDAR_POINTS = lidar_points
        self.TRACK = track
        self.ODOM_OBSERVATION_MODE = observation_mode

        # configure odom observation size:
        match observation_mode:
            case 'lidar_only':
                odom_observation_size = 2
            case 'no_position':
                odom_observation_size = 6
            case _:
                odom_observation_size = 10
        self.OBSERVATION_SIZE = odom_observation_size + self.LIDAR_POINTS

        #####################################################################################################################
        # Network params ---------------------------------------------
        self.ACTION_NUM = 2

        #####################################################################################################################
        # Environment params -----------------------------------------
        self.IS_MULTI_TRACK = 'multi_track' in self.TRACK or self.TRACK == 'staged_tracks'

        #####################################################################################################################
        # Vehicle params -------------------------------------------
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw"] = 'avg'
        # AE
        if self.LIDAR_PROCESSING == 'pretrained_ae':
            from .autoencoders.lidar_autoencoder import LidarConvAE
            self.AE_LIDAR = LidarConvAE()
            self.AE_LIDAR.load_state_dict(torch.load("/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt"))
            self.AE_LIDAR.eval()

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.MAX_ACTIONS = np.asarray([config['actions']['max_speed'], config['actions']['max_turn']])
        self.MIN_ACTIONS = np.asarray([config['actions']['min_speed'], config['actions']['min_turn']])
 
        #####################################################################################################################
        # Pub/Sub ----------------------------------------------------
        self.CMD_VEL_PUB = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            1
        )

        self.ODOM_SUB = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.LIDAR_SUB = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.PROCESSED_PUBLISHER = self.create_publisher(
            LaserScan,
            f'/{self.NAME}/processed_scan',
            1
        )

        #####################################################################################################################
        # Message filter ---------------------------------------------
        self.MESSAGE_FILTER = ApproximateTimeSynchronizer(
            [self.ODOM_SUB, self.LIDAR_SUB],
            1,
            0.1,
        )
        self.MESSAGE_FILTER.registerCallback(self.message_filter_callback)

        # Reset Client -----------------------------------------------
        self.RESET_CLIENT = self.create_client(
            Reset,
            env_name + '_reset'
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

        # Timer -------------------------------------------------------
        self.TIMER = self.create_timer(step_length, self.timer_cb)

        #####################################################################################################################
        # Initialise vars ---------------------------------------------
        
        # Track vars
        self.ALL_TRACK_MODELS = None
        self.CURR_TRACK_MODEL = None

        # Loop vars
        self.STEP_COUNTER = 0
        self.STEP_PROGRESS = 0
        self.GOALS_REACHED = 0
        self.PREV_CLOSEST_POINT = None
        self.IS_EVAL = False

        # Futures
        self.TIMER_FUTURE = Future()
        self.LAST_STATE = Future()
        self.ODOM_OBSERVATION_FUTURE = Future()


        #####################################################################################################################

    def reset(self):
        raise NotImplementedError('reset() not implemented')

    def step(self, action):
        self.STEP_COUNTER += 1
        self.call_step(pause=False)

        state = self.get_observation()
        
        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)

        while not self.TIMER_FUTURE.done():
            rclpy.spin_once(self)

        self.TIMER_FUTURE = Future()
        
        

        next_state = self.get_observation()
        self.call_step(pause=True)

        reward = self.compute_reward(state, next_state)
        terminated = self.is_terminated(next_state)
        truncated = self.STEP_COUNTER >= self.MAX_STEPS
        info = {}

        return next_state, reward, terminated, truncated, info

    def get_observation(self):
        raise NotImplementedError('get_observation() not implemented')

    def compute_reward(self, state, next_state):
        raise NotImplementedError('compute_reward() not implemented')

    def is_terminated(self, state):
        raise NotImplementedError('is_terminated() not implemented')

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.ODOM_OBSERVATION_FUTURE.set_result({'odom': odom, 'lidar': lidar})

    def get_data(self) -> tuple[Odometry,LaserScan]:
        rclpy.spin_until_future_complete(self, self.ODOM_OBSERVATION_FUTURE, timeout_sec=0.5)
        if (self.ODOM_OBSERVATION_FUTURE.result()) == None:
            future = self.LAST_STATE
            self.get_logger().info("Using previous observation")
        else:
            future = self.ODOM_OBSERVATION_FUTURE
            self.LAST_STATE = future
        self.ODOM_OBSERVATION_FUTURE = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, lin_vel, steering_angle, L=0.325):
        """
        Publish Twist Message. In place since simulator takes angular velocity commands but policies should produce ackermann steering angle.
        Takes linear velocity and steering ANGLE, NOT angular velocity.
        """
        angular = ackermann_to_twist(steering_angle, lin_vel, L)
        velocity_msg = Twist()
        velocity_msg.angular.z = float(angular)
        velocity_msg.linear.x = float(lin_vel)

        self.CMD_VEL_PUB.publish(velocity_msg)

    def sleep(self):
        while not self.TIMER_FUTURE.done():
            rclpy.spin_once(self)
        self.TIMER_FUTURE = Future()
    
    def call_step(self, pause):
        request = SetBool.Request()
        request.data = pause

        future = self.STEPPING_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def timer_cb(self):
        self.TIMER_FUTURE.set_result(True)

    def increment_stage(self):
        raise NotImplementedError('Staged training is not implemented')

    def call_reset_service(self, car_x, car_y, car_Y, goal_x, goal_y, car_name):
        """
        Reset the car and goal position
        """

        request = Reset.Request()
        request.car_name = car_name
        request.gx = float(goal_x)
        request.gy = float(goal_y)
        request.cx = float(car_x)
        request.cy = float(car_y)
        request.cyaw = float(car_Y)
        request.flag = "car_and_goal"

        future = self.RESET_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def update_goal_service(self, x, y):
        """
        Reset the goal position
        """

        request = Reset.Request()
        request.gx = x
        request.gy = y
        request.flag = "goal_only"

        future = self.RESET_CLIENT.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()