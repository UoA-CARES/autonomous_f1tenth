from abc import abstractmethod
from itertools import chain
import rclpy
from rclpy import Future
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
import rclpy.time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer, Time
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
import torch
from torch import nn
from typing import List
import scipy
import numpy as np
from typing import Literal
from collections import deque

from environments.autoencoders.lidar_beta_vae import BetaVAE1D
from environments.autoencoders.lidar_autoencoder import LidarConvAE

from environments.util import process_odom, avg_lidar, forward_reduce_lidar, ackermann_to_twist, create_lidar_msg, process_ae_lidar, process_ae_lidar_beta_vae


class Controller(Node):
    def __init__(self, node_name, car_name, step_length, isCar=False, lidar_points = 10):
        super().__init__(node_name + 'controller')

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")
          

        # Environment Details ----------------------------------------
        
        self.NAME = car_name
        self.STEP_LENGTH = step_length
        
        ######################################################
        ##### Observation configuration ######################

        self.IS_AUTOENCODER_ALG = True
        self.LIDAR_PROCESSING:Literal["avg","pretrained_ae", "raw", 'forward_reduce'] = 'raw'
        self.LIDAR_POINTS = 683 #10, 683
        self.LIDAR_OBS_STACK_SIZE = 1

        #---------------------------------------------
        if self.IS_AUTOENCODER_ALG:
            self.OBSERVATION_SIZE = (self.LIDAR_OBS_STACK_SIZE, self.LIDAR_POINTS)
        else:
            self.OBSERVATION_SIZE = 2 + (self.LIDAR_POINTS * self.LIDAR_OBS_STACK_SIZE)

         # using multiple observations
        if self.LIDAR_OBS_STACK_SIZE > 1:
            self.lidar_obs_stack = deque([], maxlen=self.LIDAR_OBS_STACK_SIZE)
        
        ######################################################
        
        
        # Pub/Sub ----------------------------------------------------
        # Ackermann pub only works for physical version
        self.ackerman_pub = self.create_publisher(
            AckermannDriveStamped,
            f'/{self.NAME}/drive',
            10
        )

        # Twist for sim
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            10
        )

        self.odom_sub = Subscriber(
            self,
            Odometry,
            f'/{self.NAME}/odometry',
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f'/{self.NAME}/scan',
        )

        self.processed_publisher = self.create_publisher(
            LaserScan,
            f'/{self.NAME}/processed_scan',
            10
        )

        ##### FOR LOCALIZED METHODS ONLY############################
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20
        )
        self.tf_sub = Subscriber(
            self,
            TFMessage,
            f'/tf'
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # -----------------------------------------------------------

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()
        
        self.firstOdom = isCar
        self.offset = [0, 0, 0, 0, 0, 0]
        
        
        #TODO:figure out what to do with this
        # # Lidar processing 
        # self.ae_lidar_model = LidarConvAE()
        # # self.ae_lidar_model = BetaVAE1D(1,10,beta=4)
        # self.ae_lidar_model.load_state_dict(torch.load("/home/anyone/autonomous_f1tenth/src/environments/environments/autoencoders/trained_models/lidar_ae_ftg_rand.pt"))
        # self.ae_lidar_model.eval()

    def step(self, action, policy):
        lin_vel, steering_angle = action
        lin_vel = self.vel_mod(lin_vel)
        self.set_velocity(lin_vel, steering_angle)

        self.sleep()

        self.timer_future = Future()

        state, full_odom = self.get_observation(policy)

        return state, full_odom # full odom contain LOCATION, especially in sim. state *should* be able to be fed to networks without processing

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    # TODO: it seems useful that get_observation *should* return tuple: 1.clean state that can be fed to other alg. 2.ground truth data (loc,speed etc.) if it can be obtained either irl or in sim
    def get_observation(self, policy):
        #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z]

        # TODO: useful later. Working with ACML node, this get the X,Y coordinate of the car with respect to map origin: can be found in the map yaml file generated by SLAM
        #                      |
        #                      V
        # if policy == 'a_star' or policy == 'd_star':
        #     now = rclpy.time.Time()
        #     transformation = self.tf_buffer.lookup_transform('map',f'{self.NAME}base_link', now)
        #     x = transformation.transform.translation.x
        #     y = transformation.transform.translation.y
        #     self.get_logger().info(f"Coord: ({x}, {y})")
        #     print(x,y)

        # TODO: delete later
        # latest = self.get_clock().now()
        # self.get_logger().info(f"getting coord")
        # try: 
        #     transformation = self.tf_buffer.lookup_transform('map', f'{self.NAME}base_link', Time())
        #     x = transformation.transform.translation.x
        #     y = transformation.transform.translation.y
        #     self.get_logger().info(f"Coord: ({x}, {y})")
        # except Exception as e:
        #     self.get_logger().info(str(e))


        odom, lidar = self.get_data()
        odom = process_odom(odom)

        limited_odom = odom[-2:]
        # if self.firstOdom:
        #     self.offset = odom[0:6]
        #     self.firstOdom = False
        # odom[0] = odom[0] - self.offset[0]
        # odom[1] = odom[1] - self.offset[1]
        # odom[2] = odom[2] - self.offset[2]
        # odom[3] = odom[3] - self.offset[3]
        # odom[4] = odom[4] - self.offset[4]
        # odom[5] = odom[5] - self.offset[5]
        
        num_points = self.LIDAR_POINTS


        match self.LIDAR_PROCESSING:
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(processed_lidar_range, posinf=-5, nan=-5, neginf=-5).tolist()  
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'forward_reduce':
                processed_lidar_range = forward_reduce_lidar(lidar)


        self.processed_publisher.publish(scan)

        # is using lidar scan stack for temporal info
        if self.LIDAR_OBS_STACK_SIZE > 1:
            # if is first observation, fill stack with current observation
            if len(self.lidar_obs_stack) <= 1:
                for _ in range(0,self.LIDAR_OBS_STACK_SIZE):
                    self.lidar_obs_stack.append(processed_lidar_range)
            # add current observation to stack.
            else:
                self.lidar_obs_stack.append(processed_lidar_range)

        #######################################################
        ####### FORMING ACTUAL STATE TO BE PASSED ON ##########

        #### Check if should pass a dict state
        if self.IS_AUTOENCODER_ALG:
            # is using lidar scan stack for temporal info
            if self.LIDAR_OBS_STACK_SIZE > 1:
                state = {
                    'image': np.array(self.lidar_obs_stack).reshape((self.LIDAR_OBS_STACK_SIZE,-1)), # e.g. shape (683,) -> (1,683)
                    'vector': np.array(limited_odom), # e.g. shape (2,)
                }
            
            # not using scan stack
            else:
                state = {
                    'image': np.array([processed_lidar_range]).reshape((self.LIDAR_OBS_STACK_SIZE,-1)), # e.g. shape (683,) -> (1,683)
                    'vector': np.array(limited_odom), # e.g. shape (2,)
                }

        #### normal algorithm: flat state
        else:
            # is using lidar scan stack for temporal info
            if self.LIDAR_OBS_STACK_SIZE > 1:
                flattened_lidar_stack = list(chain(*self.lidar_obs_stack))
                state = limited_odom + flattened_lidar_stack 
            # not using scan stack
            else:
                state = state = limited_odom + processed_lidar_range 
        
        return state, odom
        

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, lin_vel, steering_angle, L=0.325):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """

        ang_vel = ackermann_to_twist(steering_angle, lin_vel, L)

        car_velocity_msg = AckermannDriveStamped()
        sim_velocity_msg = Twist()
        sim_velocity_msg.angular.z = float(ang_vel)
        sim_velocity_msg.linear.x = float(lin_vel)

        car_velocity_msg.drive.steering_angle = float(steering_angle) #-float(angle*0.5)
        car_velocity_msg.drive.speed = float(lin_vel)

        self.ackerman_pub.publish(car_velocity_msg)
        self.cmd_vel_pub.publish(sim_velocity_msg)


    def omega_to_ackerman(self, omega, linear_v, L):
        '''
        Convert CG angular velocity to Ackerman steering angle.

        Parameters:
        - omega: CG angular velocity in rad/s
        - v: Vehicle speed in m/s
        - L: Wheelbase of the vehicle in m

        Returns:
        - delta: Ackerman steering angle in radians

        Derivation:
        R = v / omega 
        R = L / tan(delta)  equation 10 from https://www.researchgate.net/publication/228464812_Electric_Vehicle_Stability_with_Rear_Electronic_Differential_Traction#pf3
        tan(delta) = L * omega / v
        delta = arctan(L * omega/ v)
        '''
        if linear_v == 0:
            return 0

        delta = math.atan((L * omega) / linear_v)

        return delta
    
    ###############################################
    #### Action modifier ##########################

    def vel_mod(self, linear_v):
        max_vel = 3
        linear_v = min(max_vel, linear_v)
        return linear_v

    def angle_mod(self, angle):
        max_angle = 0.85
        angle = min(max_angle, angle)
        if (abs(angle)<0.2):
            angle = 0
        return angle

    #####################################################
    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)
