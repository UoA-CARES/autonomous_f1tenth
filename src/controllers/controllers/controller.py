import rclpy
from rclpy import Future
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
from typing import Literal

from environments.util import process_odom, avg_lidar, process_lidar_med_filt, forward_reduce_lidar, ackermann_to_twist, create_lidar_msg, avg_lidar_w_consensus, uneven_median_lidar


class Controller(Node):
    def __init__(self, node_name, car_name, step_length, isCar=False, lidar_points = 10):
        super().__init__(node_name + 'controller')

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")   

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.STEP_LENGTH = step_length
        self.LIDAR_POINTS = lidar_points
        self.LIDAR_PROCESSING:Literal["avg","median", "avg_w_consensus","pretrained_ae", "raw"] = 'median'
        
        # Pub/Sub ----------------------------------------------------
        # Ackermann pub only works for physical version
        self.ackerman_pub = self.create_publisher(
            AckermannDriveStamped,
            f'/{self.NAME}/drive',
            1
        )

        # Twist for sim
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.NAME}/cmd_vel',
            1
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
            1
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
            1,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()
        
        self.firstOdom = isCar
        self.offset = [0, 0, 0, 0, 0, 0]


    def step(self, action, policy):
        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)
        self.sleep()
        self.timer_future = Future()
        state = self.get_observation(policy)
        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_observation(self, policy):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        if self.firstOdom:
            self.offset = odom[0:6]
            self.firstOdom = False
        odom[0] = odom[0] - self.offset[0]
        odom[1] = odom[1] - self.offset[1]
        odom[2] = odom[2] - self.offset[2]
        odom[3] = odom[3] - self.offset[3]
        odom[4] = odom[4] - self.offset[4]
        odom[5] = odom[5] - self.offset[5]
        num_points = self.LIDAR_POINTS

        match self.LIDAR_PROCESSING:
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'median':
                processed_lidar_range = uneven_median_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = process_lidar_med_filt(lidar, 15)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'avg_w_consensus':
                processed_lidar_range = avg_lidar_w_consensus(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'forward_reduce':
                processed_lidar_range = forward_reduce_lidar(lidar)
        self.processed_publisher.publish(scan)
        state = odom+processed_lidar_range
        return state
        
    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, lin_vel, steering_angle, L=0.325):
        ang_vel = ackermann_to_twist(steering_angle, lin_vel, L)
        car_velocity_msg = AckermannDriveStamped()
        sim_velocity_msg = Twist()
        sim_velocity_msg.angular.z = float(ang_vel)
        sim_velocity_msg.linear.x = float(lin_vel)

        car_velocity_msg.drive.steering_angle = float(steering_angle)
        car_velocity_msg.drive.speed = float(lin_vel)
        
        # Add a ROS header with a timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg() 
        car_velocity_msg.header = header

        self.ackerman_pub.publish(car_velocity_msg)
        self.cmd_vel_pub.publish(sim_velocity_msg)

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)
