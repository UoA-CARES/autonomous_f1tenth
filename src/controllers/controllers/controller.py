
import rclpy
from rclpy import Future
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

from environments.util import process_odom, avg_lidar, forward_reduce_lidar, ackermann_to_twist, create_lidar_msg


class Controller(Node):
    def __init__(self, node_name, car_name, step_length, lidar_points = 10):
        #TODO: make node name dynamic
        super().__init__(node_name + 'controller')

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")
          

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.STEP_LENGTH = step_length
        self.LIDAR_POINTS = lidar_points
        
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

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            10,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)

        self.observation_future = Future()

        self.timer = self.create_timer(step_length, self.timer_cb)
        self.timer_future = Future()

    def step(self, action, policy):

        lin_vel, steering_angle = action
        lin_vel = self.vel_mod(lin_vel)
        self.set_velocity(lin_vel, steering_angle)

        self.sleep()

        self.timer_future = Future()

        state = self.get_observation(policy)

        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({'odom': odom, 'lidar': lidar})

    def get_observation(self, policy):
        #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z]
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        
        num_points = self.LIDAR_POINTS

        if policy == 'ftg':
            lidar_range = forward_reduce_lidar(lidar)
        else:
            lidar_range = avg_lidar(lidar, num_points)
        
        scan = create_lidar_msg(lidar, num_points, lidar_range)

        self.processed_publisher.publish(scan)

        state = odom+lidar_range
        return state
        

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data['odom'], data['lidar']

    def set_velocity(self, lin_vel, steering_angle):
        """
        Publish Twist messages to f1tenth cmd_vel topic
        """

        ang_vel = ackermann_to_twist(steering_angle, lin_vel, 0.25)

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

    def vel_mod(self, linear):
        max_vel = 0.5
        linear = min(max_vel, linear)
        return linear
    
    def angle_mod(self, angle):
        max_angle = 0.85
        angle = min(max_angle, angle)
        if (abs(angle)<0.2):
            angle = 0
        return angle

    def sleep(self):
        while not self.timer_future.done():
            rclpy.spin_once(self)
    
    def timer_cb(self):
        self.timer_future.set_result(True)
