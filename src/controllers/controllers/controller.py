from typing import Literal

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy import Future
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from environments.util import (
    ackermann_to_twist,
    avg_lidar,
    avg_lidar_w_consensus,
    create_lidar_msg,
    forward_reduce_lidar,
    process_lidar_med_filt,
    process_odom,
    uneven_median_lidar,
)


class Controller(Node):
    def __init__(
        self,
        node_name: str,
        car_name: str,
        step_sleep_time_ms: float,
        isCar: bool = False,
        lidar_points: int = 10,
    ):
        super().__init__(node_name + "controller")

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.step_sleep_time_ms = step_sleep_time_ms
        self.LIDAR_POINTS = lidar_points
        self.LIDAR_PROCESSING: Literal[
            "avg", "median", "avg_w_consensus", "pretrained_ae", "raw"
        ] = "median"

        # Pub/Sub ----------------------------------------------------
        self.ackerman_pub = self.create_publisher(
            AckermannDriveStamped, f"/{self.NAME}/drive", 1
        )

        self.cmd_vel_pub = self.create_publisher(Twist, f"/{self.NAME}/cmd_vel", 1)

        self.odom_sub = Subscriber(
            self,
            Odometry,
            f"/{self.NAME}/odometry",
        )

        self.lidar_sub = Subscriber(
            self,
            LaserScan,
            f"/{self.NAME}/scan",
        )

        self.processed_publisher = self.create_publisher(
            LaserScan, f"/{self.NAME}/processed_scan", 1
        )

        self.message_filter = ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub],
            1,
            0.1,
        )

        self.message_filter.registerCallback(self.message_filter_callback)
        self.observation_future = Future()
        self.firstOdom = isCar
        self.offset = [0, 0, 0, 0, 0, 0]

        ##### FOR LOCALIZED METHODS ONLY############################
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.tf_sub = Subscriber(self, TFMessage, f"/tf")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        #####################################################################################################################

    def step(self, action, policy):
        lin_vel, steering_angle = action

        self.set_velocity(lin_vel, steering_angle)
        self._sleep(self.step_sleep_time_ms)
        state = self.get_observation(policy)
        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({"odom": odom, "lidar": lidar})

    def get_observation(self, policy):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        if self.firstOdom:
            self.offset = odom[0:6]
            self.firstOdom = False
        for i in range(0, 6):
            odom[i] = odom[i] - self.offset[i]
        num_points = self.LIDAR_POINTS

        match self.LIDAR_PROCESSING:
            case "avg":
                processed_lidar_range = avg_lidar(lidar, num_points)
            case "median":
                processed_lidar_range = uneven_median_lidar(lidar, num_points)
            case "raw":
                processed_lidar_range = process_lidar_med_filt(lidar, 15)
            case "avg_w_consensus":
                processed_lidar_range = avg_lidar_w_consensus(lidar, num_points)
            case "forward_reduce":
                processed_lidar_range = forward_reduce_lidar(lidar)
        visualized_range = processed_lidar_range
        scan = create_lidar_msg(lidar, num_points, visualized_range)
        self.processed_publisher.publish(scan)
        state = odom + processed_lidar_range
        return state

    def get_data(self):
        rclpy.spin_until_future_complete(self, self.observation_future)
        future = self.observation_future
        self.observation_future = Future()
        data = future.result()
        return data["odom"], data["lidar"]

    def set_velocity(self, lin_vel, steering_angle, L=0.325):
        ang_vel = ackermann_to_twist(steering_angle, lin_vel, L)
        car_velocity_msg = AckermannDriveStamped()
        sim_velocity_msg = Twist()
        sim_velocity_msg.angular.z = float(ang_vel)
        sim_velocity_msg.linear.x = float(lin_vel)
        car_velocity_msg.drive.steering_angle = float(steering_angle)
        car_velocity_msg.drive.speed = float(lin_vel)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        car_velocity_msg.header = header

        self.ackerman_pub.publish(car_velocity_msg)
        self.cmd_vel_pub.publish(sim_velocity_msg)

    def _sleep(self, duration_ms: float) -> None:
        """
        Sleep while still processing incoming messages, to allow for callbacks to run.

        Critical that this uses self.get_clock() for timekeeping, to ensure it works properly with simulated time.
        """
        end_time = self.get_clock().now().nanoseconds + int(duration_ms * 1e6)

        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
