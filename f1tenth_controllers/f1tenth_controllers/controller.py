from typing import Literal
from threading import Thread

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy import Future
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from .runtime_utils import (
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
        state_builder=None,
        deadman_button: int | None = None,
        deadman_timeout_sec: float = 0.25,
        joy_topic: str = "/joy",
    ):
        super().__init__(node_name + "controller")

        if lidar_points < 1:
            raise Exception("Make sure number of lidar points is more than 0")

        # Environment Details ----------------------------------------
        self.NAME = car_name
        self.step_sleep_time_ms = step_sleep_time_ms
        self.LIDAR_POINTS = lidar_points
        self.state_builder = state_builder
        self.deadman_button = deadman_button
        self.deadman_timeout_ns = int(float(deadman_timeout_sec) * 1e9)
        self.deadman_pressed = False
        self.last_joy_time_ns = None
        self.deadman_node = None
        self.deadman_executor = None
        self.joy_sub = None
        self.deadman_timer = None
        self.deadman_thread = None
        if self.deadman_button is not None:
            if self.deadman_button < 0:
                raise ValueError("deadman_button must be non-negative")
            if self.deadman_timeout_ns <= 0:
                raise ValueError("deadman_timeout_sec must be greater than zero")
        self.LIDAR_PROCESSING: Literal[
            "avg", "median", "avg_w_consensus", "pretrained_ae", "raw"
        ] = "median"

        # Pub/Sub ----------------------------------------------------
        self.ackerman_pub = self.create_publisher(
            AckermannDriveStamped, f"/{self.NAME}/drive", 1
        )

        self.cmd_vel_pub = self.create_publisher(Twist, f"/{self.NAME}/cmd_vel", 1)
        if self.deadman_button is not None:
            self.deadman_node = Node(self.get_name() + "_deadman")
            self.joy_sub = self.deadman_node.create_subscription(
                Joy, joy_topic, self._joy_callback, qos_profile_sensor_data
            )
            self.deadman_timer = self.deadman_node.create_timer(
                min(float(deadman_timeout_sec) / 2.0, 0.05),
                self._deadman_watchdog,
            )
            self.deadman_executor = SingleThreadedExecutor()
            self.deadman_executor.add_node(self.deadman_node)
            self.deadman_thread = Thread(
                target=self.deadman_executor.spin, daemon=True
            )
            self.deadman_thread.start()
            self.get_logger().info(
                f"RL deadman listening on {joy_topic}, button {self.deadman_button}."
            )

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

    def destroy_node(self):
        if self.deadman_executor is not None:
            self.deadman_executor.shutdown(timeout_sec=1.0)
        if self.deadman_thread is not None:
            self.deadman_thread.join(timeout=1.0)
        if self.deadman_node is not None:
            self.deadman_node.destroy_node()
        return super().destroy_node()

    def step(self, action, policy):
        lin_vel, steering_angle = action

        self.set_velocity(lin_vel, steering_angle)
        self._sleep(self.step_sleep_time_ms)
        state = self.get_observation(policy)
        return state

    def message_filter_callback(self, odom: Odometry, lidar: LaserScan):
        self.observation_future.set_result({"odom": odom, "lidar": lidar})

    def _joy_callback(self, msg: Joy) -> None:
        was_pressed = self.deadman_pressed
        self.last_joy_time_ns = self.get_clock().now().nanoseconds
        self.deadman_pressed = (
            self.deadman_button is not None
            and self.deadman_button < len(msg.buttons)
            and msg.buttons[self.deadman_button] == 1
        )
        if not was_pressed and self.deadman_pressed:
            self.get_logger().info(
                f"RL deadman engaged on {self.deadman_button}."
            )
        if was_pressed and not self.deadman_pressed:
            self.get_logger().info("RL deadman released; commanding stop.")
            self.set_velocity(0.0, 0.0)

    def _deadman_watchdog(self) -> None:
        if self.deadman_pressed and not self._deadman_is_active():
            self.get_logger().warning("RL deadman input timed out; commanding stop.")
            self.deadman_pressed = False
            self.set_velocity(0.0, 0.0)

    def _deadman_is_active(self) -> bool:
        if self.deadman_button is None:
            return True
        if not self.deadman_pressed or self.last_joy_time_ns is None:
            return False
        age_ns = self.get_clock().now().nanoseconds - self.last_joy_time_ns
        return 0 <= age_ns <= self.deadman_timeout_ns

    def get_observation(self, policy):
        odom, lidar = self.get_data()
        if self.state_builder is not None:
            state_data = self.state_builder.build_state(odom, lidar)
            self.processed_publisher.publish(state_data.lidar_state_scan)
            return state_data.state

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
        if not self._deadman_is_active():
            lin_vel = 0.0
            steering_angle = 0.0
        angular = ackermann_to_twist(steering_angle, lin_vel, L)
        car_velocity_msg = AckermannDriveStamped()
        sim_velocity_msg = Twist()
        sim_velocity_msg.angular.z = float(angular)
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
