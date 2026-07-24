import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Int8


class RLDeadman(Node):
    """Forward RL commands only while a fresh enabled heartbeat is present."""

    def __init__(self):
        super().__init__("rl_deadman")

        self.declare_parameter("car_name", "f1tenth")
        self.declare_parameter("deadman_topic", "/rl_deadman")
        self.declare_parameter("deadman_timeout_sec", 0.1)
        self.declare_parameter("command_timeout_sec", 0.25)

        car_name = str(self.get_parameter("car_name").value)
        deadman_topic = str(self.get_parameter("deadman_topic").value)
        self.deadman_timeout_ns = int(
            float(self.get_parameter("deadman_timeout_sec").value) * 1e9
        )
        self.command_timeout_ns = int(
            float(self.get_parameter("command_timeout_sec").value) * 1e9
        )
        if self.deadman_timeout_ns <= 0 or self.command_timeout_ns <= 0:
            raise ValueError("Deadman timeouts must be greater than zero")

        self.deadman_enabled = False
        self.last_deadman_time_ns = None
        self.last_command_time_ns = None
        self.enabled_last_cycle = False

        reliable_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, f"/{car_name}/drive", 1
        )
        self.command_subscription = self.create_subscription(
            AckermannDriveStamped,
            f"/{car_name}/rl_drive",
            self._command_callback,
            1,
        )
        self.deadman_subscription = self.create_subscription(
            Int8, deadman_topic, self._deadman_callback, reliable_qos
        )
        self.watchdog_timer = self.create_timer(0.02, self._watchdog_callback)

        self.get_logger().info(
            f"RL deadman waiting for a fresh {deadman_topic}=1 heartbeat."
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _deadman_callback(self, message: Int8) -> None:
        was_enabled = self._is_enabled()
        self.last_deadman_time_ns = self._now_ns()
        self.deadman_enabled = message.data == 1
        enabled = self._is_enabled()

        if was_enabled and not enabled:
            self._publish_stop()
        if enabled != self.enabled_last_cycle:
            self.enabled_last_cycle = enabled
            if enabled:
                self.get_logger().info("RL deadman engaged.")
            else:
                self.get_logger().info("RL deadman released; commanding stop.")

    def _deadman_is_fresh(self, now_ns: int) -> bool:
        return (
            self.last_deadman_time_ns is not None
            and 0 <= now_ns - self.last_deadman_time_ns <= self.deadman_timeout_ns
        )

    def _command_is_fresh(self, now_ns: int) -> bool:
        return (
            self.last_command_time_ns is not None
            and 0 <= now_ns - self.last_command_time_ns <= self.command_timeout_ns
        )

    def _is_enabled(self, now_ns: int | None = None) -> bool:
        if now_ns is None:
            now_ns = self._now_ns()
        return self.deadman_enabled and self._deadman_is_fresh(now_ns)

    def _command_callback(self, message: AckermannDriveStamped) -> None:
        now_ns = self._now_ns()
        self.last_command_time_ns = now_ns
        if self._is_enabled(now_ns):
            message.header.stamp = self.get_clock().now().to_msg()
            self.drive_publisher.publish(message)
        else:
            self._publish_stop()

    def _watchdog_callback(self) -> None:
        now_ns = self._now_ns()
        enabled = self._is_enabled(now_ns)
        if self.enabled_last_cycle and not enabled:
            self.enabled_last_cycle = False
            self.get_logger().warning(
                "RL deadman heartbeat timed out; commanding stop."
            )

        if not enabled or not self._command_is_fresh(now_ns):
            self._publish_stop()

    def _publish_stop(self) -> None:
        stop = AckermannDriveStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        self.drive_publisher.publish(stop)


def main():
    rclpy.init()
    node = RLDeadman()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
