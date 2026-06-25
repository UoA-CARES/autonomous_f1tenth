import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Joy
from std_msgs.msg import Int8


class RLDeadman(Node):
    """Fail-closed gate between RL commands and the hardware drive topic."""

    def __init__(self):
        super().__init__("rl_deadman")

        self.declare_parameter("car_name", "f1tenth")
        self.declare_parameter("deadman_button", 5)
        self.declare_parameter("deadman_topic", "/rl_deadman")
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joy_timeout_sec", 0.25)
        self.declare_parameter("command_timeout_sec", 0.25)

        car_name = str(self.get_parameter("car_name").value)
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        deadman_topic = str(self.get_parameter("deadman_topic").value)
        joy_topic = str(self.get_parameter("joy_topic").value)
        self.joy_timeout_ns = int(
            float(self.get_parameter("joy_timeout_sec").value) * 1e9
        )
        self.command_timeout_ns = int(
            float(self.get_parameter("command_timeout_sec").value) * 1e9
        )

        if self.deadman_button < 0:
            raise ValueError("deadman_button must be non-negative")
        if self.joy_timeout_ns <= 0 or self.command_timeout_ns <= 0:
            raise ValueError("Deadman timeouts must be greater than zero")

        self.deadman_requested = False
        self.joy_button_pressed = False
        self.last_joy_time_ns = None
        self.last_command_time_ns = None
        self.enabled_last_cycle = False
        self.last_status_log_ns = 0
        self.last_input_speed = 0.0

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
        self.joy_subscription = self.create_subscription(
            Joy, joy_topic, self._joy_callback, qos_profile_sensor_data
        )
        self.watchdog_timer = self.create_timer(0.05, self._watchdog_callback)

        self.get_logger().info(
            f"RL deadman waiting for {deadman_topic}=1 and fresh button "
            f"{self.deadman_button} on {joy_topic}."
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _deadman_callback(self, message: Int8) -> None:
        requested = message.data == 1
        if self.deadman_requested and not requested:
            self._publish_stop()
        self.deadman_requested = requested
        self._log_enabled_transition()

    def _joy_callback(self, message: Joy) -> None:
        self.last_joy_time_ns = self._now_ns()
        pressed = (
            self.deadman_button < len(message.buttons)
            and message.buttons[self.deadman_button] == 1
        )
        if self.joy_button_pressed and not pressed:
            self._publish_stop()
        self.joy_button_pressed = pressed
        self._log_enabled_transition()

    def _joy_is_fresh(self, now_ns: int) -> bool:
        return (
            self.last_joy_time_ns is not None
            and 0 <= now_ns - self.last_joy_time_ns <= self.joy_timeout_ns
        )

    def _command_is_fresh(self, now_ns: int) -> bool:
        return (
            self.last_command_time_ns is not None
            and 0 <= now_ns - self.last_command_time_ns <= self.command_timeout_ns
        )

    def _is_enabled(self, now_ns: int | None = None) -> bool:
        if now_ns is None:
            now_ns = self._now_ns()
        return (
            self.deadman_requested
            and self.joy_button_pressed
            and self._joy_is_fresh(now_ns)
        )

    def _log_enabled_transition(self) -> None:
        enabled = self._is_enabled()
        if enabled == self.enabled_last_cycle:
            return
        self.enabled_last_cycle = enabled
        if enabled:
            self.get_logger().info("RL deadman engaged.")
        else:
            self.get_logger().info("RL deadman released; commanding stop.")

    def _command_callback(self, message: AckermannDriveStamped) -> None:
        now_ns = self._now_ns()
        self.last_command_time_ns = now_ns
        self.last_input_speed = float(message.drive.speed)
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
            self.get_logger().warning("RL deadman input timed out; commanding stop.")

        if not enabled:
            self._log_waiting_reason(now_ns)
        if not enabled or not self._command_is_fresh(now_ns):
            self._publish_stop()

    def _status_log_due(self, now_ns: int) -> bool:
        if now_ns - self.last_status_log_ns < int(1e9):
            return False
        self.last_status_log_ns = now_ns
        return True

    def _log_waiting_reason(self, now_ns: int) -> None:
        if not self.deadman_requested or not self._status_log_due(now_ns):
            return
        joy_age_ms = (
            None
            if self.last_joy_time_ns is None
            else (now_ns - self.last_joy_time_ns) / 1e6
        )
        command_age_ms = (
            None
            if self.last_command_time_ns is None
            else (now_ns - self.last_command_time_ns) / 1e6
        )
        self.get_logger().warning(
            "RL requested but gate closed: "
            f"button_pressed={self.joy_button_pressed}, "
            f"joy_age_ms={joy_age_ms}, command_age_ms={command_age_ms}, "
            f"last_input_speed={self.last_input_speed:.3f}"
        )

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
