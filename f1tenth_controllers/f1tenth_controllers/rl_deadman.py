import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Header


class RLDeadmanGate(Node):
    def __init__(self):
        super().__init__("rl_deadman")

        self.declare_parameter("car_name", "f1tenth")
        self.declare_parameter("deadman_button", 5)
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joy_timeout_sec", 0.25)
        self.declare_parameter("command_timeout_sec", 0.25)

        car_name = self.get_parameter("car_name").value
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        joy_topic = self.get_parameter("joy_topic").value
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

        self.deadman_pressed = False
        self.last_joy_time_ns = None
        self.last_command_time_ns = None
        self.was_enabled = False

        self.output_publisher = self.create_publisher(
            AckermannDriveStamped, f"/{car_name}/drive", 1
        )
        self.command_subscription = self.create_subscription(
            AckermannDriveStamped,
            f"/{car_name}/rl_drive",
            self._command_callback,
            1,
        )
        self.joy_subscription = self.create_subscription(
            Joy, joy_topic, self._joy_callback, qos_profile_sensor_data
        )
        self.watchdog = self.create_timer(0.05, self._watchdog_callback)

        self.get_logger().info(
            f"RL deadman gate: hold button {self.deadman_button} to forward "
            f"/{car_name}/rl_drive to /{car_name}/drive."
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _joy_callback(self, message: Joy) -> None:
        self.last_joy_time_ns = self._now_ns()
        pressed = (
            self.deadman_button < len(message.buttons)
            and message.buttons[self.deadman_button] == 1
        )

        if self.deadman_pressed and not pressed:
            self.was_enabled = False
            self.get_logger().info("RL deadman released; stopping.")
            self._publish_stop()
        elif not self.deadman_pressed and pressed:
            self.get_logger().info("RL deadman engaged.")

        self.deadman_pressed = pressed

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

    def _enabled(self, now_ns: int) -> bool:
        return self.deadman_pressed and self._joy_is_fresh(now_ns)

    def _command_callback(self, message: AckermannDriveStamped) -> None:
        now_ns = self._now_ns()
        self.last_command_time_ns = now_ns
        if self._enabled(now_ns):
            message.header.stamp = self.get_clock().now().to_msg()
            self.output_publisher.publish(message)
        else:
            self._publish_stop()

    def _watchdog_callback(self) -> None:
        now_ns = self._now_ns()
        enabled = self._enabled(now_ns)

        if self.was_enabled and not enabled:
            self.get_logger().warning("RL deadman timed out; stopping.")

        self.was_enabled = enabled
        if not enabled or not self._command_is_fresh(now_ns):
            self._publish_stop()

    def _publish_stop(self) -> None:
        stop = AckermannDriveStamped()
        stop.header = Header()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        self.output_publisher.publish(stop)


def main():
    rclpy.init()
    node = RLDeadmanGate()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
