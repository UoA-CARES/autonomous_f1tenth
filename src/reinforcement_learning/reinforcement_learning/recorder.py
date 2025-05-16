import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import time

class CmdVelRecorder(Node):
    def __init__(self, onSim):
        self.onSim = onSim
        with open(f"record_{'sim' if self.onSim else 'drive'}.txt", 'w') as log_file:
            log_file.write("")
        super().__init__('recorder')
        if onSim:
            self.topic_name = '/f1tenth/cmd_vel'
            self.subscription = self.create_subscription(
                Twist,
                self.topic_name,
                self.recorder_callback,
                10
            )
        else:
            self.topic_name = '/f1tenth/drive'
            self.subscription = self.create_subscription(
                AckermannDriveStamped,
                self.topic_name,
                self.recorder_callback_callback,
                10
            )
        self.get_logger().info(f"Subscribed to '{self.topic_name}' topic.")

    def recorder_callback(self, msg):
        timestamp = time.time()
        formatted_time = f"{timestamp:.3f}"
        # Log the received message
        self.get_logger().info(f"Received value: linear={msg.linear.x}, angular={msg.angular.z}")

        # Optionally, write to a file
        with open(f"record_{'sim' if self.onSim else 'drive'}.txt", 'a') as log_file:
            log_file.write(f"time={formatted_time},\tlinear={msg.linear.x},\tangular={msg.angular.z}\n")
        
def main(args=None):
    rclpy.init(args=args)
    recorder = CmdVelRecorder(True) # Change to False for real car
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.get_logger().info("Shutting down recorder.")
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()