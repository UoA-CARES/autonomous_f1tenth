import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import time
import os

class CmdVelRecorder(Node):
    def __init__(self, onSim):
        super().__init__('recorder')
        
        self.declare_parameter('onSim')
        self.onSim = self.get_parameter('onSim').value
        script_dir = os.path.dirname(__file__)
        self.file_path = os.path.join(script_dir, f"record_{'sim' if self.onSim else 'drive'}.txt")
        with open(self.file_path, 'w') as log_file:
            log_file.write("")
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
        
        print(f"Recording: time={formatted_time}, linear={msg.linear.x}, angular={msg.angular.z}")

        with open(self.file_path, 'a') as log_file:
            log_file.write(f"time={formatted_time},\tlinear={msg.linear.x},\tangular={msg.angular.z}\n")
        
def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRecorder(True) # Change to False for real car
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down vel recorder.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
