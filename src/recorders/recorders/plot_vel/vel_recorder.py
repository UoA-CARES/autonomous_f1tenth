import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import time
import os
from datetime import datetime
from pathlib import Path

class CmdVelRecorder(Node):
    def __init__(self):
        super().__init__('recorder')
        
        self.declare_parameter('onSim')
        self.onSim = self.get_parameter('onSim').value
        
        self.filename = f"record_{'sim' if self.onSim else 'drive'}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
        
        path = os.path.join(Path(__file__).parent.parent.parent.parent.parent,"recordings", "vel_records")
        if not os.path.exists(path):
            os.mkdir(path)
        self.file_path = os.path.join(path, self.filename)
        
        if self.onSim:
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
                self.recorder_callback,
                10
            )
        self.get_logger().info(f"Subscribed to '{self.topic_name}' topic.")

    def recorder_callback(self, msg):
        timestamp = time.time()
        formatted_time = f"{timestamp:.3f}"

        with open(self.filename, 'a') as log_file:
            if self.onSim:
                log_file.write(f"time={formatted_time},\tlinear={msg.linear.x},\tangular={msg.angular.z}\n")
            else:
                log_file.write(f"time={formatted_time},\tlinear={msg.drive.speed},\tangular={msg.drive.steering_angle}\n")
        
def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down vel recorder.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
