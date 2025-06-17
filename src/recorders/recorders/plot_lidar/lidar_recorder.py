import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

class LidarPlotter(Node):
    def __init__(self):
        super().__init__('lidar_plotter')
        self.topic_name = '/f1tenth/processed_scan'
        self.subscription = self.create_subscription(
            LaserScan,
            self.topic_name,
            self.listener_callback,
            10
        )
        self.get_logger().info(f"Subscribed to '{self.topic_name}' topic.")
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Lidar Readings - Top-Down View")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_aspect('equal')

    def listener_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Convert polar coordinates to Cartesian coordinates
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Clear the plot and redraw
        self.ax.clear()
        self.ax.plot(x, y, 'o', markersize=1)
        self.ax.set_title("Lidar Readings - Top-Down View")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_aspect('equal')
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = LidarPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down lidar recorder.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
