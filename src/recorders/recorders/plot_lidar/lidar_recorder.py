import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import time


class LidarPlotter(Node):
    def __init__(self):
        super().__init__('lidar_plotter')
        self.lidar_topic_name = '/f1tenth/processed_scan'
        self.odom_topic_name = '/f1tenth/odometry'
        
        # Subscriptions
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            self.lidar_topic_name,
            self.lidar_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic_name,
            self.odom_callback,
            10
        )
        
        script_dir = os.path.dirname(__file__)
        self.file_path = os.path.join(script_dir, f"record_lidar_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")
        
        self.get_logger().info(f"Subscribed to '{self.lidar_topic_name}' and '{self.odom_topic_name}' topics.")
        with open(self.file_path, 'w') as log_file:
            log_file.write("")
            
        
        # Initialize plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Track Walls - Top-Down View")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_aspect('equal')
        self.wall_points = []  # Store wall points for continuous plotting
        self.car_positions = []  # Store car positions for plotting

    def odom_callback(self, msg):
        # Extract car position and orientation from odometry
        self.car_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        self.car_orientation = self.quaternion_to_yaw(orientation_q)
        self.car_positions.append(self.car_position)  # Store car position for plotting

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Convert lidar readings to polar coordinates
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # Transform to global coordinates
        x_global, y_global = self.transform_to_global(x_local, y_local)

        # Store wall points
        self.wall_points.extend(zip(x_global, y_global))
        
        timestamp = time.time()
        formatted_time = f"{timestamp:.3f}"
        with open(self.file_path, 'a') as log_file:
            log_file.write(f"Time: {formatted_time},\n")
            log_file.write(f"Car Position: ({self.car_position[0]:.2f}, {self.car_position[1]:.2f})\n")
            log_file.write("Wall Points:\n")
            for x, y in zip(x_global, y_global):
                log_file.write(f"\t({x:.2f}, {y:.2f})\n")
            log_file.write("\n")

        # Update plot
        self.ax.clear()
        self.ax.plot(*zip(*self.wall_points), 'o', markersize=1, label="Walls")
        self.ax.plot(*zip(*self.car_positions), 'ro', markersize=2, label="Car Position")  # Plot car positions
        self.ax.set_title("Track Walls - Top-Down View")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_aspect('equal')
        self.ax.legend()
        plt.pause(0.01)

    def transform_to_global(self, x_local, y_local):
        # Rotate and translate lidar points to global coordinates
        rotation_matrix = np.array([
            [np.cos(self.car_orientation), -np.sin(self.car_orientation)],
            [np.sin(self.car_orientation), np.cos(self.car_orientation)]
        ])
        local_points = np.vstack((x_local, y_local))
        global_points = rotation_matrix @ local_points + self.car_position.reshape(2, 1)
        return global_points[0, :], global_points[1, :]

    def quaternion_to_yaw(self, q):
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

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