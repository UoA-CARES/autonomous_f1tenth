import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import pandas as pd
import csv
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class LidarLogger(Node):
    
    def __init__(self, node_name='lidar_logger'):
        super().__init__(node_name)

        # MIGHT LOSE SCAN. ONLY USE FOR TRAINING AUTOENCODER
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.declare_parameter('topic_name', '/f1tenth/scan')
        self.declare_parameter('save_file_path', 'lidar_record.csv')

        self.lidar_topic = self.get_parameter('topic_name').get_parameter_value().string_value
        self.save_file_path= self.get_parameter('save_file_path').get_parameter_value().string_value

        # self.lidar_topic = "/f1tenth/scan"
        # self.save_file_path= "test.csv"


        self.subscription = self.create_subscription(
            LaserScan,  
            self.lidar_topic,  
            self.listener_callback,
            qos_profile=qos_profile
        )
        
        self.data = []
        file = open(self.save_file_path, 'a+')
        self.csv_writer = csv.writer(file)

    def listener_callback(self, msg:LaserScan):
        self.csv_writer.writerow(msg.ranges)
        self.get_logger().info("Logged Scan: " + str(msg.header.stamp))


def main(args=None):
    rclpy.init(args=args)
    node = LidarLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()