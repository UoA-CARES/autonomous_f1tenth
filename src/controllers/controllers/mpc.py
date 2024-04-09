import rclpy
from rclpy import Future
from rclpy.node import Node


class MPC(Node):
    def __init__(self, node_name): 
        super().__init__(node_name)