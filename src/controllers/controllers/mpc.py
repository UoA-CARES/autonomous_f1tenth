import rclpy
from rclpy import Future
from rclpy.node import Node


class MPC():
    def __init__(self, alg): 
        self.Alg = alg