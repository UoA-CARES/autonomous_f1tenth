from rclpy.node import Node

class F1TenthReset(Node):

    def __init__(self, env_name):
        super().__init__(env_name + '_reset')
