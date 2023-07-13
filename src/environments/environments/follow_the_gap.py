import rclpy
from rclpy.node import Node
import numpy as np

class FollowTheGapNode(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

    def select_action(state):
        lin = 5
        ang = 0
        action = np.asarray([lin, ang])
        return action

def main(args):
    rclpy.init()
    node = FollowTheGapNode()
    while True:
        node.select_action()
