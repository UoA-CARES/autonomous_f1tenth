import rclpy
from rclpy.node import Node
import numpy as np

class FollowTheGapNode(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

    def select_action(state):
        # Current x: state[0], current y: state[1], current z: state[2], orientation x: state[3], orientation y: state[4], orientation z: state[5]
        # linear vel x: state[6], angular vel z: state[7], LIDAR points 1-10: state[8-17]
        
        lin = 5
        ang = 0
        action = np.asarray([lin, ang])
        return action

