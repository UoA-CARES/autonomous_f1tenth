import rclpy
from rclpy.node import Node
import numpy as np

class FollowTheGapNode(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

    def calc_func():
        turn_angle = 0.4667
        min_turn_radius = 0.625
        lidar_angle=1.396
        meeting_point= np.sqrt(2*min_turn_radius**2-2*min_turn_radius**2*np.cos(2*lidar_angle))
        return meeting_point
    
    def select_action(state):
        # Current x: state[0], current y: state[1], current z: state[2], orientation x: state[3], orientation y: state[4], orientation z: state[5]
        # linear vel x: state[6], angular vel z: state[7], LIDAR points 1-10: state[8-17]
        
        lin = 5


        # Add obstacle border values to gap array


        # Calculate nonholonomic edge constraints

        # Generate complete gap array, find max


        # Find max gap centre angle

        # Calculate final heading angle


        # Convert to angular velocity
        ang = 0
        action = np.asarray([lin, ang])
        return action

