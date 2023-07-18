import rclpy
from rclpy.node import Node
import numpy as np

class FollowTheGapNode(Node):
    def __init__(self):
        super().__init__('follow_the_gap')

    def calc_func(self):
        turn_angle = 0.4667
        min_turn_radius = 0.625
        lidar_angle=1.396
        meeting_point= np.sqrt(2*min_turn_radius**2-2*min_turn_radius**2*np.cos(2*lidar_angle))
        return meeting_point
    
    def calc_border_distances(self,range):
        obstacle_buffer = 0.0001 # value needs to be tinkered with
        chassis_width = 0.18
        d_n = np.sqrt(range**2-(obstacle_buffer+chassis_width)**2)
        return d_n
    
    def angle_to_ang_vel(self, gap_centre_angle):
        return gap_centre_angle+1
    
    def select_action(self,state):
        # Current x: state[0], current y: state[1], current z: state[2], orientation x: state[3], orientation y: state[4], orientation z: state[5]
        # linear vel x: state[6], angular vel z: state[7], LIDAR points 1-10: state[8-17] where each entry is the 64th LIDAR point
        lin = 5
        turn_angle = 0.4667
        min_turn_radius = 0.625
        lidar_angle=1.396
        min_lidar_range = 0.08
        max_lidar_range = 10
        lidar_poss_angles = np.linspace(-1.396, 1.396, 640)

        # each value in lidar_angles corresponds to a lidar range
        obstacles_angles = []
        obstacles_ranges = []
        for i in range(10):
            if (state[8+i] > min_lidar_range) & (state[8+i]<max_lidar_range):
                obstacles_ranges.append(state[8+i])
                sample = lidar_poss_angles[i*64]
                obstacles_angles.append(sample)
                

        # Add obstacle border values to array
        border_ranges = []
        border_angles = []
        for i in obstacles_ranges:
            border_dist = self.calc_border_distances(obstacles_ranges[i])
            angle = np.arccos(border_dist/obstacles_ranges[i])
            border_ranges.append(border_dist)
            border_ranges.append(border_dist)
            border_angles.append(obstacles_angles[i]-angle)
            border_angles.append(obstacles_angles[i]+angle)

        # Calculate nonholonomic edge constraints
        angle_constraint_l = -1
        angle_constraint_r = 1
        dist_constraint_l = 2
        dist_constraint_r = 2

        # Generate complete gap array, find max
        G = []
        angle_entry = angle_constraint_l-border_angles[0]
        G.append(np.abs(angle_entry))
        for i in (border_angles-1):
            angle_entry = border_angles[i]-border_angles[i+1]
            G.append(np.abs(angle_entry))
        angle_entry = border_angles[-1]-angle_constraint_r
        G.append(np.abs(angle_entry))
        greatest_gap = max(G) 
        greatest_gap_index = G.index(greatest_gap)

        # Find max gap centre angle
        if greatest_gap_index < 1:
            d1 = border_ranges[0]
            d2 = dist_constraint_l
            theta1 = border_angles[0]
            theta2 = angle_constraint_l
        elif greatest_gap_index > (len(border_angles)):
            d1 = dist_constraint_r
            d2 = border_ranges[-1]
            theta1 = border_angles[-1]
            theta2 = angle_constraint_r
        else:
            d1 = border_ranges[greatest_gap_index]
            d2 = border_ranges[greatest_gap_index-1]
            theta1 = border_angles[greatest_gap_index]
            theta2 = border_angles[greatest_gap_index-1]
        gap_centre_angle = np.arccos((d1+d2*np.cos(theta1+theta2))/(np.sqrt(d1**2+d2**2+2*d1*d2*np.cos(theta1+theta2))))-theta1
        
        # Calculate final heading angle
        ang = self.angle_to_ang_vel(gap_centre_angle)

        # Convert to angular velocity
        action = np.asarray([lin, ang])
        return action

