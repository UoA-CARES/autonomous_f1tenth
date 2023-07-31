import rclpy
from rclpy.node import Node
import numpy as np
#from tf2.transformations import euler_from_quaternion

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
    
    def angle_to_ang_vel(self, driving_angle, lin):
        return driving_angle
    
    def select_action(self,state,goal_pos):
        # Current x: state[0], current y: state[1], orientation w: state[2], orientation x: state[3], orientation y: state[4], orientation z: state[5]
        # linear vel x: state[6], angular vel z: state[7], LIDAR points 1-10: state[8-17] where each entry is the 64th LIDAR point
        lin = 0.6
        turn_angle = 0.4667
        min_turn_radius = 0.625
        lidar_angle=1.396
        min_lidar_range = 0.08
        max_lidar_range = 10
        lidar_poss_angles = np.linspace(-1.396, 1.396, 640)
        meeting_dist = self.calc_func()
        #roll, pitch, rotation = euler_from_quaternion(state[2], state[3], state[4], state[5])


        rotation = np.arctan2((2*(state[2]*state[5]+state[3]*state[4])),(1-2*(state[4]**2+state[5]**2)))
        if (goal_pos[0] > state[0]):
            if (goal_pos[1] > state[1]):
                goal_angle = np.arctan((goal_pos[1]-state[1])/(goal_pos[0]-state[0])) - rotation
            else:
                goal_angle = np.arctan((goal_pos[1]-state[1])/(goal_pos[0]-state[0])) - rotation
        else:
            if (goal_pos[1] > state[1]):
                goal_angle = abs(np.arctan((goal_pos[0]-state[0])/(goal_pos[1]-state[1]))) - rotation + np.pi/2
            else:
                goal_angle = np.arctan((goal_pos[0]-state[0])/(goal_pos[1]-state[1]))*-1 - rotation - np.pi/2

        # each value in lidar_angles corresponds to a lidar range
        obstacles_angles = []
        obstacles_ranges = []
        obstacle_max_val = 2
        for i in range(10):
            if (state[8+i] > min_lidar_range) & (state[8+i]<obstacle_max_val):
                obstacles_ranges.append(state[8+i])
                sample = lidar_poss_angles[i*64]
                obstacles_angles.append(sample)

        if (len(obstacles_angles) < 1):
            ang = self.angle_to_ang_vel(goal_angle, lin)
            action = np.asarray([lin, ang])
            return action

        #print(f"Obstacles are at: {obstacles_angles}")
        # Add obstacle border values to array
        border_ranges = []
        border_angles = []
        for i in range(len(obstacles_ranges)):
            border_dist = self.calc_border_distances(obstacles_ranges[i])
            angle = np.arccos(border_dist/obstacles_ranges[i])
            border_ranges.append(border_dist)
            border_ranges.append(border_dist)
            border_angles.append(obstacles_angles[i]-angle)
            border_angles.append(obstacles_angles[i]+angle)

        #print(f"Obstacles are at: {border_angles}")
        # Calculate nonholonomic edge constraints
        if (border_ranges[0] < meeting_dist):
            angle_constraint_l = turn_angle
        else:
            angle_constraint_l = lidar_angle

        if (border_ranges[-1] < meeting_dist):
            angle_constraint_r = turn_angle*-1
        else:
            angle_constraint_r = lidar_angle*-1
        dist_constraint_l = border_ranges[-1]*np.cos(angle_constraint_l)
        dist_constraint_r = border_ranges[0]*np.cos(angle_constraint_r)

        # Generate complete gap array, find max
        G = []
        angle_entry = angle_constraint_r-border_angles[0]
        G.append(np.abs(angle_entry))
        for i in range(1, len(border_angles)-1, 2):
            if (border_angles[i] < border_angles[i+1]):
                angle_entry = border_angles[i]-border_angles[i+1]
            else:
                angle_entry = 0
            G.append(np.abs(angle_entry))
        angle_entry = border_angles[-1]-angle_constraint_l
        G.append(np.abs(angle_entry))
        greatest_gap = max(G) 
        greatest_gap_index = G.index(greatest_gap)

        #print(f"Gap array: {G}")
        #print(f"Greatest gap: {greatest_gap}")
        #print(f"Index: {greatest_gap_index}")

        # Find whether greatest gap is between two obstacles or one obstacle and a border
        if greatest_gap_index < 1: # Between right constraint and rightmost obstacle
            d1 = dist_constraint_r
            d2 = border_ranges[0]
            theta1 = angle_constraint_r
            theta2 = border_angles[0]
        elif (greatest_gap_index*2 > (len(border_angles)-1)): # Between leftmost obstacle and left constraint
            d1 = border_ranges[-1]
            d2 = dist_constraint_l
            theta1 = angle_constraint_l
            theta2 = border_angles[-1]
        else:
            d1 = border_ranges[greatest_gap_index*2-1]
            d2 = border_ranges[greatest_gap_index*2]
            theta1 = border_angles[greatest_gap_index*2-1]
            theta2 = border_angles[greatest_gap_index*2]

        if (theta1>theta2):
            print(f"Theta1: {theta1}")
            print(f"Theta 2: {theta2}")
            print(f"R constraint: {angle_constraint_r}")
            print(f"Borders: {border_angles}")
            print(f"L constraint: {angle_constraint_l}")
        #gap_centre_angle = np.arccos((d1+d2*np.cos(theta1+theta2))/(np.sqrt(d1**2+d2**2+2*d1*d2*np.cos(theta1+theta2))))-theta1
        if (theta1 > 0): # Both obstacles to left of robot
            phi = theta2-theta1
            l = np.sqrt((d1**2+d2**2-2*d1*d2*np.cos(phi))/4)
            h = np.sqrt((d1**2+d2**2+2*d1*d2*np.cos(phi))/4)
            gap_centre_angle = (np.arccos((h**2+d1**2-l**2)/(2*h*d1))) + theta1
            
        elif (theta2 < 0): # Both obstacles to right of robot
            phi = theta1-theta2
            l = np.sqrt((d1**2+d2**2-2*d1*d2*np.cos(phi))/4)
            h = np.sqrt((d1**2+d2**2+2*d1*d2*np.cos(phi))/4)
            gap_centre_angle = (np.arccos((d2**2+h**2-l**2)/(2*d2*h)))*-1 + theta2
        else: # One obstacle on left, other on right
            phi = abs(theta1)+theta2
            l = np.sqrt((d1**2+d2**2-2*d1*d2*np.cos(phi))/4)
            h = np.sqrt((d1**2+d2**2+2*d1*d2*np.cos(phi))/4)
            if (l > (d2*np.sin(theta2))): # Turning right
                gap_centre_angle = (np.arccos((d2**2+h**2-l**2)/(2*d2*h))-theta2)*-1

            else: # Turning left
                gap_centre_angle = np.arccos((d1**2+h**2-l**2)/(2*d1*h))-abs(theta1)
        print(f"Right dist: {d1}")
        print(f"Left dist: {d2}")
        print(f"Right obstacle: {theta1}")
        print(f"Left obstacle: {theta2}")
        print(f"phi: {phi}")
        print(f"l: {l}")
        print(f"h: {h}")
        print(f"Gap centre angle: {gap_centre_angle}")
        #print(f"Goal Angle: {goal_angle}")
        # Calculate final heading angle
        dmin = min(border_ranges)
        alpha = 4
        final_heading_angle = ((alpha/dmin)*gap_centre_angle+goal_angle)/((alpha/dmin)+1)
        print(f"Final angle: {final_heading_angle}")
        # Convert to angular velocity
        ang = self.angle_to_ang_vel(final_heading_angle, lin)
        #ang = self.angle_to_ang_vel(-2)
        action = np.asarray([lin, ang])
        return action

