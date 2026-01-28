import rclpy
import numpy as np
from .controller import Controller
import os
import time

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'ftg_car'),
            ('track_name', 'multi_track'),
        ]
    )

    params = param_node.get_parameters(['car_name', 'track_name'])
    CAR_NAME, TRACK_NAME = [param.value for param in params]
    controller = Controller('ftg_policy_', CAR_NAME, 0.1)
    policy = FollowTheGapPolicy()
    policy_id = 'ftg'
    state = controller.get_observation(policy_id)
    file = open("coords.txt", "w")

    while (os.path.isfile('stateMap.pgm') == False):
        action = policy.select_action(state)
        state = controller.step(action, policy_id)
        s = '['+str(round(state[0], 2))+', '+str(round(state[1], 2)) + '], '
        file.write(s)
    action = np.asarray([0, 0])
    file.close()
    time.sleep(1)
    state = controller.step(action, policy_id)



class FollowTheGapPolicy():

    def __init__(self):
        self.turn_angle = 0.4667
        self.min_turn_radius = 0.625
        self.lidar_angle = 1.396
        self.min_lidar_range = 0.08
        self.obstacle_max_val = 4

        # Need to make this read in from config
        self.lin = 1


    def calc_func(self):
        meeting_point= np.sqrt(2*self.min_turn_radius**2-2*self.min_turn_radius**2*np.cos(2*self.lidar_angle))
        return meeting_point
    
    def calc_border_distances(self,range):
        obstacle_buffer = 0.001 # value needs to be tinkered with
        chassis_width = 0.16

        d_n = np.sqrt(max(0.001,range**2-(obstacle_buffer+chassis_width)**2))
        return d_n
    
    def constrain_angle(self, angle):
        val = angle
        while(abs(val)>(np.pi*2)):
            val -= np.pi*2*np.sign(val)
        if abs(val) > np.pi:
            val = (np.pi*2-abs(val))*-1*np.sign(val)
        return val
    
    def select_action(self, state):
        lidar_poss_angles = np.linspace(-self.lidar_angle, self.lidar_angle, 10)
        meeting_dist = self.calc_func()

        obstacles_angles = []
        obstacles_ranges = []
        for i in range(10):
            if (state[8+i] > self.min_lidar_range) & (state[8+i]<self.obstacle_max_val):
                obstacles_ranges.append(state[8+i])
                sample = lidar_poss_angles[i]
                obstacles_angles.append(sample)

        if (len(obstacles_angles) < 1):
            action = np.asarray([self.lin, 0])
            return action

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

        # Calculate nonholonomic edge constraints
        if (border_ranges[-1] < meeting_dist):
            angle_constraint_l = self.turn_angle
        else:
            angle_constraint_l = self.lidar_angle

        if (border_ranges[0] < meeting_dist):
            angle_constraint_r = self.turn_angle*-1
        else:
            angle_constraint_r = self.lidar_angle*-1
        
        r_del_index = -1 
        for i in range(1, len(border_angles), 2): 
            if (border_angles[i]<angle_constraint_r):
                r_del_index = i+1
        if (r_del_index>0):
            del border_angles[0:r_del_index]
            del border_ranges[0:r_del_index]
        if (len(border_ranges) < 1):
            action = np.asarray([self.lin, 0])
            return action
        
        l_del_index = len(border_angles)
        for i in range(len(border_angles)-2, 0, -2): 
            if (border_angles[i]>angle_constraint_l):
                l_del_index = i
        if (l_del_index<len(border_angles)):
            del border_angles[l_del_index:]
            del border_ranges[l_del_index:]

        dist_constraint_l = border_ranges[-1]*np.cos(angle_constraint_l)
        dist_constraint_r = border_ranges[0]*np.cos(angle_constraint_r)
        if (len(border_ranges) < 1):
            action = np.asarray([self.lin, 0])
            return action
        
        
        # Generate complete gap array, find max
        G = []
        if (border_angles[0] > angle_constraint_r):
            angle_entry = angle_constraint_r-border_angles[0]
        else:
            angle_entry = 0
        G.append(np.abs(angle_entry))
        for i in range(1, len(border_angles)-1, 2):
            if (border_angles[i] < border_angles[i+1]):
                angle_entry = border_angles[i]-border_angles[i+1]
            else:
                angle_entry = 0
            G.append(np.abs(angle_entry))
        if (border_angles[-1] < angle_constraint_l):
            angle_entry = border_angles[-1]-angle_constraint_l
        else:
            angle_entry = 0
        G.append(np.abs(angle_entry))
        greatest_gap = max(G)
        if (greatest_gap == 0):
            action = np.asarray([self.lin, 0])
            return action 
        greatest_gap_index = G.index(greatest_gap)

        # Find whether greatest gap is between two obstacles or one obstacle and a border
        if greatest_gap_index < 1: # Between right constraint and rightmost obstacle
            d1 = dist_constraint_r
            d2 = border_ranges[0]
            theta1 = angle_constraint_r
            theta2 = border_angles[0]
        elif (greatest_gap_index*2 > (len(border_angles)-1)): # Between leftmost obstacle and left constraint
            d1 = border_ranges[-1]
            d2 = dist_constraint_l
            theta1 = border_angles[-1]
            theta2 = angle_constraint_l
        else:
            d1 = border_ranges[greatest_gap_index*2-1]
            d2 = border_ranges[greatest_gap_index*2]
            theta1 = border_angles[greatest_gap_index*2-1]
            theta2 = border_angles[greatest_gap_index*2]
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
        gap_centre_angle = self.constrain_angle(gap_centre_angle)

        ang = gap_centre_angle 
        action = np.asarray([self.lin, ang])
        return action



if __name__ == '__main__':
    main()