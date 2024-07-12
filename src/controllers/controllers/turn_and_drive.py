import rclpy
import numpy as np
import numpy.typing as npt
import random
from rclpy.impl import rcutils_logger
from .controller import Controller
from environments.util import get_euler_from_quarternion
from .util import turn_to_goal
import time, threading

class TurnAndDrive():

    logger = rcutils_logger.RcutilsLogger(name="tnd_log")

    def __init__(self, path:npt.NDArray, next_goal_skip_cnt=15,turning_lin_vel = 0.2, turning_ang_modifier = 1, straight_lin_vel = 1, angle_diff_tolerance = 0.1, goal_tolerance = 0.2, steering_to_neutral_delay = 0.5):
        # algorithm setup
        self.path = path
        self.next_goal_skip_cnt = next_goal_skip_cnt
        self.turning_lin_vel = turning_lin_vel
        self.turning_ang_modifier = turning_ang_modifier
        self.straight_lin_vel = straight_lin_vel
        self.angle_diff_tolerance = angle_diff_tolerance
        self.goal_tolerance = goal_tolerance

        # taking into account time for wheel to align with desired angle
        self.turnedLast = False
        self.steering_to_neutral_delay = steering_to_neutral_delay
        self.is_waiting_for_steering_neutral = False
        self.steering_neutral_timer = None

        #util for getting next goal to go to
        self.next_goal = None


        
    def complete_waiting_for_steering_neutral(self):
        self.steering_neutral_timer = None
        self.is_waiting_for_steering_neutral = False

    def find_next_goal(self, location:npt.NDArray) -> npt.NDArray:
        '''Take the the next forward_cnt'th location on the path as target to turn and drive.'''
        minDist = np.inf
        row, _ = self.path.shape
        for i in range(row):
            point = self.path[i]
            distance = point - location
            hyp = np.hypot(distance[0], distance[1])
            if (hyp < minDist): # Find closest point on path to car
                closestPointInd = i
                minDist = hyp

        return self.path[ (closestPointInd+self.next_goal_skip_cnt) % len(self.path-1) ]
     


    def select_action_to_goal(self, state, goal):
        '''Select an action to drive toward target. state: loc+quarternion rot [x,y, w,x,y,z]. goal: loc [x,y]'''
        location = state[0:2]
        self_angle = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]

        self.logger.info("-------------------------------------------------")
        self.logger.info("STATE: "+str(location)+" "+str(self_angle))

        ang = turn_to_goal(location, self_angle, goal)
        distance = goal - location

        # if already at goal location
        if ((abs(distance[0]) < self.goal_tolerance) and (abs(distance[1] < self.goal_tolerance))):
            lin = 0
            action = np.asarray([lin, ang])
            return action
        
        if abs(ang) > 0:
            lin = self.turning_lin_vel
            action = np.asarray([lin, ang])
            self.turnedLast = True
            return action
        
        # if already heading toward goal
        else:

            # transitioning from turning to straight, need to wait for steering to return to neutral
            if self.turnedLast:
                self.turnedLast = False
                self.is_waiting_for_steering_neutral = True
                self.steering_neutral_timer = threading.Timer(self.steering_to_neutral_delay,self.complete_waiting_for_steering_neutral)
                self.steering_neutral_timer.start()
            
            # ONLY go full speed after allowing steering to return to neutral
            if self.is_waiting_for_steering_neutral:
                lin = self.turning_lin_vel
            else:
                lin = self.straight_lin_vel
        action = np.asarray([lin, ang*self.turning_ang_modifier])
        return action    
    
    def select_action(self, state:npt.NDArray):
        location = state[0:2]
        # first action, no current goal to move toward
        if not self.next_goal:
            self.next_goal = self.find_next_goal(location)
        # reached current goal, get new one
        elif np.linalg.norm(self.next_goal - location) < self.goal_tolerance:
            self.next_goal = self.find_next_goal(location)
        # has a goal to go to
        else:
            pass
        return self.select_action_to_goal(state,self.next_goal)

        