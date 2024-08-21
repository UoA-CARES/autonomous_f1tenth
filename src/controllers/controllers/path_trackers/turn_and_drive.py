import numpy as np
from rclpy.impl import rcutils_logger
from environments.util import get_euler_from_quarternion
from ..util import turn_to_goal
import threading

class TurnAndDrive():

    logger = rcutils_logger.RcutilsLogger(name="tnd_log")

    def __init__(self, turning_lin_vel = 0.2, turning_ang_modifier = 1, straight_lin_vel = 1, angle_diff_tolerance = 0.1, goal_tolerance = 0.2, steering_to_neutral_delay = 0.5):
        self.turning_lin_vel = turning_lin_vel
        self.turning_ang_modifier = turning_ang_modifier
        self.straight_lin_vel = straight_lin_vel
        self.angle_diff_tolerance = angle_diff_tolerance
        self.goal_tolerance = goal_tolerance
        self.multiCoord = False
        self.turnedLast = False

        self.steering_to_neutral_delay = steering_to_neutral_delay
        self.is_waiting_for_steering_neutral = False
        self.steering_neutral_timer = None
        
    def complete_waiting_for_steering_neutral(self):
        self.steering_neutral_timer = None
        self.is_waiting_for_steering_neutral = False

    # Still fixing
    def select_action(self, state, goal):
        location = state[0:2]
        self_angle = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]

        self.logger.info("-------------------------------------------------")
        self.logger.info("STATE: "+str(location)+" "+str(self_angle))
        self.logger.info("GOAL: "+str(goal))
        ang = turn_to_goal(location, self_angle, goal)
        distance = goal - location

        # if already at goal location
        if ((abs(distance[0]) < self.goal_tolerance) and (abs(distance[1] < self.goal_tolerance))):
            self.logger.info("At goal")
            lin = 0
            action = np.asarray([lin, ang])
            return action
        
        if abs(ang) > 0:
            lin = self.turning_lin_vel
            self.logger.info("Turning to goal")
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
