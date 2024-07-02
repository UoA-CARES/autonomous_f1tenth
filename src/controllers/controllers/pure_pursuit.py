import rclpy
import numpy as np
import random
from .controller import Controller
from rclpy.impl import rcutils_logger
from environments.util import get_euler_from_quarternion
   
class PurePursuit():
    logger = rcutils_logger.RcutilsLogger(name="pure_pursuit_log")

    def __init__(self, path, angle_diff_tolerance = 0.1): 
        self.logger.info("-------------------------------------------------")
        self.logger.info("Pure Pursuit Alg created")
        self.path = path
        self.angle_diff_tolerance = angle_diff_tolerance

    # Need to write
    def findGoal(self, location, yaw):
        goal = np.asarray([0, 0])
        return goal
    
    def select_action(self, state, goal):
        MAX_ACTIONS = np.asarray([1, 0.85])
        MIN_ACTIONS = np.asarray([0, -0.85])

        closestPoint = goal # Need to check if this will actually be valid
        lin = MAX_ACTIONS[0]
        location = state[0:2]
        steeringcurr = state[7]
        yawcurr = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]  

        # Calculate angle same as turn and drive
        trueGoal = self.findGoal(location, yawcurr)
        distance = trueGoal - location
        angle_to_goal = np.arctan2(distance[1], distance[0])
        if (((angle_to_goal - yawcurr) > self.angle_diff_tolerance) or ((angle_to_goal - yawcurr) < -self.angle_diff_tolerance)):
            
            # take the shortest turning angle
            self.steering_ang = angle_to_goal - yawcurr
            if self.steering_ang > np.pi:
                self.steering_ang -= 2 * np.pi
            elif self.steering_ang < -np.pi:
                self.steering_ang += 2 * np.pi

            self.logger.info("ANGLE TO GOAL: "+str(angle_to_goal))
            self.logger.info("SELF ANGLE: "+str(yawcurr))
            self.logger.info("DELTA: "+str(self.steering_ang))

            # make sure turning angle is not more than 90deg
            if self.steering_ang > 1.5:
                self.steering_ang = 1.5
            elif self.steering_ang < -1.5:
                self.steering_ang = -1.5

        ang = random.uniform(-3.14, 3.14)
        action = np.asarray([lin, ang])
        self.logger.info("DRIVE LIN_V: "+str(action[0]))
        self.logger.info("DRIVE ANGLE: "+str(action[1]))
        self.logger.info("-------------------------")
        return action
