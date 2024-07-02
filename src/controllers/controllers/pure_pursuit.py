import rclpy
import numpy as np
import random
from .controller import Controller
from rclpy.impl import rcutils_logger
from environments.util import get_euler_from_quarternion, turn_to_goal
   
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
        ang = turn_to_goal(location, yawcurr, goal)
        action = np.asarray([lin, ang])
        self.logger.info("DRIVE LIN_V: "+str(action[0]))
        self.logger.info("DRIVE ANGLE: "+str(action[1]))
        self.logger.info("-------------------------")
        return action
