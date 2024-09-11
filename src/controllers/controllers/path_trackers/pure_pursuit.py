import numpy as np
from rclpy.impl import rcutils_logger
from environments.util import get_euler_from_quarternion
from ..util import turn_to_goal
   
class PurePursuit():
    logger = rcutils_logger.RcutilsLogger(name="pure_pursuit_log")

    def __init__(self, path = np.asarray([]), look_ahead = 2, angle_diff_tolerance = 0.1): 
        self.logger.info("-------------------------------------------------")
        self.logger.info("Pure Pursuit Alg created")
        self.path = path
        self.angle_diff_tolerance = angle_diff_tolerance
        self.look_ahead = look_ahead
        self.multiCoord = True

    def loadPath(self, path):
        self.path = path
    
    # Need to write
    def findGoal(self, location):
        look_ahead = self.look_ahead
        minDist = np.inf
        lastPointInd = -1
        row, _ = self.path.shape
        for i in range(row):
            point = self.path[i]
            distance = point - location
            hyp = np.hypot(distance[0], distance[1])
            if (hyp < minDist): # Find closest point on path to car
                closestPointInd = i
                minDist = hyp
            if (hyp < look_ahead): # Find last point within lookahead distance to car
                lastPointInd = i
        if lastPointInd == (range(row-1)):
             for i in range(2):
                point = self.path[i]
                distance = point - location
                hyp = np.hypot(distance[0], distance[1])
                if (hyp < look_ahead):
                    lastPointInd = i
        if lastPointInd < 0: # If no points within lookahead range, goal1 is closest point
            self.logger.info("Nothing within lookahead")
            goal1Ind = closestPointInd
        else:  
            goal1Ind = lastPointInd
        goal1 = self.path[goal1Ind]
        try:    
            goal2 = self.path[goal1Ind+1]
        except:
            goal2 = self.path[0]    
        self.logger.info("Goal 1: "+str(goal1))
        self.logger.info("Goal 2: "+str(goal2))
        f = goal1 - location
        theta1 = np.arctan2(f[1], f[0])
        d = goal2 - goal1
        theta2 = np.arctan2(d[1], d[0])
        phi = (np.pi - theta1) + theta2 # Internal angle at goal 1

        # Distance from goal1 to final goal can be found using cosine rule: self.look_ahead**2 = F**2 + x**2 - 2*F*x*cos(phi), rearrange quadratic of x
        F = np.hypot(f[0], f[1])

        a = 1
        b = -2*F*np.cos(phi)
        print(b)
        c = F**2 - look_ahead**2
        print(c)

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            goal = goal1
            return goal
        elif discriminant == 0:
            x = (-b - np.sqrt(discriminant))/(2*a)
            goal = np.asarray([goal1[0]+x*np.cos(theta2), goal1[1]+x*np.sin(theta2)])
        else:
            x1 = (-b - np.sqrt(discriminant))/(2*a)
            x2 = (-b + np.sqrt(discriminant))/(2*a)
            goalx1 = np.asarray([goal1[0]+x1*(np.cos(theta2)), goal1[1]+x1*(np.sin(theta2))]) 
            goalx2 = np.asarray([goal1[0]+x2*(np.cos(theta2)), goal1[1]+x2*(np.sin(theta2))])
            if ((abs(goal2[0] - goalx1[0]) < abs(goal2[0]-goal1[0])) & (abs(goal2[1] - goalx1[1]) < abs(goal2[1] - goal1[1]))):
                self.logger.info("USING GOAL X1: "+str(goalx1))
                return goalx1
            else:
                self.logger.info("USING GOALX2: "+str(goalx2))
                return goalx2
        goal = np.asarray([0, 0])
        return goal
    
    def select_action(self, state, goal):
        MAX_ACTIONS = np.asarray([0.5, 0.5])
        MIN_ACTIONS = np.asarray([0, -0.7])

        lin = MAX_ACTIONS[0]
        location = state[0:2]
        yawcurr = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]

        if self.path.shape[0] == 0:
            raise Exception("No path loaded")

        # Calculate angle same as turn and drive
        trueGoal = self.findGoal(location)
        self.logger.info("CURRENT LOCATION: " + str(location))
        self.logger.info("Current yaw: "+str(yawcurr))
        
        ang = turn_to_goal(location, yawcurr, trueGoal)
        #Saturate ang
        ang = ang*MAX_ACTIONS[1]
        action = np.asarray([lin, ang])
        self.logger.info("DRIVE ANGLE: "+str(action[1]))
        self.logger.info("-------------------------")
        return action
