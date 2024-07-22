import numpy as np
from casadi import *
from rclpy.impl import rcutils_logger
from environments.util import get_euler_from_quarternion

class MPC():

    logger = rcutils_logger.RcutilsLogger(name="mpc_log")

    def __init__(self, path): 
        self.logger.info("-------------------------------------------------")
        self.logger.info("MPC Alg created")
        self.path = path
        self.deltaT = 0.1
        self.wheelbase = 0.315
        self.timeConst = 0.1

        self.multiCoord = False # Can be set to True or False. True will incorporate next coord in cost

        self.goal_tolerance = 0.5
        # MPC parameters. Options defines how many alternative actions will be considered. Prediction steps defines how far into the future each action will be assessed.
        self.predictionSteps = 5
        self.options = 10

        
    
    def newStates(self, lin, x, y, steeringAngle, desAngle, yaw):
        # Uses simple bicycle model
        x = x + lin*np.cos(yaw)*self.deltaT
        y = y + lin*np.sin(yaw)*self.deltaT
        yaw = yaw + self.deltaT*(lin*np.tan(steeringAngle))/self.wheelbase
        steeringAngle = steeringAngle - self.timeConst**(-1)*(steeringAngle - desAngle)*self.deltaT

        return x, y, yaw, steeringAngle
    
    def cost(self, xcurr, x, xdes, ycurr, y, ydes, steeringcurr, steering, yawcurr, yaw, yawdes):
        Y = np.array([[xcurr, ycurr, yawcurr, 0, x, y, yaw, 0]])
        Yref = np.array([[xdes, ydes, yawdes, 0, xdes, ydes, yawdes, 0]])
        Yarr = Y - Yref
        qx = 1
        qy = 1
        qyaw = 0
        Q = np.diag([qx, qy, qyaw, 0 , qx, qy, qyaw, 0])
        qsteer = 0

        cost = Yarr@Q@np.transpose(Yarr) + qsteer*(steeringcurr - steering)**2 #(U - Uref)*R*(U-Uref)
        return cost
    
    def select_action(self, state, goal):
        MAX_ACTIONS = np.asarray([0.2, 0.85])
        MIN_ACTIONS = np.asarray([0, -0.85])
        self.logger.info("Current Location: "+str(state[0:2]))
        self.logger.info("Current goal: "+str(goal))
        lin = MAX_ACTIONS[0]
        xcurr, ycurr = state[0:2]
        steeringcurr = state[7]
        desAngles = np.linspace(MIN_ACTIONS[1], MAX_ACTIONS[1], self.options, endpoint=True)
        yawcurr = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]       
        lowestCost = inf
        distance = goal - state[0:2]
        #Check if car is already at goal location
        if ((abs(distance[0]) < self.goal_tolerance) and (abs(distance[1]) < self.goal_tolerance)):
            lin = 0
            ang = 0
            action = np.asarray([lin, ang])
            self.logger.info("Reached goal")
            return action

        # Iterate through potential driving options
        for i in range(self.options):
            desAngle = desAngles[i]
            x = xcurr
            y = ycurr
            steeringAngle = steeringcurr
            yaw = yawcurr
            for j in range(self.predictionSteps):
                x, y, yaw, steeringAngle = self.newStates(lin, x, y, steeringAngle, desAngle, yaw)
            cost = self.cost(xcurr, x, goal[0], ycurr, y, goal[1], steeringcurr, steeringAngle, yawcurr, yaw, yawdes=0)
            if (cost < lowestCost):
                lowestCost = cost
                ang = desAngle
        
        action = np.asarray([lin, ang])
        self.logger.info("DRIVE LIN_V: "+str(lin))
        self.logger.info("DRIVE ANGLE: "+str(ang))
        self.logger.info("-------------------------")
        return action 