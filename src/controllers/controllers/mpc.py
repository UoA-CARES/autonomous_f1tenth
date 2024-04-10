import rclpy
from rclpy import Future
from rclpy.node import Node
import numpy as np


class MPC():
    def __init__(self, alg): 
        self.Alg = alg
        self.delta_t = 0.1
        self.wheelbase = 0.315
        self.time_const = 0.1
        self.predictionSteps = 2

    def calcAction(self, coord):
        lin = 0
        ang = 0
        return lin, ang
    
    def newStates(self, lin, ang, x, y, steering_angle, des_angle, yaw):
        # Uses simple bicycle model
        x = x + lin*np.cos(yaw)*self.delta_t
        y = y + lin*np.sin(yaw)*self.delta_t
        yaw = yaw + self.delta_t*(lin*np.tan(steering_angle))/self.wheelbase
        steering_angle = steering_angle - self.time_const**(-1)*(steering_angle - des_angle)*self.delta_t

        return x, y, yaw, steering_angle
    
    def cost(self, xcurr, x, xdes, ycurr, y, ydes, steeringcurr, steering, steerdes, yawcurr, yaw, yawdes):
        Y = np.array([[xcurr, ycurr, yawcurr, 0, x, y, yaw, 0]])
        Yref = np.array([[xdes, ydes, yawdes, 0, xdes, ydes, yawdes, 0]])
        Yarr = Y - Yref
        qx = 10
        qy = 100
        qyaw = 1000
        Q = np.diag([qx, qy, qyaw, 0 , qx, qy, qyaw, 0])
        qsteer = 50

        cost = Yarr@Q@np.transpose(Yarr) + qsteer*(steeringcurr - steering)**2 #(U - Uref)*R*(U-Uref)
        return cost