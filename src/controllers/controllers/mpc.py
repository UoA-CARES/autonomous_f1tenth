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