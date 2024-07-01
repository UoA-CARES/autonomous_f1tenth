import rclpy
import numpy as np
import random
from rclpy.impl import rcutils_logger
from .controller import Controller
from environments.util import get_euler_from_quarternion
import time, threading
from .turn_and_drive import TurnAndDrive

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_two'),
            ('alg', 'random'),
        ]
    )
    
    params = param_node.get_parameters(['car_name', 'alg'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    ALG = params[1]
    
    controller = Controller(ALG, CAR_NAME, 0.25)
    policy_id = ALG
    policy = policy_factory(ALG)

    # but index 5 seems to be quaternion angle??
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    # goalx = float(state[0]) + 2
    # goaly = float(state[1]) + 2
    goalx = 5
    goaly = -2
    goal = np.asarray([goalx, goaly])
    
    while True:
        # compute target [linear velocity, angular velocity]
        
        
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)

def policy_factory(alg):
    policy = 0
    match alg:
        case 'mpc':
            from .mpc import MPC
            policy = MPC()
            return policy
        case 'turn_and_drive':
            from .turn_and_drive import TurnAndDrive
            policy = TurnAndDrive(goal_tolerance=0.5)
            return policy
        case 'random':
            from .random import Random
            policy = Random()
            return policy
        case _:
            return policy


if __name__ == '__main__':
    main()