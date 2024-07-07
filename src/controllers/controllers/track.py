import rclpy
import numpy as np
import random
from rclpy.impl import rcutils_logger
from .controller import Controller
from environments.util import get_euler_from_quarternion
import time, threading

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

    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    goalx = 5
    goaly = -2
    goal = np.asarray([goalx, goaly])
    
    while True:
        
        
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)
        time.sleep(0.2)
        #time.sleep(1)

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
        case 'pure_pursuit':
            from .pure_pursuit import PurePursuit
            coordinates = np.asarray([[0, 0], [3, 1], [4, 2], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]])
            policy = PurePursuit(coordinates)
            return policy
        case _:
            return policy


if __name__ == '__main__':
    main()