import rclpy
import numpy as np
import os
from .controller import Controller
from .util import closestPointIndAhead, loadPath, furthestPointInRange
import time

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_two'),
            ('alg', 'random'),
            ('isCar', False),
            ('path_file_path', 'random')
        ]
    )
    location = []
    params = param_node.get_parameters(['car_name', 'alg', 'isCar', 'path_file_path'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    ALG = params[1]
    isCar = params[2]
    filename = params[3]
    
    controller = Controller(ALG, CAR_NAME, 0.25, isCar)
    policy_id = ALG

    while(os.path.isfile(filename) == False):
        time.sleep(1)
    time.sleep(1)
    file = open("coords.txt", 'w')
    coordinates = loadPath(filename)
    policy = policy_factory(ALG)
    state = controller.get_observation(policy_id)
    while True:
        
        if policy.multiCoord == False:
            goalInd = furthestPointInRange(state[0:2], coordinates, 0.8)
            goal = coordinates[goalInd]
            if goalInd != (len(coordinates)-1):
                nextGoal = coordinates[goalInd + 1]
            else:
                nextGoal = coordinates[0]
        else:
            policy.loadPath(coordinates)
            goal = np.asarray([[0, 0]])
            nextGoal = goal
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal, nextGoal)   
        # moves car
        controller.step(action, policy_id)
        s = '['+str(round(state[0], 2))+', '+str(round(state[1], 2)) + '], '
        file.write(s)
        time.sleep(0.15)
        action = np.asarray([0,0])
        controller.step(action, policy_id)
        time.sleep(0.1)
    file.close()

def policy_factory(alg):
    policy = 0
    match alg:
        case 'mpc':
            from .path_trackers.mpc import MPC
            policy = MPC()
            return policy
        case 'turn_and_drive':
            from .path_trackers.turn_and_drive import TurnAndDrive
            policy = TurnAndDrive(goal_tolerance=0.5)
            return policy
        case 'random':
            from .path_trackers.random import Random
            policy = Random()
            return policy
        case 'pure_pursuit':
            from .path_trackers.pure_pursuit import PurePursuit
            policy = PurePursuit()
            return policy
        case _:
            return policy


if __name__ == '__main__':
    main()
