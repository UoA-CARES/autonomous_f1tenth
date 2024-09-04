import rclpy
import numpy as np
from .controller import Controller
from .util import closestPointIndAhead
import time

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_two'),
            ('alg', 'random'),
            ('isCar', False)
        ]
    )
    
    params = param_node.get_parameters(['car_name', 'alg', 'isCar'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    ALG = params[1]
    isCar = params[2]
    
    controller = Controller(ALG, CAR_NAME, 0.25, isCar)
    policy_id = ALG
    policy = policy_factory(ALG)
    if policy.multiCoord == False:
        from .test_path import austinLap, straightLine, circleCCW, testing
        coordinates = testing()
        #coordinates = straightLine()
        #coordinates = circleCCW()
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)
    while True:
        
        if policy.multiCoord == False:
            goalInd = closestPointIndAhead(state[0:2], coordinates)
            goal = coordinates[goalInd]
        else:
            goal = np.asarray([[0, 0]])
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)
        time.sleep(0.3)
        action = np.asarray([0,0])
        controller.step(action, policy_id)
        time.sleep(0.1)

def policy_factory(alg):
    policy = 0
    match alg:
        case 'mpc':
            from .path_trackers.mpc import MPC
            from .test_path import austinLap
            coordinates = austinLap()
            policy = MPC(coordinates)
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
            from .test_path import austinLap
            coordinates = austinLap()
            policy = PurePursuit(coordinates)
            return policy
        case _:
            return policy


if __name__ == '__main__':
    main()