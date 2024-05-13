import rclpy
import numpy as np
import random
from .controller import Controller

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_two'),
        ]
    )
    
    params = param_node.get_parameters(['car_name'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    
    controller = Controller('turn_drive_', CAR_NAME, 0.25)
    policy_id = 'turn_drive'
    policy = TurnAndDrive()
    state = controller.get_observation(policy_id)
    goal = state[0:1]+2
    while True:
        action = policy.select_action(state, goal)   
        controller.step(action, policy_id)

class TurnAndDrive():

    def select_action(self, state, goal):
        location = state[0:1]
        lin = 1

        ang = state[7]
        action = np.asarray([lin, ang])
        return action    

if __name__ == '__main__':
    main()