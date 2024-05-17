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
    print(state[0:2])
    print(type(state[0]))
    goalx = float(state[0]) + 2
    goaly = float(state[1]) + 2
    goal = np.asarray([goalx, goaly])
    while True:
        action = policy.select_action(state, goal)   
        controller.step(action, policy_id)

class TurnAndDrive():
    # Still fixing
    def select_action(self, state, goal):
        location = state[0:2]
        distance = goal - location
        if ((abs(distance[0]) < 0.2)&(abs(distance[1] < 0.2))):
            lin = 0
            ang = 0
            action = np.asarray([lin, ang])
            return action
        
        angle = np.arctan2(distance[1], distance[0])
        if (((angle - state[5]) > 0.1)| ((angle - state[5]) < -0.1)):
            ang = angle - state[5]
            print(angle)
            print(state[5])
            print(ang)
            lin = 0.05
        else:
            ang = 0
            lin = 1
        
        
        action = np.asarray([lin, ang])
        return action    

if __name__ == '__main__':
    main()