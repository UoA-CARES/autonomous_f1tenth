import rclpy
from .controller import Controller
import numpy as np
import time

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

    # but index 5 seems to be quaternion angle??
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    # goalx = float(state[0]) + 2
    # goaly = float(state[1]) + 2
    lin = 1
    ang = 0.5
    action = np.asarray([lin, ang])
    controller.step(action, policy_id)
    time.sleep(0.2)
    lin = -2
    ang = -0.5
    action = np.asarray([lin, ang])
    controller.step(action, policy_id)
    time.sleep(0.2)
    
    while True:
        # compute target [linear velocity, angular velocity]
        
        
        state = controller.get_observation(policy_id) # get_observation 
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)


if __name__ == '__main__':
    main()