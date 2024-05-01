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
    
    controller = Controller('random_', CAR_NAME, 0.25)
    policy_id = 'random'
    
    while True:
        action = np.asarray([random.uniform(0, 3), random.uniform(-3.14, 3.14)])    
        controller.step(action, policy_id)
    

if __name__ == '__main__':
    main()