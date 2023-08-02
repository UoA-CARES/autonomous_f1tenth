from .controller import Controller
import rclpy
import numpy as np
import random

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
    
    while True:
        action = np.asarray([random.uniform(0, 3), random.uniform(-3.14, 3.14)])    
        controller.step(action)
    

if __name__ == '__main__':
    main()