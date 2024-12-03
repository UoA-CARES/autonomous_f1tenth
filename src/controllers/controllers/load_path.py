import rclpy
from .controller import Controller
import time
import numpy as np
import sys

def main():
    rclpy.init()
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'ld_path_car'),
        ]
    )

    params = param_node.get_parameters(['car_name'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    print("In load_path")
    controller = Controller('ld_path_policy_', CAR_NAME, 0.1, isCar=True)
    file1 = open('mypath.txt', 'w')
    try:
        createPath(controller, file1)
    except KeyboardInterrupt:
        file1.close()
    file1.close()

def createPath(controller, file):
    print("in createPath")
    while True:
        try:
            state = controller.get_observation('ld_path')
        except:
            break
        print("Escaped statement")
        s = '['+str(state[0])+', '+str(state[1]) + '], '
        file.write(s)
        
        time.sleep(1)

if __name__ == '__main__':
    main()
