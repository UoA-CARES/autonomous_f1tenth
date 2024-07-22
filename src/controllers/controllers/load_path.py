import rclpy
from .controller import Controller
import time
import numpy as np

def main():
    rclpy.init()
    print("In node")
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
    controller = Controller('ld_path_policy_', CAR_NAME, 0.1)
    file1 = open('mypath.txt', 'w')
    try:
        createPath(controller, file1)
    except KeyboardInterrupt:
        file1.close()
        print("Closed file")
    
    
    '''state = np.asarray([0, 0])
    s = '['+str(state[0])+', '+str(state[1]) + '],'
    file1.write(s)
    file1.close
    file1 = open('mypath.txt', 'w')
    state = np.asarray([1, 2])
    s = '['+str(state[0])+', '+str(state[1]) + '],'
    file1.write(s)
    file1.close'''
    '''while True:
        state = controller.get_observation('ld_path')

        # Save state


        time.sleep(1)'''

def createPath(controller, file):
    while True:
        state = np.asarray([0, 0])
        s = '['+str(state[0])+', '+str(state[1]) + '], '
        file.write(s)
        time.sleep(1)




if __name__ == '__main__':
    main()