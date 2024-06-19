import rclpy
import numpy as np
import random
from rclpy.impl import rcutils_logger
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

    # but index 5 seems to be quaternion angle??
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    # goalx = float(state[0]) + 2
    # goaly = float(state[1]) + 2
    goalx = 1
    goaly = 5
    goal = np.asarray([goalx, goaly])
    
    while True:
        # compute target [linear velocity, angular velocity]
        
        
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)

class TurnAndDrive():

    logger = rcutils_logger.RcutilsLogger(name="tnd_log")

    # Still fixing
    def select_action(self, state, goal):
        location = state[0:2]
        self.logger.info("-------------------------------------------------")
        # self.logger.info("LOCATION: "+str(location[0])+" "+str(location[1]))
        self.logger.info("STATE: "+str(state[0:6]))

        
            
        distance = goal - location
        if ((abs(distance[0]) < 0.2) and (abs(distance[1] < 0.2))):
            lin = 0
            ang = 0
            action = np.asarray([lin, ang])
            return action
        
        angle = np.arctan2(distance[1], distance[0])
        if (((angle - state[5]) > 0.1) or ((angle - state[5]) < -0.1)):
            ang = angle - state[5]
            # print(angle)
            # print(state[5])
            # print(ang)
            self.logger.info("ANGLE TO GOAL: "+str(angle))
            self.logger.info("SELF ANGLE: "+str(state[5]))
            self.logger.info("DELTA: "+str(ang))

            if ang > 3:
                ang = 3
            elif ang < -3:
                ang = -3

            lin = 0.25
        else:
            self.logger.info("ANGLE TO GOAL: "+str(angle))
            ang = 0
            lin = 1
        
        
        self.logger.info("RESULT LIN_V: "+str(lin))
        self.logger.info("RESULT ANG_V: "+str(ang))
        action = np.asarray([lin, ang])
        return action    

if __name__ == '__main__':
    main()