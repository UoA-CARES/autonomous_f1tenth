import rclpy
import numpy as np
import random
from rclpy.impl import rcutils_logger
from .controller import Controller
from environments.util import get_euler_from_quarternion
import time, threading


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
    policy = TurnAndDrive(goal_tolerance=0.5)

    # but index 5 seems to be quaternion angle??
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    # goalx = float(state[0]) + 2
    # goaly = float(state[1]) + 2
    goalx = -5
    goaly = -2
    goal = np.asarray([goalx, goaly])
    
    while True:
        # compute target [linear velocity, angular velocity]
        
        
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)

class TurnAndDrive():

    logger = rcutils_logger.RcutilsLogger(name="tnd_log")

    def __init__(self, turning_lin_vel = 0.2, turning_ang_modifier = 1, straight_lin_vel = 1, angle_diff_tolerance = 0.1, goal_tolerance = 0.2, steering_to_neutral_delay = 0.5):
        self.turning_lin_vel = turning_lin_vel
        self.turning_ang_modifier = turning_ang_modifier
        self.straight_lin_vel = straight_lin_vel
        self.angle_diff_tolerance = angle_diff_tolerance
        self.goal_tolerance = goal_tolerance

        self.steering_ang = 0

        self.steering_to_neutral_delay = steering_to_neutral_delay
        self.is_waiting_for_steering_neutral = False
        self.steering_neutral_timer = None
        
    def complete_waiting_for_steering_neutral(self):
        self.steering_neutral_timer = None
        self.is_waiting_for_steering_neutral = False

    # Still fixing
    def select_action(self, state, goal):
        location = state[0:2]
        self_angle = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]

        self.logger.info("-------------------------------------------------")
        self.logger.info("STATE: "+str(location)+" "+str(self_angle))

        
        distance = goal - location

        # if already at goal location
        if ((abs(distance[0]) < self.goal_tolerance) and (abs(distance[1] < self.goal_tolerance))):
            lin = 0
            self.steering_ang = 0
            action = np.asarray([lin, self.steering_ang])
            return action
        
        # if needing to turn
        angle_to_goal = np.arctan2(distance[1], distance[0])
        if (((angle_to_goal - self_angle) > self.angle_diff_tolerance) or ((angle_to_goal - self_angle) < -self.angle_diff_tolerance)):
            
            # take the shortest turning angle
            self.steering_ang = angle_to_goal - self_angle
            if self.steering_ang > np.pi:
                self.steering_ang -= 2 * np.pi
            elif self.steering_ang < -np.pi:
                self.steering_ang += 2 * np.pi

            self.logger.info("ANGLE TO GOAL: "+str(angle_to_goal))
            self.logger.info("SELF ANGLE: "+str(self_angle))
            self.logger.info("DELTA: "+str(self.steering_ang))

            # make sure turning angle is not more than 90deg
            if self.steering_ang > 1.5:
                self.steering_ang = 1.5
            elif self.steering_ang < -1.5:
                self.steering_ang = -1.5

            lin = self.turning_lin_vel
        
        # if already heading toward goal
        else:    
            self.logger.info("ANGLE TO GOAL: "+str(angle_to_goal))

            # transitioning from turning to straight, need to wait for steering to return to neutral
            if self.steering_ang != 0:
                self.is_waiting_for_steering_neutral = True
                self.steering_neutral_timer = threading.Timer(self.steering_to_neutral_delay,self.complete_waiting_for_steering_neutral)
                self.steering_neutral_timer.start()

            # steering to 0
            self.steering_ang = 0
            
            # ONLY go full speed after allowing steering to return to neutral
            if self.is_waiting_for_steering_neutral:
                lin = self.turning_lin_vel
            else:
                lin = self.straight_lin_vel
        
        
        self.logger.info("RESULT LIN_V: "+str(lin))
        self.logger.info("RESULT ANGLE: "+str(self.steering_ang))
        action = np.asarray([lin, self.steering_ang*self.turning_ang_modifier])
        return action    

if __name__ == '__main__':
    main()