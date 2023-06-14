from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
import rclpy
from ament_index_python import get_package_share_directory
import time
import torch
import random
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.Plot import Plot
from .DataManager import DataManager
from cares_reinforcement_learning.networks.TD3 import Actor, Critic
from datetime import datetime
import pygame as pygame
from pynput.keyboard import Key, Listener
import time
import random
import numpy as np

rclpy.init()

param_node = rclpy.create_node('params')
param_node.declare_parameters(
    '',
    [
        ('gamma', 0.95),
        ('tau', 0.005),
        ('g', 10),
        ('batch_size', 32),
        ('buffer_size', 32),
        ('seed', 123), #TODO: This doesn't do anything yet
        ('actor_lr', 1e-4),
        ('critic_lr', 1e-3),
        ('max_steps_training', 1_000_000),
        ('max_steps_exploration', 1_000),
        ('max_steps', 100)
    ]
)

params = param_node.get_parameters([
    'max_steps_training',
    'max_steps_exploration', 
    'gamma', 
    'tau', 
    'g', 
    'batch_size', 
    'buffer_size', 
    'seed', 
    'actor_lr', 
    'critic_lr',
    'max_steps'
    ])

MAX_STEPS_TRAINING,\
MAX_STEPS_EXPLORATION,\
GAMMA,\
TAU,\
G,\
BATCH_SIZE,\
BUFFER_SIZE,\
SEED,\
ACTOR_LR,\
CRITIC_LR,\
MAX_STEPS = [param.value for param in params]

print(
    f'Exploration Steps: {MAX_STEPS_EXPLORATION}\n',
    f'Training Steps: {MAX_STEPS_TRAINING}\n',
    f'Gamma: {GAMMA}\n',
    f'Tau: {TAU}\n',
    f'G: {G}\n',
    f'Batch Size: {BATCH_SIZE}\n',
    f'Buffer Size: {BUFFER_SIZE}\n',
    f'Seed: {SEED}\n',
    f'Actor LR: {ACTOR_LR}\n',
    f'Critic LR: {CRITIC_LR}\n',
    f'Steps per Episode: {MAX_STEPS}\n'
)
MAX_ACTIONS = np.asarray([3, 1])
MIN_ACTIONS = np.asarray([0, -1])

OBSERVATION_SIZE = 8 + 10 + 2 # Car position + Lidar rays + goal position
ACTION_NUM = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_NAME = 'carwall_training-' + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")





linear_vel = 0
angular_vel = 0
offset = 0.05
MAX_SPEED = 3
NETURAL_SPEED = 0
MAX_ANGULAR = 3.14
LEFT_ANGULAR = MAX_ANGULAR
RIGHT_ANGULAR = -MAX_ANGULAR
NETURAL_ANGULAR = 0

def keyboard_on_key_press(key):
    global linear_vel, angular_vel

    if key == Key.up or (hasattr(key, 'char') and key.char == "w"):
        linear_vel = MAX_SPEED  # Set linear velocity to max
    elif key == Key.down or (hasattr(key, 'char') and key.char == "s"):
        linear_vel = (-MAX_SPEED) / 2  # Set linear velocity to 1/2 max
    elif key == Key.left or (hasattr(key, 'char') and key.char == "a"):
        angular_vel = LEFT_ANGULAR  # Set angular velocity to left
    elif key == Key.right or (hasattr(key, 'char') and key.char == "d"):
        angular_vel = RIGHT_ANGULAR  # Set angular velocity to right
                
def keyboard_on_key_release(key):
    global linear_vel, angular_vel

    if key in [Key.up] or (hasattr(key, 'char') and key.char in ["w", "s"]):
        linear_vel = NETURAL_SPEED  # Stop linear movement
    elif key in [Key.left, Key.right] or (hasattr(key, 'char') and key.char in ["a", "d"]):
        angular_vel = NETURAL_ANGULAR  # Stop angular movement

def joystick_check():
    global linear_vel, angular_vel

    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            # Left Joystick X-Axis
            if event.axis == 0:
                if abs(event.value) < offset:
                    angular_vel = NETURAL_ANGULAR
                else:
                    angular_vel = -1 * event.value * MAX_ANGULAR

            # Left Trigger
            if event.axis == 2:
                if event.value < -1 + offset:
                    linear_vel = NETURAL_SPEED
                else:
                    linear_vel = (event.value + 1) / 2 * ((-MAX_SPEED) / 2)

            # Right Trigger
            if event.axis == 5:
                if event.value < -1 + offset:
                    linear_vel = NETURAL_SPEED
                else:
                    linear_vel = (event.value + 1) / 2 * MAX_SPEED
        
        if event.type == pygame.JOYHATMOTION:
            if event.hat == 0:
                x_value, y_value = event.value
                
                if x_value == 0:
                    angular_vel = NETURAL_ANGULAR
                elif x_value == 1:  # Right
                    angular_vel = RIGHT_ANGULAR
                elif x_value == -1:  # Left
                    angular_vel = LEFT_ANGULAR

                if y_value == 0:
                    linear_vel = NETURAL_SPEED
                elif y_value == 1:
                    linear_vel = MAX_SPEED
                elif y_value == -1:
                    linear_vel = (-MAX_SPEED) / 2

        


def main():
    # Init joystick control
    pygame.init()
    pygame.joystick.init()
    pygame.joystick.Joystick(0) if pygame.joystick.get_count() >= 1 else None

    # Register the callback functions for key press and release events
    listener = Listener(
        on_press=keyboard_on_key_press,
        on_release=keyboard_on_key_release)
    listener.start()

    time.sleep(3)

    env = CarWallEnvironment('f1tenth', step_length=0.25, max_steps=MAX_STEPS)

    env.reset()
    i = 0

    while True:
        joystick_check()

        if (linear_vel < 0):
            _, _, done, truncated, _ = env.step([linear_vel, -1 * angular_vel])

        else:
            _, _, done, truncated, _ = env.step([linear_vel, angular_vel])

        # print("Linear velocity:", linear_vel)
        # print("Angular velocity:", angular_vel)

        if truncated or done:
            env.reset()



if __name__ == '__main__':
    main()
