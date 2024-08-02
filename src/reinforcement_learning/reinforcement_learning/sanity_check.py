import time
from datetime import datetime

import numpy as np
import pygame as pygame
import rclpy
import torch
from pynput.keyboard import Key, Listener

from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarWallEnvironment import CarWallEnvironment

rclpy.init()

param_node = rclpy.create_node('params')
param_node.declare_parameters(
    '',
    [
        ('environment', 'CarGoal'),
        ('track', 'track_1'),
        ('gamma', 0.95),
        ('tau', 0.005),
        ('g', 10),
        ('batch_size', 32),
        ('buffer_size', 32),
        ('seed', 123),  # TODO: This doesn't do anything yet
        ('actor_lr', 1e-4),
        ('critic_lr', 1e-3),
        ('max_steps_training', 1_000_000),
        ('max_steps_exploration', 1_000),
        ('max_steps', 1000),
        ('step_length', 0.25),
        ('reward_range', 0.2),
        ('collision_range', 0.2)
    ]
)

params = param_node.get_parameters([
    'environment',
    'track',
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
    'max_steps',
    'step_length',
    'reward_range',
    'collision_range'
])

ENVIRONMENT, \
TRACK, \
MAX_STEPS_TRAINING, \
MAX_STEPS_EXPLORATION, \
GAMMA, \
TAU, \
G, \
BATCH_SIZE, \
BUFFER_SIZE, \
SEED, \
ACTOR_LR, \
CRITIC_LR, \
MAX_STEPS, \
STEP_LENGTH, \
REWARD_RANGE, \
COLLISION_RANGE = [param.value for param in params]

print(
    f'---------------------------------------------\n'
    f'Environment: {ENVIRONMENT}\n'
    f'Exploration Steps: {MAX_STEPS_EXPLORATION}\n'
    f'Training Steps: {MAX_STEPS_TRAINING}\n'
    f'Gamma: {GAMMA}\n'
    f'Tau: {TAU}\n'
    f'G: {G}\n'
    f'Batch Size: {BATCH_SIZE}\n'
    f'Buffer Size: {BUFFER_SIZE}\n'
    f'Seed: {SEED}\n'
    f'Actor LR: {ACTOR_LR}\n'
    f'Critic LR: {CRITIC_LR}\n'
    f'Steps per Episode: {MAX_STEPS}\n'
    f'Step Length: {STEP_LENGTH}\n'
    f'Reward Range: {REWARD_RANGE}\n'
    f'Collision Range: {COLLISION_RANGE}\n'
    f'---------------------------------------------\n'
)

match ENVIRONMENT:
    case 'CarWall':
        env = CarWallEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
    case 'CarBlock':
        env = CarBlockEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
    case 'CarTrack':
        env = CarTrackEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE, track=TRACK)
    case _:
        env = CarGoalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE)

MAX_ACTIONS = np.asarray([3, 1])
MIN_ACTIONS = np.asarray([0, -1])

OBSERVATION_SIZE = 8 + 10 + 2  # Car position + Lidar rays + goal position
ACTION_NUM = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_NAME = 'sanity_check-' + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

# For controlling the car
linear_vel = 0
steering_angle = 0

OFFSET = 0.05

MAX_SPEED = 0.5
NEUTRAL_SPEED = 0.001
MAX_ANGLE = 0.85

LEFT_ANGLE = MAX_ANGLE
RIGHT_ANGLE = -MAX_ANGLE
NEUTRAL_ANGLE = 0
# ====================


def keyboard_on_key_press(key):
    global linear_vel, steering_angle

    if key == Key.up or (hasattr(key, 'char') and key.char == "w"):
        linear_vel = MAX_SPEED  # Set linear velocity to max
    elif key == Key.down or (hasattr(key, 'char') and key.char == "s"):
        linear_vel = (-MAX_SPEED) / 2  # Set linear velocity to 1/2 max
    elif key == Key.left or (hasattr(key, 'char') and key.char == "a"):
        steering_angle = LEFT_ANGLE  # Set angular velocity to left
    elif key == Key.right or (hasattr(key, 'char') and key.char == "d"):
        steering_angle = RIGHT_ANGLE  # Set angular velocity to right


def keyboard_on_key_release(key):
    global linear_vel, steering_angle

    if key in [Key.up] or (hasattr(key, 'char') and key.char in ["w", "s"]):
        linear_vel = NEUTRAL_SPEED  # Stop linear movement
    elif key in [Key.left, Key.right] or (hasattr(key, 'char') and key.char in ["a", "d"]):
        steering_angle = NEUTRAL_ANGLE  # Stop angular movement


def joystick_check():
    global linear_vel, steering_angle

    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            # Left Joystick X-Axis
            if event.axis == 0:
                if abs(event.value) < OFFSET:
                    steering_angle = NEUTRAL_ANGLE
                else:
                    steering_angle = -1 * event.value * MAX_ANGLE

            # Left Trigger
            if event.axis == 2:
                if event.value < -1 + OFFSET:
                    linear_vel = NEUTRAL_SPEED
                else:
                    linear_vel = (event.value + 1) / 2 * ((-MAX_SPEED) / 2)

            # Right Trigger
            if event.axis == 5:
                if event.value < -1 + OFFSET:
                    linear_vel = NEUTRAL_SPEED
                else:
                    linear_vel = (event.value + 1) / 2 * MAX_SPEED

        if event.type == pygame.JOYHATMOTION:
            if event.hat == 0:
                x_value, y_value = event.value

                if x_value == 0:
                    steering_angle = NEUTRAL_ANGLE
                elif x_value == 1:  # Right
                    steering_angle = RIGHT_ANGLE
                elif x_value == -1:  # Left
                    steering_angle = LEFT_ANGLE

                if y_value == 0:
                    linear_vel = NEUTRAL_SPEED
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

    env.reset()
    i = 0

    while True:
        joystick_check()

        if linear_vel < 0:
            _, r, done, truncated, _ = env.step([linear_vel, -1 * steering_angle])

        else:
            _, r, done, truncated, _ = env.step([linear_vel, steering_angle])

        if truncated or done:
            print("Finished -", r)
            env.reset()


if __name__ == '__main__':
    main()
