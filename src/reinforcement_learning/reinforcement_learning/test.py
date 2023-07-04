from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarTrackOriginalEnvironment import CarTrackOriginalEnvironment
from environments.CarTrack1Environment import CarTrack1Environment
from environments.CarTrack2Environment import CarTrack2Environment

import rclpy
from ament_index_python import get_package_share_directory
import time
import torch
import random
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

import numpy as np


def main():

    rclpy.init()

    params = get_params()

    ENVIRONMENT,\
    MAX_STEPS_EVALUATION, \
    MAX_STEPS,\
    STEP_LENGTH,\
    REWARD_RANGE,\
    COLLISION_RANGE,\
    ACTOR_PATH,\
    CRITIC_PATH = [param.value for param in params]

    print(
        f'---------------------------------------------\n'
        f'Environment: {ENVIRONMENT}\n'
        f'Evaluation Steps: {MAX_STEPS_EVALUATION}\n'
        f'Steps per Episode: {MAX_STEPS}\n'
        f'Step Length: {STEP_LENGTH}\n'
        f'Reward Range: {REWARD_RANGE}\n'
        f'Collision Range: {COLLISION_RANGE}\n'
        f'Critic Path: {CRITIC_PATH}\n'
        f'Actor Path: {ACTOR_PATH}\n'
        f'---------------------------------------------\n'
    )

    if ACTOR_PATH == '' or CRITIC_PATH == '':
        raise Exception('Actor or Critic path not provided')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    time.sleep(3)

    match(ENVIRONMENT):
        case 'CarWall':
            env = CarWallEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarBlock':
            env = CarBlockEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarTrack':
            env = CarTrackOriginalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarTrack1':
            env = CarTrack1Environment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarTrack2':
            env = CarTrack2Environment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case _:
            env = CarGoalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE)
    
    actor = Actor(observation_size=env.OBSERVATION_SIZE, num_actions=env.ACTION_NUM, learning_rate=0.1)
    critic = Critic(observation_size=env.OBSERVATION_SIZE, num_actions=env.ACTION_NUM, learning_rate=0.1)

    print('Reading saved models into actor and critic')
    actor.load_state_dict(torch.load(ACTOR_PATH))
    critic.load_state_dict(torch.load(CRITIC_PATH))

    print('Successfully Loaded models')
    
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=0.999,
        tau=0.002,
        action_num=env.ACTION_NUM,
        device=DEVICE
    )

    test(env=env, agent=agent)

def test(env, agent: TD3):
    state, _ = env.reset()    
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    print('Beginning Evaluation')

    for total_step_counter in range(int(MAX_STEPS_EVALUATION)):
        episode_timesteps += 1

        action = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

        next_state, reward, done, truncated, info = env.step(action_env)

        state = next_state
        episode_reward += reward

        if done or truncated:
            print(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1



def get_params():
    '''
    This function fetches the hyperparameters passed in through the launch files
    - The hyperparameters below are defaults, to change them, you should change the train.yaml config
    '''

    param_node = rclpy.create_node('params')
    param_node.declare_parameters(
        '',
        [
            ('environment', 'CarTrack1'),
            ('max_steps_evaluation', 1_000_000),
            ('max_steps', 100),
            ('step_length', 0.25),
            ('reward_range', 0.2),
            ('collision_range', 0.2),
            ('actor_path', ''),
            ('critic_path', '')
        ]
    )

    return param_node.get_parameters([
        'environment',
        'max_steps_evaluation',
        'max_steps',
        'step_length',
        'reward_range',
        'collision_range',
        'actor_path',
        'critic_path',
        ])

if __name__ == '__main__':
    main()