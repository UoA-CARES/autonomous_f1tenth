from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
import rclpy
from ament_index_python import get_package_share_directory
import time
import torch
import random
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

import numpy as np

rclpy.init()

param_node = rclpy.create_node('params')
param_node.declare_parameters(
    '',
    [
        ('max_steps_evaluation', 1_000_000),
        ('max_steps', 100),
        ('step_length', 0.25),
        ('actor_path', ''),
        ('critic_path', '')
    ]
)

params = param_node.get_parameters([
    'max_steps_evaluation',
    'max_steps',
    'step_length',
    'actor_path',
    'critic_path',
    ])

MAX_STEPS_EVALUATION, \
MAX_STEPS,\
STEP_LENGTH,\
ACTOR_PATH,\
CRITIC_PATH = [param.value for param in params]

print(
    f'Evaluation Steps: {MAX_STEPS_EVALUATION}\n',
    f'Steps per Episode: {MAX_STEPS}\n',
    f'Step Length: {STEP_LENGTH}\n',
    f'Critic Path: {CRITIC_PATH}\n',
    f'Actor Path: {ACTOR_PATH}'
)

if ACTOR_PATH is '' or CRITIC_PATH is '':
    raise Exception('Actor or Critic path not provided')


MAX_ACTIONS = np.asarray([3, 1])
MIN_ACTIONS = np.asarray([0, -1])

OBSERVATION_SIZE = 8 + 10 + 2 # Car position + Lidar rays + goal position
ACTION_NUM = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Share Directories

    time.sleep(3)
    env = CarWallEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS)
    
    actor = Actor(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)
    critic = Critic(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)

    print('Reading saved models into actor and critic')
    actor.load_state_dict(torch.load(ACTOR_PATH))
    critic.load_state_dict(torch.load(CRITIC_PATH))

    print('Successfully Loaded models')
    
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=0.999,
        tau=0.002,
        action_num=ACTION_NUM,
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
        action_env = hlp.denormalize(action, MAX_ACTIONS, MIN_ACTIONS)

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

if __name__ == '__main__':
    main()