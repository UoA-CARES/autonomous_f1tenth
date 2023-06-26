from environments.CarTrackOriginalEnvironment import CarTrackOriginalEnvironment
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
        ('buffer_size', 1_000_000),
        ('seed', 123), #TODO: This doesn't do anything yet
        ('actor_lr', 1e-4),
        ('critic_lr', 1e-3),
        ('max_steps_training', 2_000_000),
        ('max_steps_exploration', 1_000),
        ('max_steps', 1000),
        ('step_length', 0.25)
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
    'max_steps',
    'step_length',
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
MAX_STEPS,\
STEP_LENGTH = [param.value for param in params]

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
    f'Steps per Episode: {MAX_STEPS}\n',
    f'Step Length: {STEP_LENGTH}\n'
)
MAX_ACTIONS = np.asarray([3, 3.14])
MIN_ACTIONS = np.asarray([-0.5, -3.14])

OBSERVATION_SIZE = 8 + 10 + 2 # Car position + Lidar rays + goal position
ACTION_NUM = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_NAME = 'cartrack_training-' + datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

def main():
    time.sleep(3)

    env = CarTrackOriginalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=2)
    
    actor = Actor(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=ACTOR_LR)
    critic = Critic(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=CRITIC_LR)

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=GAMMA,
        tau=TAU,
        action_num=ACTION_NUM,
        device=DEVICE
    )

    train(env=env, agent=agent)

def train(env, agent: TD3):
    ep = DataManager(name=f'{TRAINING_NAME}_episode' , checkpoint_freq=100)
    step = DataManager(name=f"{TRAINING_NAME}_steps", checkpoint_freq=10_000)
    
    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset()

    historical_reward = {"step": [], "episode_reward": []}    


    for total_step_counter in range(int(MAX_STEPS_TRAINING)):
        episode_timesteps += 1

        if total_step_counter < MAX_STEPS_EXPLORATION:
            print(f"Running Exploration Steps {total_step_counter}/{MAX_STEPS_EXPLORATION}")
            action_env = np.asarray([random.uniform(MIN_ACTIONS[0], MAX_ACTIONS[0]), random.uniform(MIN_ACTIONS[1], MAX_ACTIONS[1])]) # action range the env uses [e.g. -2 , 2 for pendulum]
            action = hlp.normalize(action_env, MAX_ACTIONS, MIN_ACTIONS)  # algorithm range [-1, 1]
        else:
            action = agent.select_action_from_policy(state) # algorithm range [-1, 1]
            action_env = hlp.denormalize(action, MAX_ACTIONS, MIN_ACTIONS)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, info = env.step(action_env)
        # reward going forward
        if action_env[0] > 0.5:
            reward += 1
        
        # small penalty for time taken
        reward -= 0.3
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        step.post(reward)

        # rate.sleep()

        if total_step_counter % 50_000 == 0:
            agent.save_models(f'{TRAINING_NAME}_{total_step_counter}')

        if total_step_counter >= MAX_STEPS_EXPLORATION:
                for _ in range(G):
                    experiences = memory.sample(BATCH_SIZE)
                    experiences = (experiences['state'], experiences['action'], experiences['reward'], experiences['next_state'], experiences['done'])
                    # print(experiences)
                    agent.train_policy(experiences)

        if done or truncated:
            print(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            ep.post(episode_reward)
            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    agent.save_models(TRAINING_NAME)
    ep.save_csv()
    step.save_csv()

if __name__ == '__main__':
    main()
