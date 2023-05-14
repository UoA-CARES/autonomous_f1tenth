from simulation.simulation_services import SimulationServices #, ResetServices
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
import rclpy
from ament_index_python import get_package_share_directory
import time
import torch
import random
from cares_reinforcement_learning.algorithm import TD3
from cares_reinforcement_learning.util import MemoryBuffer, helpers as hlp
from cares_reinforcement_learning.util.Plot import Plot
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

import numpy as np

MAX_TESTING_STEPS = 1_000_000

GAMMA = 0.95
TAU = 0.005
G = 10

BATCH_SIZE = 32
BUFFER_SIZE = 1_000_000

SEED = 123

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

MAX_ACTIONS = np.asarray([3, 1])
MIN_ACTIONS = np.asarray([0, -1])

OBSERVATION_SIZE = 8 + 0 + 2 # Car position + Lidar rays + goal position
ACTION_NUM = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    rclpy.init()
    
    # Share Directories

    time.sleep(3)
    env = CarGoalEnvironment('f1tenth', step_length=0.25, max_steps=100)
    # env = CarWallEnvironment('f1tenth', step_length=0.25)
    
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
    
    agent.load_models('training_logs/12-may-training-cargoal-success/', '12-may-cargoal-step-675000')
    test(env=env, agent=agent)

def test(env, agent: TD3):
    state, _ = env.reset()    
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    
    for total_step_counter in range(int(MAX_TESTING_STEPS)):
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