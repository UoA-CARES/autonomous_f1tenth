from simulation.simulation_services import SimulationServices #, ResetServices
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
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

import numpy as np

MAX_STEPS_TRAINING = 1_000_000
MAX_STEPS_EXPLORATION = 100_000

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
TRAINING_NAME = 'cargoal_training'
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

    train(env=env, agent=agent)

def train(env, agent: TD3):
    plot = Plot(title=f'{TRAINING_NAME}-episode' ,plot_freq=100, checkpoint_freq=100)
    step = Plot(title=f"{TRAINING_NAME}-steps", plot_freq=MAX_STEPS_TRAINING, checkpoint_freq=1_000)
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
            action_env = np.asarray([random.uniform(0, 3), random.uniform(-1, 1)]) # action range the env uses [e.g. -2 , 2 for pendulum]
            action = hlp.normalize(action_env, MAX_ACTIONS, MIN_ACTIONS)  # algorithm range [-1, 1]
        else:
            action = agent.select_action_from_policy(state) # algorithm range [-1, 1]
            action_env = hlp.denormalize(action, MAX_ACTIONS, MIN_ACTIONS)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        step.post(reward)

        if total_step_counter % 25_000 == 0:
            agent.save_models(f'{TRAINING_NAME}-{total_step_counter}')

        if total_step_counter >= MAX_STEPS_EXPLORATION:
                for _ in range(G):
                    experiences = memory.sample(BATCH_SIZE)
                    agent.train_policy(experiences)

        if done or truncated:
            print(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            plot.post(episode_reward)
            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    agent.save_models(TRAINING_NAME)
    plot.save_csv()
    step.save_csv()
    plot.plot_average()

if __name__ == '__main__':
    main()