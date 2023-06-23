import rclpy
import time
import torch
import random
import numpy as np

from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBlockEnvironment import CarBlockEnvironment

rclpy.init()

param_node = rclpy.create_node('params')
param_node.declare_parameters(
    '',
    [
        ('environment', 'CarGoal'),
        ('gamma', 0.95),
        ('tau', 0.005),
        ('g', 10),
        ('batch_size', 32),
        ('buffer_size', 1_000_000),
        ('seed', 123), #TODO: This doesn't do anything yet
        ('actor_lr', 1e-4),
        ('critic_lr', 1e-3),
        ('max_steps_training', 1_000_000),
        ('max_steps_exploration', 1_000),
        ('max_steps', 100),
        ('step_length', 0.25),
        ('reward_range', 0.2),
        ('collision_range', 0.2)
    ]
)

params = param_node.get_parameters([
    'environment',
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

ENVIRONMENT,\
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
STEP_LENGTH,\
REWARD_RANGE,\
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    time.sleep(3)

    match(ENVIRONMENT):
        case 'CarWall':
            env = CarWallEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarBlock':
            env = CarBlockEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case _:
            env = CarGoalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE)
    
    actor = Actor(observation_size=env.OBSERVATION_SIZE, num_actions=env.ACTION_NUM, learning_rate=ACTOR_LR)
    critic = Critic(observation_size=env.OBSERVATION_SIZE, num_actions=env.ACTION_NUM, learning_rate=CRITIC_LR)

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=GAMMA,
        tau=TAU,
        action_num=env.ACTION_NUM,
        device=DEVICE
    )

    networks = {'actor': actor, 'critic': critic}
    config = {
        'max_steps_training':MAX_STEPS_TRAINING,
        'max_steps_exploration': MAX_STEPS_EXPLORATION, 
        'gamma': GAMMA, 
        'tau': TAU, 
        'g': G, 
        'batch_size': BATCH_SIZE, 
        'buffer_size': BUFFER_SIZE, 
        'seed': SEED, 
        'actor_lr': ACTOR_LR, 
        'critic_lr': CRITIC_LR,
        'max_steps': MAX_STEPS,
        'step_length': STEP_LENGTH,
    }
    record = Record(networks=networks, checkpoint_freq=MAX_STEPS_TRAINING / 10, config=config)

    train(env=env, agent=agent, record=record)

def train(env, agent: TD3, record: Record):
    
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
            action_env = np.asarray([random.uniform(env.MIN_ACTIONS[0], env.MAX_ACTIONS[0]), random.uniform(env.MIN_ACTIONS[1], env.MAX_ACTIONS[1])]) # action range the env uses [e.g. -2 , 2 for pendulum]
            action = hlp.normalize(action_env, env.MAX_ACTIONS, env.MIN_ACTIONS)  # algorithm range [-1, 1]
        else:
            action = agent.select_action_from_policy(state) # algorithm range [-1, 1]
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= MAX_STEPS_EXPLORATION:
                for _ in range(G):
                    experiences = memory.sample(BATCH_SIZE)
                    experiences = (experiences['state'], experiences['action'], experiences['reward'], experiences['next_state'], experiences['done'])
                    agent.train_policy(experiences)

        record.log(
            out=done or truncated,
            Step=total_step_counter,
            Episode=episode_num,
            Step_Reward=reward,
            Episode_Reward=episode_reward if (done or truncated) else None,
        )

        if done or truncated:

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

if __name__ == '__main__':
    main()