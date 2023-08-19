import random
import time

import numpy as np
import rclpy
import torch
from cares_reinforcement_learning.algorithm.policy import TD3, PPO
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.TD3 import Actor, Critic
from cares_reinforcement_learning.util import Record, NetworkFactory
from cares_reinforcement_learning.util import helpers as hlp

from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment

def main():
    rclpy.init()

    params = get_params()

    global MAX_STEPS_TRAINING
    global MAX_STEPS_EXPLORATION
    global MAX_STEPS_PER_BATCH
    global G
    global BATCH_SIZE

    ENVIRONMENT, \
    ALGORITHM, \
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
    COLLISION_RANGE, \
    ACTOR_PATH, \
    CRITIC_PATH, \
    MAX_STEPS_PER_BATCH, \
    OBSERVATION_MODE = [param.value for param in params]

    if ACTOR_PATH != '' and CRITIC_PATH != '':
        MAX_STEPS_EXPLORATION = 0

    print(
        f'---------------------------------------------\n'
        f'Environment: {ENVIRONMENT}\n'
        f'Algorithm: {ALGORITHM}\n'
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
        f'Critic Path: {CRITIC_PATH}\n'
        f'Actor Path: {ACTOR_PATH}\n'
        f'Max Steps per Batch: {MAX_STEPS_PER_BATCH}\n'
        f'Observation Mode: {OBSERVATION_MODE}'
        f'---------------------------------------------\n'
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time.sleep(3)

    match ENVIRONMENT:
        case 'CarWall':
            env = CarWallEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarBlock':
            env = CarBlockEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE)
        case 'CarTrack':
            env = CarTrackEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE, track=TRACK, observation_mode= OBSERVATION_MODE)
        case 'CarBeat':
            env = CarBeatEnvironment('f1tenth_one', 'f1tenth_two', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE, track=TRACK)
        case _:
            env = CarGoalEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE)

    network_factory_args = {
        'observation_size': env.OBSERVATION_SIZE,
        'action_num': env.ACTION_NUM,
        'actor_lr': ACTOR_LR,
        'critic_lr': CRITIC_LR,
        'gamma': GAMMA,
        'tau': TAU,
        'device': DEVICE
    }

    agent_factory = NetworkFactory()
    agent = agent_factory.create_network(ALGORITHM, network_factory_args)
    
    if ACTOR_PATH != '' and CRITIC_PATH != '':
        print('Reading saved models into actor and critic')
        agent.actor.load_state_dict(torch.load(ACTOR_PATH))
        agent.critic.load_state_dict(torch.load(CRITIC_PATH))
        print('Successfully Loaded models')

    networks = {'actor': agent.actor_net, 'critic': agent.critic_net}
    config = {
        'environment': ENVIRONMENT,
        'algorithm': ALGORITHM,
        'max_steps_training': MAX_STEPS_TRAINING,
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
        'reward_range': REWARD_RANGE,
        'collision_range': COLLISION_RANGE,
        'max_steps_per_batch': MAX_STEPS_PER_BATCH,
        'observation_mode': OBSERVATION_MODE
    }

    if (ENVIRONMENT == 'CarTrack'):
        config['track'] = TRACK

    record = Record(networks=networks, checkpoint_freq=100, config=config)

    if ALGORITHM == 'PPO':
        train_ppo(env, agent, record=record)
    else:
        train(env=env, agent=agent, record=record)


def train(env, agent, record: Record):
    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset()

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(MAX_STEPS_TRAINING)):
        episode_timesteps += 1

        if total_step_counter < MAX_STEPS_EXPLORATION:
            print(f"Running Exploration Steps {total_step_counter}/{MAX_STEPS_EXPLORATION}")
            action_env = np.asarray([random.uniform(env.MIN_ACTIONS[0], env.MAX_ACTIONS[0]), random.uniform(env.MIN_ACTIONS[1], env.MAX_ACTIONS[1])])  # action range the env uses [e.g. -2 , 2 for pendulum]
            action = hlp.normalize(action_env, env.MAX_ACTIONS, env.MIN_ACTIONS)  # algorithm range [-1, 1]
        else:
            action = agent.select_action_from_policy(state)  # algorithm range [-1, 1]
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
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
def train_ppo(env, agent, record):
    max_steps_training = MAX_STEPS_TRAINING
    max_steps_per_batch = MAX_STEPS_PER_BATCH

    min_action_value = env.MIN_ACTIONS 
    max_action_value = env.MAX_ACTIONS

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    time_step = 1

    memory = MemoryBuffer()

    state, _ = env.reset()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward, done, truncated, _ = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob)

        state = next_state
        episode_reward += reward

        if time_step % max_steps_per_batch == 0:
            experience = memory.flush()
            
            for _ in range(G):    
                info = agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                    experience['log_prob']
                ))

            record.log(
                Train_steps = total_step_counter + 1,
                Train_episode= episode_num + 1,
                Train_timesteps=episode_timesteps,
                Train_reward= episode_reward,
                Actor_loss = info['actor_loss'].item(),
                Critic_loss = info['critic_loss'].item(),
                out=done or truncated
            )

        time_step += 1

        if done or truncated:
            print(f'Episode: {episode_num} | Reward: {episode_reward} | Steps: {time_step}')
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
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
            ('environment', 'CarGoal'),
            ('algorithm', 'TD3'),
            ('track', 'track_1'),
            ('gamma', 0.95),
            ('tau', 0.005),
            ('g', 10),
            ('batch_size', 32),
            ('buffer_size', 1_000_000),
            ('seed', 123),  # TODO: This doesn't do anything yet
            ('actor_lr', 1e-4),
            ('critic_lr', 1e-3),
            ('max_steps_training', 1_000_000),
            ('max_steps_exploration', 1_000),
            ('max_steps', 100),
            ('step_length', 0.25),
            ('reward_range', 3.0),
            ('collision_range', 0.2),
            ('actor_path', ''),
            ('critic_path', ''),
            ('max_steps_per_batch', 5000),
            ('observation_mode', 'no_position')
        ]
    )

    return param_node.get_parameters([
        'environment',
        'algorithm',
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
        'collision_range',
        'actor_path',
        'critic_path',
        'max_steps_per_batch',
        'observation_mode',
    ])


if __name__ == '__main__':
    main()
