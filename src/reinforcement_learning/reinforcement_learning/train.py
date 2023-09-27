import random
import time

import numpy as np
import rclpy
import torch
from cares_reinforcement_learning.algorithm.policy import TD3, PPO, SAC
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
    global EVALUATE_EVERY_N_STEPS
    global EVALUATE_FOR_M_EPISODES
    global ALGORITHM

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
    OBSERVATION_MODE, \
    EVALUATE_EVERY_N_STEPS, \
    EVALUATE_FOR_M_EPISODES = [param.value for param in params]

    if ACTOR_PATH != '' and CRITIC_PATH != '':
        MAX_STEPS_EXPLORATION = 0

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
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
        f'Observation Mode: {OBSERVATION_MODE}\n'
        f'Evaluate Every N Steps: {EVALUATE_EVERY_N_STEPS}\n'
        f'Evaluate For M Episodes: {EVALUATE_FOR_M_EPISODES}\n'
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
            env = CarTrackEnvironment('f1tenth', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE, track=TRACK, observation_mode=OBSERVATION_MODE)
        case 'CarBeat':
            env = CarBeatEnvironment('f1tenth_one', 'f1tenth_two', step_length=STEP_LENGTH, max_steps=MAX_STEPS, reward_range=REWARD_RANGE, collision_range=COLLISION_RANGE, track=TRACK, observation_mode=OBSERVATION_MODE)
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
        agent.actor_net.load_state_dict(torch.load(ACTOR_PATH))
        agent.critic_net.load_state_dict(torch.load(CRITIC_PATH))
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
        'observation_mode': OBSERVATION_MODE,
        'evaluate_every_n_steps': EVALUATE_EVERY_N_STEPS,
        'evaluate_for_m_episodes': EVALUATE_FOR_M_EPISODES
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
    evaluation_reward = None
    
    state, _ = env.reset()

    evaluate = False
    
    print(f'Initial State: {state}')
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

        

        if total_step_counter % EVALUATE_EVERY_N_STEPS == 0:
            evaluate = True

        
        record.log(
            out=done or truncated,
            Step=total_step_counter,
            Episode=episode_num,
            Step_Reward=reward,
            Episode_Reward=episode_reward if (done or truncated) else None,
            Evaluation_Reward=evaluation_reward
        )

        evaluation_reward = None

        if done or truncated:
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)
            
            if evaluate:
                evaluation_reward = evaluate_policy(env, agent, EVALUATE_FOR_M_EPISODES)
                evaluate = False
            
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
    evaluation_reward = None
    evaluate = False
    
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
            for _ in range(G):   
                experience = memory.sample(BATCH_SIZE) 
                info = agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                    experience['log_prob']
                ))
            
            memory.clear()

        if total_step_counter % EVALUATE_EVERY_N_STEPS == 0:
            evaluate = True
        
        record.log(
            Steps = total_step_counter,
            Episode= episode_num,
            Step_reward= reward,
            Episode_reward= episode_reward if (done or truncated) else None,
            Evaluation_reward=evaluation_reward,
            out=done or truncated,
        )
        evaluation_reward = None
        time_step += 1

        if done or truncated:
            print(f'Episode: {episode_num} | Reward: {episode_reward} | Steps: {time_step}')
            
            if evaluate:
                evaluation_reward = evaluate_policy(env, agent, EVALUATE_FOR_M_EPISODES)
                evaluate = False

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def evaluate_policy(env, agent, num_episodes):

    episode_reward_history = []

    print('Beginning Evaluation----------------------------')
    
    for ep in range(num_episodes):
        state, _ = env.reset()

        episode_timesteps = 0
        episode_reward = 0

        truncated = False
        terminated = False

        while not truncated and not terminated:

            if ALGORITHM == 'PPO':
                action = agent.select_action_from_policy(state)
            else:
                action = agent.select_action_from_policy(state, evaluation=True)
                
            action = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = next_state

        print(f'Evaluation Episode {ep + 1} Completed with a Reward of {episode_reward}')
        episode_reward_history.append(episode_reward)

    avg_reward = sum(episode_reward_history) / len(episode_reward_history)

    print(f'Evaluation Completed: Avg Reward over {num_episodes} Episodes is {avg_reward} ----------------------------')

    return avg_reward

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
            ('reward_range', 0.2),
            ('collision_range', 0.2),
            ('actor_path', ''),
            ('critic_path', ''),
            ('max_steps_per_batch', 5000),
            ('observation_mode', 'no_position'),
            ('evaluate_every_n_steps', 10000),
            ('evaluate_for_m_episodes', 5)
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
        'evaluate_every_n_steps',
        'evaluate_for_m_episodes'
    ])


if __name__ == '__main__':
    main()
