import random
from datetime import datetime
import yaml

import numpy as np
import rclpy
from rclpy.parameter import Parameter
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.Record import Record
from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
from cares_reinforcement_learning.util import helpers as hlp
import cares_reinforcement_learning.util.configurations as cfg

from .parse_args import parse_args
from .EnvironmentFactory import EnvironmentFactory

def main():
    rclpy.init()

    env_config, algorithm_config, network_config = parse_args()

    # Set Seeds
    torch.manual_seed(algorithm_config['seed'])    
    torch.cuda.manual_seed_all(algorithm_config['seed'])
    np.random.seed(algorithm_config['seed'])
    random.seed(algorithm_config['seed'])

    print(
        f'Environment Config: ------------------------------------- \n'
        f'{yaml.dump(env_config, default_flow_style=False)} \n'
        f'Algorithm Config: ------------------------------------- \n'
        f'{yaml.dump(algorithm_config, default_flow_style=False)} \n'
        f'Network Config: ------------------------------------- \n'
        f'{yaml.dump(network_config, default_flow_style=False)} \n'
    )

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()

    match network_config['algorithm']:
        case 'PPO':
            config = cfg.PPOConfig(**network_config)
        case 'DDPG':
            config = cfg.DDPGConfig(**network_config)
        case 'SAC':
            config = cfg.SACConfig(**network_config)
        case 'TD3':
            config = cfg.TD3Config(**network_config)
        case _:
            raise Exception(f'Algorithm {network_config["algorithm"]} not implemented')


    env = env_factory.create(env_config['environment'], env_config)
    agent = network_factory.create_network(env.OBSERVATION_SIZE, env.ACTION_NUM, config=config)
    memory = MemoryBuffer(algorithm_config['buffer_size'], env.OBSERVATION_SIZE, env.ACTION_NUM)


    record = Record(
        glob_log_dir='training_logs',
        log_dir= f"{network_config['algorithm']}-{env_config['environment']}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}",
        algorithm=network_config['algorithm'],
        task=env_config['environment'],
        network=agent
    )

    # TODO: Load Actor and Critic if passed. Only load if both are passed
    off_policy_train(env, agent, memory, record, algorithm_config)


def off_policy_train(env, agent, memory, record, algorithm_config):
    
    max_steps_training = algorithm_config['max_steps_training']
    max_steps_exploration = algorithm_config['max_steps_exploration']
    num_eps_evaluation = algorithm_config['evaluate_for_m_episodes']
    evaluate_every_n_steps = algorithm_config['evaluate_every_n_steps']

    batch_size = algorithm_config['batch_size']
    G = algorithm_config['g']

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    evaluate = False

    state, _ = env.reset()

    for step_counter in range(max_steps_training):
        episode_timesteps += 1

        if step_counter < max_steps_exploration:
            env.get_logger().info(f'Exploration Step #{step_counter} out of {max_steps_exploration}')
            action_env = np.random.uniform(env.MIN_ACTIONS, env.MAX_ACTIONS, env.ACTION_NUM)
            action = hlp.normalize(action_env, env.MAX_ACTIONS, env.MIN_ACTIONS)
        else:
            action = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)
        
        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if step_counter >= max_steps_exploration:
            for i in range(G):
                experience = memory.sample(batch_size)
                info = agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                ))
                memory.update_priorities(experience['indices'], info)
        
        if step_counter % evaluate_every_n_steps == 0:
            evaluate = True
        
        if done or truncated:
            env.get_logger().info(f'Episode #{episode_num} completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}')

            record.log_train(
                total_steps = step_counter + 1,
                episode = episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward = episode_reward,
                display = True
            )

            if evaluate:
                evaluate = False

                env.get_logger().info(f'*************--Begin Evaluation Loop--*************')
                off_policy_evaluate(env, agent, num_eps_evaluation, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')


            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def off_policy_evaluate(env, agent, eval_episodes, record, steps_counter):

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for episode_num in range(eval_episodes):
        state, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

            next_state, reward, done, truncated, _ = env.step(action_env)

            state = next_state
            episode_reward += reward

            if done or truncated:
                record.log_eval(
                    total_steps = steps_counter + 1,
                    episode = episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward = episode_reward,
                    display = True
                )

                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                break
