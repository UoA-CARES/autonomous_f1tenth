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
from .training_loops import off_policy_train

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

    match agent.type:
        case 'policy':
            off_policy_train(env, agent, memory, record, algorithm_config)
        case 'ppo':
            raise Exception('PPO Training not implemented')
        case _:
            raise Exception(f'Agent type {agent.type} not supported')



