import random
from datetime import datetime
import yaml

import numpy as np
import rclpy
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.record import Record
from cares_reinforcement_learning.util.network_factory import NetworkFactory

from .parse_args import parse_args
from .EnvironmentFactory import EnvironmentFactory
from .training_loops import off_policy_train, ppo_train, multi_off_policy_train

def main():
    rclpy.init()

    env_config, algorithm_config, network_config, _ = parse_args()

    # Set Seeds
    # TODO replace with cares helper set_seed
    torch.manual_seed(algorithm_config['seed'])
    torch.cuda.manual_seed_all(algorithm_config['seed'])
    np.random.seed(algorithm_config['seed'])
    random.seed(algorithm_config['seed'])

    print(
        f'Environment Config: ------------------------------------- \n'
        f'{yaml.dump(dict(env_config), default_flow_style=False)} \n'
        f'Algorithm Config: ------------------------------------- \n'
        f'{yaml.dump(dict(algorithm_config), default_flow_style=False)} \n'
        f'Network Config: ------------------------------------- \n'
        f'{yaml.dump(dict(network_config), default_flow_style=False)} \n'
    )

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()

    env = env_factory.create(env_config['environment'], env_config)
    if env_config['environment'] == 'MultiAgent':
        env2_config = env_config
        env2 = env_factory.create('MultiAgent2', env2_config)
    agent = network_factory.create_network(env.OBSERVATION_SIZE, env.ACTION_NUM, config=network_config)
    memory = MemoryBuffer(algorithm_config['buffer_size'])


    record = Record(
        base_directory=f"/home/anyone/training_logs/{env_config['car_name']}-{network_config['algorithm']}-{env_config['environment']}-{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}",
        algorithm=network_config['algorithm'],
        task=env_config['environment'],
        agent=None,
    )

    record.set_sub_directory(algorithm_config['seed'])
    record.set_agent(agent)

    record.save_config(env_config, 'env_config')
    record.save_config(algorithm_config, 'algorithm_config')
    record.save_config(network_config, 'network_config')

    # TODO: Load Actor and Critic if passed. Only load if both are passed

    match agent.policy_type:
        case 'policy':
            if network_config['algorithm'] == 'PPO':
                ppo_train(env, agent, memory, record, algorithm_config)
            elif env_config['environment'] == 'MultiAgent':
                multi_off_policy_train(env, env2, agent, memory, record, algorithm_config)
            else:
                off_policy_train(env, agent, memory, record, algorithm_config)
        case _:
            raise Exception(f'Agent type {agent.type} not supported')
    
    record.save()



