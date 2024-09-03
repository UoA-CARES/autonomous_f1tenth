import random
from datetime import datetime
import yaml

import numpy as np
import rclpy
from rclpy.parameter import Parameter
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.record import Record
from cares_reinforcement_learning.util.network_factory import NetworkFactory
import cares_reinforcement_learning.util.configurations as cares_cfg

from .parse_args import parse_args
from .EnvironmentFactory import EnvironmentFactory
from .training_loops import off_policy_train, ppo_train

def main():
    rclpy.init()

    env_config, algorithm_config, network_config, _ = parse_args()

    # Set Seeds
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

    autoencoder_config = cares_cfg.VanillaAEConfig(
        latent_dim= 10,
        is_1d= True
    )

    network_config = cares_cfg.TD3AEConfig (
        autoencoder_config=autoencoder_config,
        info_vector_size=2,
    )
    print(str(network_config))


    env = env_factory.create(env_config['environment'], env_config)
    agent = network_factory.create_network(env.OBSERVATION_SIZE, env.ACTION_NUM, config=network_config)
    memory = MemoryBuffer(algorithm_config['buffer_size'])


    record = Record(
        log_dir= f"training_logs/{network_config['algorithm']}-{env_config['environment']}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}",
        algorithm=network_config['algorithm'],
        task=env_config['environment'],
        network=agent,
    )

    record.save_config(env_config, 'env_config')
    record.save_config(algorithm_config, 'algorithm_config')
    record.save_config(network_config, 'network_config')

    # TODO: Load Actor and Critic if passed. Only load if both are passed

    match agent.type:
        case 'policy':

            if network_config['algorithm'] == 'PPO':
                ppo_train(env, agent, memory, record, algorithm_config)
            else:
                off_policy_train(env, agent, memory, record, algorithm_config)
        case _:
            raise Exception(f'Agent type {agent.type} not supported')
    
    record.save()



