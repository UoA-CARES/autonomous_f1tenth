import random
from datetime import datetime
import yaml

import rclpy
import torch

from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
import cares_reinforcement_learning.util.configurations as cfg

from .parse_args import parse_args
from .EnvironmentFactory import EnvironmentFactory
from .training_loops import off_policy_evaluate, ppo_evaluate


def main():
    rclpy.init()

    env_config, algorithm_config, network_config = parse_args()

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

    # Load models if both paths are provided
    if network_config['actor_path'] and network_config['critic_path']:
        print('Reading saved models into actor and critic')
        agent.actor_net.load_state_dict(torch.load(network_config['actor_path']))
        agent.critic_net.load_state_dict(torch.load(network_config['critic_path']))
        print('Successfully Loaded models')
    else:
        raise Exception('Both actor and critic paths must be provided')

    match network_config['algorithm']:
        case 'PPO':
            ppo_evaluate(env, agent, algorithm_config)
        case _:
            off_policy_evaluate(env, agent, algorithm_config)

if __name__ == '__main__':
    main()
