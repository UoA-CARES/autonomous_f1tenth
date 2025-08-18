import random
from datetime import datetime
import yaml

import rclpy
import torch

from cares_reinforcement_learning.util.network_factory import NetworkFactory
import cares_reinforcement_learning.util.configurations as cfg

from .parse_args import parse_args
from .EnvironmentFactory import EnvironmentFactory
from .training_loops import off_policy_evaluate, ppo_evaluate

import os

def main():
    rclpy.init()

    env_config, algorithm_config, network_config, rest = parse_args()

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

    env = env_factory.create(env_config['environment'], env_config)
    agent = network_factory.create_network(env.OBSERVATION_SIZE, env.ACTION_NUM, config=network_config)

    # Load models if both paths are provided
    if rest['actor_path'] and rest['critic_path']:
        print('Reading saved models into actor and critic')
        if torch.cuda.is_available():
            agent.actor_net.load_state_dict(torch.load(rest['actor_path']))
            agent.critic_net.load_state_dict(torch.load(rest['critic_path']))
        else:
            agent.actor_net.load_state_dict(torch.load(rest['actor_path'], map_location=torch.device('cpu')))
            agent.critic_net.load_state_dict(torch.load(rest['critic_path'], map_location=torch.device('cpu')))
        print('Successfully Loaded models')
    else:
        raise Exception('Both actor and critic paths must be provided')

    match network_config['algorithm']:
        case 'PPO':
            ppo_evaluate(env, agent, algorithm_config)
        case _:
            off_policy_evaluate(env, agent, algorithm_config['number_eval_episodes'])

if __name__ == '__main__':
    main()
