import random
from datetime import datetime
import yaml

import rclpy
import torch

#from cares_reinforcement_learning.util.network_factory import NetworkFactory
import cares_reinforcement_learning.util.configurations as cfg

from reinforcement_learning.parse_args import parse_args
from reinforcement_learning.EnvironmentFactory import EnvironmentFactory
#from .training_loops import off_policy_evaluate, ppo_evaluate


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

    env = env_factory.create(env_config['environment'], env_config)

    state, _ = env.reset()
    

if __name__ == '__main__':
    main()
