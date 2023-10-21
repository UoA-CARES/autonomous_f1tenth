import random
import time
import yaml

import numpy as np
import rclpy
from rclpy.parameter import Parameter
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.Record import Record
from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
from cares_reinforcement_learning.util import helpers as hlp

from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment

from .parse_args import parse_args

def main():
    rclpy.init()

    env_config, algorithm_config, network_config = parse_args()

    # Set Seeds
    torch.manual_seed(algorithm_config['seed'])    
    torch.cuda.manual_seed_all(algorithm_config['seed'])
    np.random.seed(algorithm_config['seed'])
    random.seed(algorithm_config['seed'])

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(
        f'Environment Config: ------------------------------------- \n'
        f'{yaml.dump(env_config, default_flow_style=False)} \n'
        f'Algorithm Config: ------------------------------------- \n'
        f'{yaml.dump(algorithm_config, default_flow_style=False)} \n'
        f'Network Config: ------------------------------------- \n'
        f'{yaml.dump(network_config, default_flow_style=False)} \n'
    )

