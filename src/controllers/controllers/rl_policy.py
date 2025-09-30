from .controller import Controller
import rclpy
import torch
import numpy as np
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from reinforcement_learning.parse_args import parse_args
import yaml
import os

def main():
    rclpy.init()
    ament_path = os.environ["AMENT_PREFIX_PATH"].split(":")[0]
    workspace_dir = os.path.dirname(ament_path)
    car_path = os.path.join(workspace_dir, "..", "src", "controllers", "config")
    config_path = os.path.join(workspace_dir, "..", "src", "environments", "config")
    
    # Load configuration from YAML file
    with open(f'{car_path}/car.yaml', 'r') as file:
        config = yaml.safe_load(file)

    env_config, _, network_config, rest = parse_args()
    MAX_ACTIONS = np.asarray([config['car']['ros__parameters']['max_speed'], config['car']['ros__parameters']['max_turn']])
    MIN_ACTIONS = np.asarray([config['car']['ros__parameters']['min_speed'], config['car']['ros__parameters']['min_turn']])
    OBSERVATION_SIZE=12
    ACTION_NUM=2

    controller = Controller('rl_policy_', env_config['car_name'], step_length=0.1)
    policy_id = 'rl'

    network_factory = NetworkFactory()
    agent = network_factory.create_network(OBSERVATION_SIZE, ACTION_NUM, config=network_config)

    # Load models if both paths are provided
    if rest['actor_path'] and rest['critic_path']:
        print('Reading saved models into actor and critic')
        agent.actor_net.load_state_dict(torch.load(rest['actor_path'], map_location=torch.device('cpu')))
        agent.critic_net.load_state_dict(torch.load(rest['critic_path'], map_location=torch.device('cpu')))
        print('Successfully Loaded models')
    else:
        raise Exception('Both actor and critic paths must be provided')
    
        
    state = controller.step([0, 0], policy_id)
    # Full state is [coordinate_x, coordinate_y, orientation_x, orientation_y, orientation_z, orientation_w, linear_velocity, angular_velocity, 10 x lidar_data]
    # 'lidar_only' only considers linear_velocity and angular_velocity, thus the first 6 elements are omitted
    state = state[6:]

    with open(f'{config_path}/config.yaml', 'r') as file:
        actions_config = yaml.safe_load(file)
    MAX_CONFIG_ACTIONS = np.asarray([actions_config['actions']['max_speed'], actions_config['actions']['max_turn']])
    MIN_CONFIG_ACTIONS = np.asarray([actions_config['actions']['min_speed'], actions_config['actions']['min_turn']])
    
    while True:
        action = agent.select_action_from_policy(state)
        action = denormalize(action, MAX_CONFIG_ACTIONS, MIN_CONFIG_ACTIONS)
        action = np.clip(action, MIN_ACTIONS, MAX_ACTIONS)
        state = controller.step(action, policy_id)
        state = state[6:]
