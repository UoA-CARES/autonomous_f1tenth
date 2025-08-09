from .controller import Controller
import rclpy
import torch
import numpy as np
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from reinforcement_learning.parse_args import parse_args
import yaml
from pathlib import Path
import os
from datetime import datetime
import time

def main():
    rclpy.init()
    
    # Load configuration from YAML file
    with open('/home/anyone/new_repo/autonomous_f1tenth/src/controllers/config/car.yaml', 'r') as file:
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
    
    path = os.path.join(Path(__file__).parent.parent.parent.parent, "recordings", "network_outputs")
    if not os.path.exists(path):
         os.mkdir(path)
    filepath = os.path.join(path, f"network_output_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt")
    with open(filepath, 'w') as f:
        f.write("time,speed,steering\n")
        
    state = controller.step([0, 0], policy_id)
    # Full state is [coordinate_x, coordinate_y, orientation_x, orientation_y, orientation_z, orientation_w, linear_velocity, angular_velocity, 10 x lidar_data]
    # 'lidar_only' only considers linear_velocity and angular_velocity, thus the first 6 elements are omitted
    state = state[6:]

    while True:
        action = agent.select_action_from_policy(state)
        action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS) 
        timestamp = time.time()
        with open(filepath, 'a') as f:
                f.write(f"{timestamp},{action[0]:.4f},{action[1]:.4f}\n")
        state = controller.step(action, policy_id)
        state = state[6:]
