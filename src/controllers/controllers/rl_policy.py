from .controller import Controller
import rclpy
import torch
import numpy as np
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from reinforcement_learning.parse_args import parse_args

def main():
    rclpy.init()

    env_config, _, network_config, rest = parse_args()
    
    # speed and turn limit
    MAX_ACTIONS = np.asarray([2, 0.45])
    MIN_ACTIONS = np.asarray([0, -0.45])

    controller = Controller('rl_policy_', env_config['car_name'], step_length=0.1)
    policy_id = 'rl'

    OBSERVATION_SIZE=12
    ACTION_NUM=2

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
    state = state[6:]

    while True:
        action = agent.select_action_from_policy(state)
     
        action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS) 
        state = controller.step(action, policy_id)
        state = state[6:]
