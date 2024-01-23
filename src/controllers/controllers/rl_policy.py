from .controller import Controller
import rclpy
import torch
import time
import numpy as np
from cares_reinforcement_learning.util.helpers import denormalize
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.TD3 import Actor, Critic
from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
from reinforcement_learning.parse_args import parse_args

def main():
    rclpy.init()

    env_config, algorithm_config, network_config, rest = parse_args()
    
    #param_node = rclpy.create_node('params')
    
    #param_node.declare_parameters(
    #    '',
    #    [
    #        ('car_name', 'f1tenth_one'),
    #        ('algorithm', 'TD3'),
    #        ('observation_mode', 'lidar_only'),
    #        ('actor_path', 'src/models/actor_checkpoint.pht'),
    #        ('critic_path', 'src/models/critic_checkpoint.pht'),
    #    ]
    #)

    #params = param_node.get_parameters(['car_name', 'algorithm', 'observation_mode', 'actor_file_path', 'critic_file_path'])
    #CAR_NAME, ALGORITHM, OBSERVATION_MODE, ACTOR_FILE_PATH, CRITIC_FILE_PATH = [param.value for param in params]

    MAX_ACTIONS = np.asarray([0.5, 0.85])
    MIN_ACTIONS = np.asarray([0, -0.85])

    controller = Controller('rl_policy_', env_config['car_name'], 0.25)
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
    

    #DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #time.sleep(3)
    
    #actor=Actor(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)
    #critic=Critic(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)
    
    #actor.load_state_dict(torch.load(ACTOR_FILE_PATH,map_location='cpu'))
    #critic.load_state_dict(torch.load(CRITIC_FILE_PATH,map_location='cpu'))
    
    #agent = TD3(
    #	actor_network=actor,
    #	critic_network=critic,
    #	gamma=0.999,
    #	tau=0.002,
    #	action_num=ACTION_NUM,
    #	device=DEVICE
    #	)

    state = controller.step([0, 0], policy_id)
    state = state[6:]

    while True:
        action = agent.select_action_from_policy(state) 
        action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS) 
        state = controller.step(action, policy_id)
        state = state[6:]
