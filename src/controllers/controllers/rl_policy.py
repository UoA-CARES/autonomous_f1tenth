from .controller import Controller
import rclpy
import torch
import time
import numpy as np
from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
from cares_reinforcement_learning.util.helpers import denormalize
from .rl_controller import RLController
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.TD3 import Actor, Critic

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_one'),
            ('algorithm', 'TD3'),
            ('observation_mode', 'lidar_only'),
            ('actor_file_path', 'src/models/actor_checkpoint.pht'),
            ('critic_file_path', 'src/models/critic_checkpoint.pht'),
        ]
    )

    params = param_node.get_parameters(['car_name', 'algorithm', 'observation_mode', 'actor_file_path', 'critic_file_path'])
    CAR_NAME, ALGORITHM, OBSERVATION_MODE, ACTOR_FILE_PATH, CRITIC_FILE_PATH = [param.value for param in params]

    MAX_ACTIONS = np.asarray([0.5, 3.14])
    MIN_ACTIONS = np.asarray([0, -3.14])

    controller = Controller('rl_policy_', CAR_NAME, 0.25)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time.sleep(3)
    OBSERVATION_SIZE=12
    ACTION_NUM=2
    actor=Actor(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)
    critic=Critic(observation_size=OBSERVATION_SIZE, num_actions=ACTION_NUM, learning_rate=0.1)
    
    actor.load_state_dict(torch.load(ACTOR_FILE_PATH,map_location='cpu'))
    critic.load_state_dict(torch.load(CRITIC_FILE_PATH,map_location='cpu'))
    
    agent = TD3(
    	actor_network=actor,
    	critic_network=critic,
    	gamma=0.999,
    	tau=0.002,
    	action_num=ACTION_NUM,
    	device=DEVICE
    	)

    state = controller.step([0, 0])

    while True:
        action = agent.select_action_from_policy(state) 
        action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS) 
        state = controller.step(action)
        state = state[6:]
