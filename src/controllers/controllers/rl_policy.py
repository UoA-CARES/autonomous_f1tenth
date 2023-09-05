from .rl_controller import RLController
import rclpy
from cares_reinforcement_learning.util.NetworkFactory import NetworkFactory
from cares_reinforcement_learning.util.helpers import denormalize

def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_one'),
            ('algorithm', 'TD3'),
            ('observation_mode', 'lidar_only'),
            ('actor_file_path', ''),
            ('critic_file_path', ''),
        ]
    )

    params = param_node.get_parameters(['car_name', 'algorithm', 'observation_mode', 'actor_file_path', 'critic_file_path'])
    CAR_NAME, ALGORITHM, OBSERVATION_MODE, ACTOR_FILE_PATH, CRITIC_FILE_PATH = [param.value for param in params]

    MAX_ACTIONS = [3, 3.14]
    MIN_ACTIONS = [0, -3.14]

    controller = RLController('ftg_policy_', CAR_NAME, 0.25, observation_mode=OBSERVATION_MODE)

    policy = NetworkFactory.create_network(
        algorithm=ALGORITHM,
        observation_size=controller.OBSERVATION_SIZE,
        action_num=controller.ACTION_NUM,
        actor_file_path=ACTOR_FILE_PATH,
        critic_file_path=CRITIC_FILE_PATH,
    )

    state = controller.step([0, 0])

    while True:
        action = policy.select_action(state) 
        action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS) 
        state = controller.step(action)