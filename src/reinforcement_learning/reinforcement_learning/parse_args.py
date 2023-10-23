import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

def parse_args():
    param_node = __declare_params()

    env_params = __get_env_params(param_node)
    algorithm_params = __get_algorithm_params(param_node)
    network_params = __get_network_params(param_node)

    return env_params, algorithm_params, network_params

def __declare_params():
    param_node = rclpy.create_node('params')
    param_node.declare_parameters(
        '',
        [
            # Environment Parameters ---------------------------
            ('environment', 'CarGoal'),
            ('car_name', 'f1tenth'),
            ('ftg_car_name', 'ftg_car'),
            ('track', 'multi_track'),
            ('max_steps', 100),
            ('step_length', 0.25),
            ('reward_range', 0.2),
            ('collision_range', 0.2),
            ('observation_mode', 'no_position'),
            ('max_goals', 500),
            ('num_lidar_points', 10),

            # Algorithm Parameters -----------------------------
            ('g', 10),
            ('batch_size', 32),
            ('buffer_size', 1_000_000),
            ('seed', 123),
            ('max_steps_training', 1_000_000),
            ('max_steps_exploration', 1_000),
            ('max_steps_per_batch', 5000),
            ('evaluate_every_n_steps', 10000),
            ('evaluate_for_m_episodes', 5),

            # Network Parameters -------------------------------
            ('actor_path', ''),
            ('critic_path', ''),
            ('algorithm', 'TD3'),
            ('gamma', 0.95),
            ('tau', 0.005),
            ('actor_lr', 1e-4),
            ('critic_lr', 1e-3),

        ]
    )

    return param_node

def __get_env_params(param_node: Node):
    
    params: list(Parameter) = param_node.get_parameters([
        'environment',
        'car_name',
        'track',
        'max_steps',
        'step_length',
        'reward_range',
        'collision_range',
        'observation_mode',
        'max_goals',
        'ftg_car_name',
        'num_lidar_points'
    ])

    # Convert to Dictionary
    params_dict = {}
    for param in params:
        params_dict[param.name] = param.value
    
    return params_dict

def __get_algorithm_params(param_node: Node):
    params = param_node.get_parameters([
        'g',
        'batch_size',
        'buffer_size',
        'seed',
        'max_steps_training',
        'max_steps_exploration',
        'max_steps_per_batch',
        'evaluate_every_n_steps',
        'evaluate_for_m_episodes'
    ])

    # Convert to Dictionary
    params_dict = {}
    for param in params:
        params_dict[param.name] = param.value
    
    return params_dict

def __get_network_params(param_node: Node):
    params = param_node.get_parameters([
        'actor_path',
        'critic_path',
        'algorithm',
        'gamma',
        'tau',
        'actor_lr',
        'critic_lr'
    ])

    # Convert to Dictionary
    params_dict = {}
    for param in params:
        params_dict[param.name] = param.value
    
    return params_dict
