import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from environments import configurations as cfg
from cares_reinforcement_learning.util import configurations as cares_cfg

def parse_args():
    param_node = __declare_params()

    env_params, rest_env = __get_env_params(param_node)
    algorithm_params, rest_alg = __get_algorithm_params(param_node)
    network_params, rest_params = __get_network_params(param_node)

    rest = {**rest_env, **rest_alg, **rest_params}

    return env_params, algorithm_params, network_params, rest


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
            ('max_steps', 1500),
            ('step_length', 0.1),
            ('reward_range', 3),
            ('collision_range', 0.2),
            ('observation_mode', 'lidar_only'),
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
            ('number_steps_per_evaluation', 10000),
            ('number_eval_episodes', 5),

            # Network Parameters -------------------------------
            ('actor_path', 'Models/SAC_actor.pht'),
            ('critic_path', 'Models/SAC_critic.pht'),
            ('algorithm', 'SAC'),
            ('gamma', 0.95),
            ('tau', 0.005),
            ('actor_lr', 1e-4),
            ('critic_lr', 1e-3),
            ('is_1d', True),
            ('latent_size',10)

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
    
    match params_dict['environment']:
        case 'CarGoal':
            config = cfg.CarGoalEnvironmentConfig(**params_dict)
        case 'CarBlock':
            config = cfg.CarBlockEnvironmentConfig(**params_dict)
        case 'CarWall':
            config = cfg.CarWallEnvironmentConfig(**params_dict)
        case 'CarTrack':
            config = cfg.CarTrackEnvironmentConfig(**params_dict)
        case 'CarBeat':
            config = cfg.CarBeatEnvironmentConfig(**params_dict)
        case _:
            raise Exception(f'Environment {params_dict["environment"]} not implemented')
    
    # Collect all the parameters that were not used into a python dictionary
    rest = set(params_dict.keys()).difference(set(config.dict().keys()))
    rest = {key: params_dict[key] for key in rest}
    
    param_node.get_logger().info(f'Rest: {rest}')
    return config, rest  

def __get_algorithm_params(param_node: Node):
    params = param_node.get_parameters([
        'g',
        'batch_size',
        'buffer_size',
        'seed',
        'max_steps_training',
        'max_steps_exploration',
        'max_steps_per_batch',
        'number_steps_per_evaluation',
        'number_eval_episodes'
    ])

    # Convert to Dictionary
    params_dict = {}
    for param in params:
        params_dict[param.name] = param.value
    
    config = cfg.TrainingConfig(**params_dict)

    rest = set(params_dict.keys()).difference(set(config.dict().keys()))
    rest = {key: params_dict[key] for key in rest}

    param_node.get_logger().info(f'Rest: {rest}')
    return config, rest

def __get_network_params(param_node: Node):
    params = param_node.get_parameters([
        'actor_path',
        'critic_path',
        'algorithm',
        'gamma',
        'tau',
        'actor_lr',
        'critic_lr',
        'is_1d',
        'latent_size'
    ])

    # Convert to Dictionary
    params_dict = {}
    for param in params:
        params_dict[param.name] = param.value
    
    match params_dict['algorithm']:
        case 'PPO':
            config = cares_cfg.PPOConfig(**params_dict)
        case 'DDPG':
            config = cares_cfg.DDPGConfig(**params_dict)
        case 'SAC':
            config = cares_cfg.SACConfig(**params_dict)
        case 'TD3':
            config = cares_cfg.TD3Config(**params_dict)
        case 'TD3AE':
            config = cares_cfg.TD3AEConfig(**params_dict)
        case 'SACAE':
            config = cares_cfg.SACAEConfig(**params_dict)
        case _:
            config = {'algorithm': 'traditional'}
    try:
        rest = set(params_dict.keys()).difference(set(config.dict().keys()))
    except:
        rest = set(params_dict.keys()).difference(set(config.keys()))
    param_node.get_logger().info(f'Rest: {rest}')
    rest = {key: params_dict[key] for key in rest}

    return config, rest

