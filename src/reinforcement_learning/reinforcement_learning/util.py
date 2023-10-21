import rclpy

def parse_args():
    params = _get_params()
    return params

def _get_params():
    '''
    This function fetches the hyperparameters passed in through the launch files
    - The hyperparameters below are defaults, to change them, you should change the train.yaml config
    '''
    param_node = rclpy.create_node('params')
    param_node.declare_parameters(
        '',
        [
            ('environment', 'CarGoal'),
            ('algorithm', 'TD3'),
            ('track', 'track_1'),
            ('gamma', 0.95),
            ('tau', 0.005),
            ('g', 10),
            ('batch_size', 32),
            ('buffer_size', 1_000_000),
            ('seed', 123),  # TODO: This doesn't do anything yet
            ('actor_lr', 1e-4),
            ('critic_lr', 1e-3),
            ('max_steps_training', 1_000_000),
            ('max_steps_exploration', 1_000),
            ('max_steps', 100),
            ('step_length', 0.25),
            ('reward_range', 0.2),
            ('collision_range', 0.2),
            ('actor_path', ''),
            ('critic_path', ''),
            ('max_steps_per_batch', 5000),
            ('observation_mode', 'no_position'),
            ('evaluate_every_n_steps', 10000),
            ('evaluate_for_m_episodes', 5)
        ]
    )

    return param_node.get_parameters([
        'environment',
        'algorithm',
        'track',
        'max_steps_training',
        'max_steps_exploration',
        'gamma',
        'tau',
        'g',
        'batch_size',
        'buffer_size',
        'seed',
        'actor_lr',
        'critic_lr',
        'max_steps',
        'step_length',
        'reward_range',
        'collision_range',
        'actor_path',
        'critic_path',
        'max_steps_per_batch',
        'observation_mode',
        'evaluate_every_n_steps',
        'evaluate_for_m_episodes'
    ])