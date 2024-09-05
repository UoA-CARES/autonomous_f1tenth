import cares_reinforcement_learning.util.configurations as cares_cfg
from environments import configurations as cfg


# Here because currently settings are passed to train and test nodes as ROS parameters,
# this is problematic as more and more implementations demand different forms of parameters.
# e.g. AE based algorithm has a whole field for AE setting which is irrelevant to other
# algorithms, putting them in will just bloat the file.
# with the current implementation these will still have to be declared for train as ROS
# parameters which does not seem intuitive.

# This file aims to parse setting files directly without being passed through ROS.


## !!!!!!!!!! THIS FILE IS VERY INCOMPLETE, not actually reading file
# TODO: actually make it read from a YAML/JSON file for settings

def parse_args_from_file():
    """INCOMPLETE: returns _, _, network config"""

    # env_params = cfg.CarTrackEnvironmentConfig(
    #     environment= "CarTrack",  #'CarTrack' # CarGoal, CarWall, CarBlock, CarTrack, CarBeat, CarTrackProgressiveGoal, CarTrack
    #     car_name = 'f1tenth',
    #     track = "multi_track_01",
    #     max_steps=1500,
    #     step_length=0.1,
    #     reward_range=3,
    #     collision_range=0.2,
    #     observation_mode='lidar_only',

    #     #NOT USED
    #     max_goals=500 
    # )

    # algorithm_params = cfg.TrainingConfig(
    #     g=1,
    #     batch_size=32,
    #     buffer_size=1000000,
    #     seed=123,
    #     max_steps_training=1000000,
    #     max_steps_exploration=1000,
    #     max_steps_per_batch=5000,
    #     number_steps_per_evaluation=10000,
    #     number_eval_episodes=10
    # )


    autoencoder_config = cares_cfg.VanillaAEConfig(
        latent_dim= 10,
        is_1d= True
    )

    network_config = cares_cfg.TD3AEConfig (
        autoencoder_config=autoencoder_config,
        info_vector_size=2,
        buffer_size=1000000,

    )
    #     env      alg     network
    return None, None, network_config


    