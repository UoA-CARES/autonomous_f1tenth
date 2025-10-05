from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarRaceEnvironment import CarRaceEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment
from environments.CarOvertakeEnvironment import CarOvertakeEnvironment
from environments.TwoCarEnvironment import TwoCarEnvironment
from environments.MultiAgentEnvironment import MultiAgentEnvironment

class EnvironmentFactory:
    def __init__(self):
        pass

    def create(self, name, config):
        print(config)
        if name == 'CarTrack':
            return CarTrackEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
                config['is_staged_training']
                )
        elif name == 'CarRace':
            return CarRaceEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
                )
        elif name == 'CarOvertake':
            return CarOvertakeEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
            )
        elif name == 'TwoCar':
            return TwoCarEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
            )
        elif name == 'CarBeat':
            return CarBeatEnvironment(
                config['car_name'],
                config['ftg_car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
                config['max_goals'],
                config['num_lidar_points']
            )
        elif name == 'MultiAgent':
            return MultiAgentEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
            )
        elif name == 'MultiAgent2':
            return MultiAgentEnvironment(
                'f2tenth', 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
            )
        else:
            raise Exception('EnvironmentFactory: Environment not found')
