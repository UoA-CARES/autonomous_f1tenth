from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarRaceEnvironment import CarRaceEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment
from environments.CarOvertakeEnvironment import CarOvertakeEnvironment
from environments.TwoCarEnvironment import TwoCarEnvironment

class EnvironmentFactory:
    def __init__(self):
        pass

    def create(self, name, config):
        print(config)
        if name == 'CarBlock':
            return CarBlockEnvironment(
                config['car_name'],
                config['reward_range'],
                config['max_steps'],
                config['collision_range'],
                config['step_length']
            )
        elif name == 'CarGoal':
            return CarGoalEnvironment(
                config['car_name'],
                config['reward_range'],
                config['max_steps'],
                config['step_length']
            )
        elif name == 'CarTrack':
            return CarTrackEnvironment(
                config['car_name'], 
                config['reward_range'], 
                config['max_steps'], 
                config['collision_range'], 
                config['step_length'], 
                config['track'], 
                config['observation_mode'], 
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
        elif name == 'CarWall':
            return CarWallEnvironment(
                config['car_name'],
                config['reward_range'],
                config['max_steps'],
                config['collision_range'],
                config['step_length']
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
        else:
            raise Exception('EnvironmentFactory: Environment not found')
