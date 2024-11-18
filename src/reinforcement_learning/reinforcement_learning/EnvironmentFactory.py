from environments.CarBlockEnvironment import CarBlockEnvironment
from environments.CarGoalEnvironment import CarGoalEnvironment
from environments.CarTrackEnvironment import CarTrackEnvironment
from environments.CarWallEnvironment import CarWallEnvironment
from environments.CarBeatEnvironment import CarBeatEnvironment

class EnvironmentFactory:
    def __init__(self):
        pass

    def create(self, name, config):
        print(config)
        if name == 'CarBlock':
            return CarBlockEnvironment(config)
        elif name == 'CarGoal':
            return CarGoalEnvironment(config)
        elif name == 'CarTrack':
            return CarTrackEnvironment(config)
        elif name == 'CarWall':
            return CarWallEnvironment(config)
        elif name == 'CarBeat':
            return CarBeatEnvironment(config)
        else:
            raise Exception('EnvironmentFactory: Environment not found')