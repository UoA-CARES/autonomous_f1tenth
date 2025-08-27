from pydantic import BaseModel
from typing import Optional

class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)

class TrainingConfig(SubscriptableClass):
    g: int
    batch_size: int
    buffer_size: int
    seed: int
    max_steps_training: int
    max_steps_exploration: int
    max_steps_per_batch: Optional[int]
    number_steps_per_evaluation: int
    number_eval_episodes: int

class EnvironmentConfig(SubscriptableClass):
    environment: str
    car_name: str
    reward_range: float
    max_steps: int
    step_length: float
    collision_range: Optional[float] # Doesn't apply to CarGoal
    is_staged_training: bool # Only applies to CarTrack for now

class CarGoalEnvironmentConfig(EnvironmentConfig):
    pass

class CarBlockEnvironmentConfig(EnvironmentConfig):
    pass

class CarWallEnvironmentConfig(EnvironmentConfig):
    pass

class CarTrackEnvironmentConfig(EnvironmentConfig):
    track: str
    observation_mode: str
    max_goals: int

class CarRaceEnvironmentConfig(EnvironmentConfig):
    track: str
    observation_mode: str
    max_goals: int

class CarOvertakeEnvironmentConfig(EnvironmentConfig):
    track: str
    observation_mode: str
    max_goals: int

class TwoCarEnvironmentConfig(EnvironmentConfig):
    track: str
    observation_mode: str
    max_goals: int

class CarBeatEnvironmentConfig(EnvironmentConfig):
    ftg_car_name: str
    track: str
    observation_mode: str
    max_goals: int
    num_lidar_points: int