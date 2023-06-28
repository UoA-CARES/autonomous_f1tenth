from environments.CarTrackParentEnvironment import CarTrackParentEnvironment


class CarTrack2Environment(CarTrackParentEnvironment):
    """
    track2.sdf
    """

    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__(car_name, reward_range, max_steps, collision_range, step_length)
        self.all_goals = [
            [0.0, -7.3],  # Right
            [7.4, 0.0],  # Top
            [0.0, 7.6],  # Left
            [-7.6, 0.0]  # Bottom
        ]

        self.car_reset_positions = {
            'x': -10.0,
            'y': 10.0,
            'yaw': 0.0
        }
