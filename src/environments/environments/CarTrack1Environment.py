from environments.CarTrackGeneralEnvironment import CarTrackGeneralEnvironment


class CarTrack1Environment(CarTrackGeneralEnvironment):
    """
    track1.sdf
    """

    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__(car_name, reward_range, max_steps, collision_range, step_length)
        self.all_goals = [
            [-4.5, -5.0],
            [-3.0, -7.0],
            [0.0, -7.5],  # Right
            [3.0, -7.0],
            [6.0, -4.0],
            [7.0, 0.0],  # Top
            [4.0, 2.0],
            [2.0, 5.0],
            [0.0, 6.0],  # Left
            [-4.0, 6.5],
            [-7.0, 4.0],
            [-4.5, 0.0]  # Bottom
        ]

        self.car_reset_positions = {
            'x': -4.5,
            'y': 0.0,
            'yaw': 3.14 * 1.6
        }
