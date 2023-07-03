from environments.CarTrackParentEnvironment import CarTrackParentEnvironment


class CarTrack2Environment(CarTrackParentEnvironment):
    """
    track2.sdf
    """

    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__(car_name, reward_range, max_steps, collision_range, step_length)
        self.all_goals = [
            [-7.6, -3.0],
            [-7.0, -6.5],
            [-4.0, -7.2],
            [-1.0, -7.3],  # Right
            [3.0, -7.1],
            [6.0, -6.0],
            [7.2, -3.0],
            [7.4, 0.0],  # Top
            [7.3, 4.0],
            [6.0, 7.7],
            [3.6, 6.0],
            [2.8, 3.0],
            [0.5, 2.0],
            [-2.0, 0.0],
            [-3.0, -3.0],
            [-4.0, -4.6],
            [-5.3, -3.0],
            [-4.9, 0.0],
            [-3.6, 3.0],
            [-2.0, 4.0],
            [0.0, 4.6],
            [0.4, 7.0],  # Left
            [-2.0, 8.0],
            [-5.0, 8.0],
            [-7.1, 7.5],
            [-7.5, 6.0],
            [-7.6, 3.0],
            [-7.6, 0.0]  # Bottom
        ]

        self.car_reset_positions = {
            'x': -7.6,
            'y': 0.0,
            'yaw': 3.14 * 1.5
        }
