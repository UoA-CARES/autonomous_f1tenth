from environments.CarTrackParentEnvironment import CarTrackParentEnvironment


class CarTrackOriginalEnvironment(CarTrackParentEnvironment):
    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__(car_name, reward_range, max_steps, collision_range, step_length)

        """
        track_original.sdf
        """
        self.all_goals = [
            [-16.5, -5.0],
            [-13.0, -7.0],
            [-10.0, -5.0],
            [-5.0, -6.0],
            [0.0, -9.0],  # Right
            [3.5, -13.0],
            [6.5, -10.0],
            [8.0, -7.0],
            [9.0, -3.0],
            [10.0, 0.0],  # Top
            [11.0, 4.0],
            [8.5, 10.0],
            [5.0, 13.0],
            [1.5, 10.0],
            [-1.5, 5.5],  # Left
            [-3.5, 6.5],
            [-6.0, 12.5],
            [-8.0, 16.0],
            [-11.5, 13.0],
            [-15.0, 0.0]  # Bottom
        ]
