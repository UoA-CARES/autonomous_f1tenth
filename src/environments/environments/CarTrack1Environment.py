from environments.CarTrackParentEnvironment import CarTrackParentEnvironment


class CarTrackOriginalEnvironment(CarTrackParentEnvironment):
    def __init__(self, car_name, reward_range=0.2, max_steps=50, collision_range=0.2, step_length=0.5):
        super().__init__(car_name, reward_range, max_steps, collision_range, step_length)

        """
        track1.sdf
        """
        # self.all_goals = [
        #     [-4.5, -5.0],
        #     [-3.0, -7.0],
        #     [0.0, -7.5],  # Right
        #     [3.0, -7.0],
        #     [6.0, -7.0],
        #     [7.0, 0.0],  # Top
        #     [0.0, 6.0],  # Left
        #     [-4.5, 0.0]  # Bottom
        # ]
