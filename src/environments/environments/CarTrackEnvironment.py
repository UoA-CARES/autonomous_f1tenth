from environments.F1tenthEnvironment import F1tenthEnvironment


class CarTrackEnvironment(F1tenthEnvironment):
    """
    CarTrack Reinforcement Learning Environment:

        Task:
            Agent learns to drive a track

        Observation:
            full:
                Car Position (x, y)
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            no_position:
                Car Orientation (x, y, z, w)
                Car Velocity
                Car Angular Velocity
                Lidar Data
            lidar_only:
                Car Velocity
                Car Angular Velocity
                Lidar Data

        Action:
            It's linear and angular velocity (Twist)

        Reward:


        Termination Conditions:
            When the agent collides with a wall or the Follow The Gap car

        Truncation Condition:
            Reaching max_steps
    """

    def __init__(
        self,
        car_name: str,
        reward_range: float = 3,
        max_steps: int = 3000,
        collision_range: float = 0.2,
        step_length: float = 0.1,
        track: str = "track_01",
        observation_mode: str = "lidar_only",
    ):
        super().__init__(
            env_name="car_track",
            car_name=car_name,
            reward_range=reward_range,
            max_steps=max_steps,
            collision_range=collision_range,
            step_sleep_time=step_length,
            lidar_observation_size=10,
            track=track,
            observation_mode=observation_mode,
        )

        self.get_logger().info("Environment Setup Complete")
