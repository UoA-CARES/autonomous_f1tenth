import math
import random
import time
from typing import List, Literal, Tuple

import numpy as np
import scipy

from environments.F1tenthEnvironment import F1tenthEnvironment

from . import util


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
        car_name,
        reward_range=3,
        max_steps=3000,
        collision_range=0.2,
        step_length=0.1,
        track="track_01",
        observation_mode="lidar_only",
    ):
        super().__init__(
            env_name="car_track",
            car_name=car_name,
            reward_range=reward_range,
            max_steps=max_steps,
            collision_range=collision_range,
            step_length=step_length,
            lidar_points=10,
            track=track,
            observation_mode=observation_mode,
        )

        #####################################################################################################################
        # Reward configuration -----------------------------------------
        self.base_reward_function: Literal["goal_hitting", "progressive"] = (
            "progressive"
        )
        self.extra_reward_terms: List[Literal["penalize_turn"]] = []
        self.reward_modifiers: List[Tuple[Literal["turn", "wall_proximity"], float]] = [
            ("turn", 0.3),
            ("wall_proximity", 0.7),
        ]

        if track == "narrow_multi_track":
            self.multi_track_train_eval_split = 12 / 15
        else:
            self.multi_track_train_eval_split = 0.5

        #####################################################################################################################
        # Environment configuration -------------------------------------
        self.progress_not_met_cnt = 0
        self.steps_since_last_goal = 0

        if self.is_multi_track:
            self.eval_track_begin_idx = int(
                len(self.all_track_waypoints) * self.multi_track_train_eval_split
            )
            self.eval_track_idx = 0

        self.get_logger().info("Environment Setup Complete")

    def _reset(self, training: bool) -> tuple[np.ndarray, dict]:
        self.steps_since_last_goal = 0

        if self.is_multi_track:
            if (
                self.eval_track_begin_idx is not None
                and self.eval_track_begin_idx >= len(self.all_track_waypoints)
            ):
                if self.is_eval:
                    all_track_keys = list(self.all_track_waypoints.keys())
                    self.current_track = all_track_keys[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(all_track_keys)
                else:
                    self.current_track = random.choice(
                        list(self.all_track_waypoints.keys())
                    )
            else:
                if self.is_eval:
                    eval_track_key_list = list(self.all_track_waypoints.keys())[
                        self.eval_track_begin_idx :
                    ]
                    self.current_track = eval_track_key_list[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(eval_track_key_list)
                else:
                    self.current_track = random.choice(
                        list(self.all_track_waypoints.keys())[
                            : self.eval_track_begin_idx
                        ]
                    )
            self.curr_waypoints = self.all_track_waypoints[self.current_track]

        if self.is_eval:
            car_x, car_y, car_yaw, index = self.curr_waypoints[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.curr_waypoints)

        self.spawn_index = index
        x, y, _, _ = self.curr_waypoints[
            (
                self.spawn_index + 1
                if self.spawn_index + 1 < len(self.curr_waypoints)
                else 0
            )
        ]  # point toward next goal

        self.goal_position = [x, y]
        self.call_reset_service(
            car_x=car_x,
            car_y=car_y,
            car_yaw=car_yaw,
            goal_x=x,
            goal_y=y,
            car_name=self.name,
        )

        self.call_step(pause=False)
        state, full_state, _ = self.get_observation()
        self.current_state = full_state
        self.call_step(pause=True)

        if self.is_multi_track:
            self.curr_track_model = self.all_track_models[self.current_track]
        self.prev_closest_point = self.curr_track_model.get_closest_point_on_spline(
            full_state[:2], t_only=True
        )

        if self.base_reward_function == "progressive":
            self.progress_not_met_cnt = 0

        info = {}
        return state, info

    def _step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:

        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)

        if not self.prev_closest_point:
            self.prev_closest_point = self.curr_track_model.get_closest_point_on_spline(
                self.current_state[:2], t_only=True
            )

        t2 = self.curr_track_model.get_closest_point_on_spline(
            full_next_state[:2], t_only=True
        )
        self.step_progress = self.curr_track_model.get_distance_along_track_parametric(
            self.prev_closest_point, t2, approximate=True
        )

        self.prev_closest_point = t2

        if abs(self.step_progress) > (full_next_state[6] / 10 * 3):
            self.step_progress = full_next_state[6] / 10 * 0.8

        reward, reward_info = self.compute_reward(
            self.current_state, full_next_state, raw_lidar_range
        )
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            "linear_velocity": ["avg", full_next_state[6]],
            "angular_velocity_diff": ["avg", abs(full_next_state[7] - self.current_state[7])],
            "traveled distance": ["sum", self.step_progress],
        }
        info.update(reward_info)

        self.current_state = full_next_state

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return util.has_collided(ranges, self.collision_range) or util.has_flipped_over(
            state[2:6]
        )

    def is_truncated(self):
        match self.base_reward_function:
            case "goal_hitting":
                return (
                    self.steps_since_last_goal >= 20
                    or self.step_counter >= self.max_steps
                )
            case "progressive":
                return (
                    self.progress_not_met_cnt >= 5
                    or self.step_counter >= self.max_steps
                )
            case _:
                raise ValueError("Unknown truncate condition for reward function.")

    def get_observation(self):
        odom, lidar = self.get_data()
        odom = util.process_odom(odom)
        num_points = self.lidar_points
        state = []

        match (self.observation_mode):
            case "no_position":
                state += odom[2:]
            case "lidar_only":
                state += odom[-2:]
            case _:
                state += odom
        match self.lidar_processing:
            case "avg":
                processed_lidar_range = util.avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(lidar, num_points, visualized_range)
            case "raw":
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(
                    processed_lidar_range, posinf=-5, nan=-1, neginf=-5
                ).tolist()
                visualized_range = processed_lidar_range
                scan = util.create_lidar_msg(lidar, num_points, visualized_range)

        self.processed_publisher.publish(scan)

        full_state = odom + processed_lidar_range

        state += processed_lidar_range
        state = np.asarray(state)

        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        reward = 0
        reward_info = {}

        match self.base_reward_function:
            case "progressive":
                base_reward, base_reward_info = self.calculate_progressive_reward(
                    state, next_state, raw_lidar_range
                )
                reward += base_reward
                reward_info.update(base_reward_info)
            case _:
                raise ValueError("Unknown reward function. Check environment.")

        for term in self.extra_reward_terms:
            match term:
                case "penalize_turn":
                    turn_penalty = abs(state[7] - next_state[7]) * 0.12
                    reward -= turn_penalty
                    reward_info.update({"turn_penalty": ("avg", turn_penalty)})

        for modifier_type, weight in self.reward_modifiers:
            match modifier_type:
                case "wall_proximity":
                    dist_to_wall = min(raw_lidar_range)
                    close_to_wall_penalize_factor = 1 / (
                        1 + np.exp(50 * (dist_to_wall - 0.3))
                    )  # y=\frac{1}{1+e^{35\left(x-0.5\right)}}
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall": ["avg", dist_to_wall]})
                    print(
                        f"--- Wall proximity penalty factor: {weight} * {close_to_wall_penalize_factor}"
                    )
                case "turn":
                    angular_vel_diff = abs(state[7] - next_state[7])
                    turning_penalty_factor = 1 - (
                        1 / (1 + np.exp(15 * (angular_vel_diff - 0.5)))
                    )  # y=1-\frac{1}{1+e^{15\left(x-0.3\right)}}
                    reward -= reward * turning_penalty_factor * weight
                    print(
                        f"--- Turning penalty factor: {weight} * {turning_penalty_factor}"
                    )
        return reward, reward_info

    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0
        goal_position = self.goal_position
        current_distance = math.dist(goal_position, next_state[:2])

        if self.step_progress < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0
        reward += self.step_progress
        self.steps_since_last_goal += 1

        if current_distance < self.reward_range:
            self.goals_reached += 1
            new_x, new_y, _, _ = self.curr_waypoints[
                (self.spawn_index + self.goals_reached) % len(self.curr_waypoints)
            ]
            self.goal_position = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 5:
            reward -= 2

        if util.has_collided(raw_range, self.collision_range) or util.has_flipped_over(
            next_state[2:6]
        ):
            reward -= 2.5

        info = {}
        return reward, info
