import math
import random
from typing import List, Literal, Tuple

import numpy as np
import scipy

from environments.F1tenthEnvironment import F1tenthEnvironment

from .util import (
    avg_lidar,
    create_lidar_msg,
    get_training_stages,
    has_collided,
    has_flipped_over,
    process_ae_lidar,
    process_odom,
    reconstruct_ae_latent,
)


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
            +2 if it comes within REWARD_RANGE units of a goal
            -25 if it collides with a wall

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
        is_staged_training=False,
    ):
        super().__init__(
            "car_track",
            car_name,
            reward_range,
            max_steps,
            collision_range,
            step_length,
            10,
            track,
            observation_mode,
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
        # Staging configuration ----------------------------------------
        self.is_staged_training = is_staged_training
        self.current_training_stage = 0
        if self.is_staged_training:
            self.training_stages = get_training_stages(track)
            self.training_idx = self.training_stages[self.current_training_stage][0]
            self.eval_idx = self.training_stages[self.current_training_stage][1]

        #####################################################################################################################
        # Environment configuration -------------------------------------
        if self.base_reward_function == "progressive":
            self.progress_not_met_cnt = 0
        self.steps_since_last_goal = 0

        if self.is_multi_track:
            if self.is_staged_training:
                self.current_track = list(self.all_track_waypoints.keys())[
                    self.training_idx[0]
                ]
                self.eval_track_begin_idx = None
                self.get_logger().info(
                    f"Track '{track}', {self.training_idx} training, {self.eval_idx} evaluation"
                )
            else:
                self.eval_track_begin_idx = int(
                    len(self.all_track_waypoints) * self.multi_track_train_eval_split
                )
            self.eval_track_idx = 0

        self.step_counter = 0

        self.get_logger().info("Environment Setup Complete")

        #####################################################################################################################

    def reset(self):
        self.step_counter = 0
        self.steps_since_last_goal = 0
        self.GOALS_REACHED = 0
        self.set_velocity(0, 0)

        if self.is_multi_track:
            if (
                self.eval_track_begin_idx is not None
                and self.eval_track_begin_idx >= len(self.all_track_waypoints)
            ):
                if self.IS_EVAL:
                    all_track_keys = list(self.all_track_waypoints.keys())
                    self.current_track = all_track_keys[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(all_track_keys)
                else:
                    self.current_track = random.choice(
                        list(self.all_track_waypoints.keys())
                    )
            else:
                if self.IS_EVAL:
                    if self.is_staged_training:
                        eval_track_key_list = list(self.all_track_waypoints.keys())[
                            self.eval_idx[0] : self.eval_idx[1] + 1
                        ]
                    else:
                        eval_track_key_list = list(self.all_track_waypoints.keys())[
                            self.eval_track_begin_idx :
                        ]
                    self.current_track = eval_track_key_list[self.eval_track_idx]
                    self.eval_track_idx += 1
                    self.eval_track_idx = self.eval_track_idx % len(eval_track_key_list)
                else:
                    if self.is_staged_training:
                        self.current_track = random.choice(
                            list(self.all_track_waypoints.keys())[
                                self.training_idx[0] : self.training_idx[1] + 1
                            ]
                        )
                    else:
                        self.current_track = random.choice(
                            list(self.all_track_waypoints.keys())[
                                : self.eval_track_begin_idx
                            ]
                        )
            self.CURR_WAYPOINTS = self.all_track_waypoints[self.current_track]

        if self.IS_EVAL:
            car_x, car_y, car_yaw, index = self.CURR_WAYPOINTS[10]
        else:
            car_x, car_y, car_yaw, index = random.choice(self.CURR_WAYPOINTS)

        self.SPAWN_INDEX = index
        x, y, _, _ = self.CURR_WAYPOINTS[
            (
                self.SPAWN_INDEX + 1
                if self.SPAWN_INDEX + 1 < len(self.CURR_WAYPOINTS)
                else 0
            )
        ]  # point toward next goal
        self.goal_position = [x, y]
        self.call_reset_service(
            car_x=car_x,
            car_y=car_y,
            car_Y=car_yaw,
            goal_x=x,
            goal_y=y,
            car_name=self.NAME,
        )

        self.call_step(pause=False)
        state, full_state, _ = self.get_observation()
        self.current_state = full_state
        self.call_step(pause=True)

        if self.is_multi_track:
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.current_track]
        self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(
            full_state[:2], t_only=True
        )

        if self.base_reward_function == "progressive":
            self.progress_not_met_cnt = 0
        info = {}
        return state, info

    def start_eval(self):
        self.eval_track_idx = 0
        self.IS_EVAL = True

    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.step_counter += 1
        full_state = self.current_state
        self.call_step(pause=False)

        lin_vel, steering_angle = action

        self.set_velocity(lin_vel, steering_angle)
        self.sleep()

        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)

        self.current_state = full_next_state
        if not self.PREV_CLOSEST_POINT:
            self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(
                full_state[:2], t_only=True
            )

        t2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(
            full_next_state[:2], t_only=True
        )
        self.STEP_PROGRESS = self.CURR_TRACK_MODEL.get_distance_along_track_parametric(
            self.PREV_CLOSEST_POINT, t2, approximate=True
        )
        # This doesn't show up anywhere?
        # self.center_line_offset = self.CURR_TRACK_MODEL.get_distance_to_spline_point(
        #     t2, full_next_state[:2]
        # )
        self.PREV_CLOSEST_POINT = t2

        if abs(self.STEP_PROGRESS) > (full_next_state[6] / 10 * 3):
            self.STEP_PROGRESS = full_next_state[6] / 10 * 0.8

        reward, reward_info = self.compute_reward(
            full_state, full_next_state, raw_lidar_range
        )
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            "linear_velocity": ["avg", full_next_state[6]],
            "angular_velocity_diff": ["avg", abs(full_next_state[7] - full_state[7])],
            "traveled distance": ["sum", self.STEP_PROGRESS],
        }
        info.update(reward_info)

        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) or has_flipped_over(
            state[2:6]
        )

    def is_truncated(self):
        match self.base_reward_function:
            case "goal_hitting":
                return (
                    self.steps_since_last_goal >= 20
                    or self.step_counter >= self.MAX_STEPS
                )
            case "progressive":
                return (
                    self.progress_not_met_cnt >= 5
                    or self.step_counter >= self.MAX_STEPS
                )
            case _:
                raise Exception("Unknown truncate condition for reward function.")

    def get_observation(self):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        num_points = self.LIDAR_POINTS
        state = {}

        match (self.ODOM_OBSERVATION_MODE):
            case "no_position":
                state["vector"] = odom[2:]
            case "lidar_only":
                state["vector"] = odom[-2:]
            case _:
                state["vector"] = odom
        match self.LIDAR_PROCESSING:
            case "pretrained_ae":
                processed_lidar_range = process_ae_lidar(
                    lidar, self.AE_LIDAR_MODEL, is_latent_only=True
                )
                visualized_range = reconstruct_ae_latent(
                    lidar, self.AE_LIDAR_MODEL, processed_lidar_range
                )
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case "ae":
                lidar_data = np.array(lidar.ranges)
                lidar_data = np.nan_to_num(lidar_data, posinf=-5)
                if not self.IS_EVAL:
                    sampled_data = scipy.signal.resample(lidar_data, 512)
                    self.train_autoencoder(sampled_data)
                processed_lidar_range = process_ae_lidar(
                    lidar, self.AE_LIDAR_MODEL, is_latent_only=True
                )
                scan = create_lidar_msg(lidar, num_points, processed_lidar_range)
            case "avg":
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case "raw":
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(
                    processed_lidar_range, posinf=-5, nan=-1, neginf=-5
                ).tolist()
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)

        self.PROCESSED_PUBLISHER.publish(scan)
        if self.LIDAR_PROCESSING == "ae":
            state["lidar"] = lidar_data.tolist()
            full_state = odom + lidar_data.tolist()
        else:
            full_state = odom + processed_lidar_range

        state = odom[-2:] + processed_lidar_range
        state = np.asarray(state)
        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        reward = 0
        reward_info = {}

        match self.base_reward_function:
            case "goal_hitting":
                base_reward, base_reward_info = self.calculate_goal_hitting_reward(
                    state, next_state, raw_lidar_range
                )
                reward += base_reward
                reward_info.update(base_reward_info)
            case "progressive":
                base_reward, base_reward_info = self.calculate_progressive_reward(
                    state, next_state, raw_lidar_range
                )
                reward += base_reward
                reward_info.update(base_reward_info)
            case _:
                raise Exception("Unknown reward function. Check environment.")

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

    def calculate_goal_hitting_reward(self, state, next_state, raw_range):
        reward = 0
        goal_position = self.goal_position
        current_distance = math.dist(goal_position, next_state[:2])
        previous_distance = math.dist(goal_position, state[:2])
        reward += previous_distance - current_distance
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            reward += 2
            self.GOALS_REACHED += 1
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[
                (self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)
            ]
            self.goal_position = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.steps_since_last_goal = 0

        if self.steps_since_last_goal >= 20:
            reward -= 10
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(
            next_state[2:6]
        ):
            reward -= 25

        info = {}
        return reward, info

    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0
        goal_position = self.goal_position
        current_distance = math.dist(goal_position, next_state[:2])

        if self.STEP_PROGRESS < 0.02:
            self.progress_not_met_cnt += 1
        else:
            self.progress_not_met_cnt = 0
        reward += self.STEP_PROGRESS
        self.steps_since_last_goal += 1

        if current_distance < self.REWARD_RANGE:
            self.GOALS_REACHED += 1
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[
                (self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)
            ]
            self.goal_position = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.steps_since_last_goal = 0

        if self.progress_not_met_cnt >= 5:
            reward -= 2
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(
            next_state[2:6]
        ):
            reward -= 2.5

        info = {}
        return reward, info

    def increment_stage(self):
        if not self.is_staged_training:
            return

        if self.current_training_stage < len(self.training_stages) - 1:
            self.current_training_stage += 1
            self.training_idx = self.training_stages[self.current_training_stage][0]
            self.eval_idx = self.training_stages[self.current_training_stage][1]
            self.get_logger().info(
                f"Incremented to training stage {self.current_training_stage}. Training indices: {self.training_idx}, Evaluation indices: {self.eval_idx}"
            )
        else:
            self.get_logger().info(
                "Already at the last training stage. No increment performed."
            )
