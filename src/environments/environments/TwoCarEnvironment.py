import math
import rclpy
import numpy as np
from rclpy import Future
import random
from environments.F1tenthEnvironment import F1tenthEnvironment
from .util import has_collided, has_flipped_over, findOccurrences
from .util import process_ae_lidar, process_odom, avg_lidar, create_lidar_msg, twist_to_ackermann, reconstruct_ae_latent, lateral_translation
from typing import Literal, List, Tuple
from std_msgs.msg import String
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry


class TwoCarEnvironment(F1tenthEnvironment):

    def __init__(self, 
                 car_name, 
                 reward_range=0.5, 
                 max_steps=200, 
                 collision_range=0.2, 
                 step_length=0.5, 
                 track='track_1',
                 observation_mode='lidar_only',
                 config_path='/home/anyone/autonomous_f1tenth/src/environments/config/config.yaml',
                 ):
        super().__init__('two_car', car_name, reward_range, max_steps, collision_range, step_length, 10, track, observation_mode)

        #####################################################################################################################
        # Reward params ----------------------------------------------
        self.REWARD_MODIFIERS:List[Tuple[Literal['turn','wall_proximity', 'racing'],float]] = [('turn', 0.3), ('wall_proximity', 0.7), ('racing', 1)]
        self.MULTI_TRACK_TRAIN_EVAL_SPLIT = 5/6

        #####################################################################################################################
        # Environment configuration -------------------------------------
        if self.IS_MULTI_TRACK:
            self.EVAL_TRACKS_IDX = int(len(self.ALL_TRACK_WAYPOINTS)*self.MULTI_TRACK_TRAIN_EVAL_SPLIT) 
            
        #####################################################################################################################
        # Pub/Sub ----------------------------------------------------
        self.ODOM_SUB_1 = Subscriber(
            self,
            Odometry,
            f'/f1tenth/odometry',
        )

        self.ODOM_SUB_2 = Subscriber(
            self,
            Odometry,
            f'/f2tenth/odometry',
        )

        self.STATUS_PUB = self.create_publisher(
            String,
            '/status',
            10
        )

        self.STATUS_SUB = self.create_subscription(
            String,
            '/status',
            self.status_callback,
            10
        )

        self.STATUS_LOCK_PUB = self.create_publisher(
            String,
            '/status_lock',
            10
        )

        self.STATUS_LOCK_SUB = self.create_subscription(
            String,
            '/status_lock',
            self.status_lock_callback,
            10
        )

        #####################################################################################################################
        # Message filter ---------------------------------------------
        self.ODOM_MESSAGE_FILTER = ApproximateTimeSynchronizer(
            [self.ODOM_SUB_1, self.ODOM_SUB_2],
            10,
            0.1,
        )
        self.ODOM_MESSAGE_FILTER.registerCallback(self.odom_message_filter_callback)

        #####################################################################################################################
        # Initialise vars ---------------------------------------------
        self.CURR_TRACK = None
        self.CURR_WAYPOINTS = None
        self.PROGRESS_NOT_MET_COUNTER = 0
        self.EP_PROGRESS1 = 0
        self.EP_PROGRESS2 = 0
        self.LAST_POS1 = [0, 0]
        self.LAST_POS2 = [0, 0]
        self.CURR_EVAL_IDX = 0
        self.STEPS_WITHOUT_GOAL = 0

        self.STATUS = 'r_f1tenth'
        self.STATUS_LOCK = 'off'

        # Futures
        self.STATUS_OBSERVATION_FUTURE = Future()
        self.ODOMS_OBSERVATION_FUTURE = Future()

        self.get_logger().info('Environment Setup Complete')

        #####################################################################################################################

    def odom_message_filter_callback(self, odom1: Odometry, odom2: Odometry):
        self.ODOMS_OBSERVATION_FUTURE.set_result({'odom1': odom1, 'odom2': odom2})                                                                            
    
    def randomize_yaw(self, yaw, percentage=0.5):
        factor = 1 + random.uniform(-percentage, percentage)
        return yaw + factor
    
    def reset(self):
        self.STEP_COUNTER = 0
        self.STEPS_WITHOUT_GOAL = 0
        self.GOALS_REACHED = 0
        self.set_velocity(0, 0)

        if self.NAME == 'f2tenth':
            while ('respawn' not in self.STATUS):
                rclpy.spin_until_future_complete(self, self.STATUS_OBSERVATION_FUTURE, timeout_sec=10)
                if (self.STATUS_OBSERVATION_FUTURE.result()) == None:
                    state, full_state , _ = self.get_observation()
                    self.CURR_STATE = full_state
                    info = {}
                    return state, info
                self.STATUS_OBSERVATION_FUTURE = Future()
            track, goal, spawn = self.parse_status(self.STATUS)
            self.CURR_TRACK = track
            self.GOAL_POS = [goal[0], goal[1]]
            self.SPAWN_INDEX = spawn
            self.CURR_WAYPOINTS = self.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]
            if self.IS_EVAL:
                eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACKS_IDX:]
                self.CURR_EVAL_IDX += 1
                self.CURR_EVAL_IDX = self.CURR_EVAL_IDX % len(eval_track_key_list)
            self.publish_status('ready')
        else:
            self.car_spawn()
            i = 0
            while(self.STATUS != 'ready'):
                rclpy.spin_until_future_complete(self, self.STATUS_OBSERVATION_FUTURE, timeout_sec=10)
                if (self.STATUS_OBSERVATION_FUTURE.result() == None):
                    break
                self.STATUS_OBSERVATION_FUTURE = Future()

        self.call_step(pause=False)
        state, full_state , _ = self.get_observation()
        self.CURR_STATE = full_state
        self.call_step(pause=True)
        info = {}

        if self.IS_MULTI_TRACK:
            self.CURR_TRACK_MODEL = self.ALL_TRACK_MODELS[self.CURR_TRACK]
        self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)
        self.EP_PROGRESS1 = 0
        self.EP_PROGRESS2 = 0
        self.LAST_POS1 = [None, None]
        self.LAST_POS2 = [None, None]
        self.PROGRESS_NOT_MET_COUNTER = 0

        self.publish_status('')
        self.change_status_lock('off')
        return state, info
    
    def car_spawn(self):
        if self.IS_MULTI_TRACK:
            if self.IS_EVAL:
                eval_track_key_list = list(self.ALL_TRACK_WAYPOINTS.keys())[self.EVAL_TRACKS_IDX:]
                self.CURR_TRACK = eval_track_key_list[self.CURR_EVAL_IDX]
                self.CURR_EVAL_IDX += 1
                self.CURR_EVAL_IDX = self.CURR_EVAL_IDX % len(eval_track_key_list)
            else:
                self.CURR_TRACK = random.choice(list(self.ALL_TRACK_WAYPOINTS.keys())[:self.EVAL_TRACKS_IDX])
            self.CURR_WAYPOINTS = self.ALL_TRACK_WAYPOINTS[self.CURR_TRACK]
        else:
            self.CURR_TRACK = self.TRACK
        if (self.CURR_TRACK[-3:]).isdigit():
            width = int(self.CURR_TRACK[-3:])
        else: 
            width = 300

        car_x, car_y, car_yaw, index = random.choice(self.CURR_WAYPOINTS)
        order = random.choice([1, 2])
        translation2 = random.random()
        translation1 = 0.15 + random.random()*0.3
        translation2 = -0.2 - random.random()*1.5

        if width > 200:
            if order == 1:
                car_2_x, car_2_y = lateral_translation((car_x, car_y), car_yaw, translation1)
                car_x, car_y = lateral_translation((car_x, car_y), car_yaw, translation2)
            else:
                car_2_x, car_2_y = lateral_translation((car_x, car_y), car_yaw, translation2) 
                car_x, car_y = lateral_translation((car_x, car_y), car_yaw, translation1)
            car_2_yaw = car_yaw
        else:
            if order == 1:
                car_2_offset = random.randint(8, 16)
                car_2_index = (index + car_2_offset) % len(self.CURR_WAYPOINTS)
                car_2_x, car_2_y, car_2_yaw, _ = self.CURR_WAYPOINTS[car_2_index]
            else:
                car_2_offset = random.randint(8, 16)
                car_2_index = (index - car_2_offset) % len(self.CURR_WAYPOINTS)
                car_2_x, car_2_y, car_2_yaw, _ = self.CURR_WAYPOINTS[car_2_index]

        self.SPAWN_INDEX = index
        x,y,_,_ = self.CURR_WAYPOINTS[self.SPAWN_INDEX+1 if self.SPAWN_INDEX+1 < len(self.CURR_WAYPOINTS) else 0]
        goal_x, goal_y, _, _ = self.CURR_WAYPOINTS[self.SPAWN_INDEX+3 if self.SPAWN_INDEX+3 < len(self.CURR_WAYPOINTS) else 0]
        self.GOAL_POS = [x,y]

        self.call_reset_service(car_x=car_x, car_y=car_y, car_Y=car_yaw, goal_x=goal_x, goal_y=goal_y, car_name='f1tenth')
        self.call_reset_service(car_x=car_2_x, car_y=car_2_y, car_Y=car_2_yaw, goal_x=goal_x, goal_y=goal_y, car_name='f2tenth')

        if self.NAME == 'f1tenth':
            string =  'respawn_' + str(self.CURR_TRACK) + '_' + str(self.GOAL_POS)+ '_' + str(self.SPAWN_INDEX) + ', car1'
        else:
            string =  'respawn_' + str(self.CURR_TRACK) + '_' + str(self.GOAL_POS)+ '_' + str(self.SPAWN_INDEX) + ', car2'
        self.publish_status(string)
    
    def start_eval(self):
        self.CURR_EVAL_IDX = 0
        self.IS_EVAL = True

    def stop_eval(self):
        self.IS_EVAL = False

    def step(self, action):
        self.STEP_COUNTER += 1
        full_state = self.CURR_STATE
        self.call_step(pause=False)
        lin_vel, steering_angle = action
        self.set_velocity(lin_vel, steering_angle)
        self.sleep()

        next_state, full_next_state, raw_lidar_range = self.get_observation()
        self.call_step(pause=True)
        self.CURR_STATE = full_next_state
        
        if not self.PREV_CLOSEST_POINT:
            self.PREV_CLOSEST_POINT = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_state[:2], t_only=True)

        t2 = self.CURR_TRACK_MODEL.get_closest_point_on_spline(full_next_state[:2], t_only=True)
        self.STEP_PROGRESS = self.CURR_TRACK_MODEL.get_distance_along_track_parametric(self.PREV_CLOSEST_POINT, t2, approximate=True)
        self.center_line_offset = self.CURR_TRACK_MODEL.get_distance_to_spline_point(t2, full_next_state[:2])
        self.PREV_CLOSEST_POINT = t2

        if abs(self.STEP_PROGRESS) > (full_next_state[6]/10*3): 
            self.STEP_PROGRESS = full_next_state[6]/10*0.8 

        reward, reward_info = self.compute_reward(full_state, full_next_state, raw_lidar_range)
        terminated = self.is_terminated(full_next_state, raw_lidar_range)
        truncated = self.is_truncated()

        info = {
            'linear_velocity':["avg", full_next_state[6]],
            'angular_velocity_diff':["avg", abs(full_next_state[7] - full_state[7])],
            'traveled distance': ['sum', self.STEP_PROGRESS]
        }
        info.update(reward_info)

        if self.IS_EVAL and (terminated or truncated):
            self.CURR_EVAL_IDX
        if ((terminated or truncated) and self.STATUS_LOCK == 'off'):
            self.change_status_lock('on')
            string = 'r_' + str(self.NAME)
            self.publish_status(string)
            self.STATUS=string
        if ((not truncated) and ('r' in self.STATUS)):
            truncated = True
        return next_state, reward, terminated, truncated, info

    def is_terminated(self, state, ranges):
        return has_collided(ranges, self.COLLISION_RANGE) \
            or has_flipped_over(state[2:6])

    def is_truncated(self):
        return self.PROGRESS_NOT_MET_COUNTER >= 5 or \
        self.STEP_COUNTER >= self.MAX_STEPS

    def get_observation(self):
        odom, lidar = self.get_data()
        odom = process_odom(odom)
        num_points = self.LIDAR_POINTS
        state = []

        match (self.ODOM_OBSERVATION_MODE):
            case 'no_position':
                state += odom[2:]
            case 'lidar_only':
                state += odom[-2:] 
            case _:
                state += odom 
        match self.LIDAR_PROCESSING:
            case 'pretrained_ae':
                processed_lidar_range = process_ae_lidar(lidar, self.AE_LIDAR, is_latent_only=True)
                visualized_range = reconstruct_ae_latent(lidar, self.AE_LIDAR, processed_lidar_range)
                scan = create_lidar_msg(lidar, 682, visualized_range)
            case 'avg':
                processed_lidar_range = avg_lidar(lidar, num_points)
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
            case 'raw':
                processed_lidar_range = np.array(lidar.ranges.tolist())
                processed_lidar_range = np.nan_to_num(processed_lidar_range, posinf=-5, nan=-1, neginf=-5).tolist()  
                visualized_range = processed_lidar_range
                scan = create_lidar_msg(lidar, num_points, visualized_range)
        self.PROCESSED_PUBLISHER.publish(scan)

        state += processed_lidar_range
        full_state = odom + processed_lidar_range
        return state, full_state, lidar.ranges

    def compute_reward(self, state, next_state, raw_lidar_range):
        reward = 0
        reward_info = {}
        odom1, odom2 = self.get_odoms()

        if self.LAST_POS1[0] == None:
            self.LAST_POS1 = odom1[:2]
        if self.LAST_POS2[0] == None:
            self.LAST_POS2 = odom2[:2]

        progression1 = self.CURR_TRACK_MODEL.get_distance_along_track(self.LAST_POS1, odom1[:2])
        progression2 = self.CURR_TRACK_MODEL.get_distance_along_track(self.LAST_POS2, odom2[:2])

        if abs(progression1) < 1:
            self.EP_PROGRESS1 += progression1
        if abs(progression2) < 1:
            self.EP_PROGRESS2 += progression2
        if self.NAME == 'f1tenth':
            base_reward, base_reward_info = self.calculate_total_progress_reward(self.EP_PROGRESS1, progression1, next_state, raw_lidar_range)
        else:
            base_reward, base_reward_info = self.calculate_total_progress_reward(self.EP_PROGRESS2, progression2, next_state, raw_lidar_range)
        reward += base_reward
        reward_info.update(base_reward_info)
        
        for modifier_type, weight in self.REWARD_MODIFIERS:
            match modifier_type:
                case 'wall_proximity':
                    dist_to_wall = min(raw_lidar_range)
                    close_to_wall_penalize_factor = 1 / (1 + np.exp(35 * (dist_to_wall - 0.5)))
                    reward -= reward * close_to_wall_penalize_factor * weight
                    reward_info.update({"dist_to_wall":["avg",dist_to_wall]}) 
                case 'turn':
                    steering_angle1 = twist_to_ackermann(state[7], state[6], L=0.325)
                    steering_angle2 = twist_to_ackermann(next_state[7], next_state[6], L=0.325)
                    angle_diff = abs(steering_angle1 - steering_angle2)
                    if angle_diff > 3:
                        turning_penalty_factor = 0
                    else:
                        turning_penalty_factor = 1 - (1 / (1 + np.exp(15 * (angle_diff - 0.3))))
                    
                    reward -= reward * turning_penalty_factor * weight
                case 'racing':
                    if self.NAME == 'f1tenth':
                        if self.EP_PROGRESS1 == self.EP_PROGRESS2:
                            modifier=0
                        else:
                            modifier = (self.EP_PROGRESS1 - self.EP_PROGRESS2)/abs(self.EP_PROGRESS1 - self.EP_PROGRESS2)
                    else:
                        if self.EP_PROGRESS1 == self.EP_PROGRESS2:
                            modifier=0
                        else:
                            modifier = (self.EP_PROGRESS2 - self.EP_PROGRESS1)/abs(self.EP_PROGRESS2 - self.EP_PROGRESS1)
                    reward += reward * modifier * weight
                    self.LAST_POS1 = odom1[:2]
                    self.LAST_POS2 = odom2[:2]  
        return reward, reward_info
    
    def calculate_progressive_reward(self, state, next_state, raw_range):
        reward = 0
        goal_position = self.GOAL_POS
        current_distance = math.dist(goal_position, next_state[:2])

        if self.STEP_PROGRESS < 0.02:
            self.PROGRESS_NOT_MET_COUNTER += 1
        else:
            self.PROGRESS_NOT_MET_COUNTER = 0
        reward += self.STEP_PROGRESS
        self.STEPS_WITHOUT_GOAL += 1

        if current_distance < self.REWARD_RANGE:
            self.GOALS_REACHED += 1
            new_x, new_y, _, _ = self.CURR_WAYPOINTS[(self.SPAWN_INDEX + self.GOALS_REACHED) % len(self.CURR_WAYPOINTS)]
            self.GOAL_POS = [new_x, new_y]
            self.update_goal_service(new_x, new_y)
            self.STEPS_WITHOUT_GOAL = 0
        if self.PROGRESS_NOT_MET_COUNTER >= 5:
            reward -= 2
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5

        info = {}
        return reward, info

    def calculate_total_progress_reward(self, total_progression, step_progression, next_state, raw_range):
        reward = 0
        if step_progression < 0.02:
            self.PROGRESS_NOT_MET_COUNTER += 1
        else:
            self.PROGRESS_NOT_MET_COUNTER = 0
        reward += total_progression

        if self.PROGRESS_NOT_MET_COUNTER >= 5:
            reward -= 2
        if has_collided(raw_range, self.COLLISION_RANGE) or has_flipped_over(next_state[2:6]):
            reward -= 2.5
        info = {}
        return reward, info
    
    def get_odoms(self):
        rclpy.spin_until_future_complete(self, self.ODOMS_OBSERVATION_FUTURE)
        future = self.ODOMS_OBSERVATION_FUTURE
        self.ODOMS_OBSERVATION_FUTURE = Future()
        data = future.result()
        odom1 = process_odom(data['odom1'])
        odom2 = process_odom(data['odom2'])
        return odom1, odom2
    
    def publish_status(self, status):
        msg = String()
        msg.data = str(status)
        self.STATUS_PUB.publish(msg)

    def status_callback(self, msg):
        self.STATUS = msg.data
        self.STATUS_OBSERVATION_FUTURE.set_result({'status': msg})

    def change_status_lock(self, change):
        msg = String()
        msg.data = str(change)
        self.STATUS_LOCK_PUB.publish(msg)
        self.STATUS_LOCK = change

    def status_lock_callback(self, msg):
        self.STATUS_LOCK = msg.data

    def parse_status(self, msg):
        indexes = findOccurrences(msg, '_')
        comma = findOccurrences(msg, ',')
        track = msg[(indexes[0]+1):indexes[2]]
        goalx = float(msg[(indexes[2]+2):(comma[0])])
        goaly = float(msg[(comma[0]+2):(indexes[3]-1)])
        goal = goalx, goaly
        spawn_index = int(msg[(indexes[3]+1):comma[1]])
        return track, goal, spawn_index