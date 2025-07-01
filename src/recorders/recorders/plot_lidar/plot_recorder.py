import math
import json
import torch
import numpy as np
from typing import List
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import sys
import os


root_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(root_dir, 'cares_reinforcement_learning'))

from cares_reinforcement_learning.util.network_factory import NetworkFactory
from cares_reinforcement_learning.util.configurations import TD3Config
from cares_reinforcement_learning.util.helpers import denormalize

VEL_RECORD = 'C:/Users/eason/Desktop/Code/autonomous_f1tenth/src/recorders/recorders/plot_lidar/record_drive_2025-06-30 14_07_58.txt'
LIDAR_RECORD = 'C:/Users/eason/Desktop/Code/autonomous_f1tenth/src/recorders/recorders/plot_lidar/record_lidar_2025-06-30 14_07_59.txt'
MODEL_PATH = 'C:/Users/eason/Desktop/Code/autonomous_f1tenth/src/recorders/recorders/plot_lidar'
ACTOR = os.path.join(MODEL_PATH, 'TD3_actor.pht')
CRITIC = os.path.join(MODEL_PATH, 'TD3_critic.pht')
NETWORK_CONFIG_PATH = os.path.join(MODEL_PATH, 'network_config.json')


class LidarData:
    def __init__(self, odom):
        self.odom = odom    # Car position (x, y)
        self.scans = []     # List of Lidar distances
        self.wall_points = []  # List of wall points (x, y)

    def add_scan(self, scan):
        self.scans.append(scan)

    def add_wall_point(self, point):
        self.wall_points.append(point)


class VelData:
    def __init__(self, linear, angular):
        self.linear = linear
        self.angular = angular


def read_lidar_data(lidar_file_path):
    lidar_data_list: List[LidarData] = []
    data = None
    with open(lidar_file_path, 'r') as lidar_file:
        lines = lidar_file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                if data:
                    lidar_data_list.append(data)
                    data = None
            elif line.startswith("Car Position:"):
                position = line.split(":")[1].strip("() /n")
                x_str, y_str = [p.strip() for p in position.split(",")]
                data = LidarData((float(x_str), float(y_str)))
            elif line.startswith("("):
                line = line.strip("(), /t/n")
                x_str, y_str = [p.strip() for p in line.split(",")]
                x, y = float(x_str), float(y_str)
                distance = math.sqrt(
                    (x-data.odom[0]) ** 2 + (y-data.odom[1]) ** 2)
                data.add_scan(distance)
                data.add_wall_point((x, y))

        # Add the last data point if it exists
        if data:
            lidar_data_list.append(data)

    return lidar_data_list


def read_vel_data(vel_file_path):
    vel_data_list: List[VelData] = []
    with open(vel_file_path, 'r') as vel_file:
        lines = vel_file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                _, linear, angular = (
                    float(part.split('=')[1]) for part in parts)
                vel_data_list.append(VelData(linear, angular))
    return vel_data_list


def plot():
    lidar_data_list = read_lidar_data(LIDAR_RECORD)
    vel_data_list = read_vel_data(VEL_RECORD)

    if not lidar_data_list:
        print("No lidar data!")
        return

    if not vel_data_list:
        print("No vel data!")
        return

    print(f"Loaded {len(lidar_data_list)} lidar data points")
    print(f"Loaded {len(vel_data_list)} vel data points")

    if len(lidar_data_list) < len(vel_data_list):
        print("Lidar data is shorter than vel data, truncating vel data to match lidar data length.")
        vel_data_list = vel_data_list[:len(lidar_data_list)]
    elif len(vel_data_list) < len(lidar_data_list):
        print("Vel data is shorter than lidar data, truncating lidar data to match vel data length.")
        lidar_data_list = lidar_data_list[:len(vel_data_list)]

    fig, ax = plt.subplots(figsize=(10, 6))

    fig.text(0.85, 0.85, 'Recorded Values', fontsize=10, va='center')
    linear_text = fig.text(0.85, 0.8, '', fontsize=10, va='center')
    angular_text = fig.text(0.85, 0.75, '', fontsize=10, va='center')
    fig.text(0.85, 0.70, 'Network Output', fontsize=10, va='center')
    speed_text = fig.text(0.85, 0.65, '', fontsize=10, va='center')
    steering_text = fig.text(0.85, 0.60, '', fontsize=10, va='center')

    plt.subplots_adjust(bottom=0.2)  # Adjust space for the slider

    wall_plot, = ax.plot([], [], 'o', markersize=1, label="Walls")
    car_plot, = ax.plot([], [], 'ro', markersize=2, label="Car Position")

    ax.set_title("Track Walls and Car Position - Top-Down View")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()
    ax.grid()
    ax.axis('equal')

    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position of the slider
    slider = Slider(ax_slider, 'Index', 1, min(
        len(lidar_data_list), len(vel_data_list))-1, valinit=1, valstep=1)

    def update(val):
        index = int(slider.val)
        lidar_data_sublist = lidar_data_list[:index]
        last_lidar_data = lidar_data_list[index]
        recorded_vel_data = vel_data_list[index-1]

        state = data_to_state(last_lidar_data, recorded_vel_data)
        steering_angle, speed = get_network_output(state, AGENT)

        linear_text.set_text(f"Linear: {recorded_vel_data.linear:.2f}")
        angular_text.set_text(f"Angular: {recorded_vel_data.angular:.2f}")
        speed_text.set_text(f"Linear: {speed:.2f}")
        steering_text.set_text(f"Angular: {steering_angle:.2f}")

        # Flatten wall points into x and y coordinates
        wall_points = [
            point for data in lidar_data_sublist for point in data.wall_points]
        if wall_points:
            wall_x, wall_y = zip(*wall_points)
        else:
            wall_x, wall_y = [], []  # Handle empty wall points

        # Extract car positions
        car_positions = [data.odom for data in lidar_data_sublist]
        car_x, car_y = zip(*car_positions) if car_positions else ([], [])

        # Update the plots
        wall_plot.set_data(wall_x, wall_y)
        car_plot.set_data(car_x, car_y)

        # Auto-scale the axes to fit the data
        ax.relim()
        ax.autoscale_view()

        # Force a redraw
        fig.canvas.draw()

    slider.on_changed(update)

    # Initialize the plot with the first point
    update(1)
    plt.show()


def get_network_config():
    with open(NETWORK_CONFIG_PATH, 'r') as config_file:
        config_dict = json.load(config_file)
    return TD3Config(**config_dict)


def data_to_state(lidar_data: LidarData, vel_data: VelData):
    linear_velocity = vel_data.linear
    angular_velocity = vel_data.angular
    scans = lidar_data.scans
    state = [linear_velocity, angular_velocity] + scans

    return np.array(state)


def get_network_output(state, agent):
    MAX_ACTIONS = np.asarray([1.5, 0.45])
    MIN_ACTIONS = np.asarray([0, -0.45])

    action = agent.select_action_from_policy(state)
    action = denormalize(action, MAX_ACTIONS, MIN_ACTIONS)
    angular, linear = action
    angular = ackermann_to_twist(angular, linear, 0.325)
    return angular, linear

def ackermann_to_twist(delta, linear_v, L):
    try: 
        omega = math.tan(delta)*linear_v/L
    except ZeroDivisionError:
        print("Wheelbase must be greater than zero")
        return 0
    return omega

if __name__ == "__main__":
    OBSERVATION_SIZE = 2 + 10   # 2 Odom, 10 Lidar
    ACTION_NUM = 2
    network_config = get_network_config()
    network_factory = NetworkFactory()
    agent = network_factory.create_network(
        OBSERVATION_SIZE, ACTION_NUM, config=network_config)
    print('Reading saved models into actor and critic')
    agent.actor_net.load_state_dict(torch.load(
        ACTOR, map_location=torch.device('cpu')))
    agent.critic_net.load_state_dict(torch.load(
        CRITIC, map_location=torch.device('cpu')))
    print('Successfully Loaded models')
    AGENT = agent

    plot()
