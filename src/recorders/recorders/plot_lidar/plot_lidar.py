import pandas as pd
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.ndimage

"""
This script plots the lidar reading of the car based on the logged csv file ('lidar_*.csv').
Lidar distances are converted into x and y coordinates and plotted in a top-down view to represent the obstacle around the car.
The blue dots are the original (valid) readings, the red lines are NaN values and the yelloe dots are processed values.
"""

def gaussian_window_sizes(total_points, num_windows, std_ratio):
    """
    total_points: total number of lidar points (e.g., 1088)
    num_windows: number of windows (e.g., 10)
    std_ratio: controls how concentrated the windows are in the center (0.25 = 25% of range)
    """
    # Generate window centers in normalized [-1, 1] space
    centers = np.linspace(-1, 1, num_windows)
    # Gaussian weights, peak at center (0)
    std = std_ratio * 2  # since range is [-1, 1]
    weights = np.exp(-0.5 * (centers / std) ** 2)
    # Invert weights so center gets smaller windows (higher resolution)
    inv_weights = 1 / (weights + 1e-6)
    # Normalize to sum to total_points
    window_sizes = inv_weights / np.sum(inv_weights) * total_points
    window_sizes = np.round(window_sizes).astype(int)
    # Adjust last window to ensure sum matches exactly
    window_sizes[-1] += total_points - np.sum(window_sizes)
    print(f"Window sizes: {window_sizes}")
    return window_sizes

window_size = gaussian_window_sizes(683, 10, 0.30)
# window_size = [121, 70, 60 ,50, 40, 40, 50, 60, 70, 122]
# window_size = [68, 68, 68, 68, 68, 68, 68, 69, 69, 69]

def plot_lidar_scan(csv_file):
    data = pd.read_csv(csv_file)

    time = data['time']
    scans = data.drop(columns=['time']).values

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plt.grid()

    line, = ax.plot([], [], 'o')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time', time.min(), time.max(), valinit=time.min())

    # Add forward/backward buttons below the slider
    ax_back10 = plt.axes([0.18, 0.02, 0.12, 0.05])
    ax_back1 = plt.axes([0.32, 0.02, 0.12, 0.05])
    ax_forward1 = plt.axes([0.56, 0.02, 0.12, 0.05])
    ax_forward10 = plt.axes([0.70, 0.02, 0.12, 0.05])
    btn_back10 = Button(ax_back10, 'Backward (-10)')
    btn_back1 = Button(ax_back1, 'Backward (-1)')
    btn_forward1 = Button(ax_forward1, 'Forward (+1)')
    btn_forward10 = Button(ax_forward10, 'Forward (+10)')

    def step_backward(event):
        current_val = slider.val
        idx = (time - current_val).abs().idxmin()
        if idx > 0:
            slider.set_val(time.iloc[max(idx - 1, 0)])

    def step_forward(event):
        current_val = slider.val
        idx = (time - current_val).abs().idxmin()
        if idx < len(time) - 1:
            slider.set_val(time.iloc[min(idx + 1, len(time) - 1)])

    def step_backward10(event):
        current_val = slider.val
        idx = (time - current_val).abs().idxmin()
        slider.set_val(time.iloc[max(idx - 10, 0)])

    def step_forward10(event):
        current_val = slider.val
        idx = (time - current_val).abs().idxmin()
        slider.set_val(time.iloc[min(idx + 10, len(time) - 1)])

    btn_back1.on_clicked(step_backward)
    btn_forward1.on_clicked(step_forward)
    btn_back10.on_clicked(step_backward10)
    btn_forward10.on_clicked(step_forward10)

    def update(val):
        current_time = slider.val
        
        idx = (time - current_time).abs().idxmin()
        
        scan_values = scans[idx]
        processed_scan_values, processed_angles = avg_lidar(scan_values, 10)
        processed_scan_values_2, processed_angles_2 = median_lidar(scan_values, 10)
        # processed_scan_values_3, processed_angles_3 = uneven_avg_lidar(scan_values, 10)
        processed_scan_values_4, processed_angles_4 = uneven_median_lidar(scan_values, 10)
        
        angles = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(scan_values))
        # Separate valid and NaN values
        valid_indices = ~np.isnan(scan_values)
        invalid_indices = np.isnan(scan_values)
        
        valid_scan_values = scan_values[valid_indices]
        valid_angles = angles[valid_indices]
        
        invalid_angles = angles[invalid_indices]
        
        x = valid_scan_values * np.cos(valid_angles)
        y = valid_scan_values * np.sin(valid_angles)
        
        processed_x = processed_scan_values * np.cos(processed_angles)
        processed_y = processed_scan_values * np.sin(processed_angles)
        
        processed_x_2 = processed_scan_values_2 * np.cos(processed_angles_2)
        processed_y_2 = processed_scan_values_2 * np.sin(processed_angles_2)
        
        # processed_x_3 = processed_scan_values_3 * np.cos(processed_angles_3)
        # processed_y_3 = processed_scan_values_3 * np.sin(processed_angles_3)
        
        processed_x_4 = processed_scan_values_4 * np.cos(processed_angles_4)
        processed_y_4 = processed_scan_values_4 * np.sin(processed_angles_4)
        
        max_x = np.max(np.abs(np.nan_to_num(scan_values, nan=float(10))*np.cos(angles))) + 1
        max_y = np.max(np.abs(np.nan_to_num(scan_values, nan=float(10))*np.sin(angles))) + 1
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)
        
        line.set_xdata(x)
        line.set_ydata(y)
        
        # Clear previous
        [line.remove() for line in ax.lines[1:]]
        
        # Draw red lines for NaN values
        for angle in invalid_angles:
            ax.plot([0, 10 * np.cos(angle)], [0, 10 * np.sin(angle)], 'r-')

        # colors = plt.cm.get_cmap('tab10', len(window_size))
        # start = 0
        # for i, w in enumerate(window_size):
        #     end = start + w
        #     # Only plot valid points in this window
        #     window_indices = np.arange(start, end)
        #     valid_in_window = valid_indices[window_indices]
        #     if np.any(valid_in_window):
        #         xw = scan_values[window_indices][valid_in_window] * np.cos(angles[window_indices][valid_in_window])
        #         yw = scan_values[window_indices][valid_in_window] * np.sin(angles[window_indices][valid_in_window])
        #         ax.plot(xw, yw, 'o', color=colors(i), label=f'Window {i+1}' if idx == 0 else "")
        #     start = end
        
        ax.plot(processed_x, processed_y, 'mo', label='Average')            # avg
        # ax.plot(processed_x_2, processed_y_2, 'co', label='Median')         # median
        # ax.plot(processed_x_3, processed_y_3, 'ko', label='Uneven Average') # uneven avg
        # ax.plot(processed_x_4, processed_y_4, 'yo', label='Uneven Median')  # uneven median

        # Add legend
        ax.legend(loc='upper right')
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def avg_lidar(lidar, num_points: int):
        step = len(lidar) // num_points
        averaged_lidar = []
        
        for i in range(num_points):
            start = i * step
            end = start + step if i < num_points - 1 else len(lidar)
            segment = lidar[start:end]
            segment = np.nan_to_num(segment, nan=10)
            averaged_lidar.append(np.mean(segment))

        fov_rad = np.radians(240)
        angle_edges = np.linspace(-fov_rad/2, fov_rad/2, num_points + 1)
        angles = (angle_edges[:-1] + angle_edges[1:]) / 2
        return np.array(averaged_lidar), angles
    
def uneven_avg_lidar(lidar, num_points: int):
        ranges = lidar
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))  # Lidar only sees up to 4 meters
        new_range = []
        
        if len(ranges) != sum(window_size):
            raise Exception("Lidar length and window size do not match")
        
        if len(window_size) != num_points:
            raise Exception("Window size length and num_points do not match")
        
        start = 0
        for window in window_size:
            end = start + window
            window_ranges = ranges[start:end]
            new_range.append(float(np.mean(window_ranges)))
            start = end
            
        return new_range
    
def uneven_median_lidar(lidar, num_points: int):
        ranges = lidar
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))  # Lidar only sees up to 4 meters
        new_range = []
        angles = []
        
        if len(ranges) != sum(window_size):
            raise Exception("Lidar length and window size do not match")
        
        if len(window_size) != num_points:
            raise Exception("Window size length and num_points do not match")
        
        fov_deg = 240
        angle_per_index = fov_deg / len(lidar)
        start = 0
        for window in window_size:
            end = start + window
            window_ranges = ranges[start:end]
            new_range.append(float(np.median(window_ranges)))

            window_center_idx = (start + end) // 2
            angle_deg = -fov_deg / 2 + angle_per_index * window_center_idx
            angles.append(np.radians(angle_deg))

            start = end
        return new_range, angles

def median_lidar(lidar, num_points: int):
    ranges = lidar
    ranges = np.nan_to_num(ranges, nan=float(
        10), posinf=float(10), neginf=float(10))

    # Apply median filter first to reduce spikes from nan values
    window_size = math.ceil(len(ranges)/num_points) #refer to line 78 in CarTrackEnvironment.py
    filtered_ranges = scipy.ndimage.median_filter(
        ranges, window_size, mode='nearest')

    new_range = []
    angle = 240/num_points
    iter = 240/len(filtered_ranges)
    num_ind = np.ceil(angle/iter)
    x = 1
    sum_val = filtered_ranges[0]

    while (x < len(filtered_ranges)):
        if (x % num_ind == 0):
            new_range.append(float(sum_val/num_ind))
            sum_val = 0
        sum_val += filtered_ranges[x]
        x += 1
    if (sum_val > 0):
        new_range.append(float(sum_val/(len(filtered_ranges) % num_ind)))

    fov_rad = np.radians(240)
    angle_edges = np.linspace(-fov_rad/2, fov_rad/2, num_points + 1)
    angles = (angle_edges[:-1] + angle_edges[1:]) / 2
    return new_range, angles

    
if __name__ == "__main__":
    csv_file = '/home/anyone/autonomous_f1tenth/temp_2.csv'
    plot_lidar_scan(csv_file)
