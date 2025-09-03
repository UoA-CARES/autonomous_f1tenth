import pandas as pd
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.ndimage

"""
This script plots the lidar reading of the car based on the logged csv file ('lidar_*.csv').
Lidar distances are converted into x and y coordinates and plotted in a top-down view to represent the obstacle around the car.
The blue dots are the original (valid) readings, the red lines are NaN values and the yelloe dots are processed values.
"""

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

    def update(val):
        current_time = slider.val
        
        idx = (time - current_time).abs().idxmin()
        
        scan_values = scans[idx]
        processed_scan_values = avg_lidar(scan_values, 10)
        processed_scan_values_2 = median_lidar(scan_values, 10)
        processed_scan_values_3 = uneven_avg_lidar(scan_values, 10)
        processed_scan_values_4 = uneven_median_lidar(scan_values, 10)
        
        angles = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(scan_values))
        processed_angles = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(processed_scan_values))
        processed_angles_2 = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(processed_scan_values_2))
        processed_angles_3 = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(processed_scan_values_3))
        processed_angles_4 = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(processed_scan_values_4))
        
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
        
        processed_x_3 = processed_scan_values_3 * np.cos(processed_angles_3)
        processed_y_3 = processed_scan_values_3 * np.sin(processed_angles_3)
        
        processed_x_4 = processed_scan_values_4 * np.cos(processed_angles_4)
        processed_y_4 = processed_scan_values_4 * np.sin(processed_angles_4)
        
        max_x = np.max(np.abs(x)) + 1
        max_y = np.max(np.abs(y)) + 1
        max_range = max(max_x, max_y)
        ax.set_xlim(min(0, -max_range), max_range)
        ax.set_ylim(min(0, -max_range), max_range)
        
        line.set_xdata(x)
        line.set_ydata(y)
        
        # Clear previous
        [line.remove() for line in ax.lines[1:]]
        
        # Draw red lines for NaN values
        for angle in invalid_angles:
            ax.plot([0, 10 * np.cos(angle)], [0, 10 * np.sin(angle)], 'r-')
        
        avg_plot, = ax.plot(processed_x, processed_y, 'mo', label='Average')         # avg
        median_plot, = ax.plot(processed_x_2, processed_y_2, 'co', label='Median')   # median
        uneven_avg_plot, = ax.plot(processed_x_3, processed_y_3, 'ko', label='Uneven Average')  # uneven avg
        uneven_median_plot, = ax.plot(processed_x_4, processed_y_4, 'yo', label='Uneven Median')  # uneven median
        
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
        
        return np.array(averaged_lidar)
    
def uneven_avg_lidar(lidar, num_points: int):
        ranges = lidar
        ranges = np.nan_to_num(ranges, nan=float(10), posinf=float(10), neginf=float(10))  # Lidar only sees up to 4 meters
        new_range = []
        
        window_size = [121, 70, 60 ,50, 40, 40, 50, 60, 70, 122]
        
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
        
        window_size = [121, 70, 60 ,50, 40, 40, 50, 60, 70, 122]
        
        if len(ranges) != sum(window_size):
            raise Exception("Lidar length and window size do not match")
        
        if len(window_size) != num_points:
            raise Exception("Window size length and num_points do not match")
        
        start = 0
        for window in window_size:
            end = start + window
            window_ranges = ranges[start:end]
            new_range.append(float(np.median(window_ranges)))
            start = end
            
        return new_range

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

    return new_range
    
if __name__ == "__main__":
    csv_file = '/home/anyone/autonomous_f1tenth/recordings/Sep3rd/lidar_records/staged_training_new.csv'
    plot_lidar_scan(csv_file)
