# Load a CSV file, the first column is the time and the rest 683 are scan values
# Have a slider that allows you to select a time and plot the scan values at that time
# The scan covers 240 degrees starting from the left side, plot the scans as xy coordinates assuming the car is at 0, 0

import pandas as pd
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

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
        
        angles = np.linspace(-120 * np.pi / 180, 120 * np.pi / 180, len(scan_values))
        
        # Separate valid and NaN values
        valid_indices = ~np.isnan(scan_values)
        invalid_indices = np.isnan(scan_values)
        
        valid_scan_values = scan_values[valid_indices]
        valid_angles = angles[valid_indices]
        
        invalid_angles = angles[invalid_indices]
        
        # Compute valid x, y coordinates
        x = valid_scan_values * np.cos(valid_angles)
        y = valid_scan_values * np.sin(valid_angles)
        
        # Update valid points
        line.set_xdata(x)
        line.set_ydata(y)
        
        # Clear previous red lines
        [line.remove() for line in ax.lines[1:]]
        
        # Draw red lines for NaN values
        for angle in invalid_angles:
            ax.plot([0, 10 * np.cos(angle)], [0, 10 * np.sin(angle)], 'r-')
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    
    
if __name__ == "__main__":
    csv_file = '/home/anyone/autonomous_f1tenth/src/recorders/recorders/plot_lidar/lidar_2025-08-18_13_11_43.csv'
    plot_lidar_scan(csv_file)