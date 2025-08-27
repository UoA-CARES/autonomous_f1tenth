import matplotlib.pyplot as plt
import numpy as np

"""
For waypoints of simulated tracks
"""

# Filepaths to the waypoints files
waypoints_file = 'waypoints.txt'
original_file = 'original_waypoints.txt'

# Initialize lists to store x, y, and yaw values for both sets of waypoints
x_coords = []
y_coords = []
yaw_values = []

original_x_coords = []
original_y_coords = []
original_yaw_values = []

# Read the current waypoints file
with open(waypoints_file, 'r') as file:
    for line in file:
        # Extract x, y, and yaw values from lines containing "Waypoint"
        if "Waypoint" in line:
            parts = line.strip().split(',')
            x = float(parts[0].split('(')[1])  # Extract x value
            y = float(parts[1])               # Extract y value
            yaw = float(parts[2])             # Extract yaw value
            x_coords.append(x)
            y_coords.append(y)
            yaw_values.append(yaw)

# Read the original waypoints file
with open(original_file, 'r') as file:
    for line in file:
        # Extract x, y, and yaw values from lines containing "Waypoint"
        if "Waypoint" in line:
            parts = line.strip().split(',')
            x = float(parts[0].split('(')[1])  # Extract x value
            y = float(parts[1])               # Extract y value
            yaw = float(parts[2])             # Extract yaw value
            original_x_coords.append(x)
            original_y_coords.append(y)
            original_yaw_values.append(yaw)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot the current waypoints on the first subplot
axes[0].plot(x_coords, y_coords, marker='o', markersize=2, linestyle='-', color='blue', label='Current Track')
arrow_interval = 10
for i in range(0, len(x_coords), arrow_interval):
    x = x_coords[i]
    y = y_coords[i]
    yaw = yaw_values[i]
    dx = np.cos(yaw)  # Arrow direction based on yaw
    dy = np.sin(yaw)
    axes[0].quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.5, color='red', label='Yaw' if i == 0 else "")
axes[0].set_title('Current Track with Yaw Arrows')
axes[0].set_xlabel('X Coordinate')
axes[0].set_ylabel('Y Coordinate')
axes[0].axis('equal')  # Ensure equal scaling for x and y axes
axes[0].grid(True)
axes[0].legend()

# Plot the original waypoints on the second subplot
axes[1].plot(original_x_coords, original_y_coords, marker='o', markersize=2, linestyle='-', color='green', label='Original Track')
arrow_interval = 10
for i in range(0, len(original_x_coords), arrow_interval):
    x = original_x_coords[i]
    y = original_y_coords[i]
    yaw = original_yaw_values[i]
    dx = np.cos(yaw)  # Arrow direction based on yaw
    dy = np.sin(yaw)
    axes[1].quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.5, color='orange', label='Yaw' if i == 0 else "")
axes[1].set_title('Original Track with Yaw Arrows')
axes[1].set_xlabel('X Coordinate')
axes[1].set_ylabel('Y Coordinate')
axes[1].axis('equal')  # Ensure equal scaling for x and y axes
axes[1].grid(True)
axes[1].legend()

# Show the plots
plt.tight_layout()
plt.show()