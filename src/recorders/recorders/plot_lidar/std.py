import os
import numpy as np

"""Finds the standard deviation of each lidar reading when the car is standing still."""

def calculate_std_for_elements(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    wall_points_sets = []
    current_wall_points = []
    for line in lines:
        line = line.strip()
        if line.startswith("Car Position:"):
            if line == "Car Position: (0.00, 0.00)" and current_wall_points:
                wall_points_sets.append(current_wall_points)
            current_wall_points = []
        elif line.startswith("(") and "," in line:
            # Parse wall point coordinates
            point = tuple(map(float, line.strip("()").split(", ")))
            current_wall_points.append(point)
    if current_wall_points:  # Add the last set if it exists
        wall_points_sets.append(current_wall_points)
    
    # Transpose wall points to group by index
    max_length = max(len(points) for points in wall_points_sets)
    grouped_points = [[] for _ in range(max_length)]
    
    for wall_points in wall_points_sets:
        for i, point in enumerate(wall_points):
            grouped_points[i].append(point)
    
    # Calculate standard deviation for each group
    std_results = []
    for group in grouped_points:
        x_coords = [point[0] for point in group]
        y_coords = [point[1] for point in group]
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        std_results.append((x_std, y_std))
    
    return std_results

def process_all_files(directory):
    std_results_all_files = []
    for filename in os.listdir(directory):
        if filename.startswith("record_lidar"):
            file_path = os.path.join(directory, filename)
            std_results = calculate_std_for_elements(file_path)
            std_results_all_files.append(std_results)
    
    # Calculate average standard deviation across all files
    avg_std_x = []
    avg_std_y = []
    for std_results in std_results_all_files:
        avg_std_x.append(np.mean([x_std for x_std, _ in std_results]))
        avg_std_y.append(np.mean([y_std for _, y_std in std_results]))
    
    overall_avg_std_x = np.mean(avg_std_x)
    overall_avg_std_y = np.mean(avg_std_y)
    
    return overall_avg_std_x, overall_avg_std_y

# Example usage
directory = "/home/anyone/autonomous_f1tenth/src/recorders/recorders/plot_lidar/Lidar records"
avg_std_x, avg_std_y = process_all_files(directory)

print(f"Average X Standard Deviation across all files: {avg_std_x:.2f}")
print(f"Average Y Standard Deviation across all files: {avg_std_y:.2f}")