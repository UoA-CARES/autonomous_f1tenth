import matplotlib.pyplot as plt

def plot_lidar_from_file(file_path):
    wall_points = []
    car_positions = []

    # Read the file and extract data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Car Position:"):
                # Extract car position
                position = line.split(":")[1].strip("() \n")
                x_str, y_str = [p.strip() for p in position.split(",")]
                car_positions.append((float(x_str), float(y_str)))
            elif line.startswith("\t("):
                # Extract wall points
                line = line.strip("(), \t\n")
                x_str, y_str = [p.strip() for p in line.split(",")]
                wall_points.append((float(x_str), float(y_str)))


    # Plot the data
    plt.figure(figsize=(10, 6))
    if wall_points:
        plt.plot(*zip(*wall_points), 'o', markersize=1, label="Walls")
    if car_positions:
        plt.plot(*zip(*car_positions), 'ro', markersize=2, label="Car Position")
    plt.title("Track Walls and Car Position - Top-Down View")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    file_path = '/home/anyone/new_repo/autonomous_f1tenth/src/recorders/recorders/plot_lidar/record_lidar_1970-01-01 13:40:55.txt'
    plot_lidar_from_file(file_path)