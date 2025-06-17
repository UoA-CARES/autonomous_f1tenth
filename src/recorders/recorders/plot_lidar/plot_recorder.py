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
                position = line.split(":")[1].strip("()").split(", ")
                car_positions.append((float(position[0]), float(position[1])))
            elif line.startswith("\t("):
                # Extract wall points
                point = line.strip("()\t").split(", ")
                wall_points.append((float(point[0]), float(point[1])))

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
    file_path = ''
    plot_lidar_from_file(file_path)