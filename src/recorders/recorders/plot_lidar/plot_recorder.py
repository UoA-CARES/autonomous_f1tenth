import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List


class RecordedData:
    def __init__(self, car_position):
        self.wall_points = []
        self.car_position = car_position

    def add_wall_point(self, point):
        self.wall_points.append(point)

    def get_wall_points(self):
        return self.wall_points

    def get_car_position(self):
        return self.car_position


def read_data(file_path):
    data_list: List[RecordedData] = []
    data = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if data:
                    data_list.append(data)
                    data = None
            elif line.startswith("Car Position:"):
                position = line.split(":")[1].strip("() \n")
                x_str, y_str = [p.strip() for p in position.split(",")]
                data = RecordedData((float(x_str), float(y_str)))
            elif line.startswith("("):
                line = line.strip("(), \t\n")
                x_str, y_str = [p.strip() for p in line.split(",")]
                if data:
                    data.add_wall_point((float(x_str), float(y_str)))
        
        # Add the last data point if it exists
        if data:
            data_list.append(data)
    
    return data_list


def plot(file_path):
    data_list = read_data(file_path)
    
    if not data_list:
        print("No data found in the file!")
        return
    
    print(f"Loaded {len(data_list)} data points")
    
    fig, ax = plt.subplots(figsize=(10, 6))
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
    slider = Slider(ax_slider, 'Index', 0, len(
        data_list) - 1, valinit=0, valstep=1)

    def update(val):
        index = int(slider.val)
        data_sublist = data_list[:index]

        # Flatten wall points into x and y coordinates
        wall_points = [point for data in data_sublist for point in data.get_wall_points()]
        if wall_points:
            wall_x, wall_y = zip(*wall_points)
        else:
            wall_x, wall_y = [], []  # Handle empty wall points

        # Extract car positions
        car_positions = [data.get_car_position() for data in data_sublist]
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
    update(0)
    plt.show()


if __name__ == "__main__":
    file_path = '/home/anyone/autonomous_f1tenth/src/recorders/recorders/plot_lidar/record_lidar_2025-06-30 14_06_44.txt'
    plot(file_path)
