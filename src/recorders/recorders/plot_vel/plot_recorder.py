import matplotlib.pyplot as plt

def read_recorded_data(file_name):
    """
    Reads the recorded data from the specified file and normalizes the time axis.
    """
    times = []
    linear_velocities = []
    angular_velocities = []

    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Parse the line to extract time, linear, and angular velocities
                parts = line.strip().split(',')
                time = float(parts[0].split('=')[1])
                linear = float(parts[1].split('=')[1])
                angular = float(parts[2].split('=')[1])
                times.append(time)
                linear_velocities.append(linear)
                angular_velocities.append(angular)
        
        # Normalize the time axis
        start_time = times[0]
        times = [t - start_time for t in times]

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None, None, None

    return times, linear_velocities, angular_velocities

def plot_data(times, linear_velocities, angular_velocities, file_name):
    """
    Plots the linear and angular velocities against the time.
    """
    if not times or not linear_velocities or not angular_velocities:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot linear velocities
    plt.plot(times, linear_velocities, label='Linear Velocity (m/s)', color='blue')

    # Plot angular velocities
    plt.plot(times, angular_velocities, label='Angular Velocity (rad/s)', color='orange')

    plt.title(f"Recorded Velocities from {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_name = "/home/anyone/autonomous_f1tenth/src/recorders/recorders/plot_vel/Vel records/record_drive_2025-06-18 17_13_48.txt"

    times, linear_velocities, angular_velocities = read_recorded_data(file_name)

    plot_data(times, linear_velocities, angular_velocities, file_name)

if __name__ == '__main__':
    main()