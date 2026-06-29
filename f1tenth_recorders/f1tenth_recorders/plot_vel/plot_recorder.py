import argparse
import os

import matplotlib.pyplot as plt


def read_recorded_data(file_name):
    times = []
    linear_velocities = []
    angular_velocities = []

    try:
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                # Parse the line to extract time, linear, and angular velocities
                parts = line.strip().split(",")
                time = float(parts[0].split("=")[1])
                linear = float(parts[1].split("=")[1])
                angular = float(parts[2].split("=")[1])
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
    plt.plot(times, linear_velocities, label="Linear Velocity (m/s)", color="blue")

    # Plot angular velocities
    plt.plot(
        times, angular_velocities, label="Angular Velocity (rad/s)", color="orange"
    )

    plt.title(f"Recorded Velocities from {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot recorded velocity data")
    parser.add_argument(
        "file_name",
        nargs="?",
        default=os.environ.get("VEL_RECORDER_FILE"),
        help="Path to the recorded velocity file.",
    )
    args = parser.parse_args()

    if not args.file_name:
        raise SystemExit(
            "Provide a velocity record file path or set VEL_RECORDER_FILE."
        )

    times, linear_velocities, angular_velocities = read_recorded_data(args.file_name)

    plot_data(times, linear_velocities, angular_velocities, args.file_name)


if __name__ == "__main__":
    main()
