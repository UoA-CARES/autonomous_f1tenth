import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
path = "/home/anyone/training_logs/new_multi_track/123/data/eval.csv"

# Load and clean CSV
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

# Ensure proper data types and drop invalid rows
df = df.dropna(subset=["total_steps", "traveled distance", "episode"])
df["traveled distance"] = pd.to_numeric(df["traveled distance"], errors='coerce')
df = df.dropna(subset=["traveled distance"])

# Sort by total_steps (optional but nice)
df = df.sort_values("total_steps")

# Compute the average traveled distance for each total_steps value (not cumulative)
average_per_step = df.groupby("total_steps")["traveled distance"].mean().reset_index()

# Plot per-episode traveled distance
plt.figure(figsize=(10, 6))
for ep, group in df.groupby("episode"):
    plt.plot(group["total_steps"], group["traveled distance"], label=f"Episode {ep}")

# Plot the average traveled distance at each step
plt.plot(
    average_per_step["total_steps"],
    average_per_step["traveled distance"],
    label="Average of All Distances at Same Step",
    color="black",
    linewidth=2,
    linestyle="--"
)

# Labels, legend, layout
plt.xlabel("Total Steps")
plt.ylabel("Traveled Distance (m)")
plt.title("Per-Episode and Average Distance per Total Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
