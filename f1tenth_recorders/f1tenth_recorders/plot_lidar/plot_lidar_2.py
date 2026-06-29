import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = '/media/anyone/0BD1-8D90/Sep24th/recordings/lidar_records/lidar_2025-09-23_19_48_52.csv'
data = pd.read_csv(csv_file)
scan = data.drop(columns=['time']).values[0]  # first scan

indices = np.arange(len(scan))
valid = ~np.isnan(scan)

plt.plot(indices[valid], scan[valid], label='Valid LIDAR readings')
plt.xlabel('Reading index')
plt.ylabel('Distance (m)')
plt.title('LIDAR Scan (valid readings only)')
plt.legend()
plt.grid(True)
plt.show()