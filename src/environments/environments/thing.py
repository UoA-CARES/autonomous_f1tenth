import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 500)
y = 1 - 1 / (1 + np.exp(15 * (x - 0.3)))
x2 = np.linspace(0, 1, 500)
y2 = 1 - 1 / (1 + np.exp(15 * (x - 0.5)))

plt.plot(x, y)
plt.plot(x2, y2) 
plt.xlabel('Δω (rad/s)')
plt.ylabel('Penalty')
plt.grid(True)
plt.legend(['original', 'modified'])
plt.show()