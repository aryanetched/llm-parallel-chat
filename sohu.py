import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x = 5 * np.sin(u) * np.cos(v)
y = 5 * np.sin(u) * np.sin(v)
z = 5 * np.cos(u)

# Plot the sphere
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

# Define the angle of rotation
theta = np.linspace(0, 2 * np.pi, 100)

# Create a 3D plot of the revolving sphere
for t in theta:
    sphere_x = 5 * np.sin(u) * np.cos(v + t)
    sphere_y = 5 * np.sin(u) * np.sin(v + t)
    sphere_z = 5 * np.cos(u)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, rstride=1, cstride=1, color='b', alpha=0.1)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-6, 6)
plt.show()
