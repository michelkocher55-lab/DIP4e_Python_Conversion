import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Parameters
low = -0.5
high = 0.5
np_points = 40
K = 10
C = 0.6
r = 0.2

# Function phi (centered so that the bowl intersects near the plane center)
x, y = np.meshgrid(np.linspace(low, high, np_points), np.linspace(low, high, np_points))
phi = K * (x ** 2 + y ** 2 - r ** 2)

# Display
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Mesh-style surface (lines only, like MATLAB mesh)
ax.plot_wireframe(
    x, y, phi,
    rstride=1, cstride=1,
    color='black',
    linewidth=0.3
)

# Plane
plane = 0 * x + C
ax.plot_surface(
    x, y, plane,
    rstride=1, cstride=1,
    facecolor=(0.5, 0.5, 0.5),
    edgecolor='none',
    linewidth=0.0,
    antialiased=False,
    shade=False,
    alpha=0.9
)

# Intersection of phi and plane as a 3D curve at z = C:
# K*(x^2 + y^2 - r^2) = C  ->  x^2 + y^2 = r^2 + C/K
radius_sq = r ** 2 + C / K
if radius_sq > 0:
    th = np.linspace(0, 2 * np.pi, 600)
    rr = np.sqrt(radius_sq)
    xc = rr * np.cos(th)
    yc = rr * np.sin(th)
    zc = np.full_like(th, C)
    ax.plot(xc, yc, zc, color='black', linewidth=2.0)

ax.grid(False)
ax.view_init(elev=8, azim=35)
plt.savefig('Figure1114.png')
plt.show()
