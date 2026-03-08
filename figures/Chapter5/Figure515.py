
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libDIPUM.cnotch import cnotch


def plot_mesh(ax, H, title):
    # Shift for display
    H_disp = np.fft.fftshift(H)
    rows, cols = H_disp.shape

    # Meshgrid for plotting
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)

    ax.plot_wireframe(X, Y, H_disp, color='black', rstride=1, cstride=1, linewidth=0.7)

    # Setup axis
    ax.set_title(title)
    ax.set_axis_off()

    # View point? Matlab default.
    ax.view_init(elev=30, azim=-60)  # Default-ish


# Transfer functions
# Hideal = double(cnotch('ideal','reject',40,40,[28 11],3));
Hideal = cnotch('ideal', 'reject', 40, 40, [28, 11], 3)

# Hgauss = double(cnotch('gaussian','reject',40,40,[28 11],3));
Hgauss = cnotch('gaussian', 'reject', 40, 40, [28, 11], 3)

# Hbw = double(cnotch('butterworth','reject',40,40,[28 11],3,2));
Hbw = cnotch('butterworth', 'reject', 40, 40, [28, 11], 3, n=2)

# Display
fig = plt.figure(figsize=(15, 6))

# Plot 1
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
plot_mesh(ax1, Hideal, 'Ideal Notch')

# Plot 2
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
plot_mesh(ax2, Hgauss, 'Gaussian Notch')

# Plot 3
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
plot_mesh(ax3, Hbw, 'Butterworth Notch')

plt.tight_layout()
plt.savefig('Figure515.png')
print("Saved Figure515.png")
plt.show()

