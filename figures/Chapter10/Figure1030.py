import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line

print("Running Figure1030 (Hough Transform)...")

# Data
# f = zeros(101,101);
f = np.zeros((101, 101), dtype=float)

# f(1,1) = 1; f(101,1) = 1; f(1,101) = 1; f(101,101) = 1; f(51,51) = 1;
# MATLAB indices are 1-based (row, col).
# Python indices are 0-based.
# MATLAB (1,1) -> Python (0,0)
# MATLAB (101,1) -> Python (100,0)
# MATLAB (1,101) -> Python (0,100)
# MATLAB (101,101) -> Python (100,100)
# MATLAB (51,51) -> Python (50,50)

f[0, 0] = 1
f[100, 0] = 1
f[0, 100] = 1
f[100, 100] = 1
f[50, 50] = 1

# Hough transform
# [H,theta,rho]=hough(f);
# Skimage hough_line returns H, theta, distances
# Default theta is -pi/2 to pi/2

H, theta, d = hough_line(f)
print(f"H shape: {H.shape}")
print(f"Theta range: {np.degrees(theta.min())} to {np.degrees(theta.max())}")
print(f"Distance range: {d.min()} to {d.max()}")

# H = H > 0; %Convert to binary for display (in MATLAB code).
# MATLAB imshow scaling might display it binary-like or just scaled.
# The script says "H = H > 0; %Convert to binary for display."
H_binary = H > 0

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(f, cmap="gray")
axes[0].set_title("Original Image (5 points)")
# axes[0].axis('off')

# Display Hough Transform
# imshow (H,[],'Xdata',theta,'Ydata',rho,'InitialMagnification','fit');
# axis on, axis square, xlabel('\theta'), ylabel('\rho')

# We display H_binary or H? MATLAB script says H=H>0, then imshow(H). So binary.
# Use degree for x-axis

# Extent for imshow: [left, right, bottom, top]
# theta is radians. Convert to degrees for display usually.
theta_deg = np.degrees(theta)
extent = [theta_deg.min(), theta_deg.max(), d.min(), d.max()]

# Note: imshow default origin is 'upper'. MATLAB origin for plot axes implies y grows up usually?
# Actually for images ('imshow'), y grows down.
# But for 'rho', it might be from -diag to +diag.
# Let's adjust aspect

axes[1].imshow(H_binary, cmap="gray", extent=extent, aspect="auto", origin="lower")
axes[1].set_title("Hough Transform")
axes[1].set_xlabel("Theta (degrees)")
axes[1].set_ylabel("Rho (pixels)")

plt.tight_layout()
plt.savefig("Figure1030.png")
print("Saved Figure1030.png")
plt.show()
