
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.feature import canny
from libDIPUM.hough import hough
from libDIPUM.houghpeaks import houghpeaks
from libDIPUM.houghlines import houghlines
from libDIPUM.data_path import dip_data

print("Running Figure1031 (Hough Line Detection with custom utils)...")

# Parameters
NPeaks = 1
RatioHoughThreshold = 0.3
FillGap = 50
MinLength = 7

# Data
img_path = dip_data('Fig1034(a)(marion_airport).tif')
I = img_as_float(imread(img_path))

# Edge detection
BW = canny(I, sigma=1.0)

# Hough transform
H, theta, rho = hough(BW)

# Find peaks
threshold = np.ceil(RatioHoughThreshold * H.max())

# Python houghpeaks returns r, c indices (rho_idx, theta_idx)
r_idx, c_idx = houghpeaks(H, numpeaks=NPeaks, threshold=threshold)

print(f"Detected peaks (theta, rho indices):")
if len(r_idx) > 0:
    for r, c in zip(r_idx, c_idx):
        print(f"  Theta: {theta[c]} deg, Rho: {rho[r]}")

# Convert peaks to list of [rho_idx, theta_idx] for houghlines
peaks = np.column_stack((r_idx, c_idx))

# Get lines
lines = houghlines(BW, theta, rho, peaks, fill_gap=FillGap, min_length=MinLength)

print(f"Found {len(lines)} line segments.")

# Display
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

axes[0].imshow(I, cmap='gray')
axes[0].set_title('Original image')
axes[0].axis('off')

axes[1].imshow(BW, cmap='gray')
axes[1].set_title('Edge detection')
axes[1].axis('off')

axes[2].imshow(H, cmap='gray', extent=[theta[0], theta[-1], rho[0], rho[-1]], aspect='auto', origin='lower')
axes[2].set_title(f'Hough Peaks ({len(peaks)})')
axes[2].set_xlabel('Theta (deg)')
axes[2].set_ylabel('Rho')

# Plot peaks
for r, c in zip(r_idx, c_idx):
    axes[2].plot(theta[c], rho[r], 's', color='white', markeredgecolor='white', markersize=5)

axes[3].imshow(I, cmap='gray')
axes[3].set_title('Overlay')
axes[3].axis('off')

# Overlay lines
for line in lines:
    p1 = line['point1']
    p2 = line['point2']
    # p1 is (row, col) -> (y, x) for plot
    axes[3].plot([p1[1], p2[1]], [p1[0], p2[0]], linewidth=2, color='green')
    axes[3].plot(p1[1], p1[0], 'x', linewidth=2, color='yellow')
    axes[3].plot(p2[1], p2[0], 'x', linewidth=2, color='red')

plt.tight_layout()
plt.savefig('Figure1031.png')
print("Saved Figure1031.png")
plt.show()