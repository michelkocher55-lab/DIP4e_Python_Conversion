import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from libDIP.imRecon4e import imRecon4e

# Parameters
NR = 256
Center = NR / 2  # 128.0

x = np.arange(1, NR + 1)
y = np.arange(1, NR + 1)
Col, Row = np.meshgrid(x, y)

Col = Col - Center
Row = Row - Center

Radius = np.sqrt(Col**2 + Row**2)
X = Radius < 30

# Angles
Angles = 5.625 * np.arange(32)

# Reconstructions
print("Computing Reconstructions...")

# 1. BackProj (0)
print("- BackProj (0)")
recon_0 = imRecon4e(X, [0])

# 2. BackProj (0, 45)
print("- BackProj (0, 45)")
recon_0_45 = imRecon4e(X, [0, 45])

# 3. BackProj (0, 45, 90)
print("- BackProj (0, 45, 90)")
recon_0_45_90 = imRecon4e(X, [0, 45, 90])

# 4. BackProj (0, 45, 90, 135)
print("- BackProj (0, 45, 90, 135)")
recon_0_45_90_135 = imRecon4e(X, [0, 45, 90, 135])

# 5. BackProj (31 angles)
print(f"- BackProj ({len(Angles)} angles)")
recon_angles = imRecon4e(X, Angles)

# Display
print("Displaying results. Please close plot to continue.")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1. X
axes[0].imshow(X, cmap='gray')
axes[0].set_title('X')
axes[0].axis('off')

# 2. BackProj (0)
axes[1].imshow(recon_0, cmap='gray')
axes[1].set_title('BackProj (0)')
axes[1].axis('off')

# 3. BackProj (0, 45)
axes[2].imshow(recon_0_45, cmap='gray')
axes[2].set_title('BackProj (0, 45)')
axes[2].axis('off')

# 4. BackProj (0, 45, 90)
axes[3].imshow(recon_0_45_90, cmap='gray')
axes[3].set_title('BackProj (0, 45, 90)')
axes[3].axis('off')

# 5. BackProj (0, 45, 90, 135)
axes[4].imshow(recon_0_45_90_135, cmap='gray')
axes[4].set_title('BackProj (0, 45, 90, 135)')
axes[4].axis('off')

# 6. BackProj (31 angles) -> 32 actually
axes[5].imshow(recon_angles, cmap='gray')
axes[5].set_title('BackProj (31 angles)') # Keeping MATLAB title
axes[5].axis('off')

plt.tight_layout()

# Save to file
filename = 'Figure533.png'
plt.savefig(filename)
print(f"Saved figure to {filename}")

plt.show()
