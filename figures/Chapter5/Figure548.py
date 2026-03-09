import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from libDIPUM.fanbeam import fanbeam
from libDIPUM.ifanbeam import ifanbeam
from libDIP.intScaling4e import intScaling4e
from libDIPUM.data_path import dip_data

# Parameters
NR = 256
Diag = np.sqrt(2) * NR

# Rotation increments for the 4 cases
FanRotInc = [1.0, 0.5, 0.25, 0.125]
# Corresponding Sensor Spacings (based on MATLAB code usage)
# F1: Default (likely 1, 1).
# F2: 0.5, 0.5
# F3: 0.25, 0.25
# F4: 0.125, 0.125
FanSensorSpacing = [1.0, 0.5, 0.25, 0.125]

# Data
img_name = dip_data("vertical_rectangle.tif")

g = imread(img_name)

g = g.astype(float)
if g.max() > 1:
    g /= 255.0

M, N = g.shape
# D: Distance source to center
d_source = np.sqrt(M**2 + N**2) + 10

results = []

for i in range(4):
    rot_inc = FanRotInc[i]
    sensor_spacing = FanSensorSpacing[i]

    print(f"Case {i + 1}: FanRotInc={rot_inc}, FanSensorSpacing={sensor_spacing}...")

    # Fanbeam
    # Note: D corresponds to 'd' in MATLAB code
    F, gamma, beta = fanbeam(
        g, D=d_source, FanRotationIncrement=rot_inc, FanSensorSpacing=sensor_spacing
    )

    # Inverse Fanbeam
    # OutputSize=600 in MATLAB
    g_recon = ifanbeam(
        F,
        D=d_source,
        FanRotationIncrement=rot_inc,
        FanSensorSpacing=sensor_spacing,
        filter="Hamming",
        OutputSize=600,
    )

    # Scale
    g_scaled = intScaling4e(g_recon, "full")
    results.append(g_scaled)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(results[i], cmap="gray")
    ax.set_title(f"Inc: {FanRotInc[i]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("Figure548.png")
print("Saved Figure548.png")
plt.show()
