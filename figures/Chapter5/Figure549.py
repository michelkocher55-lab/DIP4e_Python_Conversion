import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from libDIPUM.fanbeam import fanbeam
from libDIPUM.ifanbeam import ifanbeam
from libDIP.intScaling4e import intScaling4e

# Parameters
# f = phantom(600);
NR = 600
base_phantom = shepp_logan_phantom()
f = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

# d = sqrt(size(f,1)^2 + size(f,2)^2) + 10;
M, N = f.shape
d_source = np.sqrt(M**2 + N**2) + 10

# Fan Parameters
FanRotInc = [1.0, 0.5, 0.25, 0.125]
FanSensorSpacing = [1.0, 0.5, 0.25, 0.125]

results = []

for i in range(4):
    rot_inc = FanRotInc[i]
    sensor_spacing = FanSensorSpacing[i]

    print(f"Case {i + 1}: FanRotInc={rot_inc}, FanSensorSpacing={sensor_spacing}...")

    # Fanbeam
    F, gamma, beta = fanbeam(
        f, D=d_source, FanRotationIncrement=rot_inc, FanSensorSpacing=sensor_spacing
    )

    # Inverse Fanbeam
    # OutputSize=600
    g_recon = ifanbeam(
        F,
        D=d_source,
        FanRotationIncrement=rot_inc,
        FanSensorSpacing=sensor_spacing,
        filter="Hamming",
        OutputSize=NR,
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
plt.savefig("Figure549.png")
print("Saved Figure549.png")
plt.show()
