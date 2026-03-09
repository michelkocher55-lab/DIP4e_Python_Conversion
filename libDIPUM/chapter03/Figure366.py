import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.triangmf import triangmf
from libDIPUM.trapezmf import trapezmf
from libDIPUM.sigmamf import sigmamf
from libDIPUM.smf import smf
from libDIPUM.bellmf import bellmf
from libDIPUM.truncgaussmf import truncgaussmf

print("Running Figure366 (Fuzzy Membership Functions)...")

z = np.linspace(0, 255, 500)

# 1. Triangle
# u.triangle = triangmf (z, 20, 70, 200);
u_triangle = triangmf(z, 20, 70, 200)

# 2. Trapezoid
# u.trapez = trapezmf (z, 20, 50, 200, 220);
u_trapez = trapezmf(z, 20, 50, 200, 220)

# 3. Sigma
# u.sigma = sigmamf (z, 30, 70);
u_sigma = sigmamf(z, 30, 70)

# 4. S-shape
# u.s = smf (z, 30, 226);
u_s = smf(z, 30, 226)

# 5. Bell
# u.bell = bellmf (z, 50, 100);
u_bell = bellmf(z, 50, 100)

# 6. Truncated Gaussian
# u.gauss = truncgaussmf (z, 50, 100, 20);
u_gauss = truncgaussmf(z, 50, 100, 20)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

plots = [
    (u_triangle, "Triangle"),
    (u_trapez, "Trapezoid"),
    (u_sigma, "Sigma"),
    (u_s, "S-shape"),
    (u_bell, "Bell"),
    (u_gauss, "Truncated Gaussian"),
]

for i, (data, title) in enumerate(plots):
    ax = axes[i]
    ax.plot(z, data)
    ax.set_title(title)
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1.05])
    # ax.axis('tight') # MATLAB axis tight

plt.tight_layout()
plt.savefig("Figure366.png")
print("Saved Figure366.png")
plt.show()
