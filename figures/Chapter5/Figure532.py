import numpy as np
import matplotlib.pyplot as plt
from libDIPUM.lpfilter import lpfilter
from libDIP.imRecon4e import imRecon4e

H = lpfilter("ideal", 480, 480, 40)
f = np.fft.fftshift(H)

# Horizontal back projection
gh = imRecon4e(f, 90)

# Vertical back projection
gv = imRecon4e(f, 0)

# Now get a horizontal and a vertical projection.
ghv = imRecon4e(f, [90, 0])

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Original
axes[0, 0].imshow(f, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

# 2. Horizontal BackProj
axes[0, 1].imshow(gh, cmap="gray")
axes[0, 1].set_title("BackProj (90)")
axes[0, 1].axis("off")

# 3. Vertical BackProj
axes[1, 0].imshow(gv, cmap="gray")
axes[1, 0].set_title("BackProj (0)")
axes[1, 0].axis("off")

# 4. Both
axes[1, 1].imshow(ghv, cmap="gray")
axes[1, 1].set_title("BackProj (90, 0)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("Figure532.png")
print("Saved Figure532.png")
plt.show()
