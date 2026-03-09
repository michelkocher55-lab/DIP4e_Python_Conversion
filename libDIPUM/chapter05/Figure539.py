import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

# Parameters
NR = 256
# Theta = 0 : 0.5 : 179.5;
Theta = np.arange(0, 180, 0.5)

# Data
Rectangle = np.zeros((NR, NR))
r_start = int(NR / 4)
r_end = int(3 * NR / 4)
c_center = int(NR / 2)
c_start = c_center - 20
c_end = c_center + 20

Rectangle[r_start:r_end, c_start:c_end] = 1

# SheppLogan
base_phantom = shepp_logan_phantom()

# Resize to NR x NR
SheppLogan = resize(base_phantom, (NR, NR), anti_aliasing=True, preserve_range=True)

rt_rectangle = radon(Rectangle, theta=Theta, circle=False)
rt_shepp = radon(SheppLogan, theta=Theta, circle=False)

# Display
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Rectangle
axes[0, 0].imshow(Rectangle, cmap="gray")
axes[0, 0].set_title("A rectangle")
axes[0, 0].axis("off")

# 2. Sinogram Rectangle
# MATLAB: axis xy.
# Theta on x-axis? No.
# ylabel('theta'). xlabel('rho').
# MATLAB radon returns [rho x theta].
# MATLAB plotting: 'XData' [xp(1) xp(end)] -> rho range. 'YData' [Theta(end) Theta(1)].
# So Y is Theta?
# If MATLAB imshow shows it with YData as Theta, then Y axis is Theta.
# skimage radon returns (projection_size, len(theta)). Rows are rho, cols are theta.
# So x-axis is Theta (cols), y-axis is rho (rows).
# MATLAB code: RadonTransform.Rectangle = flipud (RadonTransform.Rectangle');
# MATLAB Transpose: (rho x theta)' -> (theta x rho).
# Flipud: flips theta direction?
# So in MATLAB result image, Rows are Theta, Cols are Rho.
# Python radon: Rows are Rho, Cols are Theta.
# To match MATLAB display (Y=Theta, X=Rho):
# We need (Theta, Rho) array -> Python (Rows=Theta, Cols=Rho).
# So we should transpose `radon` output.

sinogram_rect = rt_rectangle.T  # (theta, rho)
# flipud? Theta 0 to 180.
# MATLAB YData: [Theta(end), Theta(1)]. 179.5 down to 0? Or 0 down to 179.5?
# Usually sinograms are shown with 0 at top or bottom.
# Let's just transpose to get Theta on Y-axis.

axes[0, 1].imshow(sinogram_rect, cmap="gray", aspect="auto", origin="lower")
axes[0, 1].set_title("Sinogram (Rect)")
axes[0, 1].set_xlabel("rho")
axes[0, 1].set_ylabel("theta")

# 3. Shepp Logan
axes[1, 0].imshow(SheppLogan, cmap="gray")
axes[1, 0].set_title("Shepp Logan")
axes[1, 0].axis("off")

# 4. Sinogram Shepp
sinogram_shepp = rt_shepp.T
axes[1, 1].imshow(sinogram_shepp, cmap="gray", aspect="auto", origin="lower")
axes[1, 1].set_title("Sinogram (Shepp)")
axes[1, 1].set_xlabel("rho")
axes[1, 1].set_ylabel("theta")

plt.tight_layout()
plt.savefig("Figure539.png")
print("Saved Figure539.png")
plt.show()
