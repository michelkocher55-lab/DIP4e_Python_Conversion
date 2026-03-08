"""Figure 12.66 - Match rotated and half-size corners against original building."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt

from libDIPUM.match import match
from libDIPUM.data_path import dip_data


print("Running Figure1266 (matching rotated and half-size corners)...")

img_base = dip_data('building-600by600.pgm')
img_rot_corner = dip_data('building-rot-corner.pgm')
img_half_corner = dip_data('building-halfsize-corner.pgm')

# Match rotated corner against original building
num1, match1 = match(img_base, img_rot_corner)

# Match half-size corner against original building
num2, match2 = match(img_base, img_half_corner)

# Display
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(match1)
axs[0].set_title(f"Matching, {num1}, match")
axs[0].axis("off")

axs[1].imshow(match2)
axs[1].set_title(f"Matching, {num2}, match")
axs[1].axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1266.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

plt.show()
