"""Figure 12.57 - Gaussian octaves without white padded frames."""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from libDIPUM.gaussKernel4e import gaussKernel4e
from libDIPUM.data_path import dip_data


print("Running Figure1257...")

# Parameters
sigma_first_oct = np.sqrt(2.0) / 2.0
sigma_second_oct = 2.0 * sigma_first_oct
sigma_third_oct = 2.0 * sigma_second_oct
k = np.sqrt(2.0)
images_per_octave = 5

# Data
img_path = dip_data("building-600by600.tif")
f = plt.imread(img_path)
if f.ndim == 3:
    f = f[..., 0]
f = np.asarray(f)

nr, nc = f.shape

# First octave
size_kernel_first_oct = [5, 7, 9, 13, 17]
first_octave = np.zeros((nr, nc, images_per_octave), dtype=np.float64)
les_sigma_first_oct = np.zeros(images_per_octave, dtype=np.float64)

for i in range(images_per_octave):
    if i == 0:
        les_sigma_first_oct[i] = sigma_first_oct
    else:
        les_sigma_first_oct[i] = les_sigma_first_oct[i - 1] * k

    kernel = gaussKernel4e(size_kernel_first_oct[i], les_sigma_first_oct[i], 1)
    if i == 0:
        first_octave[:, :, i] = ndimage.convolve(f, kernel, mode="reflect")
    else:
        first_octave[:, :, i] = ndimage.convolve(
            first_octave[:, :, i - 1], kernel, mode="reflect"
        )

# Second octave
size_kernel_second_oct = [9, 13, 17, 25, 35]
second_base = f[::2, ::2]
second_octave = np.zeros(
    (second_base.shape[0], second_base.shape[1], images_per_octave), dtype=np.float64
)
les_sigma_second_oct = np.zeros(images_per_octave, dtype=np.float64)

for i in range(images_per_octave):
    if i == 0:
        les_sigma_second_oct[i] = sigma_second_oct
    else:
        les_sigma_second_oct[i] = les_sigma_second_oct[i - 1] * k

    kernel = gaussKernel4e(size_kernel_second_oct[i], les_sigma_second_oct[i], 1)
    if i == 0:
        second_octave[:, :, i] = ndimage.convolve(second_base, kernel, mode="reflect")
    else:
        second_octave[:, :, i] = ndimage.convolve(
            second_octave[:, :, i - 1], kernel, mode="reflect"
        )

# Third octave
size_kernel_third_oct = [17, 25, 35, 49, 67]
third_base = f[::4, ::4]
third_octave = np.zeros(
    (third_base.shape[0], third_base.shape[1], images_per_octave), dtype=np.float64
)
les_sigma_third_oct = np.zeros(images_per_octave, dtype=np.float64)

for i in range(images_per_octave):
    if i == 0:
        les_sigma_third_oct[i] = sigma_third_oct
    else:
        les_sigma_third_oct[i] = les_sigma_third_oct[i - 1] * k

    kernel = gaussKernel4e(size_kernel_third_oct[i], les_sigma_third_oct[i], 1)
    if i == 0:
        third_octave[:, :, i] = ndimage.convolve(third_base, kernel, mode="reflect")
    else:
        third_octave[:, :, i] = ndimage.convolve(
            third_octave[:, :, i - 1], kernel, mode="reflect"
        )


# Display helper: place image at top-left in full-size coordinates
# without padded white image data.
def show_octave(ax: plt.Axes, img: np.ndarray, nr_: int, nc_: int, title: str) -> None:
    """show_octave."""
    ax.imshow(
        img,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=(1, img.shape[1], img.shape[0], 1),
    )
    ax.set_xlim(1, nc_)
    ax.set_ylim(nr_, 1)
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_title(title)


fig, axes = plt.subplots(3, images_per_octave, figsize=(18, 10))

for i in range(images_per_octave):
    show_octave(
        axes[0, i],
        first_octave[:, :, i],
        nr,
        nc,
        f"Oct = 1, $\\sigma$ = {les_sigma_first_oct[i]:.3g}",
    )

for i in range(images_per_octave):
    show_octave(
        axes[1, i],
        second_octave[:, :, i],
        nr,
        nc,
        f"Oct = 2, $\\sigma$ = {les_sigma_second_oct[i]:.3g}",
    )

for i in range(images_per_octave):
    show_octave(
        axes[2, i],
        third_octave[:, :, i],
        nr,
        nc,
        f"Oct = 3, $\\sigma$ = {les_sigma_third_oct[i]:.3g}",
    )

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "Figure1257.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
