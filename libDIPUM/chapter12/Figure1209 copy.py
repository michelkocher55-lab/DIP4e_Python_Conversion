import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from im2minperpoly import im2minperpoly
from connectpoly import connectpoly
from bound2im import bound2im
from skimage.morphology import dilation, square
from bwboundaries import bwboundaries
from libDIPUM.data_path import dip_data


def Figure1209():
    """Figure1209."""
    print("Running Figure1209 (Minimum Perimeter Polygon)...")

    # 1. Data
    path = dip_data("mapleleaf.tif")
    if not os.path.exists(path):
        # Alternatives
        alts = [dip_data("mapleleaf.tif"), dip_data("mapleleaf.tif")]
        for a in alts:
            if os.path.exists(a):
                path = a
                break

    if not os.path.exists(path):
        print("mapleleaf.tif not found.")
        return

    B = imread(path)
    # B should be binary?
    # Ensure binary
    if B.ndim == 3:
        B = B[:, :, 0]
    B = B > 0  # Assume >0 is object

    LesCellSize = [4, 6, 8, 16, 32]
    M, N = B.shape

    # Boundaries
    boundaries = bwboundaries(B, conn=8)
    if boundaries:
        b = boundaries[0]
        # b is [(r,c)...]. bwboundaries returns start=end?
        # My impl appends start at end?
        # trace_boundary logic usually repeats start.
        # MATLAB bwboundaries also repeats start.
    else:
        b = np.array([])

    # bound2im with autoscale logic?
    # Logic in MATLAB: bIm = bound2im (b, M, N);
    # My bound2im creates empty image size M,N and puts b.
    bIm = bound2im(b, M, N)

    B2_list = []
    LesX_list = []

    for cellsize in LesCellSize:
        print(f"Processing Cell Size {cellsize}...")
        X, Y, R = im2minperpoly(B, cellsize)
        LesX_list.append(len(X))

        # ConnectPoly
        if len(X) > 0:
            b2 = connectpoly(X, Y)
            B2 = bound2im(b2, M, N)
        else:
            B2 = np.zeros_like(B)

        B2_list.append(B2)

    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ax = axes.ravel()

    # 1. Original
    ax[0].imshow(B, cmap="gray")
    ax[0].set_title(f"X, size = {B.shape}")

    # 2. Boundary
    ax[1].imshow(bIm, cmap="gray")
    ax[1].set_title(f"8 conn., N_Ver = {len(b)}")

    # Loops
    SE = square(4)  # mmsecross(4) ~ approx square? Or diamond?
    # mmsecross is plus shape.
    # Use cross structuring element.
    se_cross = np.array(
        [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=bool,
    )  # 4x4?
    # mmsecross(R): usually radius R.
    # Let's use simple square(4) for visibility.

    for i, cellsize in enumerate(LesCellSize):
        idx = i + 2
        if idx < 8:
            B2 = B2_list[i]
            # Dilate for visibility
            B2_dil = dilation(B2, SE)

            # mmshow(B, green=B2_dil, blue=B2_dil)
            # mmshow implementation only supports one mask currently?
            # My mmshow supports (f, mask, color).
            # MATLAB mmshow(f, mask1, mask2) overlays both.
            # I can overlay manually.

            # Plot
            plt.sca(ax[idx])
            # Base
            ax[idx].imshow(B, cmap="gray")
            # Overlay
            # Using alpha blending or contour?
            # My mmshow overlays mask.
            # Let's just overlay B2_dil in Red.

            # Create RGB
            rgb = np.dstack((B, B, B)).astype(float)
            # If B is bool, 0/1.

            # Overlay
            # Red channel for mask
            mask = B2_dil > 0
            if mask.any():
                rgb[mask] = [1, 0, 0]  # Red

            ax[idx].imshow(rgb)
            ax[idx].set_title(f"CS={cellsize}, N_Ver={LesX_list[i]}")
            ax[idx].axis("off")

    plt.tight_layout()
    plt.savefig("Figure1209.png")
    plt.show()


if __name__ == "__main__":
    Figure1209()
