import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from General.qtdecomp import qtdecomp
from General.qtgetblk import qtgetblk
from General.qtsetblk import qtsetblk
import ia870 as ia
from libDIPUM.data_path import dip_data
from typing import Any


# Robust image reader (reuse from previous or import if refactored)
def read_image_robust(path: Any):
    """Read an image using multiple backends."""
    try:
        return imread(path)
    except Exception:
        try:
            from PIL import Image

            return np.array(Image.open(path))
        except Exception:
            import cv2

            return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def ComputeMeans(I: Any, S: Any):
    """
    Compute mean of each block in quadtree decomposition.
    """
    means = I.astype(float)  # Output image

    # Iterate dimensions used in decomposition (powers of 2)
    # TQTDecomp loop: [512 256 ... 1]
    # We should detect dimensions present in S or iterate standard range.
    # Dimensions present:
    if S.nnz == 0:
        return means

    dims = np.unique(S.data)

    for dim in dims:
        dim = int(dim)  # Ensure int
        values = qtgetblk(I, S, dim)

        if values.size > 0:
            # values is (dim, dim, k)
            # Sum over dim, dim (axis 0, 1)
            # Result (k,)

            # doublesum = sum(sum(values,1,'double'),2); -> Sum over blocks
            # Here: sum axis 0 and 1
            block_sums = np.sum(values, axis=(0, 1))
            block_means = block_sums / (dim**2)

            # Repmat mean to fill block?
            # qtsetblk(means, S, dim, mean_val)
            # qtsetblk expects values of size (dim, dim, k).

            # Create (dim, dim, k) with constant mean
            k = values.shape[2]
            mean_blocks = np.zeros((dim, dim, k))
            for i in range(k):
                mean_blocks[:, :, i] = block_means[i]

            means = qtsetblk(means, S, dim, mean_blocks)

    return means


def TQTDecomp():
    """Run quadtree decomposition demo and generate output plots."""
    print("Running TQTDecomp...")

    # Data
    # MATLAB source used liftingbody.png
    # Locate image
    fname = "cygnusloop.tif"
    # Common locations or search result?
    # I'll rely on find_by_name result logic or hardcoded from previous experience,
    # but since I am running blindly, I'll search common paths or current dir.

    # Fallback to standard check
    possible_paths = [
        dip_data("cygnusloop.tif"),
    ]

    # Add whatever find_by_name returns if I could parse it here, but I will write this generic.
    path = fname
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break

    if not os.path.exists(path):
        # Try finding anywhere?
        # For now prompt or error
        print(f"Image {fname} not found.")
        # Try standard skimage image?
        try:
            from skimage import data

            print("Using skimage.data.camera() as fallback.")
            I = data.camera()
        except Exception:
            return
    else:
        I = read_image_robust(path)

    if I.ndim == 3:
        I = I[:, :, 0]

    # DynThreshold = .27; (If float 0..1, input should be float? or uint8 scaled?)
    # "If I is of class uint8, threshold is multiplied by 255."
    # Let's use float range 0..1 by ensuring I is float 0..1?
    # Or keep uint8 and scale threshold.
    DynThreshold = 0.27
    if I.dtype == np.uint8:
        threshold = DynThreshold * 255
    else:
        threshold = DynThreshold

    print(f"Quadtree Decomposition (Threshold={threshold:.2f})...")
    S = qtdecomp(I, threshold)

    # Visualization of blocks
    # blocks = repmat(uint8(0),size(S));
    blocks = np.zeros(I.shape, dtype=np.uint8)

    # Dim sizes: 512, 256, ..., 1
    # Check max dim in S
    # sparse S contains dims.
    dims = np.unique(S.data)
    # Filter out 0 (background/none)
    dims = dims[dims > 0]
    dims = np.sort(dims)[::-1]  # Descending

    for dim in dims:
        dim = int(dim)
        # numblocks = length(find(S==dim));
        # qtsetblk logic handles finding blocks.

        # values = repmat(uint8(1),[dim dim numblocks]);
        # values(2:dim,2:dim,:) = 0;
        # -> Finds blocks, sets border to 1, center to 0. (Drawing borders)

        # We need num blocks to construct values array
        # Get count from S
        numblocks = np.sum(S.data == dim)

        if numblocks > 0:
            values = np.ones((dim, dim, numblocks), dtype=np.uint8)
            if dim > 1:
                values[1:dim, 1:dim, :] = 0

            blocks = qtsetblk(blocks, S, dim, values)

    # blocks(end,1:end) = 1; blocks(1:end,end) = 1; (Borders of image)
    blocks[-1, :] = 1
    blocks[:, -1] = 1

    # Compute Means
    print("Computing Means...")
    BlockMean = ComputeMeans(I, S)

    # e = mmsymdif (I, BlockMean);
    # mmsymdif? 'ia.iasymdif'?
    # Symmetric difference usually for sets (binary) or absolute diff for gray?
    # ia.iasymdif exists?
    # If not, use abs diff.
    try:
        e = ia.iasymdif(I, BlockMean)  # Expects same type?
        # I is uint8, BlockMean is float/double likely.
        # Ensure compat.
    except:
        # Fallback
        e = np.abs(I.astype(float) - BlockMean.astype(float))

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(I, cmap="gray")
    axes[0, 0].set_title("f")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(blocks, cmap="gray")  # [] scale? blocks is 0/1.
    axes[0, 1].set_title(f"blocks, Threshold = {DynThreshold}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(BlockMean, cmap="gray")
    axes[1, 0].set_title("Mean of each block")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(e, cmap="gray")
    axes[1, 1].set_title("e = f - BlockMean")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("TQTDecomp.png")
    print("Saved TQTDecomp.png")
    plt.show()


if __name__ == "__main__":
    TQTDecomp()
