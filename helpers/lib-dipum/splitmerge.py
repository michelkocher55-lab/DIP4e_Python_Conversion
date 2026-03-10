from typing import Any
import numpy as np
from skimage.measure import label
from skimage.morphology import reconstruction
from helpers.qtdecomp import qtdecomp
from helpers.qtgetblk import qtgetblk


def splitmerge(f: Any, mindim: Any, predicate: Any):
    """
    Segment an image using a split-and-merge algorithm.

    Parameters:
    f : ndarray
        Input image.
    mindim : int
        Minimum dimension of quadtree regions.
    predicate : callable
        Function `flag = predicate(region)` returning True if region satisfies criteria.

    Returns:
    g : ndarray
        Labeled image.
    """

    # Pad image to nearest power of 2 square
    M, N = f.shape
    # paddedsize utility usually calculates dim for FFT (P,Q).
    # Here we need a square power of 2 large enough to hold M, N.
    # Standard: 2^nextpow2(max(M,N))
    dim = int(2 ** np.ceil(np.log2(max(M, N))))

    # Pad to (dim, dim)
    # padarray in MATLAB adds 'post'.
    pad_rows = dim - M
    pad_cols = dim - N

    fp = np.pad(f, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

    # Wrapper for qtdecomp predicate
    # qtdecomp in MATLAB usually takes a threshold or a function.
    # splitmerge.m defines a split_test function that returns TRUE if ANY block should be split.
    # But my qtdecomp expects `predicate(block)` -> True/False (Split/Keep).
    # splitmerge.m's split_test logic:
    # "Perform split test on each block. If predicate(fun) is TRUE, split."
    # Wait, splitmerge.m logic: "returns TRUE for blocks that SHOULD BE SPLIT".
    # And it splits if: size > mindim AND fun(region) is TRUE.
    # Wait, check splitmerge.m:
    # function v = split_test(B, mindim, fun)
    #    if size <= mindim: false
    #    flag = fun(quadregion)
    #    if flag: v = true
    # So if predicate(region) is True, we SPLIT.
    # This implies the predicate defines NON-HOMOGENEITY (e.g. "std > 10").
    # If std > 10 (True), we Split.
    # If std <= 10 (False), we don't split (Merge/Keep).

    def qt_predicate(block: Any):
        """qt_predicate."""
        h, w = block.shape
        if h <= mindim:
            return False
        return predicate(block)

    # Perform splitting
    Z = qtdecomp(fp, predicate=qt_predicate, min_dim=mindim)

    # Merging
    Lmax = int(Z.max())  # Largest block size
    g_pad = np.zeros_like(fp, dtype=int)
    MARKER = np.zeros_like(fp, dtype=int)

    for K in range(1, Lmax + 1):
        # We only care about powers of 2 ideally, or whatever sizes exist in Z
        # qtgetblk will return empty if size doesn't exist.

        vals, r, c = qtgetblk(fp, Z, K)

        if len(vals) > 0:
            for i in range(len(r)):
                xlow, ylow = r[i], c[i]
                xhigh = xlow + K
                yhigh = ylow + K

                region = fp[xlow:xhigh, ylow:yhigh]

                # Check predicate again?
                # splitmerge.m: "Check the predicate... if flag: g=1, MARKER=1"
                # "Looking at each quadregion and setting all its elements to 1 if the block satisfies the predicate"
                # Wait. In splitting, we split if Predicate is True (Non-homogeneous).
                # In merging, if Predicate is True, we mark it?
                # That sounds contradictory if Predicate means "Non-homogeneous".
                # Standard split-merge:
                # 1. Split until all blocks are Homogeneous (Pred=False).
                # 2. Merge adjacent Homogeneous blocks.

                # Let's re-read splitmerge.m logic carefully.
                # L74: flag = fun(region);
                # L75: if flag: g(...)=1; MARKER(...)=1;

                # And split_test (splitting stage):
                # L112: flag = fun(quadregion);
                # L113: if flag: v(I) = true (SPLIT)

                # So if `fun` (Predicate) returns True:
                # - During Split: It SPLITS the block (implies it's NOT ready/uniform).
                # - During Merge: It MARKS the block as part of the region "1"?

                # Example 11.11 Predicate:
                # flag = (sd > 10) & (m > 0) & (m < 125);
                # This predicate selects regions with High Variance? That seems odd for a region merging criteria.
                # Typically we want regions with LOW variance to form an object.
                # UNLESS the object of interest IS the high-texture area.

                # If the predicate identifies the "Object of Interest", then:
                # - Splitting: If part of the object is mixed with background, or if the block is too large and heterogeneous?
                # - If the predicate assumes "Homogeneity", then `fun` returns False means Homogeneous.
                #   Then Split if NOT Homogeneous (True).
                #   But then Merge if... True? No, usually Merge if Homogeneous.

                # Let's look at the Example 11.11 given in comments:
                # flag = (sd > 10) & (m > 0) & (m < 125);
                # "Sets FLAG to TRUE if intensities ... have std > 10 AND mean between 0 and 125".
                # If this returns True, `splitmerge` sets the output g=1 for that region.
                # And MARKER=1.
                # Finally `imreconstruct(MARKER, g)`.

                # It seems `splitmerge` as defined here is looking for regions that SATISFY the property `fun`.
                # And it forces splitting until blocks satisfy it (or are too small)?
                # Wait, if `fun` returns True, we SPLIT (L113).
                # So we keep splitting regions that have std > 10.
                # We STOP splitting when std <= 10 (Homogeneous).
                # Then in Merging (L75), if `fun` returns True, we mark as 1.
                # Taking the Example:
                # If a block has std=5 (Homogeneous), `fun` is False. Split stops.
                # Merging loop: `fun` is False. g=0.
                # So the Homogeneous regions become Background (0)?
                # And the Heterogeneous regions (std > 10) would have been split further?
                # Until they become small enough to have std <= 10? Or hit mindim.
                # If they hit mindim and still std > 10, then func returns True.
                # Then in Merging: func is True. g=1.

                # So this logic seems to isolate "Complex/Textured" regions that defy homogeneity down to `mindim`.
                # OR, maybe the Example 11.11 predicate is specific.

                # Regardless, I must rigorously follow the code structure:
                # 1. `qtdecomp` using provided predicate. (Splits if Pred=True).
                # 2. Iterate leaves (blocks in Z).
                # 3. If Pred(leaf) is True -> Mark.

                # This implementation follows that logic.

                flag = predicate(region)
                if flag:
                    g_pad[xlow:xhigh, ylow:yhigh] = 1
                    MARKER[xlow, ylow] = 1

    # Connected components
    # g = bwlabel(imreconstruct(MARKER, g))
    # reconstruction in skimage: reconstruction(seed, mask) -> seed=MARKER, mask=g_pad
    # But wait, g_pad is binary (0/1). MARKER is binary subset.
    # morphological reconstruction will dilate MARKER inside g_pad.
    # This connects adjacent marked blocks.

    rec = reconstruction(MARKER, g_pad, method="dilation")
    # The result `rec` is a binary mask of connected valid regions.

    # Label them
    g_labeled = label(rec, connectivity=2)  # 8-connectivity default in MATLAB usually

    # Crop back
    g_final = g_labeled[:M, :N]

    return g_final
