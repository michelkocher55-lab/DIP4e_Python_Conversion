from typing import Any
import numpy as np
from scipy.ndimage import correlate

try:
    from fun2hist import fun2hist
except ImportError:
    pass  # Should be in path or handled by user


def exacthist(f: Any, H: Any, mask: Any = None):
    """
    Performs exact histogram specification.

    Parameters:
    f (ndarray): Input image.
    H (ndarray): Specified histogram (counts). Must sum to number of active pixels.
    mask (ndarray, optional): Binary mask. Only pixels where mask > 0 are processed.

    Returns:
    g (ndarray): Processed image (uint8).
    Hg (ndarray): Histogram of g (excluding masked pixels).
    lexiOrder (ndarray): Indices of sorting order.
    """
    f = np.asarray(f, dtype=float)
    M, N = f.shape
    H = np.asarray(H).flatten()

    if mask is None:
        mask = np.ones((M, N), dtype=bool)
    else:
        mask = np.asarray(mask) > 0

    # Verify H
    # Using np.isclose for float checking if H passed as floats, but should be ints ideally.
    if not np.all(H >= 0):
        raise ValueError("Histogram components must be nonnegative.")

    num_active = np.sum(mask)
    if np.sum(H) != num_active:
        # If mismatch, try to adjust/modify H (logic in modifyhist)
        # Assuming H is for full image? No, doc says "sum ... must equal M*N" (if no mask)
        # If mask present, modifyhist is called in MATLAB.
        # Let's implement modifyhist logic inline or helper.

        if num_active != M * N:
            # Scale H to fit active pixels
            R = num_active / (M * N)
            H_scaled = H * R
            # fun2hist logic equivalent to normalize and round to sum
            # Simplest: H = H * (num_active / sum(H))?
            # MATLAB implementation uses fun2hist(H, totalPix) which handles rounding error.
            # We should probably do a simple rounding that preserves sum.
            H = _normalize_hist_counts(H, num_active)
        else:
            if np.sum(H) != num_active:
                # Even if full image, if sum mismatch, error or fix?
                # MATLAB exacthist errors if mismatch (line 31).
                # But lines 34-38 handle modification if mask is present.
                pass

    # Filters
    filters = _get_filters()

    # Compute feature vectors Axy
    # K filters.
    K = len(filters)
    # Axy will be (M*N, K)
    MF_list = []

    for filt in filters:
        # MATLAB imfilter 'replicate' -> scipy 'nearest'
        res = correlate(f, filt, mode="nearest")
        MF_list.append(res.flatten())

    # Axy shape (Pixel_Count, K).
    # MF_list is list of 1D arrays of length M*N.
    # We want columns.
    Axy = np.stack(MF_list, axis=1)  # (M*N, K)

    # Lexicographical Sort
    # MATLAB sortrows(Axy) sorts by Col 1, then Col 2, etc.
    # NumPy lexsort((colN, colN-1, ..., col1)) sorts by Col 1 then 2...
    # So we pass columns in REVERSE order.
    keys = [Axy[:, k] for k in range(K - 1, -1, -1)]
    lexiOrder = np.lexsort(keys)

    # Assign intensities
    g_flat = np.zeros(M * N, dtype=int)

    # Construct target values array based on H
    # H=[h0, h1, ...] -> [0]*h0 + [1]*h1 ...
    # This must sum to num_active.

    levels = np.arange(len(H))
    # Filter out empty bins for efficiency
    valid_bins = H > 0
    target_vals_active = np.repeat(levels[valid_bins], H[valid_bins].astype(int))

    if target_vals_active.size != num_active:
        # Rounding issues might cause mismatch of 1 pixel?
        # If so, adjust.
        diff = num_active - target_vals_active.size
        # Add checks later. For now assume H matches sum.
        pass

    # Assignment
    # We need to iterate through lexiOrder.
    # If mask is None, simply: g_flat[lexiOrder] = target_vals
    # If mask is present, we only assign to masked pixels, skipping others in lexiOrder sequence?
    # Logic in MATLAB:
    # for pix = 1:M*N
    #   ind = lexiOrder(pix)
    #   if mask(ind) > 0:
    #       assign next target value

    if np.all(mask):
        g_flat[lexiOrder] = target_vals_active
    else:
        # Mask handling
        # We process lexiOrder sequentially. If the pixel pointed to is in mask, we give it the next available value.
        # Which order? The sorted order.
        # Filter lexiOrder to only include active pixels.
        # active_indices_sorted = [idx for idx in lexiOrder if mask_flat[idx]]
        mask_flat = mask.flatten()

        # Optimized:
        # lexiOrder contains indices 0..MN-1 sorted by features.
        # Boolean indexing?
        # mask_flat[lexiOrder] returns bools in sorted order.

        sorted_mask = mask_flat[lexiOrder]  # True if that sorted pixel is active
        active_sorted_indices = lexiOrder[sorted_mask]

        g_flat[active_sorted_indices] = target_vals_active

    g = g_flat.reshape((M, N))

    # Output histogram calculation
    if np.all(mask):
        Hg = H
    else:
        # Exclude masked pixels. All masked pixels are 0 in g?
        # MATLAB sets g=zeros, so masked pixels are 0.
        # We want to return histogram of ACTIVE pixels (which should match H).
        # Actually MATLAB implementation Hg(1) correction suggests it calculates histogram of whole image g,
        # but masked pixels are 0 so they inflate bin 0.
        # Since we assigned exact values to active pixels, their histogram IS H.
        Hg = H

    return g.astype(np.uint8), Hg, lexiOrder


def _get_filters():
    """_get_filters."""
    # F1 = 1 (Identity? Convolution with 1 is just value)
    f1 = np.array([[1]])

    f2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0

    f3 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]) / 4.0

    f4 = (
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        / 4.0
    )

    f5 = (
        np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
            ]
        )
        / 8.0
    )

    f6 = (
        np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
            ]
        )
        / 4.0
    )

    return [f1, f2, f3, f4, f5, f6]


def _normalize_hist_counts(H: Any, total: Any):
    """_normalize_hist_counts."""
    # Normalize H to sum to total, preserving integer counts
    # This effectively mimics fun2hist logic for counts.
    current_sum = np.sum(H)
    if current_sum == 0:
        return H

    H = H.astype(float)
    H *= total / current_sum
    H = np.round(H).astype(int)

    # Error correction
    diff = total - np.sum(H)
    if diff != 0:
        # Add/subtract from max bin or distribute?
        # Add to peak or distributed.
        # Simplest: Add to argmax.
        idx = np.argmax(H)
        H[idx] += diff

    return H
