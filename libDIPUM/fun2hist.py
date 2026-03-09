from typing import Any
import numpy as np


def fun2hist(fun: Any, S: Any = None):
    """
    Generates a histogram from a given digital function.

    Parameters:
    fun (ndarray): Input function (1D array).
    S (scalar, optional): Scaling factor.
                          If None (default), returns normalized histogram (sum=1).
                          If provided, returns unnormalized histogram (sum approx S).

    Returns:
    h (ndarray): Histogram.
    """

    fun = np.asarray(fun, dtype=float)
    L = fun.size
    h = np.zeros(L, dtype=float)

    # Indices of non-zero elements
    idx = np.flatnonzero(fun)
    IDXL = idx.size

    if S is None:
        # Normalized mode
        if np.sum(fun) == 0:
            return h  # All zeros
        h = fun / np.sum(fun)
    else:
        # Unnormalized mode
        # Convert to integers.
        # h = round(S * (fun / sum(fun)))
        if np.sum(fun) == 0:
            return np.zeros(L, dtype=int)

        norm_fun = fun / np.sum(fun)
        h = np.round(S * norm_fun).astype(int)  # Start as int

        # Correction for rounding errors
        current_sum = np.sum(h)
        D = current_sum - S

        if D < 0:
            count = abs(D)
            while count > 0:
                # Distribute difference
                K = IDXL
                if count < IDXL:
                    K = count

                # Add 1 to first K non-zero bins
                # idx contains indices of non-zero elements in ORIGINAL fun.
                # Logic: iterate 1 to K.
                for i in range(K):
                    bin_idx = idx[i]
                    h[bin_idx] += 1
                    count -= 1
                    if count == 0:
                        break

        elif D > 0:
            count = D
            while count > 0:
                K = IDXL
                if count < IDXL:
                    K = count

                for i in range(K):
                    bin_idx = idx[i]
                    # Subtract but ensure >= 0
                    h[bin_idx] -= 1
                    if h[bin_idx] < 0:
                        # Restore
                        h[bin_idx] += 1  # Back to what it was (presumably 0?)
                        # Actually MATLAB logic:
                        # "Restore... And reduce count so that the count will be subtracted elsewhere."
                        # Wait, MATLAB code:
                        # h(idx(I)) = h(idx(I)) - 1;
                        # if h < 0: h = h + 1 (restore); count = count + 1 (retry elsewhere).
                        # But loop continues.
                        # This logic in MATLAB seems to handle retries by increasing count, effectively extending the while loop.

                        # Python translation:
                        count += 1
                        # We need to ensure we don't get stuck if all are 0?
                        # fun elements were non-zero. S > 0. h should have some counts.
                    else:
                        count -= 1

                    if count == 0:
                        break

    return h
