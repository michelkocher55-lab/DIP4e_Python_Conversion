from typing import Any
import numpy as np
from lib.mat2huff import mat2huff


def im2lpc4e(f: Any, coefs: Any = None):
    """
    Compresses a matrix using 1-D lossless predictive coding.

    Parameters:
    -----------
    f : numpy.ndarray
        Input matrix (image).
    coefs : list or numpy.ndarray, optional
        Prediction coefficients. Default is [1].

    Returns:
    --------
    c : dict
        Compressed structure with fields:
        'coefs': coefficients used.
        'huffman': Huffman encoded prediction error (from mat2huff).
    """
    if coefs is None:
        coefs = [1]

    f = np.array(f, dtype=float)
    coefs = np.array(coefs, dtype=float).flatten()

    m, n = f.shape
    p = np.zeros((m, n))
    xs = f.copy()
    zerocol = np.zeros((m, 1))

    # Compute prediction
    # MATLAB loop:
    # for j = 1:length(coefs)
    #    xs = [zc xs(:, 1:end-1)]
    #    p = p + coefs(j) * xs
    # end

    # In Python, we can simulate this shift.
    # xs starts as f.
    # In iteration 1: xs shifts right by 1 (col 0 becomes 0, col 1 becomes col 0...)
    # We multiply by coefs[0].
    # In iteration 2: xs shifts right again.

    # Let's manage xs state carefully.

    current_xs = f

    for c_val in coefs:
        # Shift right
        # New column 0 is 0s
        # Keep columns 0 to n-2
        shifted = np.hstack((zerocol, current_xs[:, :-1]))

        # Accumulate prediction
        p = p + c_val * shifted

        # Update current_xs for next iteration?
        # Wait, the MATLAB code uses `xs` which is UPDATED in the loop
        # `xs = [zc xs(:, 1:end-1)]`.
        # So for the NEXT coefficient, it uses the Further shifted version.
        # Yes.
        current_xs = shifted

    # Compute prediction error
    # Round prediction to nearest integer
    # f should ideally be integer-like if we want lossless
    # but the inputs are double.
    # The error is encoded.
    error = f - np.round(p)

    # Huffman encode the error using mat2huff
    huff_struct = mat2huff(error)

    c = {"coefs": coefs, "huffman": huff_struct}

    return c
