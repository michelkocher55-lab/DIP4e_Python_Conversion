from typing import Any
import numpy as np
from scipy.fftpack import dct, idct


def watermark4e(f: Any, m: Any):
    """
    Inserts a watermark into an image using the Cox method (alpha=0.1).

    g, w = watermark4e(f, m)

    Parameters
    ----------
    f : numpy.ndarray
        Input image.
    m : numpy.ndarray
        Watermark sequence (pseudo-random sequence).

    Returns
    -------
    g : numpy.ndarray
        Watermarked image (uint8).
    w : dict
        Watermark decoding structure:
        - 'm': original mark
        - 'index': linear indices of modified coefficients (0-based)
        - 'coef': original DCT coefficients at those indices
    """

    f = np.array(f, dtype=float)
    m = np.array(m).flatten()  # Ensure 1D sequence

    alpha = 0.1
    k = m.size

    # helper for 2D DCT (Type 2, orthonormalized matches MATLAB default often?
    # MATLAB dct2 is simply dct along cols then rows.
    # MATLAB dct(x): x is unscaled. Inverse scaled by 2/N.
    # SciPy dct(x, norm='ortho'): scaled by sqrt(2/N), etc.
    # To match MATLAB 'dct2' exactly without 'ortho' or with specific scaling?
    # Usually for watermarking, relative scaling (alpha) matters, but absolute values matter for reconstruction.
    # Let's try to match MATLAB behavior.
    # MATLAB dct2(A) = dct(dct(A).').'
    # SciPy dct(x, type=2, norm=None) * 2 matches MATLAB?
    # Actually, let's use norm='ortho' for both valid forward/inverse.
    # It preserves energy. Watermarking relies on this.
    # If MATLAB verification is strict, I might need to adjust, but for "update test" user request,
    # functional correctness (embedding works) is key.

    # 2D DCT
    # Axis 0 then Axis 1.
    tfrm = dct(dct(f, axis=0, norm="ortho"), axis=1, norm="ortho")

    # Sort absolute coefficients (Descending)
    # flattened array
    flat_tfrm = tfrm.flatten()
    sorted_indices = np.argsort(np.abs(flat_tfrm))[::-1]  # Descending

    # Select top k
    # m length determines k
    # We should skip the DC coefficient (usually index 0)?
    # MATLAB code: `[coef, index] = sort(abs(tfrm(:)), 'descend')`.
    # `index = index(1:k)`.
    # It blindly takes the top K. DC is usually largest.
    # Cox method usually skips DC. But MATLAB code DOES NOT skip it explicitly.
    # It takes 1:k.
    # I will follow MATLAB code logic exactly.

    top_indices = sorted_indices[:k]
    top_coefs = flat_tfrm[top_indices]

    # Embed Watermark
    # coef = coef * (1 + m * alpha)
    new_coefs = top_coefs * (1 + m * alpha)

    # Inject back
    flat_tfrm[top_indices] = new_coefs

    # Reshape
    tfrm_mod = flat_tfrm.reshape(tfrm.shape)

    # Inverse DCT
    g = idct(idct(tfrm_mod, axis=1, norm="ortho"), axis=0, norm="ortho")

    # Clip and Cast
    g = np.clip(g, 0, 255).astype(np.uint8)

    # Output structure
    w = {
        "m": m,
        "index": top_indices,  # 0-based linear indices
        "coef": top_coefs,
    }

    return g, w
