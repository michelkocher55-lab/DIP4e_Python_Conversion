from typing import Any
import numpy as np
from scipy.ndimage import correlate
from skimage.transform import resize


def pyramid4e(I: Any, h: Any = None):
    """
    Computes a single level of Gaussian/Laplacian pyramid decomposition.

    [a, d, dScale] = pyramid4e(I, h)

    Parameters
    ----------
    I : numpy.ndarray
        Input image.
    h : numpy.ndarray, optional
        Filter kernel. Defaults to Gaussian 5x5 with sigma=0.75 by default (MATLAB DIP4e).

    Returns
    -------
    a : numpy.ndarray (uint8)
        Approximation image (Downsampled Gaussian).
    d : numpy.ndarray (uint8)
        Detail image (Laplacian), scaled such that 0 diff matches 128.
    dScale : numpy.ndarray (uint8)
        Detail image scaled to full range [0, 255] for display.
    """

    # Default Filter: Gaussian 5x5, sigma=0.5 (MATLAB fspecial default)
    # The source code says: h = fspecial ('gaussian', [5,5], 0.75); NOTE: 0.75 not 0.5 default.
    if h is None:
        sigma = 0.75
        size = 5
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        h = g / g.sum()

    # Ensure float for processing
    I_double = I.astype(float)

    # 1. Compute Approximation 'a'
    # MATLAB: a = imresize (uint8 (round (filter2 (h, i))), 0.5);
    # filter2 is correlation. 'h' usually symmetric.
    # We use correlate with 'constant' padding (0) or 'reflect'?
    # MATLAB filter2 default is zero-padding ('near' 0).
    try:
        filtered_I = correlate(I_double, h, mode="constant", cval=0.0)
    except Exception:
        # Fallback if I_double dimension issues
        filtered_I = correlate(I_double, h, mode="constant", cval=0.0)

    # Convert to uint8 before resizing as per MATLAB code strict order
    # MATLAB: uint8(round(...))
    filtered_I_uint8 = np.round(filtered_I).clip(0, 255).astype(np.uint8)

    # Resize 0.5
    # MATLAB imresize uses bicubic by default.
    # skimage.transform.resize expects output shape or scale.
    # We need to replicate explicit output shape calculation: floor(dim * 0.5) or ceil?
    # MATLAB: output_size = ceil(input_size * scale)
    out_shape_a = (int(np.ceil(I.shape[0] * 0.5)), int(np.ceil(I.shape[1] * 0.5)))

    # skimage resize returns float [0, 1] by default if we don't handle preserve_range
    # We want to operate on image values.
    # order=3 is smooth (cubic).
    a_float = resize(
        filtered_I_uint8,
        out_shape_a,
        order=3,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=False,
    )
    # Note: anti_aliasing might be implicit in MATLAB's lowpass before resize, but here we did filter2 manually.
    # MATLAB imresize does anti-aliasing internally unless turned off.
    # But the code applies 'h' filter first.
    # However, 'a' calculation in MATLAB: `imresize( ... , 0.5 )`
    # The `filter2` step is manual smoothing.

    a = np.round(a_float).clip(0, 255).astype(np.uint8)

    # 2. Compute Detail 'd' and 'dScale'
    # MATLAB: double (i) - filter2 (h, imresize (a, 2))

    # Upsample 'a' back to 'I' size
    # MATLAB: imresize(a, 2) -> might not exactly match I size if odds involved.
    # Usually matches 2*size.
    # BUT we need it to match 'I' for subtraction.
    # If I is 255x255, a is 128x128. a*2 is 256x256. Mismatch.
    # In MATLAB, operations on mismatching sizes usually error or truncate?
    # Actually, imresize(a, 2) creates a specific size.
    # If standard pyramid, usually we crop or pad to match I.
    # Let's target I.shape explicitly.

    upsampled_a_float = resize(
        a, I.shape, order=3, mode="reflect", preserve_range=True, anti_aliasing=False
    )

    # Filter upsampled
    filtered_upsampled = correlate(upsampled_a_float, h, mode="constant", cval=0.0)

    # Difference matrix
    diff = I_double - filtered_upsampled

    # dScale calculation
    # MATLAB: dScale = uint8 (round (mat2gray (diff) * 255));
    # mat2gray scales [min, max] -> [0, 1]
    if diff.max() > diff.min():
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff_norm = np.zeros_like(diff)  # Flat image

    dScale = np.round(diff_norm * 255).clip(0, 255).astype(np.uint8)

    # d calculation
    # MATLAB: d = uint8 (round (mat2gray (diff, [-255, 255]) * 255));
    # mat2gray(A, [L1, L2]): Values < L1 -> 0, > L2 -> 1.
    # Map [-255, 255] to [0, 1].
    # formula: (val - (-255)) / (255 - (-255)) = (val + 255) / 510

    d_norm = (diff + 255.0) / 510.0
    d_norm = np.clip(d_norm, 0, 1)

    d = np.round(d_norm * 255).clip(0, 255).astype(np.uint8)

    return a, d, dScale
