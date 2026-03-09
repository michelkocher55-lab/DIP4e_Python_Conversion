from typing import Any
import numpy as np
from skimage import img_as_float


def rgb2hsi4e(rgb: Any):
    """
    Converts an RGB image to HSI.

    hsi = rgb2hsi4e(rgb)

    Parameters
    ----------
    rgb : numpy.ndarray
        Input RGB image (H x W x 3). Can be uint8 or float.
        Values are normalized to [0, 1] internally.

    Returns
    -------
    hsi : numpy.ndarray
        HSI image (H x W x 3) of type float [0, 1].
        hsi[:, :, 0] is Hue (normalized to [0, 1] by dividing angle by 2*pi).
        hsi[:, :, 1] is Saturation [0, 1].
        hsi[:, :, 2] is Intensity [0, 1].
    """

    # 1. Normalize input to [0, 1] double
    rgb = img_as_float(rgb)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Input must be an RGB image (H x W x 3).")

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Epsilon for numerical stability
    eps = np.finfo(float).eps

    # 2. Compute Hue (H)
    # theta = acos( (0.5*((r-g) + (r-b))) / sqrt((r-g)^2 + (r-b)*(g-b)) )

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))

    # Avoid division by zero
    theta = np.arccos(num / (den + eps))

    H = theta.copy()

    # If B > G, H = 2*pi - theta
    mask_b_gt_g = b > g
    H[mask_b_gt_g] = 2 * np.pi - H[mask_b_gt_g]

    # Normalize to [0, 1]
    H = H / (2 * np.pi)

    # 3. Compute Saturation (S)
    # S = 1 - 3/(R+G+B) * min(R, G, B)

    # min across channels
    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b

    # Avoid division by zero
    den_s = sum_rgb.copy()
    den_s[den_s == 0] = eps

    S = 1.0 - 3.0 * min_rgb / den_s

    # 4. Handle Singularity: H is defined as 0 when S=0 (achromatic)
    # MATLAB code: H(S == 0) = 0;
    # Note: S can be very close to zero. Exact 0 check matches MATLAB behavior.
    H[S == 0] = 0

    # 5. Compute Intensity (I)
    I = sum_rgb / 3.0

    # Combine
    hsi = np.stack((H, S, I), axis=2)

    return hsi
