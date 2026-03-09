from typing import Any
import numpy as np


def hsi2rgb4e(hsi: Any):
    """
    Converts an HSI image to RGB.

    Parameters:
    -----------
    hsi : numpy.ndarray
        HSI image (height, width, 3).
        hsi[:,:,0] = Hue, normalized to [0, 1].
        hsi[:,:,1] = Saturation, in [0, 1].
        hsi[:,:,2] = Intensity, in [0, 1].

    Returns:
    --------
    rgb : numpy.ndarray
        RGB image (height, width, 3) with values in range [0, 1].
    """
    # Extract components
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # RG sector (0 <= H < 2*pi/3)
    idx = (0 <= H) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # BG sector (2*pi/3 <= H < 4*pi/3)
    idx = (2 * np.pi / 3 <= H) & (H < 4 * np.pi / 3)
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (
        1 + S[idx] * np.cos(H[idx] - 2 * np.pi / 3) / np.cos(np.pi - H[idx])
    )
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # BR sector (4*pi/3 <= H <= 2*pi)
    idx = (4 * np.pi / 3 <= H) & (H <= 2 * np.pi)
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (
        1 + S[idx] * np.cos(H[idx] - 4 * np.pi / 3) / np.cos(5 * np.pi / 3 - H[idx])
    )
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    # Combine results
    rgb = np.dstack((R, G, B))

    # Clip to [0, 1]
    rgb = np.clip(rgb, 0, 1)

    return rgb
