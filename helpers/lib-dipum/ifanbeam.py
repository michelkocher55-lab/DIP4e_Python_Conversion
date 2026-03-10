from typing import Any
import numpy as np
from skimage.transform import iradon, resize
from scipy.ndimage import map_coordinates


def ifanbeam(
    F: Any,
    D: Any,
    FanRotationIncrement: Any = 1.0,
    FanSensorSpacing: Any = 1.0,
    filter: Any = "ramp",
    OutputSize: Any = None,
):
    """
    Inverse fan-beam reconstruction via rebinning to parallel beam.
    Optimized to use scipy.ndimage.map_coordinates.
    """

    num_sensors, num_beta = F.shape

    # Fan coordinates
    center_sensor = (num_sensors - 1) / 2.0
    gamma_indices = np.arange(num_sensors)
    gamma = (gamma_indices - center_sensor) * FanSensorSpacing

    beta_indices = np.arange(num_beta)
    # beta = beta_indices * FanRotationIncrement

    # 1. Define Target Parallel Beam Coordinates
    max_gamma_rad = np.radians(np.max(np.abs(gamma)))
    t_max = D * np.sin(max_gamma_rad)

    # Use moderately fine theta spacing
    d_theta = 1.0  # Reverted to coarser resolution for speed
    theta_parallel = np.arange(0, 180, d_theta)

    num_t = num_sensors
    t_parallel = np.linspace(-t_max, t_max, num_t)

    # Meshgrid for Target Parallel P
    Theta_p, T_p = np.meshgrid(theta_parallel, t_parallel, indexing="xy")
    # Theta_p shape: (num_t, num_theta), T_p shape: (num_t, num_theta)

    # 2. Map Parallel (Theta, t) to Fan (Beta, Gamma)
    # Gamma = arcsin(t / D)
    t_ratio = T_p / D
    t_ratio = np.clip(t_ratio, -1.0, 1.0)
    Gamma_req_rad = np.arcsin(t_ratio)
    Gamma_req_deg = np.degrees(Gamma_req_rad)

    # Beta = Theta - Gamma
    Beta_req_deg = Theta_p - Gamma_req_deg
    Beta_req_mod = np.mod(Beta_req_deg, 360)

    # 3. Map to Indices for F
    # F axis 0: gamma. range [-gamma_lim, +gamma_lim] mapped to 0..num_sensors-1
    # gamma = (idx - center) * spacing => idx = gamma/spacing + center
    Gamma_indices = Gamma_req_deg / FanSensorSpacing + center_sensor

    # F axis 1: beta. range [0, 360). 0..num_beta-1.
    # beta = idx * inc => idx = beta / inc
    Beta_indices = Beta_req_mod / FanRotationIncrement

    # Handle Beta wrapping for interpolation
    # map_coordinates 'wrap' mode works for integer boundaries, but F beta axis is [0, 360-inc].
    # If Beta_indices goes beyond num_beta-1, it should wrap.
    # map_coordinates(..., mode='wrap') wraps nicely?
    # The 'wrap' mode in map_coordinates wraps at the boundary.
    # range is 0 to N-1. 'wrap' extends by repeating.
    # We need circular wrapping 360->0.
    # Actually, RegularGridInterpolator logic required padding.
    # map_coordinates mode='grid-wrap'?
    # 'wrap': (a, b, c, c, b, a) (reflect).
    # 'grid-wrap': (a, b, c, a, b, c). THIS is what we want for beta axis.
    # But gamma axis should probably be 'constant' (0) outside.
    # map_coordinates only accepts one mode for all axes?
    # Solution: We pad F along beta manually to handle wrapping or use 'nearest' for gamma?
    # Gamma should fall off to 0. So 'constant' 0 is good for gamma.
    # Beta needs 'grid-wrap'.
    # Modes conflict.
    # Let's use 'nearest' for beta by keeping indices in [0, num_beta)?
    # Or just pad F manually for beta and use constant 0.

    pad_b = 2
    F_pad = np.pad(F, ((0, 0), (pad_b, pad_b)), mode="wrap")
    # Shift Beta_indices by pad_b
    Beta_indices_shifted = Beta_indices + pad_b

    coords = np.array([Gamma_indices.ravel(), Beta_indices_shifted.ravel()])

    P_flat = map_coordinates(F_pad, coords, order=1, mode="constant", cval=0.0)
    P = P_flat.reshape(Theta_p.shape)

    # 4. Inverse Radon
    # Map filters
    filter_map = {
        "Ram-Lak": "ramp",
        "Hamming": "hamming",
        "Hann": "hann",
        "Cosine": "cosine",
        "Shepp-Logan": "shepp-logan",
    }
    sk_filter = filter_map.get(filter, "ramp")

    g_recon = iradon(P, theta=theta_parallel, filter_name=sk_filter, circle=False)

    # 5. OutputSize
    if OutputSize is not None:
        if isinstance(OutputSize, (int, float)):
            target_shape = (int(OutputSize), int(OutputSize))
        else:
            target_shape = tuple(OutputSize)

        if g_recon.shape != target_shape:
            g_recon = resize(
                g_recon, target_shape, anti_aliasing=True, preserve_range=True
            )

    return g_recon
