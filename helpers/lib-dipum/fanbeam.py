from typing import Any
import numpy as np
from skimage.transform import radon
from scipy.ndimage import map_coordinates


def fanbeam(
    g: Any, D: Any, FanRotationIncrement: Any = 1.0, FanSensorSpacing: Any = 1.0
):
    """
    Computes the fan-beam projection data (sinogram) of an image.
    Optimized to use scipy.ndimage.map_coordinates.
    """

    g = np.array(g)

    # 1. Compute High-Resolution Parallel Radon Transform
    # Resolution for parallel radon
    d_theta_parallel = 0.5  # Reverted to coarser resolution for speed
    theta_parallel = np.arange(0, 180, d_theta_parallel)

    # P shape: (num_t, num_theta)
    P = radon(g, theta=theta_parallel, circle=False)

    num_t, num_theta = P.shape
    center_t = num_t // 2

    # 2. Define Fan-Beam Coordinates
    beta = np.arange(0, 360, FanRotationIncrement)

    M, N = g.shape
    R_img = np.sqrt(M**2 + N**2) / 2
    if D <= R_img:
        raise ValueError("Distance D must be greater than image radius.")

    gamma_max_radians = np.arcsin(R_img / D)
    gamma_max_degrees = np.degrees(gamma_max_radians)
    gamma_limit = np.ceil(gamma_max_degrees + FanSensorSpacing)
    gamma = np.arange(-gamma_limit, gamma_limit + 1e-6, FanSensorSpacing)

    # 3. Map Fan (Beta, Gamma) to Parallel (Theta, t)
    # Target grid: Beta (cols), Gamma (rows) -> matches P's axes if swapped?
    # P axes: 0=t, 1=theta
    # We produce output (num_gamma, num_beta).

    Beta, Gamma = np.meshgrid(beta, gamma, indexing="xy")
    # Beta: (num_gamma, num_beta), Gamma: (num_gamma, num_beta)

    Theta_req = Beta + Gamma
    t_req = D * np.sin(np.radians(Gamma))

    # Handle wrapping/reflection for Parallel Sinogram Lookup
    Theta_req_mod = np.mod(Theta_req, 360)

    mask_flip = Theta_req_mod >= 180
    Theta_lookup = Theta_req_mod.copy()
    Theta_lookup[mask_flip] -= 180

    t_lookup = t_req.copy()
    t_lookup[mask_flip] *= -1

    # Map t_lookup to index coordinates for P
    # t range for P corresponds to approx coordinate indices centered at center_t
    # skimage radon t-axis roughly matches pixel coordinates if projected?
    # Actually radon returns size typically sqrt(2)*N.
    # The 't' axis in radon is indexed 0..num_t-1.
    # Center is at center_t.
    # So t=0 is at center_t.
    # We need to scaling factor? skimage radon projection doesn't explicitly return 't' coords, just the array.
    # The 't' bins are 1 pixel wide usually?
    # Yes, for unit pixel spacing, t bins are 1 unit.
    t_indices = t_lookup + center_t

    # Map theta_lookup to index coordinates for P
    # theta starts at 0, step d_theta_parallel
    theta_indices = Theta_lookup / d_theta_parallel

    # Combine coordinates: (row_coords, col_coords) -> (t_indices, theta_indices)
    coords = np.array([t_indices.ravel(), theta_indices.ravel()])

    # Interpolate
    # order=1 (linear) is fast and usually sufficient.
    F_flat = map_coordinates(P, coords, order=1, mode="constant", cval=0.0)
    F = F_flat.reshape(Beta.shape)

    return F, gamma, beta
