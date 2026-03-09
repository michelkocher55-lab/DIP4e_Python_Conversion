from typing import Any
import numpy as np


def levelSetIterate4e(phi: Any, F: Any, delT: Any = None):
    """
    Iterative solution of level set equation.

    phin = levelsetIterate4e(phi, F, delT)
    """

    phi = np.array(phi, dtype=float)

    if delT is None:
        max_F = np.max(np.abs(F))
        if max_F == 0:
            delT = 0.5
        else:
            delT = 0.5 * (1.0 / max_F)

    # Compute Upwind Derivatives
    # Pad phi to prevent wrap-around from affecting edges
    phi_pad = np.pad(phi, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    # MATLAB: circshift(phi, 1, 1) - phi
    # Python np.roll(A, 1, 0)
    # Dxplus (Backward Row?): phi(i-1) - phi(i).
    # Wait, in MATLAB file I analyzed earlier:
    # Dxplus = circshift(phi, 1, 1) - phi
    # Dxminus = phi - circshift(phi, -1, 1)

    # Implementing using slices on padded array for speed/clarity if possible,
    # but to match logic exactly I'll use roll on the padded array then crop.

    # Actually, simpler with slices (matches Upwind def):
    # dx_fwd = phi(i+1) - phi(i)
    # dx_bwd = phi(i) - phi(i-1)

    # MATLAB: `Dxplus` = `circshift(1)` - `phi`.
    # `circshift(1)` puts `phi(i-1)` at `i`.
    # So `Dxplus` = `phi(i-1) - phi(i)`. This is negative Backward diff. `-D_bwd`.

    # MATLAB: `Dxminus` = `phi` - `circshift(-1)`.
    # `circshift(-1)` puts `phi(i+1)` at `i`.
    # So `Dxminus` = `phi(i) - phi(i+1)`. This is negative Forward diff. `-D_fwd`.

    # Standard LS Evolution:
    # phi_t = F * |grad(phi)|.
    # If F > 0 (Expand): Use Entropy condition / Upwind.
    # Grad+ = ...

    # Let's blindly replicate the algebra:
    s_phi = phi_pad

    # Axis 0 (Rows)
    roll_p1_ax0 = np.roll(s_phi, 1, axis=0)  # Shift Down. i -> i+1. (Old i-1 is at i)
    roll_m1_ax0 = np.roll(s_phi, -1, axis=0)  # Shift Up. i -> i-1. (Old i+1 is at i)

    Dxplus = roll_p1_ax0 - s_phi
    Dxminus = s_phi - roll_m1_ax0

    # Axis 1 (Cols)
    roll_p1_ax1 = np.roll(s_phi, 1, axis=1)
    roll_m1_ax1 = np.roll(s_phi, -1, axis=1)

    Dyplus = roll_p1_ax1 - s_phi
    Dyminus = s_phi - roll_m1_ax1

    # Crop border
    Dxplus = Dxplus[1:-1, 1:-1]
    Dxminus = Dxminus[1:-1, 1:-1]
    Dyplus = Dyplus[1:-1, 1:-1]
    Dyminus = Dyminus[1:-1, 1:-1]

    # Upwind Norm Grad
    # gNormPlus
    gNormPlus = np.sqrt(
        np.maximum(Dxminus, 0) ** 2
        + np.minimum(Dxplus, 0) ** 2
        + np.maximum(Dyminus, 0) ** 2
        + np.minimum(Dyplus, 0) ** 2
    )

    # gNormMinus
    gNormMinus = np.sqrt(
        np.maximum(Dxplus, 0) ** 2
        + np.minimum(Dxminus, 0) ** 2
        + np.maximum(Dyplus, 0) ** 2
        + np.minimum(Dyminus, 0) ** 2
    )

    # Update
    # phin = phi - delT*(max(F,0).*gNormPlus + min(F,0).*gNormMinus);
    term = np.maximum(F, 0) * gNormPlus + np.minimum(F, 0) * gNormMinus
    phin = phi - delT * term

    return phin
