from typing import Any
import numpy as np
import math
from skimage.transform import rotate

try:
    from helpers.libdipum.lpfilter import lpfilter
except ImportError:
    # If lpfilter not in path yet, might need adjustment or inline.
    # For now assume it is in path/environment as provided in previous steps.
    pass


def ishade(
    M: Any, N: Any, ILOW: Any, IHIGH: Any, mode: Any = "spot", param: Any = None
):
    """
    Generates an intensity shading function.

    Parameters:
        M, N: Size of output image (rows, cols).
        ILOW, IHIGH: Range of intensity values [ILOW, IHIGH]. Must be in [0, 1].
        mode: 'spot' for Gaussian spot, 'ramp' for linear ramp.
        param:
            For 'spot', param is variance (sigma). Default min(M, N).
            For 'ramp', param is angle in degrees. Default 0.

    Returns:
        g: Shading image of size (M, N).
    """

    if (ILOW >= IHIGH) or (ILOW < 0) or (IHIGH > 1):
        raise ValueError("ILOW and IHIGH must be in the range [0 1] with ILOW < IHIGH")

    if param is None:
        if mode == "spot":
            param = min(M, N)
        elif mode == "ramp":
            param = 0

    g = None

    if mode == "spot":
        # g = lpfilter('gaussian', M, N, param);
        # g = fftshift(g);
        # We need to ensure lpfilter is available or implement equivalent gaussian.
        # Assuming lpfilter follows standard meshgrid distance.
        # Gaussian filter H(u,v) = exp(-D^2 / (2*sigma^2))?
        # Typically lpfilter returns H in freq domain centered depending on implementation.
        # If it returns centered H, then fftshift might uncenter it or v.v.
        # MATLAB lpfilter usually returns centered transfer function if generated via paddedsize/dftuv centered?
        # Actually in dipum, lpfilter returns H.
        # If we just want a spatial gaussian spot, we can generate it directly.
        # But let's try to verify if we have lpfilter.

        # Spatial generation is safer/easier if just a spot.
        # Center: (M/2, N/2). Sigma = param.
        # H(x,y) = exp(-((x-cx)^2 + (y-cy)^2) / (2*param^2))

        x = np.arange(0, N)
        y = np.arange(0, M)
        X, Y = np.meshgrid(x, y)
        cx, cy = N / 2, M / 2
        # Use simple gaussian formula
        # Note: MATLAB lpfilter('gaussian', M, N, D0) -> H = exp(-(D.^2)./(2*(D0^2)))
        g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (param**2)))

        # The MATLAB code does `g = fftshift(g)`. `lpfilter` usually generates filter with origin at (1,1) or center?
        # In DIPUM `lpfilter` uses `dftuv` which puts origin at corners if not centered??
        # Actually `dftuv` usually computes meshgrid suitable for FFT (0 freq at corners).
        # So `lpfilter` output has peak at corners. `fftshift` moves it to center.
        # My spatial generation above puts peak at center. So no fftshift needed.
        pass

    elif mode == "ramp":
        # LIM = 2*max(M, N);
        LIM = 2 * max(M, N)

        # inc = 1/(LIM - 1);
        # row = 0:inc:1;
        row = np.linspace(0, 1, LIM)

        # g = repmat(row, LIM, 1);
        g = np.tile(row, (LIM, 1))

        # g = imrotate(g, param, 'bilinear', 'crop');
        # Python rotate: angle in degrees. 'crop' behavior: standard rotate fills background?
        # skimage.transform.rotate(image, angle, resize=False) keeps original size (crop).
        # Note: skimage rotates counter-clockwise? Yes.
        # MATLAB imrotate also CCW.
        g = rotate(
            g, param, resize=False, mode="reflect"
        )  # 'crop' in matlab means keep size.
        # mode='constant' fills with 0. 'reflect' or 'edge' might be safer for ramp?
        # MATLAB default 'bilinear', 'crop' fills with 0 (parameter 'bbox' implicitly 'crop').
        # But since we made image 2x larger, center crop will be safe from edges.

        # Crop M-by-N out of center
        # ctr = LIM/2;
        ctr = LIM / 2  # float center

        # r = floor(M/2); c = floor(N/2);
        r = math.floor(M / 2)
        c = math.floor(N / 2)

        rinc = 0 if (M % 2 != 0) else 1
        cinc = 0 if (N % 2 != 0) else 1

        # Python slicing: start:stop
        # center index roughly LIM//2
        # We want width M, height N
        # If shape is (LIM, LIM).
        # Centered crop

        start_row = int((LIM - M) // 2)
        start_col = int((LIM - N) // 2)

        g = g[start_row : start_row + M, start_col : start_col + N]

    else:
        raise ValueError("Unknown mode. Use 'spot' or 'ramp'.")

    # Scale to intensity range
    g_min = g.min()
    g_max = g.max()

    # Avoid div by zero if flat
    if g_max - g_min < 1e-10:
        g = np.full(g.shape, ILOW)
    else:
        g = (g - g_min) / (g_max - g_min)  # Norm to [0,1]
        g = g * (IHIGH - ILOW) + ILOW  # Scale

    return g
