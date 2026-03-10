from typing import Any
import numpy as np
from lib.intScaling4e import intScaling4e
from lib.twodConv4e import twodConv4e
from lib.morphoReconDilate4e import morphoReconDilate4e
from lib.morphoConComp4e import morphoConComp4e


def diskCounter(f: Any):
    """
    Counts small and large disks in an image.
    Assumes objects are DARKER than background.

    imOut, numSmallDisks, numLargeDisks = diskCounter(f)

    Parameters
    ----------
    f : numpy.ndarray
        Input image.

    Returns
    -------
    imOut : numpy.ndarray
        Image with small disks labeled 0.5 and large disks labeled 1.0.
    numSmallDisks : int
        Count of small disks.
    numLargeDisks : int
        Count of large disks.
    """

    # 1. Scale
    f_scaled = intScaling4e(f)  # Returns float [0, 1]? Or [0, 255]?
    # Assuming intScaling4e returns float usually in DIP4E utils, unless specified.
    # Let's verify intScaling4e return type if needed, but assuming standard normalization.

    # 2. Smooth
    w = np.ones((3, 3)) / 9.0
    f_smooth = twodConv4e(f_scaled, w)

    # 3. Automatic Threshold
    # Objects are assumed Dark (< T)
    T = (np.max(f_smooth) + np.min(f_smooth)) / 2.0

    # Binarize (Objects become 1, Background 0)
    f_bin = (f_smooth < T).astype(float)  # boolean -> 0.0/1.0

    # 4. Clear Border Objects
    F_marker = np.zeros_like(f_bin)
    rows, cols = f_bin.shape

    # Set border markers
    F_marker[0, :] = f_bin[0, :]
    F_marker[-1, :] = f_bin[-1, :]
    F_marker[:, 0] = f_bin[:, 0]
    F_marker[:, -1] = f_bin[:, -1]

    # Reconstruction
    R, _ = morphoReconDilate4e(F_marker, f_bin)

    # Subtract (Remove border objects)
    f_clean = f_bin - R
    f_clean[f_clean < 0] = 0  # Safety

    # 5. Connected Components
    C, NC = morphoConComp4e(f_clean)

    if NC == 0:
        return C, 0, 0

    # 6. Compute Areas
    areas = []
    # C contains labels 1..NC
    # We can iterate or use unique/bincount
    # Since morphoConComp4e returns labels 1..NC steps 1.

    # To match MATLAB loop logic exactly:
    for i in range(1, NC + 1):
        area = np.sum(C == i)
        areas.append(area)

    areas = np.array(areas)

    if len(areas) == 0:
        return C, 0, 0

    Alarge = np.max(areas)
    Asmall = np.min(areas)

    # 7. Classify
    # Modify C to have 0.5 or 1.0
    # Copy C to avoid modifying loop references?
    # C is reusing values.
    # MATLAB loop:
    # for I=1:NC
    #    if A(I) < 0.9*Alarge -> C(C==I) = 0.5
    #    else -> C(C==I) = 1

    C_out = np.zeros_like(C)

    for i in range(1, NC + 1):
        area = areas[i - 1]

        mask = C == i
        if area < 0.9 * Alarge:
            C_out[mask] = 0.5
        else:
            C_out[mask] = 1.0

    # 8. Count
    # numSmall = round(TotalGrayArea / Asmall)
    # numLarge = round(TotalWhiteArea / Alarge)

    grayArea = np.sum(C_out == 0.5)
    whiteArea = np.sum(C_out == 1.0)

    # Avoid division by zero
    if Asmall == 0:
        Asmall = 1
    if Alarge == 0:
        Alarge = 1

    numSmallDisks = round(grayArea / Asmall)
    numLargeDisks = round(whiteArea / Alarge)

    # Note: If there's only one sizes of disks (e.g. all equal), Asmall == Alarge.
    # The condition < 0.9*Alarge will be false. All become 1.0.
    # So all counted as Large.

    return C_out, numSmallDisks, numLargeDisks
