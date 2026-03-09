from typing import Any
import numpy as np
from libDIP.intScaling4e import intScaling4e
from libDIP.covmatrix4e import covmatrix4e
from libDIP.boundaryTracer4e import boundaryTracer4e


def binaryRegionProps4e(I: Any):
    """
    Properties of a single binary region.

    Computes properties of a single region in binary image I.
    Returns a dictionary with the following keys:
      - 'comp': compactness
      - 'circ': circularity
      - 'ecc': eccentricity

    Parameters:
    -----------
    I : numpy.ndarray
        Input binary image.

    Returns:
    --------
    P : dict
        Dictionary with fields 'comp', 'circ', 'ecc'.
    """
    # Convert to floating point in the range [0,1]
    I = intScaling4e(I)

    # Find the coordinates of the 1-valued pixels.
    # np.where returns tuple (rows, cols)
    # matching [x,y] = find(I==1) in MATLAB (where x=rows, y=cols)
    rows, cols = np.where(I == 1)

    # helper for concatenation
    # MATLAB: [x y] -> N x 2 matrix
    points = np.column_stack((rows, cols))

    if points.size == 0:
        return {"comp": 0.0, "circ": 0.0, "ecc": 0.0}

    # Compute the covariance matrix of the points.
    # covmatrix4e returns (C, m). We only need C.
    C, _ = covmatrix4e(points)

    # Find the eigenvalues.
    # MATLAB: [~,evals] = eig(C,'vector');
    # np.linalg.eigh is appropriate for symmetric matrices (covariance)
    # and returns eigenvalues in ascending order, but we take min/max anyway.
    evals, _ = np.linalg.eigh(C)

    # Find the largest and smallest eigenvalues.
    evalmax = np.max(evals)
    evalmin = np.min(evals)

    # Find the boundary and compute its perimeter.
    # boundaryTracer4e returns a list of boundaries.
    # The MATLAB utility implies processing a single region or taking the first found.
    B = boundaryTracer4e(I)

    if not B:
        # Handle case with no region found
        return {"comp": 0.0, "circ": 0.0, "ecc": 0.0}

    # Boundary.
    b = B[0]

    # Approximate perimeter as number of points in the boundary.
    p = b.shape[0]

    # Area of number of 1-valued pixels in I.
    # sum(I(:)) implies summing all pixels. If I is 0/1 float:
    A = np.sum(I)

    # Initialize P dictionary
    P = {}

    # Compactness
    if A == 0:
        P["comp"] = 0.0
        P["circ"] = 0.0
    else:
        P["comp"] = (p**2) / A
        P["circ"] = 4 * np.pi * A / (p**2) if p > 0 else 0.0

    # Eccentricity
    # evalmax technically could be 0 if single point or line?
    if evalmax == 0:
        P["ecc"] = 0.0
    else:
        # Avoid slight negative due to precision
        ratio_sq = (evalmin / evalmax) ** 2
        P["ecc"] = np.sqrt(max(0, 1 - ratio_sq))

    return P
