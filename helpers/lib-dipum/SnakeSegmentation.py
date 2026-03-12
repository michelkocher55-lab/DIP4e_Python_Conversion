import numpy as np
from skimage.util import img_as_float

from DIP4eFigures.intScaling4e import intScaling4e
from DIP4eFigures.snakeMap4e import snakeMap4e
from DIP4eFigures.snakeForce4e import snakeForce4e
from DIP4eFigures.snakeIterate4e import snakeIterate4e
from DIP4eFigures.snakeReparam4e import snakeReparam4e
from helpers.coord2mask import coord2mask


def SnakeSegmentation(f, x, y, T, Sig, NSig, Mu, NIterForce, NIterConvergence, Alpha, Beta, Gamma):
    """
    Snake segmentation using GVF force.

    Transcoding of MATLAB:
      [x, y, emap] = SnakeSegmentation(f, x, y, T, Sig, NSig, Mu, NIterForce,
                                 NIterConvergence, Alpha, Beta, Gamma)
    """
    M, N = f.shape

    # Binary mask (kept for parity with MATLAB code path).
    binmask = coord2mask(M, N, x, y)
    _ = binmask

    # Snake force from original image edge map.
    emap = snakeMap4e(f, T, Sig, NSig, 'both')
    emap = img_as_float(intScaling4e(emap))

    FTx, FTy = snakeForce4e(emap, 'gvf', Mu, NIterForce)
    mag = np.sqrt(FTx**2 + FTy**2)
    FTx = FTx / (mag + 1e-10)
    FTy = FTy / (mag + 1e-10)

    # Snake convergence.
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    for k in range(1, int(NIterConvergence) + 1):
        x, y = snakeIterate4e(Alpha, Beta, Gamma, x, y, 1, FTx, FTy)
        if k % 5 == 0:
            x, y = snakeReparam4e(x, y)

        # Keep snake in image (prevents out-of-bounds force=0 collapse).
        x = np.minimum(np.maximum(x, 1), M)
        y = np.minimum(np.maximum(y, 1), N)

    x, y = snakeReparam4e(x, y)
    return x, y, emap
