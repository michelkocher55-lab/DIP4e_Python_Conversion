from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def snakeIterate4e(
    alpha: Any, beta: Any, gamma: Any, x: Any, y: Any, NI: Any, Fx: Any, Fy: Any
):
    """
    SNAKEITERATE4E Iterative computation of segmentation snake.
    Literal transcoding of snakeIterate4e.m.
    """

    # -Preliminaries.
    # K = numel(x);
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    K = len(x)

    # -Construct matrix inv((I - delT*A)) (see Eq. (11-45) in DIP4E).
    # --First construct matrix D2 in Eq. (11-34).
    # a = -2*ones(K,1);
    # b = 1*ones(K-1,1);
    # D2 = diag(a) + diag(b,-1) + diag(b, 1);
    a = -2 * np.ones(K)
    b = np.ones(K - 1)
    D2 = np.diag(a) + np.diag(b, -1) + np.diag(b, 1)

    # D2(1,K) = 1; D2(K,1) = 1;
    D2[0, K - 1] = 1
    D2[K - 1, 0] = 1

    # --Next construct D4 in Eq. (11-36).
    # a = 6*ones(K,1);
    # b = -4*ones(K-1,1);
    # c = 1*ones(K-2,1);
    # D4 = diag(a) + diag(b,-1) + diag(b,1) + diag(c,-2) + diag(c,2);
    a4 = 6 * np.ones(K)
    b4 = -4 * np.ones(K - 1)
    c4 = np.ones(K - 2)
    D4 = (
        np.diag(a4)
        + np.diag(b4, -1)
        + np.diag(b4, 1)
        + np.diag(c4, -2)
        + np.diag(c4, 2)
    )

    # D4(1,K) = -4; D4(K,1) = -4;
    D4[0, K - 1] = -4
    D4[K - 1, 0] = -4

    # D4(1,K-1) = 1; D4(K-1,1) = 1;
    D4[0, K - 2] = 1
    D4[K - 2, 0] = 1

    # D4(2,K) = 1; D4(K,2) = 1;
    D4[1, K - 1] = 1
    D4[K - 1, 1] = 1

    # --Matrix D in Eq. (11-39)
    # D = alpha*D2 - beta*D4;
    D = alpha * D2 - beta * D4

    # --Construct the final matrix inv(I - A))
    # A = inv(eye(K) - D);
    A = np.linalg.inv(np.eye(K) - D)

    # -Multiply the forces by gamma.
    # Fx = gamma*Fx;
    # Fy = gamma*Fy;
    # (Note: we scale applied forces inside loop or pre-scale F)
    # Here we should scale field values. But we can just scale interp result.

    # Setup Interpolators
    # Fx, Fy are images.
    H, W = Fx.shape
    r_grid = np.arange(H)
    c_grid = np.arange(W)

    # Note: interp2 uses 'linear', 0 (fill value).
    interp_fx = RegularGridInterpolator(
        (r_grid, c_grid), Fx, bounds_error=False, fill_value=0
    )
    interp_fy = RegularGridInterpolator(
        (r_grid, c_grid), Fy, bounds_error=False, fill_value=0
    )

    # -Iterative solution based on Eq. (11-46).
    # for I = 1:NI
    for i in range(NI):
        # --Interpolate...
        # fx = interp2(Fx,y,x,'linear',0);
        # fy = interp2(Fy,y,x,'linear',0);

        # MATLAB interp2(V, Xq, Yq). Xq=Col, Yq=Row.
        # It passes Y as Xq, X as Yq.
        # Assuming Y=Col, X=Row.
        # Python RegularGridInterpolator((row, col), data).
        # We pass (row, col) = (X, Y).

        pts = np.column_stack((x, y))
        fx_val = interp_fx(pts)
        fy_val = interp_fy(pts)

        # Apply Gamma
        fx_val *= gamma
        fy_val *= gamma

        # -Compute new values of x and y using Eq. (11-46).
        # x = A*(x + fx);
        # y = A*(y + fy);
        x = A @ (x + fx_val)
        y = A @ (y + fy_val)

    # -Form the snake.
    # xs = x; ys = y;
    return x, y
