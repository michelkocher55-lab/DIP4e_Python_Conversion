from typing import Any
import numpy as np
from scipy.ndimage import laplace


def snakeForce4e(emap: Any, mode: Any = "gradient", mu: Any = 0, niter: Any = 0):
    """
    SNAKEFORCE4E Computes external force for use in snake algorithm.
    Literal transcoding of snakeForce4e.m.
    """

    # -Check default.
    # if nargin == 1 -> mode='gradient'
    # Handled by default arg.

    # -Compute force components. Note that function gradient works with
    # (c,r) instead of (r,c), as we do in the book.

    if mode == "gradient":
        # case 'gradient'
        # [Fy,Fx] = gradient(emap);

        # MATLAB gradient(emap) returns [Gx, Gy].
        # Gx is horizontal gradient (d/dx, across cols).
        # Gy is vertical gradient (d/dy, across rows).
        # So Fy = Gx (Col Gradient).
        #    Fx = Gy (Row Gradient).

        # Python np.gradient(emap) returns [GradRow, GradCol] (axis0, axis1).
        grad = np.gradient(emap)
        grad_row = grad[0]
        grad_col = grad[1]

        # Matching MATLAB assignment:
        # Fx (Row Force) = Gy = grad_row
        # Fy (Col Force) = Gx = grad_col

        Fx = grad_row
        Fy = grad_col

    elif mode == "gvf":
        # case 'gvf'
        # [egy, egx] = gradient(emap);
        # egy = Col Grad. egx = Row Grad.

        grad = np.gradient(emap)
        egx = grad[0]  # Row Grad
        egy = grad[1]  # Col Grad

        # gradMagSq = (egx.^2 + egy.^2);
        gradMagSq = egx**2 + egy**2

        # Initialize GVF
        # vx = egx; vy = egy;
        vx = egx.copy()
        vy = egy.copy()

        # Iterate
        # for I = 1:niter
        for i in range(niter):
            # vx = vx + mu*4*del2(vx) - gradMagSq.*(vx - egx);
            # vy = vy + mu*4*del2(vy) - gradMagSq.*(vy - egy);

            # 4*del2(U) is approx Laplacian?
            # MATLAB del2 computes finite difference approximation of Laplacian / 4.
            # So 4*del2(U) approx Laplacian(U).
            # We use scipy.ndimage.laplace(U).
            # Verify kernel [0 1 0; 1 -4 1; 0 1 0] which matches 4*del2 for grid spacing 1.

            # Using constant mode for boundary to match del2 default?
            Lvx = laplace(vx, mode="constant", cval=0.0)
            Lvy = laplace(
                vy, mode="constant", cval=0.0
            )  # check boundary conditions of del2?
            # MATLAB del2 handles boundaries.

            vx = vx + mu * Lvx - gradMagSq * (vx - egx)
            vy = vy + mu * Lvy - gradMagSq * (vy - egy)

        # Fx = vx; Fy = vy;
        Fx = vx
        Fy = vy

    return Fx, Fy
