import numpy as np
import matplotlib.pyplot as plt


def compare(f1, f2, scale=1):
    """
    Compute RMSE between two matrices and optionally display error histogram/image.
    Mirrors MATLAB compare.m behavior.
    """
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    if f1.shape != f2.shape:
        raise ValueError("Input matrices must have the same dimensions.")

    e = f1 - f2
    rmse = np.sqrt(np.mean(e**2))

    if rmse != 0 and scale > 0:
        emax = np.max(np.abs(e))
        if emax > 0:
            bins = int(np.ceil(emax))
            if bins < 1:
                bins = 1
            h, edges = np.histogram(e.ravel(), bins=bins)
            plt.figure()
            plt.bar(edges[:-1], h, color='k', width=edges[1] - edges[0])

            # Scale error image symmetrically
            emax = emax / scale
            e_scaled = (e + emax) / (2 * emax)
            e_scaled = np.clip(e_scaled, 0, 1)
            plt.figure()
            plt.imshow(e_scaled, cmap='gray')

    return rmse
